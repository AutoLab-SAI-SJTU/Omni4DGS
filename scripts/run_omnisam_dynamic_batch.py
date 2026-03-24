#!/usr/bin/env python3
"""
Single-file OmniSAM batch runner for fixed dataset layout:

  <ROOT>/<CATEGORY>/<CLIP>/images

For each clip, it writes:
  <CLIP>/omnisam_1920/masks/*.png      (class-id mask)
  <CLIP>/dynamic_mask/*.png            (binary dynamic mask)

This script merges:
- inference (previous inference_custom.py usage)
- dynamic mask extraction (previous extract_dynamic_mask.py usage)
"""

import argparse
import os
from pathlib import Path
from typing import Any, List, Sequence, Tuple

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SAM_CFGS_KEYS = ("sam2_l", "sam2_b+", "sam2_s", "sam2_t")

# Lazy-loaded runtime deps (assigned in lazy_import_runtime_deps).
cv2 = None
np = None
torch = None
Image = None
transforms = None
OmniSAM = None
OmniSAM_adapt = None
build_sam2 = None
SAM_CFGS = None
merge_width_scatter_np = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run OmniSAM segmentation + dynamic-mask extraction in one script."
    )
    p.add_argument("root", help="Dataset root: <ROOT>/<CATEGORY>/<CLIP>/images")
    p.add_argument("--gpu", default=None, help="GPU id, e.g. 0. Default: keep current env.")
    p.add_argument("--ckpt", default="omnisam_l.pth", help="OmniSAM .pth checkpoint path.")
    p.add_argument("--backbone", default="sam2_l", choices=list(SAM_CFGS_KEYS))
    p.add_argument("--model-type", default="adapt", choices=["adapt", "base"])
    p.add_argument("--num-classes", type=int, default=13)
    p.add_argument("--num-maskmem", type=int, default=9)

    # Fixed pipeline defaults (same as your current shell script)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--patch-size", type=int, default=960)
    p.add_argument("--stride", type=int, default=320)
    p.add_argument("--passes", type=int, default=2, choices=[1, 2])

    p.add_argument("--dynamic-classes", default="11,12", help="CSV class ids, e.g. 11,12")
    p.add_argument("--open-ksize", type=int, default=3)
    p.add_argument("--close-ksize", type=int, default=5)
    p.add_argument("--min-area", type=int, default=200)

    p.add_argument("--output-subdir", default="omnisam_1920")
    p.add_argument("--dynamic-subdir", default="dynamic_mask")

    p.add_argument("--force", action="store_true", help="Rerun even when dynamic_mask is complete.")
    p.add_argument("--limit", type=int, default=0, help="Process first N clips only (0=all).")
    p.add_argument("--dry-run", action="store_true", help="Only print planned actions.")
    return p.parse_args()


def parse_class_csv(text: str) -> List[int]:
    out: List[int] = []
    for x in text.split(","):
        t = x.strip()
        if not t:
            continue
        out.append(int(t))
    if not out:
        raise ValueError("--dynamic-classes is empty")
    return out


def lazy_import_runtime_deps() -> None:
    global cv2, np, torch, Image, transforms
    global OmniSAM, OmniSAM_adapt, build_sam2, SAM_CFGS, merge_width_scatter_np

    import cv2 as _cv2
    import numpy as _np
    import torch as _torch
    from PIL import Image as _Image
    from torchvision import transforms as _transforms

    from model.model import OmniSAM as _OmniSAM, OmniSAM_adapt as _OmniSAM_adapt
    from sam.sam.build_sam import build_sam2 as _build_sam2
    from utils.build import SAM_CFGS as _SAM_CFGS
    from utils.tools import merge_width_scatter_np as _merge_width_scatter_np

    cv2 = _cv2
    np = _np
    torch = _torch
    Image = _Image
    transforms = _transforms

    OmniSAM = _OmniSAM
    OmniSAM_adapt = _OmniSAM_adapt
    build_sam2 = _build_sam2
    SAM_CFGS = _SAM_CFGS
    merge_width_scatter_np = _merge_width_scatter_np


def list_images(image_dir: Path) -> List[Path]:
    return sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS])


def count_pngs(d: Path) -> int:
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".png")


def discover_clips(root: Path) -> List[Tuple[Path, Path]]:
    # fixed layout: <root>/<category>/<clip>/images
    clips: List[Tuple[Path, Path]] = []
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        for clip_dir in sorted(cat_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            image_dir = clip_dir / "images"
            if image_dir.is_dir():
                clips.append((clip_dir, image_dir))
    return clips


def build_kernels(open_ksize: int, close_ksize: int):
    open_kernel = None
    close_kernel = None
    if open_ksize > 0:
        if open_ksize % 2 == 0:
            raise ValueError("--open-ksize must be odd or 0")
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    if close_ksize > 0:
        if close_ksize % 2 == 0:
            raise ValueError("--close-ksize must be odd or 0")
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    return open_kernel, close_kernel


def label_to_dynamic_mask(
    label: Any,
    class_ids: Sequence[int],
    open_kernel,
    close_kernel,
    min_area: int,
) -> Any:
    dyn = np.isin(label, class_ids).astype(np.uint8) * 255

    if open_kernel is not None:
        dyn = cv2.morphologyEx(dyn, cv2.MORPH_OPEN, open_kernel)
    if close_kernel is not None:
        dyn = cv2.morphologyEx(dyn, cv2.MORPH_CLOSE, close_kernel)

    if min_area > 0:
        num, comp, stats, _ = cv2.connectedComponentsWithStats((dyn > 0).astype(np.uint8), connectivity=8)
        for cid in range(1, num):
            if stats[cid, cv2.CC_STAT_AREA] < min_area:
                dyn[comp == cid] = 0

    return dyn


def load_checkpoint(model, ckpt_path: Path) -> None:
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    inc = model.load_state_dict(state, strict=False)
    if inc.missing_keys or inc.unexpected_keys:
        print(
            f"[warn] missing_keys={len(inc.missing_keys)} "
            f"unexpected_keys={len(inc.unexpected_keys)}"
        )


def make_patches(img_t: Any, patch_size: int, stride: int, passes: int):
    _, _, w = img_t.shape
    if w < patch_size:
        raise ValueError(f"image width {w} < patch_size {patch_size}")

    starts = list(range(0, w - patch_size + 1, stride))
    patches = [img_t[:, :, x : x + patch_size] for x in starts]
    if passes == 2:
        patches = patches + list(reversed(patches))
    return patches, len(starts)


def infer_one(
    model,
    img_t: Any,
    num_classes: int,
    patch_size: int,
    stride: int,
    passes: int,
    device: str,
) -> Any:
    with torch.no_grad():
        patches, frames = make_patches(img_t, patch_size, stride, passes)
        _, h, w_full = img_t.shape
        b = len(patches)

        logits_np = np.zeros((1, b, num_classes, h, patch_size), dtype=np.float32)

        # pass 1
        model.memory_bank._init_output_dict_source()
        for i in range(frames):
            out = model(i, patches[i].unsqueeze(0).to(device))
            if isinstance(out, (tuple, list)):
                out = out[0]
            logits_np[0, i] = out.detach().cpu().numpy()

        # pass 2
        if passes == 2:
            model.memory_bank._init_output_dict_source()
            for i in range(frames):
                out = model(i, patches[frames + i].unsqueeze(0).to(device))
                if isinstance(out, (tuple, list)):
                    out = out[0]
                logits_np[0, frames + i] = out.detach().cpu().numpy()

        pred = merge_width_scatter_np(
            logits_np=logits_np,
            orig_size=(h, w_full),
            W_patch=patch_size,
            stride_w=stride,
            passes=passes,
            device=device,
            return_probs=False,
        )[0]

        return pred.astype(np.uint8)


def prepare_model(repo_root: Path, args: argparse.Namespace, device: str):
    sam_ckpt_rel, model_cfg = SAM_CFGS[args.backbone]

    sam_ckpt = Path(sam_ckpt_rel)
    if not sam_ckpt.is_absolute():
        sam_ckpt = repo_root / sam_ckpt

    model_cfg = model_cfg.lstrip("/")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = repo_root / ckpt_path

    if not sam_ckpt.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {sam_ckpt}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"OmniSAM checkpoint not found: {ckpt_path}")

    sam = build_sam2(model_cfg, str(sam_ckpt), device=device)
    if args.model_type == "adapt":
        model = OmniSAM_adapt(sam, num_classes=args.num_classes, num_maskmem=args.num_maskmem).to(device)
    else:
        model = OmniSAM(sam, num_classes=args.num_classes, num_maskmem=args.num_maskmem).to(device)

    model.eval()
    load_checkpoint(model, ckpt_path)
    return model


def main() -> int:
    args = parse_args()

    if not args.dry_run:
        try:
            lazy_import_runtime_deps()
        except Exception as e:
            print(f"[error] import runtime dependencies failed: {e}")
            print("[hint] install OmniSAM runtime deps (torch/torchvision/opencv-python/pillow/numpy/safetensors/timm/wandb).")
            return 1

    if args.gpu is not None and args.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"[error] root not found: {root}")
        return 1

    repo_root = Path(__file__).resolve().parent

    class_ids = parse_class_csv(args.dynamic_classes)
    open_kernel = None
    close_kernel = None
    if not args.dry_run:
        open_kernel, close_kernel = build_kernels(args.open_ksize, args.close_ksize)

    clips = discover_clips(root)
    if args.limit > 0:
        clips = clips[: args.limit]

    if not clips:
        print(f"[error] no clips found under: {root}")
        return 1

    print(f"[info] root={root}")
    print(f"[info] clips={len(clips)}")
    print(f"[info] dynamic_classes={class_ids}")

    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    print(f"[info] device={device}")

    model = None
    tfm = None
    if not args.dry_run:
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    total = 0
    ok = 0
    skip = 0
    fail = 0

    for clip_dir, image_dir in clips:
        total += 1
        images = list_images(image_dir)
        img_n = len(images)

        if img_n == 0:
            print(f"[skip] {clip_dir} (no images)")
            skip += 1
            continue

        out_dir = clip_dir / args.output_subdir
        mask_dir = out_dir / "masks"
        dyn_dir = clip_dir / args.dynamic_subdir

        if not args.force:
            dyn_n = count_pngs(dyn_dir)
            if dyn_n >= img_n:
                print(f"[skip] {clip_dir} (dynamic_mask complete: {dyn_n}/{img_n})")
                skip += 1
                continue

        print(f"[run] {clip_dir} ({img_n} images)")

        if args.dry_run:
            continue

        if model is None:
            model = prepare_model(repo_root, args, device)

        mask_dir.mkdir(parents=True, exist_ok=True)
        dyn_dir.mkdir(parents=True, exist_ok=True)

        clip_failed = False
        for p in images:
            try:
                img = Image.open(p).convert("RGB")
                resized = img.resize((args.width, args.height), Image.BICUBIC)
                img_t = tfm(resized)

                _, _, w_full = img_t.shape
                if (w_full - args.patch_size) % args.stride != 0:
                    raise ValueError(
                        f"(W_full - patch_size) % stride != 0: "
                        f"W_full={w_full}, patch_size={args.patch_size}, stride={args.stride}"
                    )

                pred = infer_one(
                    model=model,
                    img_t=img_t,
                    num_classes=args.num_classes,
                    patch_size=args.patch_size,
                    stride=args.stride,
                    passes=args.passes,
                    device=device,
                )

                Image.fromarray(pred, mode="L").save(mask_dir / f"{p.stem}.png")

                dyn = label_to_dynamic_mask(
                    label=pred,
                    class_ids=class_ids,
                    open_kernel=open_kernel,
                    close_kernel=close_kernel,
                    min_area=args.min_area,
                )
                Image.fromarray(dyn, mode="L").save(dyn_dir / f"{p.stem}.png")

            except Exception as e:
                clip_failed = True
                print(f"[fail] {clip_dir} ({p.name}) {e}")
                break

        if clip_failed:
            fail += 1
            continue

        dyn_n = count_pngs(dyn_dir)
        if dyn_n >= img_n:
            ok += 1
            print(f"[ok] {clip_dir} (dynamic_mask: {dyn_n}/{img_n})")
        else:
            fail += 1
            print(f"[fail] {clip_dir} (dynamic_mask incomplete: {dyn_n}/{img_n})")

    print(f"[summary] total={total} ok={ok} skip={skip} fail={fail}")
    return 2 if fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
