# Run OmniSAM Dynamic Mask

Only two steps are needed.

## 1. Install OmniSAM first

Set up the environment by following the official [OmniSAM](https://github.com/Ding-Zhong/OmniSAM) repository instructions, and make sure the required checkpoints are ready in the repo root.

This script expects:

- `sam/checkpoints/sam2.1_hiera_large.pt`
- `omnisam_l.pth`

Script file:

- `run_omnisam_dynamic_batch.py`

Expected dataset layout:

```text
<ROOT>/<CATEGORY>/<CLIP>/images/*.jpg
```

Outputs per clip:

```text
<ROOT>/<CATEGORY>/<CLIP>/omnisam_1920/masks/*.png
<ROOT>/<CATEGORY>/<CLIP>/dynamic_mask/*.png
```

## 2. Run with one command

From the OmniSAM repo root:

```bash
python run_omnisam_dynamic_batch.py /path/to/dataset_root \
  --gpu 0 \
  --ckpt omnisam_l.pth \
  --height 960 \
  --width 1920 \
  --patch-size 960 \
  --stride 320 \
  --passes 2 \
  --dynamic-classes 11,12 \
  --output-subdir omnisam_1920 \
  --dynamic-subdir dynamic_mask
```

All other options use the script defaults.

Add `--force` if you want to rerun clips that already have `dynamic_mask`.
