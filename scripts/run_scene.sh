CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/openmvg.yaml \
source_path=data/waymo_scenes/0145050 \
model_path=eval_output/waymo_nvs/0145050

CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/openmvg.yaml \
source_path=data/waymo_scenes/0147030 \
model_path=eval_output/waymo_nvs/0147030

CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/openmvg.yaml \
source_path=data/waymo_scenes/0158150 \
model_path=eval_output/waymo_nvs/0158150

