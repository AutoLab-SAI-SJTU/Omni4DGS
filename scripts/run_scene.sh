CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/openmvg.yaml \
source_path=data/openmvg/space \
model_path=outputs/openmvg/space

CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/openmvg.yaml \
source_path=data/openmvg/lake \
model_path=outputs/openmvg/lake



