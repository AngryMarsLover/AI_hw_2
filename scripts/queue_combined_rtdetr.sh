#!/usr/bin/env bash
# All three YOLO changes together, then RT-DETR without mosaic: rtdetr_base jumped when mosaic closed.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
    name=$1; shift
    .venv/bin/python src/train_ultralytics.py --name "$name" --fold 4 --epochs 30 "$@" > "work/logs/$name.log" 2>&1
}

run yolo11s_aug_lr_nwd --model yolo11s.pt --batch 4 --nwd-ratio 0.5 \
    --notes "scale 0.9, mixup 0.1, AdamW lr 0.0003, 0.5 CIoU + 0.5 NWD" --set scale=0.9 mixup=0.1 optimizer=AdamW lr0=0.0003
run rtdetr_nomosaic --model rtdetr-l.pt --batch 2 \
    --notes "no mosaic, otherwise rtdetr_base" --set optimizer=AdamW lr0=0.0001 mosaic=0.0
