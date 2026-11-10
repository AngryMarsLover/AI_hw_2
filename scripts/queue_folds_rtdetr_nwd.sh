#!/usr/bin/env bash
# Check the best YOLO setup against its baseline on two more folds, then RT-DETR with NWD.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
    name=$1; fold=$2; shift 2
    .venv/bin/python src/train_ultralytics.py --name "$name" --fold "$fold" --epochs 30 "$@" \
        > "work/logs/${name}_fold${fold}.log" 2>&1
}

for fold in 1 2; do
    run yolo11s_base "$fold" --model yolo11s.pt --batch 4 --notes "baseline: COCO weights, defaults"
    run yolo11s_aug_lr_nwd "$fold" --model yolo11s.pt --batch 4 --nwd-ratio 0.5 \
        --notes "scale 0.9, mixup 0.1, AdamW lr 0.0003, 0.5 CIoU + 0.5 NWD" \
        --set scale=0.9 mixup=0.1 optimizer=AdamW lr0=0.0003
done
run rtdetr_nwd 4 --model rtdetr-l.pt --batch 2 --nwd-ratio 0.5 \
    --notes "loss: 0.5 GIoU + 0.5 NWD, C=37 px" --set optimizer=AdamW lr0=0.0001
