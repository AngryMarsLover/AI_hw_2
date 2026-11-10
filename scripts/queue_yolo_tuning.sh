#!/usr/bin/env bash
# One change at a time against yolo11s_base, on the same fold, frames and epochs.
# P2 runs out of memory at batch 3 on 6 GB, so it uses batch 2.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
    name=$1; shift
    .venv/bin/python src/train_ultralytics.py --name "$name" --fold 4 --epochs 30 "$@" > "work/logs/$name.log" 2>&1
}

run yolo11s_p2 --model configs/yolo11s-p2.yaml --pretrained yolo11s.pt --batch 2 --notes "P2 head, batch 2"
run yolo11s_nwd --model yolo11s.pt --batch 4 --nwd-ratio 0.5 --notes "loss: 0.5 CIoU + 0.5 NWD, C=37 px"
run yolo11s_aug --model yolo11s.pt --batch 4 --notes "augmentation: scale 0.9, mixup 0.1" --set scale=0.9 mixup=0.1
run yolo11s_960 --model yolo11s.pt --batch 4 --imgsz 960 --notes "input 960 instead of 1280"
