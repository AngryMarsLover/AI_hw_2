#!/usr/bin/env bash
# YOLO follow-ups on the best setup so far (scale 0.9, mixup 0.1), then the RT-DETR baseline.
# Dense lists have twice the frames, so 15 epochs keep the same number of training steps.
# RT-DETR runs out of memory at batch 4 on 6 GB, so it uses batch 2.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
    name=$1; shift
    .venv/bin/python src/train_ultralytics.py --name "$name" --fold 4 "$@" > "work/logs/$name.log" 2>&1
}

run yolo11s_aug_nwd --model yolo11s.pt --batch 4 --epochs 30 --nwd-ratio 0.5 \
    --notes "scale 0.9, mixup 0.1, 0.5 CIoU + 0.5 NWD" --set scale=0.9 mixup=0.1
run yolo11s_aug_lr --model yolo11s.pt --batch 4 --epochs 30 \
    --notes "scale 0.9, mixup 0.1, AdamW lr 0.0003" --set scale=0.9 mixup=0.1 optimizer=AdamW lr0=0.0003
run yolo11s_aug_dense --model yolo11s.pt --batch 4 --epochs 15 --lists _dense \
    --notes "scale 0.9, mixup 0.1, every 5th frame, 15 epochs" --set scale=0.9 mixup=0.1
run rtdetr_base --model rtdetr-l.pt --batch 2 --epochs 30 \
    --notes "baseline: COCO weights, AdamW lr 0.0001, batch 2" --set optimizer=AdamW lr0=0.0001
