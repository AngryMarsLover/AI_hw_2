#!/usr/bin/env bash
# Final model: the tuned YOLO11s on every clip. The fold runs peaked at epochs 10, 14 and 15 of a
# 30-epoch schedule, so this keeps that schedule and stops after epoch 14. No validation is left.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/python src/train_ultralytics.py --name yolo11s_aug_lr_nwd --all-clips --epochs 30 --stop-epoch 14 \
    --model yolo11s.pt --batch 4 --nwd-ratio 0.5 \
    --set scale=0.9 mixup=0.1 optimizer=AdamW lr0=0.0003 val=False > work/logs/yolo11s_aug_lr_nwd_all.log 2>&1
