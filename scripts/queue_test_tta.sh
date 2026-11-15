#!/usr/bin/env bash
# Test-set flip and tile predictions of the final YOLO, then the five-source fused submission
# (weights 2:2:2:1:1 scored 0.2763 on fold 4). Waits for queue_inference.sh to free the GPU.
export PYTHONPATH=src
while pgrep -f "scripts/queue_inference.sh" > /dev/null; do sleep 60; done
PY=.venv/bin/python
RUN="--run runs/yolo11s_aug_lr_nwd_all --weights last.pt"
$PY src/predict_test.py $RUN --mode flip > work/logs/test_yolo_flip.log 2>&1
$PY src/predict_test.py $RUN --mode tile > work/logs/test_yolo_tile.log 2>&1
$PY src/predict_test.py --fuse yolo11s_aug_lr_nwd_all yolo11s_aug_lr_nwd_all_tile yolo11s_aug_lr_nwd_all_flip \
    frcnn_anchors_fold4 retina_focal_fold4 --fuse-weights 2 2 2 1 1 \
    --out submissions/ens_all5.csv > work/logs/ens_all5.log 2>&1
