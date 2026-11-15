#!/usr/bin/env bash
# Inference-time tests on fold 4, then raw test predictions and fused submissions.
# Waits for the final YOLO training to free the GPU.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src
while pgrep -f "scripts/queue_final.sh" > /dev/null; do sleep 60; done
PY=.venv/bin/python
YOLO4=runs/yolo11s_aug_lr_nwd_fold4/weights/best.pt

$PY src/inference_tests.py tta --mode flip --weights $YOLO4 > work/logs/tta_flip.log 2>&1
$PY src/inference_tests.py tta --mode ultralytics --weights $YOLO4 > work/logs/tta_ultralytics.log 2>&1
$PY src/inference_tests.py tile --weights $YOLO4 > work/logs/tile_2x2.log 2>&1

$PY src/predict_test.py --run runs/yolo11s_aug_lr_nwd_all --weights last.pt > work/logs/test_yolo_all.log 2>&1
$PY src/predict_test.py --run runs/frcnn_anchors_fold4 > work/logs/test_frcnn.log 2>&1
$PY src/predict_test.py --run runs/retina_focal_fold4 > work/logs/test_retina.log 2>&1
$PY src/predict_test.py --fuse yolo11s_aug_lr_nwd_all frcnn_anchors_fold4 --fuse-weights 2 1 \
    --out submissions/ens_yolo_all_frcnn.csv > work/logs/ens_test.log 2>&1
$PY src/predict_test.py --fuse yolo11s_aug_lr_nwd_all frcnn_anchors_fold4 retina_focal_fold4 --fuse-weights 2 1 1 \
    --out submissions/ens_yolo_all_frcnn_retina.csv >> work/logs/ens_test.log 2>&1
