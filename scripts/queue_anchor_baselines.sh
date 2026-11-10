#!/usr/bin/env bash
# Baselines for the two anchor-based families, on the same fold and frames as the YOLO baseline.
.venv/bin/python src/train_torchvision.py --name frcnn_base --arch frcnn --fold 4 --epochs 12 --batch 4 \
    --notes "baseline: COCO weights, default anchors, Smooth L1" > work/logs/frcnn_base.log 2>&1
.venv/bin/python src/train_torchvision.py --name retina_base --arch retina --fold 4 --epochs 12 --batch 4 \
    --notes "baseline: COCO weights, default anchors, focal 0.25/2.0, GIoU" > work/logs/retina_base.log 2>&1
