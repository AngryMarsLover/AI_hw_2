#!/usr/bin/env bash
# Anchors at 2x the stride of each level, after 1.5x (12-192 px) lost large-object accuracy.
.venv/bin/python src/train_torchvision.py --name retina_anchors_16 --fold 4 --epochs 12 --batch 4 --arch retina \
    --anchor-sizes 16,32,64,128,256 --notes "anchors 16-256 px, 2x stride; k-means sizes were 1.5x" \
    > work/logs/retina_anchors_16.log 2>&1
