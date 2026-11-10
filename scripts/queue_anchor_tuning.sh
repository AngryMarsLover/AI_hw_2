#!/usr/bin/env bash
# One change at a time against each baseline, on the same fold, frames and epochs.
# Anchor sizes come from src/anchors.py: k-means smallest cluster 13 px, doubling per level.
while pgrep -f '[q]ueue_anchor_baselines' > /dev/null; do sleep 30; done

run() {
    name=$1; shift
    .venv/bin/python src/train_torchvision.py --name "$name" --fold 4 --epochs 12 --batch 4 "$@" > "work/logs/$name.log" 2>&1
}

run frcnn_anchors --arch frcnn --anchor-sizes 12,24,48,96,192 --notes "anchors from k-means, 12-192 px"
run frcnn_giou --arch frcnn --box-loss giou --notes "loss: GIoU instead of Smooth L1"
run retina_anchors --arch retina --anchor-sizes 12,24,48,96,192 --notes "anchors from k-means, 12-192 px"
run retina_focal --arch retina --focal-gamma 1.5 --notes "loss: focal gamma 1.5 instead of 2.0"
