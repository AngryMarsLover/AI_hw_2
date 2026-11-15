#!/usr/bin/env bash
# Resume queue_gaps.sh after it was stopped: fold 2 is done, run the remaining two experiments.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python src/train_torchvision.py --name frcnn_rpn --arch frcnn --fold 4 --epochs 12 --batch 4 \
    --anchor-sizes 12,24,48,96,192 --rpn-post-train 4000 --rpn-post-test 2000 \
    --notes "k-means anchors, RPN proposals 4000 train / 2000 test" > work/logs/frcnn_rpn.log 2>&1
.venv/bin/python src/train_ultralytics.py --name rtdetr_nwd --fold 4 --epochs 30 --model rtdetr-l.pt --batch 2 \
    --nwd-ratio 0.5 --notes "loss: 0.5 GIoU + 0.5 NWD, C=37 px" --set optimizer=AdamW lr0=0.0001 \
    > work/logs/rtdetr_nwd.log 2>&1
