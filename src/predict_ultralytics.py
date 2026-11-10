"""Inference with a trained YOLO or RT-DETR model, for fold scoring and test submissions.

Usage: python src/predict_ultralytics.py --weights runs/x/weights/best.pt --out submissions/x.csv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import RTDETR, YOLO

from metric import image_key
from submission import write_submission

CONF = 0.001  # low threshold: AP is computed over the whole ranked list
IOU = 0.65
BATCH = 8


def load_model(weights):
    """RT-DETR needs its own wrapper; the checkpoint says which model it holds."""
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    is_detr = "RTDETR" in type(checkpoint["model"]).__name__
    return (RTDETR if is_detr else YOLO)(str(weights))


def to_rows(result):
    """One Ultralytics result to rows [class, score, x, y, w, h] in original pixels."""
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    return np.column_stack([
        boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(),
        xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1],
    ])


def predict_frames(model, frames, imgsz=1280, batch=BATCH, conf=CONF, iou=IOU):
    """Predict image paths in batches; boxes come back in each frame's original pixels."""
    preds = {}
    for start in range(0, len(frames), batch):
        chunk = [str(p) for p in frames[start:start + batch]]
        results = model.predict(
            chunk, imgsz=imgsz, conf=conf, iou=iou, max_det=500, batch=batch, device=0, verbose=False
        )
        for result in results:
            preds[image_key(result.path)] = to_rows(result)
    return preds


def time_inference(model, frames, imgsz, n_images=100):
    """Milliseconds per image at batch size 1, the same procedure for every model."""
    sample = [str(p) for p in frames[:: max(1, len(frames) // n_images)]][:n_images]
    model.predict(sample[0], imgsz=imgsz, device=0, verbose=False)  # warm-up
    torch.cuda.synchronize()
    start = time.perf_counter()
    for path in sample:
        model.predict(path, imgsz=imgsz, conf=CONF, iou=IOU, max_det=500, device=0, verbose=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / len(sample) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    test = pd.read_csv(args.data_root / "test_images.csv")
    frames = [args.data_root / name for name in test["file_name"]]
    preds = predict_frames(load_model(args.weights), frames, imgsz=args.imgsz)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(preds, args.data_root / "test_images.csv", args.out)
    print(f"wrote {args.out}: {sum(map(len, preds.values()))} detections on {len(preds)} frames")


if __name__ == "__main__":
    main()
