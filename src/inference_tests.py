"""Inference-time experiments on fold 4, scored with the shared scorer and logged like training runs.

  ensemble  fuse saved fold predictions of two models with weighted box fusion (CPU only)
  tta       Ultralytics test-time augmentation (flips and scales), or horizontal flip fused by WBF
  tile      full frame plus a 2x2 grid of overlapping tiles, each upscaled to the model size

Usage:
  python src/inference_tests.py ensemble --preds yolo11s_aug_lr_nwd_fold4 frcnn_anchors_fold4 --weights 2 1
  python src/inference_tests.py tta --weights runs/yolo11s_aug_lr_nwd_fold4/weights/best.pt --mode flip
  python src/inference_tests.py tile --weights runs/yolo11s_aug_lr_nwd_fold4/weights/best.pt
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate import fold_frames, fold_gt, load_preds, log_result, save_preds
from fusion import wbf
from metric import cap_per_image, evaluate, image_key
from predict_ultralytics import CONF, IOU, load_model, predict_frames, to_rows

PREDS = Path("work/preds")


def fuse_all(sources, weights, iou_thr):
    """WBF per image over several prediction dicts with the same keys."""
    keys = set().union(*sources)
    empty = np.empty((0, 6))
    return {k: cap_per_image(wbf([s.get(k, empty) for s in sources], weights, iou_thr)) for k in keys}


def flip_predict(model, frames, imgsz):
    """Predictions on horizontally flipped frames, mapped back to original coordinates."""
    preds = {}
    for path in frames:
        image = Image.open(path).convert("RGB")
        width = image.width
        flipped = np.ascontiguousarray(np.asarray(image)[:, ::-1, ::-1])  # mirror, RGB -> BGR for Ultralytics
        result = model.predict(flipped, imgsz=imgsz, conf=CONF, iou=IOU, max_det=500, device=0, verbose=False)[0]
        rows = to_rows(result)
        rows[:, 2] = width - rows[:, 2] - rows[:, 4]
        preds[image_key(path)] = rows
    return preds


def tile_predict(model, frames, imgsz, overlap=0.2, margin=4):
    """Predict a 2x2 grid of overlapping tiles; drop boxes cut by an inner tile edge."""
    preds = {}
    for path in frames:
        image = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1]
        height, width = image.shape[:2]
        tile_w, tile_h = int(width / 2 * (1 + overlap)), int(height / 2 * (1 + overlap))
        rows = []
        for x0 in (0, width - tile_w):
            for y0 in (0, height - tile_h):
                tile = np.ascontiguousarray(image[y0:y0 + tile_h, x0:x0 + tile_w])
                result = model.predict(tile, imgsz=imgsz, conf=CONF, iou=IOU, max_det=500, device=0, verbose=False)[0]
                part = to_rows(result)
                x1, y1, x2, y2 = part[:, 2], part[:, 3], part[:, 2] + part[:, 4], part[:, 3] + part[:, 5]
                # An inner edge is one that is not also the frame edge; boxes touching it are partial objects.
                cut = np.zeros(len(part), dtype=bool)
                if x0 > 0:
                    cut |= x1 < margin
                if x0 + tile_w < width:
                    cut |= x2 > tile_w - margin
                if y0 > 0:
                    cut |= y1 < margin
                if y0 + tile_h < height:
                    cut |= y2 > tile_h - margin
                part = part[~cut]
                part[:, 2] += x0
                part[:, 3] += y0
                rows.append(part)
        preds[image_key(path)] = np.concatenate(rows)
    return preds


def score(name, preds, notes):
    save_preds(preds, PREDS / f"{name}_fold4.npz")
    row = log_result(name, 4, evaluate(fold_gt(4), preds), notes=notes)
    print({k: row[k] for k in ["name", "mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test", choices=["ensemble", "tta", "tile"])
    parser.add_argument("--name", default=None)
    parser.add_argument("--preds", nargs="*", default=[], help="saved prediction names in work/preds")
    parser.add_argument("--weights", nargs="*", default=[], help="model weights, or fusion weights for ensemble")
    parser.add_argument("--mode", default="flip", choices=["flip", "ultralytics"])
    parser.add_argument("--iou-thr", type=float, default=0.65)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--base", default="yolo11s_aug_lr_nwd_fold4", help="saved plain predictions of the model")
    args = parser.parse_args()

    if args.test == "ensemble":
        sources = [load_preds(PREDS / f"{name}.npz") for name in args.preds]
        weights = [float(w) for w in args.weights] or None
        name = args.name or "ens_" + "+".join(args.preds).replace("_fold4", "")
        score(name, fuse_all(sources, weights, args.iou_thr),
              f"WBF of {', '.join(args.preds)}, weights {weights}, IoU {args.iou_thr}")
        return

    model = load_model(args.weights[0])
    frames = fold_frames(4)
    base = load_preds(PREDS / f"{args.base}.npz")
    if args.test == "tta" and args.mode == "ultralytics":
        preds = {}
        for start in range(0, len(frames), 8):
            for result in model.predict([str(p) for p in frames[start:start + 8]], imgsz=args.imgsz, conf=CONF,
                                        iou=IOU, max_det=500, augment=True, device=0, verbose=False):
                preds[image_key(result.path)] = to_rows(result)
        score(args.name or "tta_ultralytics", preds, "Ultralytics augment=True (flip, scales 1/0.83/0.67)")
    elif args.test == "tta":
        flipped = flip_predict(model, frames, args.imgsz)
        save_preds(flipped, PREDS / "flip_only_fold4.npz")
        score(args.name or "tta_flip", fuse_all([base, flipped], None, args.iou_thr),
              f"original + horizontal flip, WBF IoU {args.iou_thr}")
    else:
        tiles = tile_predict(model, frames, args.imgsz)
        save_preds(tiles, PREDS / "tiles_only_fold4.npz")
        score(args.name or "tile_2x2", fuse_all([base, tiles], None, args.iou_thr),
              f"full frame + 2x2 tiles (20% overlap) at {args.imgsz}, WBF IoU {args.iou_thr}")


if __name__ == "__main__":
    main()
