"""Save raw test-set predictions of one run, so several models can be fused into one submission.

Usage:
  python src/predict_test.py --run runs/frcnn_anchors_fold4
  python src/predict_test.py --run runs/yolo11s_aug_lr_nwd_all --weights last.pt
  python src/predict_test.py --run runs/yolo11s_aug_lr_nwd_all --weights last.pt --mode tile
  python src/predict_test.py --fuse yolo11s_aug_lr_nwd_all frcnn_anchors_fold4 --fuse-weights 2 1 \
      --out submissions/ens_yolo_frcnn.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from evaluate import load_preds, save_preds
from inference_tests import fuse_all
from submission import write_submission

TEST_CSV = Path("data/test_images.csv")
PREDS = Path("work/preds")


def test_frames():
    return [Path("data") / name for name in pd.read_csv(TEST_CSV)["file_name"]]


def predict_run(run, weights, mode="plain"):
    """Torchvision runs keep args.json and best.pth; Ultralytics runs keep weights/*.pt."""
    if (run / "args.json").exists():
        from train_torchvision import build_model, predict
        args = argparse.Namespace(**json.loads((run / "args.json").read_text()))
        model = build_model(args)
        model.load_state_dict(torch.load(run / "weights" / (weights or "best.pth"), map_location="cpu"))
        return predict(model.cuda(), test_frames(), torch.device("cuda"))
    from inference_tests import flip_predict, tile_predict
    from predict_ultralytics import load_model, predict_frames
    model = load_model(run / "weights" / (weights or "best.pt"))
    predict = {"plain": predict_frames, "flip": flip_predict, "tile": tile_predict}[mode]
    return predict(model, test_frames(), 1280)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--weights", default=None, help="checkpoint file name inside the run's weights folder")
    parser.add_argument("--mode", default="plain", choices=["plain", "flip", "tile"], help="Ultralytics runs only")
    parser.add_argument("--fuse", nargs="*", default=[], help="names of saved test predictions to fuse")
    parser.add_argument("--fuse-weights", nargs="*", type=float, default=None)
    parser.add_argument("--iou-thr", type=float, default=0.65)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.run:
        preds = predict_run(args.run, args.weights, args.mode)
        suffix = "" if args.mode == "plain" else f"_{args.mode}"
        path = PREDS / f"test_{args.run.name}{suffix}.npz"
        save_preds(preds, path)
        print(f"saved {path}: {len(preds)} frames")
    if args.fuse:
        preds = fuse_all([load_preds(PREDS / f"test_{name}.npz") for name in args.fuse], args.fuse_weights,
                         args.iou_thr)
        args.out.parent.mkdir(exist_ok=True)
        write_submission(preds, TEST_CSV, args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
