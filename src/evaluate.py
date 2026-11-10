"""Score saved fold predictions with the shared scorer and append them to the experiment log.

Predictions are .npz files with one array per image_key, rows [class, score, x, y, w, h].
"""
import csv
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from metric import CLASS_NAMES, evaluate, load_gt

LOG_PATH = Path("results/experiments.csv")
EVAL_STRIDE = 5  # every prepared frame of the held-out clips, the same for every model


def fold_gt(fold, data_root=Path("data"), folds_csv=Path("folds/folds.csv")):
    """Ground truth for the validation frames of one fold."""
    folds = pd.read_csv(folds_csv)
    clips = set(folds.loc[folds["fold"] == fold, "clip"])
    paths = [data_root / split / "annotations_coco.json" for split in ["train", "val"]]
    return load_gt(paths, clips=clips, stride=EVAL_STRIDE)


def fold_frames(fold, data_root=Path("data"), folds_csv=Path("folds/folds.csv")):
    """Original (unpainted) image paths for the validation frames of one fold."""
    folds = pd.read_csv(folds_csv)
    frames = []
    for row in folds[folds["fold"] == fold].itertuples():
        frames += sorted((data_root / row.split / "images" / row.clip).glob("*.jpg"))[::EVAL_STRIDE]
    return frames


def save_preds(preds, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **preds)


def load_preds(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def log_result(name, fold, metrics, ms_per_image=None, notes="", public_score=None):
    """Append one row to results/experiments.csv, so each result links to a commit."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    row = {
        "time": datetime.now().isoformat(timespec="minutes"),
        "name": name,
        "fold": fold,
        **{k: round(metrics[k], 4) for k in ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]},
        "ms_per_image": None if ms_per_image is None else round(ms_per_image, 1),
        **{f"AP_{c}": round(metrics["per_class"][c], 4) for c in CLASS_NAMES},
        "public_score": public_score,
        "commit": commit,
        "notes": notes,
    }
    LOG_PATH.parent.mkdir(exist_ok=True)
    is_new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return row
