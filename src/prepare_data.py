"""Build the shared training copy of the data used by every detector.

For every `stride`-th frame of each train + val clip:
- paint ignored regions grey, so they cannot be learned as background, keeping the pixels
  of labelled boxes that overlap them;
- shrink frames wider than the widest test frame (1920 px), so object scales resemble test;
- write the image and its YOLO label to work/frames/{images,labels}/<clip>/<frame>.

Frame lists per fold go to work/lists, with an Ultralytics data yaml for each.
Validation lists keep every prepared frame; training lists take every `train_every`-th one.

Usage: python src/prepare_data.py --folds folds/folds.csv
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MAX_WIDTH = 1920
PAINT = 114  # Ultralytics' letterbox grey, so painted areas look like padding


def read_boxes(path):
    """Read normalised `... cx cy w h` rows; the ignore files have no class column."""
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 4))
    return np.loadtxt(path, ndmin=2)[:, -4:]


def to_pixels(boxes, width, height):
    """Normalised centres to integer x1, y1, x2, y2, clipped to the image."""
    scale = np.array([width, height, width, height])
    centre, size = boxes[:, :2], boxes[:, 2:]
    corners = np.hstack([centre - size / 2, centre + size / 2]) * scale
    corners = np.round(corners).astype(int)
    corners[:, [0, 2]] = corners[:, [0, 2]].clip(0, width)
    corners[:, [1, 3]] = corners[:, [1, 3]].clip(0, height)
    return corners


def paint_ignored(image, ignored, labelled):
    """Grey out ignored regions, then restore the pixels of labelled boxes."""
    height, width = image.shape[:2]
    original = image.copy()
    for x1, y1, x2, y2 in to_pixels(ignored, width, height):
        image[y1:y2, x1:x2] = PAINT
    for x1, y1, x2, y2 in to_pixels(labelled, width, height):
        image[y1:y2, x1:x2] = original[y1:y2, x1:x2]
    return image


def process_frame(task):
    image_path, label_path, ignore_path, out_image, out_label = task
    image = cv2.imread(str(image_path))
    label_text = label_path.read_text() if label_path.exists() else ""
    labelled = np.loadtxt(label_path, ndmin=2)[:, 1:] if label_text.strip() else np.empty((0, 4))
    image = paint_ignored(image, read_boxes(ignore_path), labelled)
    if image.shape[1] > MAX_WIDTH:
        new_height = round(image.shape[0] * MAX_WIDTH / image.shape[1])
        image = cv2.resize(image, (MAX_WIDTH, new_height), interpolation=cv2.INTER_AREA)
    out_image.parent.mkdir(parents=True, exist_ok=True)
    out_label.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_image), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    out_label.write_text(label_text)  # labels are normalised, so resizing keeps them valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--folds", type=Path, default=Path("folds/folds.csv"))
    parser.add_argument("--out", type=Path, default=Path("work"))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--train-every", type=int, default=2, help="training lists use stride * this")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    folds = pd.read_csv(args.folds)
    frames_root = args.out / "frames"
    tasks, kept = [], {}
    for row in folds.itertuples():
        split_root = args.data_root / row.split
        paths = sorted((split_root / "images" / row.clip).glob("*.jpg"))[:: args.stride]
        kept[row.clip] = [frames_root / "images" / row.clip / p.name for p in paths]
        for path in paths:
            out_image = frames_root / "images" / row.clip / path.name
            out_label = frames_root / "labels" / row.clip / f"{path.stem}.txt"
            if not out_image.exists():
                tasks.append((
                    path,
                    split_root / "labels" / row.clip / f"{path.stem}.txt",
                    split_root / "ignore" / row.clip / f"{path.stem}.txt",
                    out_image,
                    out_label,
                ))
    print(f"{len(tasks)} frames to write")
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(process_frame, tasks, chunksize=16))

    lists = args.out / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    names = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
    for fold in sorted(folds["fold"].unique()):
        train_clips = folds.loc[folds["fold"] != fold, "clip"]
        val_clips = folds.loc[folds["fold"] == fold, "clip"]
        train = [p.resolve() for clip in train_clips for p in kept[clip][:: args.train_every]]
        val = [p.resolve() for clip in val_clips for p in kept[clip]]
        (lists / f"fold{fold}_train.txt").write_text("\n".join(map(str, train)) + "\n")
        (lists / f"fold{fold}_val.txt").write_text("\n".join(map(str, val)) + "\n")
        (lists / f"fold{fold}.yaml").write_text(
            f"path: {lists.resolve()}\ntrain: fold{fold}_train.txt\nval: fold{fold}_val.txt\n"
            f"nc: {len(names)}\nnames: {names}\n"
        )
        print(f"fold {fold}: {len(train)} train frames, {len(val)} val frames")


if __name__ == "__main__":
    main()
