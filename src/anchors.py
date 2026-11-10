"""Choose anchor sizes and aspect ratios from the training boxes with k-means (as in YOLOv2).

Boxes are measured at the 1280-wide training scale, on the training clips of one fold only.
Sizes are clustered in log space, because detection levels double in size at each step.

Usage: python src/anchors.py --fold 4
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

TRAIN_WIDTH = 1280
TRAIN_HEIGHT = 720


def load_boxes(list_path):
    """Width and height in pixels at the training scale, from the YOLO labels of a frame list."""
    boxes = []
    for image_path in Path(list_path).read_text().split():
        label_path = Path(image_path.replace("/images/", "/labels/")).with_suffix(".txt")
        if label_path.read_text().strip():
            boxes.append(np.loadtxt(label_path, ndmin=2)[:, 3:5])
    return np.vstack(boxes) * [TRAIN_WIDTH, TRAIN_HEIGHT]


def cluster_1d(values, k, seed=42):
    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(np.log(values).reshape(-1, 1))
    return np.sort(np.exp(model.cluster_centers_.ravel()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results/anchors.json"))
    args = parser.parse_args()

    boxes = load_boxes(Path("work/lists") / f"fold{args.fold}_train.txt")
    sizes = cluster_1d(np.sqrt(boxes.prod(axis=1)), 5)
    ratios = cluster_1d(boxes[:, 1] / boxes[:, 0], 3)  # height / width, as in torchvision
    summary = {
        "fold": args.fold,
        "boxes": len(boxes),
        "sizes": [int(round(s)) for s in sizes],
        "ratios": [round(float(r), 2) for r in ratios],
        "share_below_default_32px": float((np.sqrt(boxes.prod(axis=1)) < 32).mean()),
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
