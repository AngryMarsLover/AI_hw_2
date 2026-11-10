"""Build fixed folds over train + val clips, grouped by source video.

VisDrone clip names look like uav0000013_00000_v, and clips sharing the first part may
come from one flight, so they stay in the same fold. Among random group assignments,
keep the one with the most even frame counts and class mix across folds.

Usage: python src/folds.py --data-root data --out folds/folds.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def clip_stats(json_paths):
    """One row per clip: frame count and box count per class."""
    rows = {}
    for split, json_path in json_paths.items():
        with open(json_path) as f:
            coco = json.load(f)
        image_clip = {}
        for image in coco["images"]:
            clip = Path(image["file_name"]).parent.name
            image_clip[image["id"]] = clip
            row = rows.setdefault(clip, {"clip": clip, "split": split, "frames": 0, **{f"c{i}": 0 for i in range(1, 9)}})
            row["frames"] += 1
        for ann in coco["annotations"]:
            if not ann.get("iscrowd", 0):
                rows[image_clip[ann["image_id"]]][f"c{ann['category_id']}"] += 1
    stats = pd.DataFrame(rows.values())
    stats["group"] = stats["clip"].str.split("_").str[0]
    return stats


def assign_folds(stats, n_folds=5, n_trials=20000, seed=42):
    """Random search over group-to-fold assignments, minimising imbalance across folds."""
    class_cols = [f"c{i}" for i in range(1, 9)]
    groups = stats.groupby("group")[["frames", *class_cols]].sum()
    values = groups.to_numpy(dtype=float)
    values = values / values.sum(axis=0).clip(min=1)  # each column becomes a share of its total

    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    for _ in range(n_trials):
        assignment = np.resize(np.arange(n_folds), len(groups))
        rng.shuffle(assignment)
        totals = np.stack([values[assignment == k].sum(axis=0) for k in range(n_folds)])
        # Frame balance counts as much as all class columns together.
        cost = totals[:, 0].std() * 8 + totals[:, 1:].std(axis=0).sum()
        if cost < best_cost:
            best, best_cost = assignment, cost
    return dict(zip(groups.index, best.tolist())), best_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("folds/folds.csv"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = clip_stats({
        "train": args.data_root / "train/annotations_coco.json",
        "val": args.data_root / "val/annotations_coco.json",
    })
    print(f"{len(stats)} clips, {stats['group'].nunique()} source videos")
    crossing = stats.groupby("group")["split"].nunique()
    print("source videos present in both train and val:", list(crossing[crossing > 1].index))

    mapping, cost = assign_folds(stats, args.n_folds, seed=args.seed)
    stats["fold"] = stats["group"].map(mapping)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.out, index=False)

    class_cols = [f"c{i}" for i in range(1, 9)]
    print(f"balance cost {cost:.4f}")
    print(stats.groupby("fold")[["frames", *class_cols]].sum().assign(clips=stats.groupby("fold").size()))


if __name__ == "__main__":
    main()
