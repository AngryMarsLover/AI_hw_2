"""Write training lists with every prepared frame (every 5th original), twice the default density.

Usage: python src/dense_lists.py --fold 4
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--frames", type=Path, default=Path("work/frames/images"))
    parser.add_argument("--lists", type=Path, default=Path("work/lists"))
    args = parser.parse_args()

    folds = pd.read_csv("folds/folds.csv")
    clips = folds.loc[folds["fold"] != args.fold, "clip"]
    frames = [p.resolve() for clip in clips for p in sorted((args.frames / clip).glob("*.jpg"))]
    (args.lists / f"fold{args.fold}_train_dense.txt").write_text("\n".join(map(str, frames)) + "\n")

    config = (args.lists / f"fold{args.fold}.yaml").read_text()
    config = config.replace(f"fold{args.fold}_train.txt", f"fold{args.fold}_train_dense.txt")
    (args.lists / f"fold{args.fold}_dense.yaml").write_text(config)
    print(f"fold {args.fold}: {len(frames)} train frames")


if __name__ == "__main__":
    main()
