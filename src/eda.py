"""Exploratory analysis of train + val: writes figures to images/ and numbers to results/eda.json.

Usage: python src/eda.py --folds folds/folds.csv
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from metric import CLASS_NAMES  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, MUTED, GRID = "#fcfcfb", "#52514e", "#e4e3df"
SEQUENTIAL = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"]
)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titlelocation": "left",
})


def load_tables(data_root):
    """Merge train and val COCO files into image and annotation tables."""
    images, annotations = [], []
    for split in ["train", "val"]:
        with open(data_root / split / "annotations_coco.json") as f:
            coco = json.load(f)
        split_images = pd.DataFrame(coco["images"])
        split_anns = pd.DataFrame(coco["annotations"])
        split_images["split"] = split
        split_images["uid"] = split + "_" + split_images["id"].astype(str)
        split_anns["uid"] = split + "_" + split_anns["image_id"].astype(str)
        images.append(split_images)
        annotations.append(split_anns)
    images = pd.concat(images, ignore_index=True)
    anns = pd.concat(annotations, ignore_index=True)
    anns[["x", "y", "w", "h"]] = np.array(anns["bbox"].tolist(), dtype=float)
    anns = anns.merge(images[["uid", "width", "height", "video", "frame_id", "split"]], on="uid")
    anns["size"] = np.sqrt(anns["w"] * anns["h"])
    return images, anns


def plot_classes(objects, out):
    counts = objects["category_id"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = [CLASS_NAMES[i - 1] for i in counts.index]
    order = np.argsort(counts.to_numpy())
    ax.barh([labels[i] for i in order], counts.to_numpy()[order], color=BLUE, height=0.65)
    for y, i in enumerate(order):
        ax.text(counts.iloc[i], y, f"  {counts.iloc[i] / counts.sum():.1%}", va="center", color=MUTED)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("labelled boxes")
    ax.set_title("Cars and people dominate; bus is about 1 box in 100")
    fig.tight_layout()
    fig.savefig(out / "eda_classes.png", dpi=150)
    plt.close(fig)


def plot_sizes(objects, out):
    """Box size at native resolution and after resizing the frame to the training widths."""
    at_1280 = objects["size"] / objects["width"] * 1280
    bins = np.geomspace(2, 512, 60)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for values, label, color in [
        (objects["size"], "native frame", BLUE),
        (at_1280, "frame resized to 1280 wide", ORANGE),
        (at_1280 / 2, "frame resized to 640 wide", AQUA),
    ]:
        ax.hist(values, bins=bins, histtype="step", linewidth=2, label=label, color=color, density=True)
    ax.set_xscale("log")
    ax.set_xticks([4, 8, 16, 32, 64, 128, 256])
    ax.set_xticklabels([4, 8, 16, 32, 64, 128, 256])
    ax.set_yticks([])
    ax.set_xlabel("box size, sqrt(area) in pixels")
    ax.set_title("Resizing to 640 pushes most boxes below 16 px")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "eda_sizes.png", dpi=150)
    plt.close(fig)


def plot_per_clip(images, objects, out):
    """Density and class mix per clip, ordered by density."""
    frames = images.groupby("video").size()
    density = (objects.groupby("video").size() / frames).fillna(0).sort_values()
    mix = pd.crosstab(objects["video"], objects["category_id"], normalize="index").reindex(density.index)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, height_ratios=[1, 1.2])
    top.bar(range(len(density)), density.to_numpy(), color=BLUE, width=0.75)
    top.grid(axis="x", visible=False)
    top.set_ylabel("objects per frame")
    top.set_title("Density and class mix change strongly between clips")
    mesh = bottom.imshow(mix.to_numpy().T, aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=1)
    bottom.grid(False)
    bottom.set_yticks(range(8))
    bottom.set_yticklabels(CLASS_NAMES)
    bottom.set_xlabel("clips, sorted by density")
    bottom.set_xticks([])
    fig.colorbar(mesh, ax=[top, bottom], label="share of the clip's boxes", fraction=0.025)
    fig.savefig(out / "eda_per_clip.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return density


def plot_resolutions(images, test, out):
    """Share of frames per frame width in each split: test has sizes training never shows."""
    shares = pd.DataFrame({
        "train": images[images["split"] == "train"]["width"].value_counts(normalize=True),
        "val": images[images["split"] == "val"]["width"].value_counts(normalize=True),
        "test": test["width"].value_counts(normalize=True),
    }).fillna(0).sort_index()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    positions = np.arange(len(shares))
    for offset, (name, color) in zip([-0.27, 0, 0.27], [("train", BLUE), ("val", ORANGE), ("test", AQUA)]):
        ax.bar(positions + offset, shares[name], width=0.25, label=name, color=color)
    ax.set_xticks(positions)
    ax.set_xticklabels(shares.index)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("frame width, px")
    ax.set_ylabel("share of frames")
    ax.set_title("Test includes low-resolution clips and nothing above 1920 px")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "eda_resolution.png", dpi=150)
    plt.close(fig)
    return shares


def track_overlap(objects, lags):
    """Mean IoU of one object with itself `lag` frames later, shown as frame redundancy."""
    tracks = objects[["video", "track_id", "frame_id", "x", "y", "w", "h"]]
    result = {}
    for lag in lags:
        later = tracks.assign(frame_id=tracks["frame_id"] - lag)
        pairs = tracks.merge(later, on=["video", "track_id", "frame_id"], suffixes=("", "_b"))
        ix = np.minimum(pairs.x + pairs.w, pairs.x_b + pairs.w_b) - np.maximum(pairs.x, pairs.x_b)
        iy = np.minimum(pairs.y + pairs.h, pairs.y_b + pairs.h_b) - np.maximum(pairs.y, pairs.y_b)
        inter = ix.clip(lower=0) * iy.clip(lower=0)
        union = pairs.w * pairs.h + pairs.w_b * pairs.h_b - inter
        result[lag] = float((inter / union).mean())
    return result


def plot_overlap(overlap, out):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(list(overlap), list(overlap.values()), color=BLUE, linewidth=2, marker="o", markersize=8)
    for lag in [5, 10]:
        if lag in overlap:
            ax.annotate(f"{overlap[lag]:.2f}", (lag, overlap[lag]), textcoords="offset points", xytext=(6, 6), color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks(list(overlap))
    ax.set_xticklabels(list(overlap))
    ax.set_ylim(0, 1)
    ax.set_xlabel("frames apart")
    ax.set_ylabel("mean IoU of the same object")
    ax.set_title("Neighbouring frames are redundant; overlap fades by 10 frames apart")
    fig.tight_layout()
    fig.savefig(out / "eda_frame_overlap.png", dpi=150)
    plt.close(fig)


def ignored_share(images, ignored, n_frames=500, seed=0):
    """Share of frame area under ignored regions, as a true union on a sample of frames."""
    sample = images.sample(min(n_frames, len(images)), random_state=seed)
    groups = dict(tuple(ignored.groupby("uid")))
    shares = []
    for row in sample.itertuples():
        mask = np.zeros((row.height // 4, row.width // 4), dtype=bool)
        if row.uid in groups:
            for box in groups[row.uid].itertuples():
                mask[int(box.y) // 4: int(box.y + box.h) // 4 + 1, int(box.x) // 4: int(box.x + box.w) // 4 + 1] = True
        shares.append(mask.mean())
    return float(np.mean(shares)), float(np.percentile(shares, 90))


def check_yolo_labels(images, objects, data_root, n_frames=300, seed=0):
    """Compare YOLO text labels with COCO boxes; the two must describe the same objects."""
    sample = images.sample(n_frames, random_state=seed)
    groups = dict(tuple(objects.groupby("uid")))
    worst = 0.0
    for row in sample.itertuples():
        path = data_root / row.split / "labels" / Path(row.file_name.replace("images/", "", 1)).with_suffix(".txt")
        text = path.read_text().strip()
        rows = np.loadtxt(path, ndmin=2) if text else np.empty((0, 5))
        coco = groups.get(row.uid)
        n_coco = 0 if coco is None else len(coco)
        if len(rows) != n_coco:
            raise ValueError(f"{path}: {len(rows)} YOLO rows, {n_coco} COCO boxes")
        if not n_coco:
            continue
        boxes = rows[:, 1:] * [row.width, row.height, row.width, row.height]
        boxes[:, :2] -= boxes[:, 2:] / 2
        ours = np.c_[rows[:, 0] + 1, boxes]
        theirs = coco[["category_id", "x", "y", "w", "h"]].to_numpy(dtype=float)
        ours, theirs = (a[np.lexsort(np.round(a, 0).T)] for a in (ours, theirs))
        worst = max(worst, float(np.abs(ours - theirs).max()))
    return worst


def plot_ground_truth(images, objects, ignored, data_root, out, seed=1):
    """Draw converted YOLO labels and ignored regions on frames from four clips."""
    videos = pd.Series(images["video"].unique()).sample(4, random_state=seed)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, video in zip(axes.flat, videos):
        rows = images[images["video"] == video]
        row = rows.iloc[len(rows) // 2]
        image = cv2.imread(str(data_root / row["split"] / row["file_name"]))
        thickness = max(2, row["width"] // 700)
        stem = Path(row["file_name"]).with_suffix(".txt").relative_to("images")
        for folder, color, has_class in [("ignore", (52, 104, 235), False), ("labels", (0, 215, 90), True)]:
            path = data_root / row["split"] / folder / stem
            if not path.exists() or not path.read_text().strip():
                continue
            for values in np.loadtxt(path, ndmin=2):
                cx, cy, w, h = values[1:] if has_class else values
                x, y = (cx - w / 2) * row["width"], (cy - h / 2) * row["height"]
                corner = (int(x + w * row["width"]), int(y + h * row["height"]))
                cv2.rectangle(image, (int(x), int(y)), corner, color, thickness)
        ax.imshow(image[..., ::-1])
        ax.set_title(f"{video}  ({row['width']}x{row['height']})", fontsize=9)
        ax.axis("off")
    fig.suptitle("Converted labels (green) and ignored regions (orange) on sample frames", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "gt_check.png", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("images"))
    parser.add_argument("--results", type=Path, default=Path("results/eda.json"))
    args = parser.parse_args()

    images, anns = load_tables(args.data_root)
    test = pd.read_csv(args.data_root / "test_images.csv")
    objects = anns[anns["iscrowd"] == 0]
    # The COCO files repeat identical ignored regions within a frame.
    ignored = anns[anns["iscrowd"] == 1].drop_duplicates(["uid", "x", "y", "w", "h"])
    args.out.mkdir(exist_ok=True)
    args.results.parent.mkdir(exist_ok=True)

    plot_classes(objects, args.out)
    plot_sizes(objects, args.out)
    density = plot_per_clip(images, objects, args.out)
    shares = plot_resolutions(images, test, args.out)
    overlap = track_overlap(objects, [1, 2, 5, 10, 20, 50])
    plot_overlap(overlap, args.out)
    plot_ground_truth(images, objects, ignored, args.data_root, args.out)
    union_mean, union_p90 = ignored_share(images, ignored)
    at_1280 = objects["size"] / objects["width"] * 1280
    per_frame = objects.groupby("uid").size()

    summary = {
        "frames": len(images),
        "boxes": len(objects),
        "class_share": {CLASS_NAMES[k - 1]: float(v) for k, v in objects["category_id"].value_counts(normalize=True).items()},
        "size_native_p10_p50_p90": np.percentile(objects["size"], [10, 50, 90]).tolist(),
        "size_1280_p10_p50_p90": np.percentile(at_1280, [10, 50, 90]).tolist(),
        "share_below_16px": {"native": float((objects["size"] < 16).mean()), "at_1280": float((at_1280 < 16).mean()), "at_640": float((at_1280 < 32).mean())},
        "coco_area_share": {
            "small": float((objects["size"] < 32).mean()),
            "medium": float(((objects["size"] >= 32) & (objects["size"] < 96)).mean()),
            "large": float((objects["size"] >= 96).mean()),
        },
        "objects_per_frame": {"median": float(per_frame.median()), "p99": float(per_frame.quantile(0.99)), "max": int(per_frame.max())},
        "objects_per_frame_by_clip": {"min": float(density.min()), "median": float(density.median()), "max": float(density.max())},
        "ignored": {
            "regions_per_frame": len(ignored) / len(images),
            "frames_with_any": ignored["uid"].nunique() / len(images),
            "union_area_share_mean": union_mean,
            "union_area_share_p90": union_p90,
        },
        "same_object_iou_by_lag": overlap,
        "frame_width_share": shares.round(3).to_dict(),
        "yolo_vs_coco_max_abs_difference_px": check_yolo_labels(images, objects, args.data_root),
    }
    args.results.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
