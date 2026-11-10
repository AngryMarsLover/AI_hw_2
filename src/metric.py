"""Single scorer used for every model, so results stay comparable.

Assumes the leaderboard metric is COCO mAP@[.5:.95], with ignored regions and a
per-image cap of 500 detections as described on the competition data page.

Predictions are {image_key: array of rows [class, score, x, y, w, h]}, with 0-based
classes and top-left pixel boxes. This is the same layout as the submission file.
"""
import contextlib
import copy
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

MAX_DETS = 500
CLASS_NAMES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]


def image_key(file_name):
    """clip/0000001.jpg -> clip_0000001, matching the test image_id format."""
    path = Path(file_name)
    return f"{path.parent.name}_{path.stem}"


def load_gt(json_paths, clips=None, stride=1):
    """Merge COCO files into one dataset keyed by image_key, optionally filtered by clip.

    Keeps every `stride`-th frame of each clip, the same frames as prepare_data.py.
    Ignored regions have no class, but COCOeval only applies a crowd to detections of
    the same category, so each region is expanded to all 8 categories. The COCO files
    repeat identical regions within a frame, so duplicates are dropped first.
    """
    images, annotations = [], []
    for json_path in json_paths:
        with open(json_path) as f:
            coco = json.load(f)
        local_to_key = {}
        by_clip = {}
        for image in coco["images"]:
            clip = Path(image["file_name"]).parent.name
            if clips is None or clip in clips:
                by_clip.setdefault(clip, []).append(image)
        for clip_images in by_clip.values():
            for image in sorted(clip_images, key=lambda item: item["file_name"])[::stride]:
                key = image_key(image["file_name"])
                local_to_key[image["id"]] = key
                images.append({**image, "id": key})
        seen_regions = set()
        for ann in coco["annotations"]:
            key = local_to_key.get(ann["image_id"])
            if key is None:
                continue
            x, y, w, h = ann["bbox"]
            if ann.get("iscrowd", 0):
                if (key, x, y, w, h) in seen_regions:
                    continue
                seen_regions.add((key, x, y, w, h))
            record = {
                "image_id": key,
                "bbox": [x, y, w, h],
                "area": ann.get("area", w * h),
                "iscrowd": ann.get("iscrowd", 0),
            }
            categories = range(1, 9) if record["iscrowd"] else [ann["category_id"]]
            annotations.extend({**record, "category_id": c} for c in categories)
    for index, ann in enumerate(annotations, start=1):
        ann["id"] = index
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(CLASS_NAMES)]
    return {"images": images, "annotations": annotations, "categories": categories}


def cap_per_image(rows, max_dets=MAX_DETS):
    """Keep the highest-scoring rows; the leaderboard cap is per image, not per class."""
    rows = np.asarray(rows, dtype=float).reshape(-1, 6)
    return rows[np.argsort(-rows[:, 1], kind="stable")[:max_dets]]


def evaluate(gt, preds):
    """Return mAP, AP50, AP75, AP_small/medium/large and per-class AP."""
    results = []
    known = {image["id"] for image in gt["images"]}
    for key, rows in preds.items():
        if key not in known:
            continue
        for cls, score, x, y, w, h in cap_per_image(rows):
            results.append({
                "image_id": key,
                "category_id": int(cls) + 1,
                "score": float(score),
                "bbox": [float(x), float(y), float(w), float(h)],
            })
    names = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
    if not results:
        return {**dict.fromkeys(names, 0.0), "per_class": dict.fromkeys(CLASS_NAMES, 0.0)}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = copy.deepcopy(gt)
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(results)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.maxDets = [1, 10, MAX_DETS]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    metrics = {name: float(value) for name, value in zip(names, evaluator.stats[:6])}
    # precision axes: [IoU, recall, category, area range, maxDets]
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    # stats[0] is hard-coded to maxDets=100 inside COCOeval, which is not in our list and
    # reads as -1, so compute mAP at MAX_DETS from the precision array directly.
    metrics["mAP"] = float(precision[precision > -1].mean())
    metrics["per_class"] = {}
    for index, name in enumerate(CLASS_NAMES):
        values = precision[:, :, index]
        values = values[values > -1]
        metrics["per_class"][name] = float(values.mean()) if values.size else float("nan")
    return metrics
