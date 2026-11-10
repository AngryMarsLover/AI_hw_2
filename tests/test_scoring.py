"""Checks for the scorer and submission writer on small synthetic data."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from folds import assign_folds  # noqa: E402
from nwd import nwd  # noqa: E402
from metric import MAX_DETS, cap_per_image, evaluate, image_key, load_gt  # noqa: E402
from submission import format_rows, validate_submission, write_submission  # noqa: E402

CAR_BOX = [100, 100, 40, 20]
PERSON_BOX = [300, 300, 10, 30]
IGNORED = [500, 500, 100, 100]
KEY = "clip_a_0000001"


@pytest.fixture
def gt(tmp_path):
    """One frame: a car and a person, plus an ignored region labelled with one class only."""
    coco = {
        "images": [{"id": 7, "file_name": "images/clip_a/0000001.jpg", "width": 1000, "height": 800}],
        "annotations": [
            {"id": 1, "image_id": 7, "category_id": 3, "bbox": CAR_BOX, "area": 800, "iscrowd": 0},
            {"id": 2, "image_id": 7, "category_id": 1, "bbox": PERSON_BOX, "area": 300, "iscrowd": 0},
            {"id": 3, "image_id": 7, "category_id": 3, "bbox": IGNORED, "area": 10000, "iscrowd": 1},
        ],
        "categories": [],
    }
    path = tmp_path / "coco.json"
    path.write_text(json.dumps(coco))
    return load_gt([path])


def perfect():
    # 0-based classes in predictions, 1-based in COCO: car is 2, person is 0.
    return [[2, 0.5, *CAR_BOX], [0, 0.5, *PERSON_BOX]]


def test_image_key_matches_test_ids():
    assert image_key("test/images/clip_001/0000001.jpg") == "clip_001_0000001"


def test_perfect_predictions_score_one(gt):
    assert evaluate(gt, {KEY: perfect()})["mAP"] == pytest.approx(1.0)


def test_wrong_class_ids_score_zero(gt):
    # Guards the 0-based/1-based trap: person predicted as class 1 is a wrong class.
    rows = [[3, 0.5, *CAR_BOX], [1, 0.5, *PERSON_BOX]]
    assert evaluate(gt, {KEY: rows})["mAP"] == pytest.approx(0.0)


def test_detections_in_ignored_region_are_not_penalised_for_any_class(gt):
    rows = perfect() + [[2, 0.9, 510, 510, 50, 50], [0, 0.9, 510, 510, 50, 50]]
    assert evaluate(gt, {KEY: rows})["mAP"] == pytest.approx(1.0)


def test_false_positive_outside_ignored_region_lowers_score(gt):
    rows = perfect() + [[2, 0.9, 700, 100, 50, 50]]
    assert evaluate(gt, {KEY: rows})["mAP"] < 1.0


def test_empty_predictions_score_zero(gt):
    assert evaluate(gt, {})["mAP"] == 0.0


def test_cap_is_per_image_and_keeps_best_scores():
    rows = [[i % 8, i / 1000, 0, 0, 5, 5] for i in range(MAX_DETS + 100)]
    capped = cap_per_image(rows)
    assert len(capped) == MAX_DETS
    assert capped[:, 1].min() == pytest.approx(100 / 1000)


def test_format_clips_boxes_and_drops_empty_ones():
    string = format_rows([[2, 0.5, -10, 790, 50, 50], [1, 0.4, 2000, 0, 5, 5]], 1000, 800)
    assert string == "2 0.50000 0.00 790.00 40.00 10.00"


def test_format_drops_boxes_that_round_to_zero_size():
    assert format_rows([[2, 0.5, 10, 10, 0.001, 20]], 1000, 800) == ""


@pytest.fixture
def test_csv(tmp_path):
    path = tmp_path / "test_images.csv"
    pd.DataFrame({
        "image_id": ["clip_001_0000001", "clip_001_0000002"],
        "file_name": ["a", "b"], "clip": ["clip_001"] * 2, "frame": [1, 2],
        "width": [1000, 1000], "height": [800, 800],
    }).to_csv(path, index=False)
    return path


def test_submission_round_trip_with_empty_frame(tmp_path, test_csv):
    out = tmp_path / "sub.csv"
    write_submission({"clip_001_0000001": np.array([[2, 0.9, 10, 20, 30, 40]])}, test_csv, out)
    assert validate_submission(out, test_csv) == {"images": 2, "detections": 1}
    lines = out.read_text().splitlines()
    assert lines[1] == "clip_001_0000001,2 0.90000 10.00 20.00 30.00 40.00"
    assert lines[2] == "clip_001_0000002,"


@pytest.mark.parametrize("string, message", [
    ("9 0.5 1 1 5 5", "class"),
    ("2 1.5 1 1 5 5", "score"),
    ("2 0.5 995 1 50 5", "outside"),
    ("2 0.5 1 1 5", "multiple"),
])
def test_validator_rejects_bad_rows(tmp_path, test_csv, string, message):
    out = tmp_path / "bad.csv"
    pd.DataFrame({
        "image_id": ["clip_001_0000001", "clip_001_0000002"],
        "PredictionString": [string, ""],
    }).to_csv(out, index=False)
    with pytest.raises(ValueError, match=message):
        validate_submission(out, test_csv)


def test_folds_keep_source_videos_together():
    stats = pd.DataFrame({
        "clip": [f"uav{i // 2}_{i}_v" for i in range(20)],
        "group": [f"uav{i // 2}" for i in range(20)],
        "frames": 100, **{f"c{k}": 10 for k in range(1, 9)},
    })
    mapping, _ = assign_folds(stats, n_folds=5, n_trials=50)
    assert set(mapping) == set(stats["group"])
    assert np.bincount(list(mapping.values())).tolist() == [2, 2, 2, 2, 2]


def test_nwd_is_one_for_identical_boxes_and_smoother_than_iou_on_tiny_boxes():
    box = torch.tensor([[10.0, 10.0, 16.0, 16.0]])
    assert nwd(box, box, 37.0).item() == pytest.approx(1.0, abs=1e-3)
    shifted = box + 8  # no overlap, so IoU is 0, but NWD still orders by distance
    far = box + 80
    assert 0 < nwd(far, box, 37.0).item() < nwd(shifted, box, 37.0).item() < 1
