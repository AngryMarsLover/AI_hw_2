"""Write and validate Kaggle submission files."""
import numpy as np
import pandas as pd

from metric import MAX_DETS, cap_per_image


def format_rows(rows, width, height):
    """Clip boxes to the image and format as `class score x y w h` groups."""
    groups = []
    for cls, score, x, y, w, h in cap_per_image(rows):
        x1, y1 = max(x, 0.0), max(y, 0.0)
        x2, y2 = min(x + w, width), min(y + h, height)
        box_w, box_h = round(x2 - x1, 2), round(y2 - y1, 2)
        if box_w <= 0 or box_h <= 0:  # also drops boxes that would print as size 0.00
            continue
        groups.append(f"{int(cls)} {score:.5f} {x1:.2f} {y1:.2f} {box_w:.2f} {box_h:.2f}")
    return " ".join(groups)


def write_submission(preds, test_csv, out_path):
    """Write one row per test image; images without predictions get an empty string."""
    test = pd.read_csv(test_csv)
    strings = [
        format_rows(preds.get(row.image_id, np.empty((0, 6))), row.width, row.height)
        for row in test.itertuples()
    ]
    pd.DataFrame({"image_id": test["image_id"], "PredictionString": strings}).to_csv(
        out_path, index=False
    )
    validate_submission(out_path, test_csv)


def validate_submission(path, test_csv):
    """Raise ValueError on anything that would be scored wrongly; return summary counts."""
    test = pd.read_csv(test_csv).set_index("image_id")
    submission = pd.read_csv(path, keep_default_na=False, dtype=str)
    if list(submission.columns) != ["image_id", "PredictionString"]:
        raise ValueError(f"unexpected columns: {list(submission.columns)}")
    if set(submission["image_id"]) != set(test.index) or submission["image_id"].duplicated().any():
        raise ValueError("image_id values do not match test_images.csv exactly")

    total = 0
    for image_id, string in zip(submission["image_id"], submission["PredictionString"]):
        values = string.split()
        if len(values) % 6:
            raise ValueError(f"{image_id}: value count is not a multiple of 6")
        if len(values) > 6 * MAX_DETS:
            raise ValueError(f"{image_id}: more than {MAX_DETS} detections")
        total += len(values) // 6
        if not values:
            continue
        rows = np.array(values, dtype=float).reshape(-1, 6)
        cls, score, x, y, w, h = rows.T
        info = test.loc[image_id]
        if not np.all((cls == cls.astype(int)) & (cls >= 0) & (cls <= 7)):
            raise ValueError(f"{image_id}: class outside 0-7")
        if not np.all((score >= 0) & (score <= 1)):
            raise ValueError(f"{image_id}: score outside 0-1")
        # Allow rounding from the two-decimal output format.
        if np.any(w <= 0) or np.any(h <= 0) or np.any(x < 0) or np.any(y < 0) \
                or np.any(x + w > info["width"] + 0.02) or np.any(y + h > info["height"] + 0.02):
            raise ValueError(f"{image_id}: box outside the image")
    return {"images": len(submission), "detections": total}
