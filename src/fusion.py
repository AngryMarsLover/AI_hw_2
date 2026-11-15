"""Weighted box fusion (Solovyev et al., 2021) for merging detections from several sources.

Rows are [class, score, x, y, w, h] in pixels, as everywhere else in this project. Unlike NMS,
which keeps one box of each overlapping group, WBF averages the group's coordinates weighted
by score, so agreeing sources both move the box and raise its confidence.
"""
import numpy as np


def _iou(box, boxes):
    """IoU of one xyxy box with an array of xyxy boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-9)


def _fuse_class(boxes, scores, n_sources, iou_thr):
    """Cluster boxes of one class greedily in score order and average each cluster."""
    order = np.argsort(-scores)
    boxes, scores = boxes[order], scores[order]
    fused = np.empty((0, 4))
    members = []  # indices into boxes, per cluster
    for i, box in enumerate(boxes):
        if len(fused):
            ious = _iou(box, fused)
            best = int(ious.argmax())
            if ious[best] > iou_thr:
                members[best].append(i)
                idx = members[best]
                w = scores[idx][:, None]
                fused[best] = (boxes[idx] * w).sum(0) / w.sum()
                continue
        fused = np.vstack([fused, box])
        members.append([i])
    # Average score, scaled down when fewer sources than expected found the object.
    out_scores = np.array([scores[idx].mean() * min(len(idx), n_sources) / n_sources for idx in members])
    return fused, out_scores


def wbf(sources, weights=None, iou_thr=0.65, skip_score=0.0):
    """Fuse lists of rows from several sources for one image; returns rows in the same layout."""
    weights = np.ones(len(sources)) if weights is None else np.asarray(weights, dtype=float)
    weights = weights / weights.mean()
    rows = []
    for rows_k, weight in zip(sources, weights):
        rows_k = np.asarray(rows_k, dtype=float).reshape(-1, 6)
        rows_k = rows_k[rows_k[:, 1] >= skip_score].copy()
        rows_k[:, 1] = rows_k[:, 1] * weight
        rows.append(rows_k)
    rows = np.concatenate(rows) if rows else np.empty((0, 6))
    out = []
    for cls in np.unique(rows[:, 0]):
        part = rows[rows[:, 0] == cls]
        xyxy = np.column_stack([part[:, 2], part[:, 3], part[:, 2] + part[:, 4], part[:, 3] + part[:, 5]])
        fused, scores = _fuse_class(xyxy, part[:, 1], len(sources), iou_thr)
        out.append(np.column_stack([np.full(len(fused), cls), np.minimum(scores, 1.0), fused[:, :2], fused[:, 2:] - fused[:, :2]]))
    return np.concatenate(out) if out else np.empty((0, 6))
