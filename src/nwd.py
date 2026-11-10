"""Normalized Wasserstein Distance box loss for small objects (Wang et al., arXiv:2110.13389).

IoU changes sharply with small shifts on tiny boxes; NWD models each box as a Gaussian and
stays smooth. It is blended with the IoU term of YOLO (CIoU) or RT-DETR (GIoU) by patching.
"""
import torch
from ultralytics.models.utils import loss as detr_loss
from ultralytics.utils import loss as ultralytics_loss


def nwd(boxes1, boxes2, constant, eps=1e-7):
    """Similarity in (0, 1] for matching xyxy boxes in pixels; `constant` is the mean box size."""
    center = ((boxes1[..., :2] + boxes1[..., 2:]) - (boxes2[..., :2] + boxes2[..., 2:])) / 2
    size = ((boxes1[..., 2:] - boxes1[..., :2]) - (boxes2[..., 2:] - boxes2[..., :2])) / 2
    distance = (center.pow(2).sum(-1) + size.pow(2).sum(-1) + eps).sqrt()
    return torch.exp(-distance / constant)


def patch_box_loss(ratio, constant):
    """Make the box term (1 - ratio) * CIoU + ratio * NWD. Boxes arrive in units of the stride."""
    original = ultralytics_loss.BboxLoss.forward

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                target_scores_sum, fg_mask, imgsz, stride):
        loss_iou, loss_dfl = original(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                                      target_scores, target_scores_sum, fg_mask, imgsz, stride)
        foreground = fg_mask.nonzero(as_tuple=True)
        weight = target_scores[foreground].sum(-1)
        scale = stride[foreground[1]]  # stride has shape (anchors, 1)
        similarity = nwd(pred_bboxes[foreground] * scale, target_bboxes[foreground] * scale, constant)
        loss_nwd = ((1.0 - similarity) * weight).sum() / target_scores_sum
        return (1 - ratio) * loss_iou + ratio * loss_nwd, loss_dfl

    ultralytics_loss.BboxLoss.forward = forward


def patch_detr_loss(ratio, constant, imgsz):
    """Make RT-DETR's GIoU term (1 - ratio) * GIoU + ratio * NWD. Boxes arrive as normalised xywh."""
    original = detr_loss.DETRLoss._get_loss_bbox

    def get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=""):
        loss = original(self, pred_bboxes, gt_bboxes, postfix)
        if len(gt_bboxes) == 0:
            return loss
        to_xyxy = lambda b: torch.cat([b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2], -1) * imgsz
        loss_nwd = (1.0 - nwd(to_xyxy(pred_bboxes), to_xyxy(gt_bboxes), constant)).sum() / len(gt_bboxes)
        name = f"loss_giou{postfix}"
        loss[name] = (1 - ratio) * loss[name] + ratio * self.loss_gain["giou"] * loss_nwd
        return loss

    detr_loss.DETRLoss._get_loss_bbox = get_loss_bbox
