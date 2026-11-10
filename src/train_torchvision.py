"""Train Faster R-CNN or RetinaNet from COCO weights on one fold, then score with the shared scorer.

Usage:
  python src/train_torchvision.py --name frcnn_base --arch frcnn --fold 4
  python src/train_torchvision.py --name frcnn_giou --arch frcnn --box-loss giou
  python src/train_torchvision.py --name retina_anchors --arch retina --anchor-sizes 16,32,64,128,256
"""
import argparse
import csv
import functools
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
    roi_heads,
)
from torchvision.models.detection import retinanet as retinanet_module
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.ops import generalized_box_iou_loss, sigmoid_focal_loss

from evaluate import fold_frames, fold_gt, log_result, save_preds
from metric import evaluate, image_key
from submission import write_submission

GIOU_WEIGHT = 10.0  # MMDetection's weight for GIoU in Faster R-CNN, comparable to Smooth L1 here
RCNN_BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)  # torchvision's defaults for encoding box deltas
MAX_LOG_SIZE = math.log(1000.0 / 16)


class TrainFrames(Dataset):
    """Prepared frames with YOLO labels, random horizontal flip and light colour jitter."""

    def __init__(self, list_path):
        self.paths = Path(list_path).read_text().split()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = Path(self.paths[index])
        image = cv2.imread(str(path))[..., ::-1].copy()
        height, width = image.shape[:2]
        label_path = Path(str(path).replace("/images/", "/labels/")).with_suffix(".txt")
        text = label_path.read_text().strip()
        rows = np.loadtxt(label_path, ndmin=2) if text else np.empty((0, 5))
        centre, size = rows[:, 1:3], rows[:, 3:5]
        boxes = np.hstack([centre - size / 2, centre + size / 2]) * [width, height, width, height]
        boxes = boxes.clip(0, [width, height, width, height])
        if random.random() < 0.5:
            image = image[:, ::-1].copy()
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        image = np.clip(image * random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
        keep = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
        target = {
            "boxes": torch.as_tensor(boxes[keep], dtype=torch.float32),
            "labels": torch.as_tensor(rows[keep, 0], dtype=torch.int64),  # shifted per model below
        }
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255, target


class InferenceFrames(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(str(self.paths[index]))[..., ::-1].copy()
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255, image_key(self.paths[index])


def collate(batch):
    return tuple(zip(*batch))


def giou_rcnn_loss(class_logits, box_regression, labels, regression_targets):
    """Replacement for torchvision's fastrcnn_loss, using GIoU instead of Smooth L1.

    Deltas are relative to each proposal, and GIoU does not change under per-axis scaling and
    shifting, so decoding both against the same unit box gives the true GIoU.
    """
    labels = torch.cat(labels)
    regression_targets = torch.cat(regression_targets)
    classification_loss = nn.functional.cross_entropy(class_logits, labels)
    positive = torch.where(labels > 0)[0]
    count, _ = class_logits.shape
    deltas = box_regression.reshape(count, -1, 4)[positive, labels[positive]]

    def to_corners(values):
        scale = values.new_tensor(RCNN_BOX_WEIGHTS)
        dx, dy, dw, dh = (values / scale).unbind(-1)
        half_w, half_h = dw.clamp(max=MAX_LOG_SIZE).exp() / 2, dh.clamp(max=MAX_LOG_SIZE).exp() / 2
        return torch.stack([dx - half_w, dy - half_h, dx + half_w, dy + half_h], -1)

    box_loss = generalized_box_iou_loss(
        to_corners(deltas).float(), to_corners(regression_targets[positive]).float(), reduction="sum"
    )
    return classification_loss, GIOU_WEIGHT * box_loss / labels.numel()


def build_model(args):
    """COCO-pretrained model with an 8-class head; the training scale matches YOLO's 1280 input."""
    sizes = [int(s) for s in args.anchor_sizes.split(",")] if args.anchor_sizes else None
    ratios = tuple(float(r) for r in args.anchor_ratios.split(","))
    common = {"min_size": (540, 630, 720), "max_size": args.imgsz}
    if args.arch == "frcnn":
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
            box_score_thresh=0.001,
            box_detections_per_img=500,
            rpn_post_nms_top_n_train=args.rpn_post_train,
            rpn_post_nms_top_n_test=args.rpn_post_test,
            box_batch_size_per_image=args.roi_batch,
            box_positive_fraction=args.roi_positive,
            **common,
        )
        predictor = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(predictor, 9)
        if sizes:
            model.rpn.anchor_generator = AnchorGenerator(tuple((s,) for s in sizes), (ratios,) * len(sizes))
        if args.box_loss == "giou":
            roi_heads.fastrcnn_loss = giou_rcnn_loss
    else:
        model = retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
            score_thresh=0.001,
            detections_per_img=500,
            topk_candidates=2000,
            fg_iou_thresh=args.fg_iou,
            bg_iou_thresh=args.bg_iou,
            **common,
        )
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            model.backbone.out_channels, num_anchors, 8, norm_layer=functools.partial(nn.GroupNorm, 32)
        )
        if sizes:
            scales = tuple((s, int(s * 2 ** (1 / 3)), int(s * 2 ** (2 / 3))) for s in sizes)
            model.anchor_generator = AnchorGenerator(scales, (ratios,) * len(sizes))
        if args.box_loss:
            model.head.regression_head._loss_type = args.box_loss  # v2 default is giou
        retinanet_module.sigmoid_focal_loss = functools.partial(
            sigmoid_focal_loss, alpha=args.focal_alpha, gamma=args.focal_gamma
        )
    return model


@torch.no_grad()
def predict(model, frames, device, batch=4, workers=4):
    """Predict image paths; torchvision maps boxes back to each frame's original pixels."""
    model.eval()
    is_rcnn = hasattr(model, "roi_heads")
    loader = DataLoader(InferenceFrames(frames), batch_size=batch, num_workers=workers, collate_fn=collate)
    preds = {}
    for images, keys in loader:
        with torch.autocast(device.type):
            outputs = model([image.to(device) for image in images])
        for key, output in zip(keys, outputs):
            xyxy = output["boxes"].float().cpu().numpy()
            classes = output["labels"].cpu().numpy() - (1 if is_rcnn else 0)  # back to 0-7
            preds[key] = np.column_stack([
                classes, output["scores"].float().cpu().numpy(),
                xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1],
            ])
    return preds


def time_inference(model, frames, device, n_images=100):
    """Milliseconds per image at batch size 1, the same procedure for every model."""
    sample = frames[:: max(1, len(frames) // n_images)][:n_images]
    predict(model, sample[:1], device, batch=1, workers=0)  # warm-up
    start = time.perf_counter()
    predict(model, sample, device, batch=1, workers=0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / len(sample) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--arch", choices=["frcnn", "retina"], required=True)
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.02 / 16, help="per image; multiplied by batch size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="use only this many frames, for smoke tests")
    parser.add_argument("--notes", default="")
    parser.add_argument("--submission", type=Path, default=None, help="also write a test submission here")
    parser.add_argument("--anchor-sizes", default=None, help="base anchor size per FPN level, comma separated")
    parser.add_argument("--anchor-ratios", default="0.5,1.0,2.0")
    parser.add_argument("--box-loss", default=None, help="frcnn: giou; retina: l1 or giou")
    parser.add_argument("--rpn-post-train", type=int, default=2000)
    parser.add_argument("--rpn-post-test", type=int, default=1000)
    parser.add_argument("--roi-batch", type=int, default=512)
    parser.add_argument("--roi-positive", type=float, default=0.25)
    parser.add_argument("--fg-iou", type=float, default=0.5)
    parser.add_argument("--bg-iou", type=float, default=0.4)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run_dir = Path("runs") / f"{args.name}_fold{args.fold}"
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    model = build_model(args).to(device)
    label_shift = 1 if args.arch == "frcnn" else 0  # Faster R-CNN reserves label 0 for background
    dataset = TrainFrames(Path("work/lists") / f"fold{args.fold}_train.txt")
    frames = fold_frames(args.fold)
    if args.limit:
        dataset.paths = dataset.paths[: args.limit]
        frames = frames[: args.limit]
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=8, collate_fn=collate,
        drop_last=True, persistent_workers=True,
        worker_init_fn=lambda worker: random.seed(args.seed + worker),
    )
    gt = fold_gt(args.fold)
    if args.limit:
        keys = {image_key(path) for path in frames}
        gt = {**gt, "images": [i for i in gt["images"] if i["id"] in keys],
              "annotations": [a for a in gt["annotations"] if a["image_id"] in keys]}

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr * args.batch, momentum=0.9, weight_decay=1e-4)
    total_steps = args.epochs * len(loader)
    warmup = min(500, total_steps // 10)

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history_path = run_dir / "results.csv"
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "time", "loss", "mAP", "AP50", "AP_small"])
    best_map, start_time = -1.0, time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [
                {"boxes": t["boxes"].to(device), "labels": t["labels"].to(device) + label_shift} for t in targets
            ]
            with torch.autocast(device.type):
                loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        row = [epoch, round(time.time() - start_time), round(running / len(loader), 4), "", "", ""]
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(gt, predict(model, frames, device))
            row[3:] = [round(metrics[k], 4) for k in ["mAP", "AP50", "AP_small"]]
            if metrics["mAP"] > best_map:
                best_map = metrics["mAP"]
                torch.save(model.state_dict(), run_dir / "weights" / "best.pth")
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"epoch {epoch}/{args.epochs}", row, flush=True)

    # Official numbers: best weights on the original frames with the shared scorer.
    model.load_state_dict(torch.load(run_dir / "weights" / "best.pth", map_location=device))
    preds = predict(model, frames, device)
    save_preds(preds, Path("work/preds") / f"{args.name}_fold{args.fold}.npz")
    row = log_result(args.name, args.fold, evaluate(gt, preds), time_inference(model, frames, device), args.notes)
    print(row)

    if args.submission:
        test_csv = Path("data/test_images.csv")
        test_frames = [Path("data") / name for name in np.loadtxt(test_csv, dtype=str, delimiter=",", skiprows=1, usecols=1)]
        write_submission(predict(model, test_frames, device), test_csv, args.submission)


if __name__ == "__main__":
    main()
