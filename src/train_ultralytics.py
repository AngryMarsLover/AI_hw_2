"""Train a YOLO or RT-DETR model on one fold, then score it with the shared scorer.

Usage:
  python src/train_ultralytics.py --name yolo11s_base --model yolo11s.pt --fold 4
  python src/train_ultralytics.py --name yolo11s_p2 --model configs/yolo11s-p2.yaml \
      --pretrained yolo11s.pt --set box=9.0
"""
import argparse
import ast
from pathlib import Path

from ultralytics import RTDETR, YOLO

from evaluate import fold_frames, fold_gt, log_result, save_preds
from metric import evaluate
from nwd import patch_box_loss, patch_detr_loss
from predict_ultralytics import load_model, predict_frames, time_inference


def parse_overrides(pairs):
    """Turn key=value strings into typed training arguments."""
    overrides = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        try:
            overrides[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            overrides[key] = value
    return overrides


def stop_on_batch_drop(batch):
    """Ultralytics halves the batch after an out-of-memory error, which would change the protocol."""
    def check(trainer):
        if trainer.batch_size != batch:
            raise RuntimeError(f"batch dropped from {batch} to {trainer.batch_size}")
    return check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pretrained", default=None, help="weights to load into a yaml model")
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lists", default="", help="suffix of the fold list file, e.g. _dense")
    parser.add_argument("--nwd-ratio", type=float, default=0.0, help="share of NWD in the box loss")
    parser.add_argument("--nwd-constant", type=float, default=37.0, help="mean box size at 1280 wide, in pixels")
    parser.add_argument("--notes", default="")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", help="extra Ultralytics args, key=value")
    args = parser.parse_args()

    if args.nwd_ratio and "rtdetr" in args.model:
        patch_detr_loss(args.nwd_ratio, args.nwd_constant, args.imgsz)
    elif args.nwd_ratio:
        patch_box_loss(args.nwd_ratio, args.nwd_constant)

    model_class = RTDETR if "rtdetr" in args.model else YOLO
    model = model_class(args.model)
    if args.pretrained:
        model.load(args.pretrained)

    model.add_callback("on_train_batch_end", stop_on_batch_drop(args.batch))
    model.train(
        data=str(Path("work/lists").resolve() / f"fold{args.fold}{args.lists}.yaml"),
        project=str(Path("runs").resolve()),
        name=f"{args.name}_fold{args.fold}",
        exist_ok=True,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        seed=args.seed,
        deterministic=True,
        workers=8,
        close_mosaic=5,
        patience=0,
        plots=True,
        **parse_overrides(args.overrides),
    )

    # Official numbers: best weights on the original frames with the shared scorer.
    weights = Path("runs") / f"{args.name}_fold{args.fold}" / "weights" / "best.pt"
    best = load_model(weights)
    frames = fold_frames(args.fold)
    preds = predict_frames(best, frames, imgsz=args.imgsz)
    save_preds(preds, Path("work/preds") / f"{args.name}_fold{args.fold}.npz")
    metrics = evaluate(fold_gt(args.fold), preds)
    row = log_result(args.name, args.fold, metrics, time_inference(best, frames, args.imgsz), args.notes)
    print(row)


if __name__ == "__main__":
    main()
