"""Train a YOLO or RT-DETR model on one fold, then score it with the shared scorer.

Usage:
  python src/train_ultralytics.py --name yolo11s_base --model yolo11s.pt --fold 4
  python src/train_ultralytics.py --name yolo11s_p2 --model configs/yolo11s-p2.yaml \
      --pretrained yolo11s.pt --set box=9.0
  python src/train_ultralytics.py --name yolo11s_final --model yolo11s.pt --all-clips --stop-epoch 14
"""
import argparse
import ast
from pathlib import Path

import pandas as pd
from ultralytics import RTDETR, YOLO

from evaluate import fold_frames, fold_gt, log_result, save_preds
from metric import evaluate
from nwd import patch_box_loss, patch_detr_loss
from predict_ultralytics import load_model, predict_frames, time_inference
from submission import write_submission

LISTS = Path("work/lists").resolve()


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


def stop_after(epoch):
    """End training after this epoch while keeping the learning-rate schedule of the full run."""
    def check(trainer):
        if trainer.epoch + 1 >= epoch:
            trainer.stop = True
    return check


def all_clips_yaml():
    """Every train and val clip at the usual frame stride: the union of the fold training lists.

    Ultralytics still needs a validation list; fold 4 is reused, so its numbers mean nothing here.
    """
    frames = sorted({line for k in range(5) for line in (LISTS / f"fold{k}_train.txt").read_text().split()})
    (LISTS / "all_train.txt").write_text("\n".join(frames) + "\n")
    text = (LISTS / "fold4.yaml").read_text().replace("fold4_train.txt", "all_train.txt")
    (LISTS / "all.yaml").write_text(text)
    print(f"all clips: {len(frames)} training frames")
    return LISTS / "all.yaml"


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
    parser.add_argument("--all-clips", action="store_true", help="train on every clip and write a test submission")
    parser.add_argument("--stop-epoch", type=int, default=0, help="stop after this epoch, schedule unchanged")
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
    if args.stop_epoch:
        model.add_callback("on_fit_epoch_end", stop_after(args.stop_epoch))
    data = all_clips_yaml() if args.all_clips else LISTS / f"fold{args.fold}{args.lists}.yaml"
    run_name = f"{args.name}_all" if args.all_clips else f"{args.name}_fold{args.fold}"
    model.train(
        data=str(data),
        project=str(Path("runs").resolve()),
        name=run_name,
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

    if args.all_clips:
        # No held-out clips are left, so the last epoch is the model; it goes straight to the test set.
        weights = Path("runs") / run_name / "weights" / "last.pt"
        test_csv = Path("data/test_images.csv")
        frames = [Path("data") / name for name in pd.read_csv(test_csv)["file_name"]]
        out = Path("submissions") / f"{run_name}.csv"
        out.parent.mkdir(exist_ok=True)
        write_submission(predict_frames(load_model(weights), frames, imgsz=args.imgsz), test_csv, out)
        print(f"wrote {out}")
        return

    # Official numbers: best weights on the original frames with the shared scorer.
    weights = Path("runs") / run_name / "weights" / "best.pt"
    best = load_model(weights)
    frames = fold_frames(args.fold)
    preds = predict_frames(best, frames, imgsz=args.imgsz)
    save_preds(preds, Path("work/preds") / f"{args.name}_fold{args.fold}.npz")
    metrics = evaluate(fold_gt(args.fold), preds)
    row = log_result(args.name, args.fold, metrics, time_inference(best, frames, args.imgsz), args.notes)
    print(row)


if __name__ == "__main__":
    main()
