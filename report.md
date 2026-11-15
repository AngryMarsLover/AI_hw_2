---
title: Homework 2 -- Multiple Object Video Detection (VisDrone2019-VID)
author: Mykola Balyk
course: Artificial Intelligence · UCU
date: 2026-11-15
---

## 1. Task and evaluation

Detect 8 classes (person, bicycle, car, van, truck, tricycle, bus, motor) in drone video frames. Submissions list `class score x y w h` in pixels. All results below come from one scorer (`src/metric.py`): COCO mAP@[.5:.95] with up to 500 detections per image, ignored regions treated as crowd boxes for every class, scored on the original (unpainted) frames, using every 5th frame of the held-out clips. Inference speed is ms per image at batch 1 on an RTX 4050 Laptop (6 GB).

## 2. Data understanding and preprocessing

Code: `src/eda.py`, `src/prepare_data.py`, `src/folds.py`, `src/anchors.py`. Figures: `images/eda_*.png`, `images/gt_check.png`.

| Observation | Value | Decision it informed |
|---|---|---|
| Class balance | car 44%, person 31%, motor 9%, bus 0.8% | Report per-class AP; rare classes (bus, truck) are noisy |
| Box size at 1280 px width | p10 / p50 / p90 = 16 / 31 / 66 px | Train at 1280 px (at 640 px, 54% of boxes would be under 16 px) |
| COCO size share | small 24%, medium 64%, large 12% | Track AP_small separately |
| Objects per frame | median 42, p99 140, max 180 | 500 detections per image; RT-DETR's 300 queries are enough |
| Same-object IoU between frames | lag 1: 0.85, lag 5: 0.56, lag 10: 0.40 | Train on every 10th frame; adjacent frames are near-duplicates |
| Test widths | 680 to 1920 px | Keep frames at their native aspect ratio, letterboxed to 1280 |

![Object sizes](images/eda_sizes.png)

**Preprocessing.** Every 5th frame is extracted. Ignored regions are painted grey (114), with labelled boxes restored on top, so the detector is not taught that unlabelled objects are background. Frames wider than 1920 px are downscaled. Annotations are converted to YOLO txt and COCO json from the same source. `images/gt_check.png` shows the converted boxes on sample frames.

**Validation design.** The 63 train+val clips come from 55 source videos. Five folds are split by source video, so clips cut from the same flight never appear on both sides (`folds/folds.csv`, seed 42). Fold 4 is the tuning fold. The best YOLO setup was re-checked on folds 1 and 2. Anchors were fitted only on fold 4's training clips.

## 3. Common protocol

Same frames, fold and input size for every family: 1280 px, COCO-pretrained weights, AMP, batch 4 where memory allows. YOLO and RT-DETR use Ultralytics 8.4 for 30 epochs, keeping the best epoch. Faster R-CNN and RetinaNet use torchvision 0.29 for 12 epochs (SGD lr 0.02×batch/16, warmup + cosine). Each experiment changes one thing from its family's baseline.

## 4. Two-stage anchor-based: Faster R-CNN (ResNet-50 FPN v2)

Code: `src/train_torchvision.py --arch frcnn`.

| Run | mAP | AP50 | AP_small | ms/img |
|---|---|---|---|---|
| Baseline (default anchors 32–512, Smooth L1) | 0.2168 | 0.414 | 0.040 | 97 |
| **Anchors from k-means (12–192 px)** | **0.2206** | **0.441** | **0.051** | 100 |
| GIoU box loss (weight 10) | 0.2146 | 0.412 | 0.043 | 98 |
| K-means anchors + 4,000 / 2,000 RPN proposals | 0.2185 | 0.438 | 0.050 | 140 |

K-means on box sizes (`results/anchors.json`) showed that 54% of boxes are below the default 32 px anchor. Smaller anchors raised AP50 by 0.027 and AP_small by 28%, but not AP75. The model finds more objects but does not localise them better. GIoU did not help at this weight. Doubling the RPN proposals (frames hold up to 180 objects) changed nothing and cost 40% more time, so the proposal count is not the bottleneck. The model peaked at epoch 4 of 12, so longer training would not help either.

**What each stage contributes.** The RPN turns about 200k anchors into 1,000–2,000 class-agnostic proposals, so recall is decided here. Smaller anchors therefore help small objects most. The second stage re-pools features for each proposal and refines class and box. That second refinement is why Faster R-CNN tolerates poorly matched anchors (see RetinaNet below) and has the best AP_large (0.439).

**Failure cases** (from per-class and per-size AP):
1. **Van vs car** (van AP 0.11 vs car 0.44). Seen from above, vans and cars differ mainly in length, and both are labelled in the same scenes. Classification, not localisation, is the likely error.
2. **Tiny people and bicycles** (AP_small 0.05). A 10–15 px person covers 2–4 cells of the stride-4 P2 map. A box that is off by 2 px drops below IoU 0.5, so detections count as false positives at the stricter thresholds.
3. **Rare large vehicles** (bus and truck have about 500–900 instances per fold). Scores swing by ±0.1 between runs, and a bus cut off at the image edge is often split into several boxes.

## 5. One-stage anchor-based: RetinaNet (ResNet-50 FPN v2)

| Run | Anchor size / stride | mAP | AP75 | AP_small | AP_large |
|---|---|---|---|---|---|
| **Baseline (32–512)** | 4× | 0.1874 | **0.178** | 0.040 | **0.397** |
| K-means anchors (12–192) | 1.5× | 0.1348 | 0.100 | 0.044 | 0.272 |
| Anchors 16–256 | 2× | 0.1625 | 0.137 | 0.046 | 0.327 |
| **Focal gamma 1.5** (default anchors) | 4× | **0.1885** | 0.177 | 0.042 | 0.401 |

The anchors that helped Faster R-CNN hurt RetinaNet badly, and the effect grows with anchor size relative to stride. RetinaNet starts at P3 (stride 8). With anchors only 1.5× the stride, a 200 px bus is assigned to P7 (stride 128). I simulated anchor matching on the training boxes: the median best-anchor IoU for 32–64 px boxes fell from 0.77 to 0.63. Matching an anchor set to box sizes (the data) is not enough; it must also match the feature strides (the model).

**Removing the proposal stage.** Without the RPN, the classifier sees every anchor: about 100k background anchors per frame against about 40 objects. Focal loss addresses this imbalance; changing gamma 2.0→1.5 made no difference (+0.001). The other loss is that boxes are never re-pooled and refined, so the anchor-to-object fit directly limits AP75. Fixes are better positive assignment (ATSS), a refinement step (Cascade RPN), or a stride-4 level for small objects. Tuned RetinaNet (0.1885) stays 0.032 below tuned Faster R-CNN (0.2206) and is only about 1.4× faster.

## 6. One-stage anchor-free: YOLO11s

**How locations are assigned.** No anchors exist: every grid cell on P3–P5 predicts distances to the 4 box sides. The Task-Aligned Assigner (TOOD) picks the top-10 cells per object by classification score × IoU, among cells whose centre falls inside the box. A 12 px object contains only about 2 cells at stride 8, and neighbouring objects in a dense car park compete for the same cells, which limits small and crowded objects.

**Search range** (fold 4, one change at a time, then combined):

| Run | mAP | AP50 | AP_small | AP_medium | AP_large |
|---|---|---|---|---|---|
| Baseline (auto AdamW lr 8.3e-4) | 0.2105 | 0.398 | 0.063 | 0.165 | 0.374 |
| Input 960 instead of 1280 | 0.1966 | 0.369 | 0.043 | 0.146 | 0.374 |
| P2 head (stride 4; batch 2 for memory) | 0.1982 | 0.394 | 0.048 | 0.168 | 0.342 |
| Loss: 0.5 CIoU + 0.5 NWD, C = 37 px | 0.2116 | 0.399 | 0.067 | 0.162 | 0.386 |
| Augmentation: scale 0.9, mixup 0.1 | 0.2172 | 0.413 | 0.068 | 0.171 | 0.370 |
| Every 5th frame, 15 epochs (same steps) | 0.2145 | 0.400 | 0.065 | 0.168 | 0.395 |
| Augmentation + NWD | 0.2309 | 0.426 | 0.074 | 0.181 | 0.388 |
| Augmentation + lr 3e-4 | 0.2316 | 0.430 | **0.078** | 0.184 | **0.427** |
| **Augmentation + lr 3e-4 + NWD** | **0.2403** | **0.434** | 0.069 | **0.191** | 0.417 |

NWD (Normalized Wasserstein Distance, `src/nwd.py`) is the loss modification. It treats boxes as Gaussians, so a few-pixel shift of a tiny box changes the loss smoothly instead of collapsing IoU. Alone it raised AP_small by 0.004. Combined with augmentation and a lower learning rate, it added 0.009 mAP. The baseline peaked at epoch 8 and then overfit (training loss falling while validation classification loss rose). Augmentation and the lower learning rate addressed that; denser frames did not, because they are near-duplicates. The P2 head failed: only 297 of 593 weight tensors loaded from the pretrained checkpoint, and memory forced batch 2.

**Check on other folds:**

| Fold | Baseline | Tuned | Gain |
|---|---|---|---|
| 4 (tuning) | 0.2105 | 0.2403 | +0.030 |
| 1 | 0.1984 | 0.2214 | +0.023 |
| 2 | 0.2351 | 0.2559 | +0.021 |

The gain holds on all three folds (+0.021 to +0.030), so it is not tuning noise. It comes from medium and large objects (truck AP on fold 2: 0.16 → 0.24). AP_small fell on folds 1 and 2 (0.080 → 0.074, 0.156 → 0.142), which is the gap that tiling addresses in Section 8. Against the anchor-based models, tuned YOLO beats tuned Faster R-CNN by 0.020 mAP and on AP_small (0.069 vs 0.051), but not on AP_large, at about 4.5× the speed.

## 7. Query-based: RT-DETR-L

Batch 4 at 1280 px ran out of memory, so RT-DETR used batch 2 (AdamW lr 1e-4, 30 epochs, 300 queries, Hungarian matching on cost weights class 2, L1 5, GIoU 2).

| Run | mAP | AP_small | AP_large | ms/img | Train time |
|---|---|---|---|---|---|
| Baseline | 0.1684 | 0.030 | 0.335 | 122 | 286 min |
| No mosaic | 0.1495 | 0.032 | 0.290 | 123 | 287 min |
| **Loss: 0.5 GIoU + 0.5 NWD** | **0.1766** | **0.036** | 0.330 | 127 | 290 min |

![Validation mAP over training time](images/rtdetr_vs_yolo.png)

**Bipartite matching** assigns each ground-truth box to exactly one query, so duplicates are penalised during training and no NMS is needed. The cost is a slower, unstable early phase: matches change between iterations, and most of the 300 queries learn "no object". At YOLO's full budget (81 min), RT-DETR had reached about 0.07 validation mAP against YOLO's 0.22. **RT-DETR had not converged:** its best epoch was the last one, with a jump in the final 5 epochs. The jump also appears without mosaic, so it comes from the decaying learning rate rather than from switching mosaic off. NWD helped here too (+0.008 mAP, AP_small 0.030 → 0.036, AP_large unchanged), the same small-object pattern as in YOLO, so the loss change transfers across detector families. On small objects RT-DETR is still the weakest model. The encoder works at stride 8–32, and 300 queries must cover frames with up to 180 objects.

## 8. Pipeline, final selection and leaderboard

**Final model.** The tuned YOLO (augmentation + lr 3e-4 + NWD) was retrained on all 63 clips (2,734 frames, `--all-clips`). No clips are left for picking an epoch, so it keeps the 30-epoch schedule and stops after epoch 14. On the three folds, the best epochs were 10, 14 and 15.

**Inference-time methods** (`src/inference_tests.py`, `src/fusion.py`; fold-4 models, scored on fold 4):

| Method | mAP | AP75 | AP_small | AP_large |
|---|---|---|---|---|
| Tuned YOLO, plain | 0.2403 | 0.235 | 0.069 | 0.417 |
| + horizontal flip (TTA), fused | 0.2522 | 0.245 | 0.073 | 0.430 |
| Ultralytics TTA (flip + 0.83×, 0.67× scales) | 0.2527 | 0.243 | 0.071 | 0.438 |
| + 2×2 tiles (20% overlap, each upscaled to 1280), fused | 0.2589 | 0.250 | 0.080 | 0.437 |
| YOLO + Faster R-CNN (2:1) | 0.2640 | 0.251 | 0.073 | 0.482 |
| YOLO + Faster R-CNN + RetinaNet (2:1:1) | 0.2664 | 0.255 | 0.074 | 0.491 |
| YOLO + Faster R-CNN + RT-DETR (2:1:1) | 0.2617 | 0.248 | 0.066 | 0.478 |
| **YOLO + flip + tiles + Faster R-CNN + RetinaNet (2:2:2:1:1)** | **0.2763** | **0.269** | **0.084** | **0.494** |

Sources are merged with **weighted box fusion** (WBF): overlapping same-class boxes are averaged, weighted by score, instead of all but one being dropped as in NMS. The IoU threshold matters in dense scenes. At the usual 0.55, fusing YOLO with itself lost 0.010 mAP because it merged neighbouring cars that YOLO's own NMS (IoU 0.65) had kept apart. At 0.65, self-fusion reproduces the plain score exactly, so 0.65 is used throughout. Tile boxes touching an inner tile edge are dropped as cut-off objects.

Each source fixes a different weakness. Tiles show small objects 2.4× larger (AP_small +0.011). Flip averages out localisation noise (AP75). The two-stage detector adds the best large-object boxes (AP_large +0.065). RT-DETR made the ensemble worse: its small-object predictions are weaker than everything else. All weights were chosen on fold 4, so the +0.036 gain is somewhat optimistic.

**Submissions and final selection:**

| Submission | Local mAP, fold 4 (same setup) | Public LB |
|---|---|---|
| YOLO11s baseline (folds 0–3) | 0.2105 | 0.2605 |
| YOLO11s augmentation + lr 3e-4 (folds 0–3) | 0.2316 | 0.2406 |
| Tuned YOLO11s, all clips | 0.2403 | 0.2415 |
| **Five-source WBF ensemble** (`ens_all5.csv`) | **0.2763** | **0.2678** |

The two submissions selected for the private leaderboard are:

1. **`ens_all5.csv`, the main pick.** Test frames go through the all-clips YOLO three times (full frame, horizontal flip, 2×2 tiles), plus the fold-4 Faster R-CNN (k-means anchors) and RetinaNet (focal gamma 1.5) checkpoints. All five are fused with WBF at weights 2:2:2:1:1 and IoU 0.65, keeping the top 500 boxes per frame. It is the best setup locally and the best public score. The ensemble adds the same amount on both: +0.036 locally and +0.026 publicly over the all-clips YOLO alone. Inference costs 6 YOLO passes (full, flip, 4 tiles) plus one Faster R-CNN and one RetinaNet pass per frame. From the single-model timings that is about 0.3 s per frame on the laptop GPU, against 22 ms for YOLO alone.
2. **`yolo11s_base_fold4.csv`, the backup.** The plain YOLO11s baseline is a single model and the second-best public score. Its public score is 0.05 above its local score, the only submission with such a gap. That points to a favourable public subset (per-clip mAP ranges 0.05–0.36, and random 30% subsets of clips moved fold-4 mAP by ±0.06) rather than a better model, which is why it is the backup and not the main pick.

The all-clips retrain alone gave no public gain over the fold-4 model (0.2415 vs 0.2406). As with denser frames, more footage of the same kind of scenes adds little to this model.

**Not done:** temporal rescoring across neighbouring frames, and Grad-CAM/W&B. With more time, the first step would be temporal rescoring: objects persist across frames (IoU 0.85 at lag 1), so it could recover the weak small-object detections.

## 9. How to run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
ln -s multiple-object-video-detection-hw-2-ucu-ai-2026 data      # Kaggle data
python src/folds.py && python src/prepare_data.py                  # folds, frames, lists
python src/eda.py && python src/anchors.py --fold 4                # figures, anchors
python src/train_torchvision.py --name frcnn_anchors --arch frcnn --anchor-sizes 12,24,48,96,192
python src/train_ultralytics.py --name yolo11s_aug_lr_nwd --model yolo11s.pt --fold 4 \
    --nwd-ratio 0.5 --set scale=0.9 mixup=0.1 optimizer=AdamW lr0=0.0003
python src/train_torchvision.py --name retina_focal --arch retina --focal-gamma 1.5
# final model on every clip, then the five test sources and the fused submission
python src/train_ultralytics.py --name yolo11s_aug_lr_nwd --model yolo11s.pt --all-clips --stop-epoch 14 \
    --nwd-ratio 0.5 --set scale=0.9 mixup=0.1 optimizer=AdamW lr0=0.0003 val=False
export PYTHONPATH=src; R="--run runs/yolo11s_aug_lr_nwd_all --weights last.pt"
for m in plain flip tile; do python src/predict_test.py $R --mode $m; done
python src/predict_test.py --run runs/frcnn_anchors_fold4 && python src/predict_test.py --run runs/retina_focal_fold4
python src/predict_test.py --fuse yolo11s_aug_lr_nwd_all yolo11s_aug_lr_nwd_all_tile yolo11s_aug_lr_nwd_all_flip \
    frcnn_anchors_fold4 retina_focal_fold4 --fuse-weights 2 2 2 1 1 --out submissions/ens_all5.csv
python src/inference_tests.py tile --weights runs/yolo11s_aug_lr_nwd_fold4/weights/best.pt   # fold-4 checks
pytest tests
```

Every run appends its metrics to `results/experiments.csv`; public scores are in `results/public_scores.csv`. The exact settings for each experiment are in `scripts/queue_*.sh`. Seed 42 everywhere; Ultralytics runs use `deterministic=True`.

## References

VisDrone (Zhu et al., 2021); Faster R-CNN (Ren et al., 2015); FPN (Lin et al., 2017); RetinaNet / focal loss (Lin et al., 2017); ATSS (Zhang et al., 2020); TOOD (Feng et al., 2021); YOLO11, Ultralytics (Jocher et al., 2024); RT-DETR (Zhao et al., 2024); Weighted Boxes Fusion (Solovyev et al., 2021); SAHI slicing (Akyon et al., 2022); NWD (Wang et al., arXiv:2110.13389); GIoU (Rezatofighi et al., 2019); YOLOv2 k-means anchors (Redmon & Farhadi, 2017). Libraries: PyTorch, torchvision, Ultralytics, pycocotools. Code was written with the help of an AI assistant (Claude); all experiments were run and checked locally.
