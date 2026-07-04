# Consolidated Benchmark Report — Operation Cyber-Histology

## Section 1 — Pipeline Reconstruction Baseline

**Run configuration** (from `results/full_grid_results.csv`, patch `0017`'s `full_grid` grid): `BATCH_SIZE=128`, `LEARNING_RATE=0.001`, `OPTIMIZER=Adam`, `LOSS=CrossEntropyLoss`, `DROPOUT=0.5`, `ACTIVATION=ReLU`, `VAL_SPLIT=0.1`, `EPOCHS=20`.

### Summary table

| Dataset | Model | Accuracy | Precision | Recall | Macro F1 |
|---------|-------|---------:|----------:|-------:|---------:|
| cells   | AlexNet   | 93.86% | 93.60% | 92.81% | 93.16% |
| cells   | VGG16     | 96.40% | 96.90% | 95.83% | 96.33% |
| cells   | ResNet18  | 97.25% | 97.22% | 97.25% | 97.21% |
| chest   | AlexNet   | 90.71% | 92.61% | 88.03% | 89.59% |
| chest   | VGG16     | 88.14% | 91.35% | 84.44% | 86.39% |
| chest   | ResNet18  | 87.18% | 90.54% | 83.25% | 85.20% |
| lesions | AlexNet   | 75.76% | 54.86% | 48.98% | 49.73% |
| lesions | VGG16     | 72.67% | 38.76% | 33.28% | 33.68% |
| lesions | ResNet18  | 75.76% | 45.53% | 40.08% | 41.23% |
| orgs    | AlexNet   | 89.97% | 89.08% | 88.47% | 88.52% |
| orgs    | VGG16     | 89.57% | 88.00% | 88.46% | 88.05% |
| orgs    | ResNet18  | 92.05% | 91.17% | 91.05% | 90.88% |

### Efficiency 

| Dataset | Model | Train Runtime (s) | Train Peak Mem (MB) | Inference Runtime (s) | Inference Latency (ms/sample) | Inference Peak Mem (MB) |
|---------|-------|-------------------:|----------------------:|------------------------:|--------------------------------:|---------------------------:|
| cells   | AlexNet   | 26.17  | 402.4  | 0.22 | 0.063 | 269.0 |
| cells   | VGG16     | 128.95 | 2072.8 | 0.61 | 0.178 | 717.8 |
| cells   | ResNet18  | 296.28 | 3179.0 | 1.16 | 0.340 | 856.7 |
| chest   | AlexNet   | 8.58   | 399.5  | 0.03 | 0.054 | 265.1 |
| chest   | VGG16     | 48.50  | 2070.1 | 0.10 | 0.166 | 715.4 |
| chest   | ResNet18  | 112.64 | 3174.4 | 0.21 | 0.329 | 851.9 |
| lesions | AlexNet   | 15.20  | 402.7  | 0.11 | 0.055 | 267.8 |
| lesions | VGG16     | 75.72  | 2073.5 | 0.34 | 0.170 | 996.4 |
| lesions | ResNet18  | 173.88 | 3179.0 | 0.69 | 0.344 | 1028.3 |
| orgs    | AlexNet   | 24.57  | 399.4  | 0.30 | 0.037 | 264.1 |
| orgs    | VGG16     | 141.36 | 2068.9 | 1.29 | 0.157 | 715.2 |
| orgs    | ResNet18  | 329.20 | 3174.5 | 2.62 | 0.319 | 851.2 |

### Minimum accuracy targets

| Dataset | Target | AlexNet | VGG16 | ResNet18 | All pass? |
|---------|-------:|--------:|------:|---------:|:---------:|
| cells   | 90% | 93.86% | 96.40% | 97.25% | Yes |
| chest   | 87% | 90.71% | 88.14% | 87.18% | Yes |
| lesions | 67% | 75.76% | 72.67% | 75.76% | Yes |
| orgs    | 83% | 89.97% | 89.57% | 92.05% | Yes |

All 12 dataset/model combinations meet their minimum test accuracy.

### Architectural recommendations

- **chest → AlexNet.** AlexNet is simultaneously the *most* accurate model on this dataset (90.71%, F1 89.59%) and by far the cheapest (8.58s train, 399MB). VGG16 and ResNet18 add 6-13x the training cost and 5-8x the memory while scoring *lower* on every metric — their extra capacity actively hurts here (likely overfitting at 20 epochs on a comparatively simple/lower-resolution modality). Clearest result in the whole grid.
- **lesions → AlexNet.** AlexNet ties ResNet18 for best accuracy (75.76%) and has the best F1 by a wide margin (49.73% vs. ResNet18's 41.23%, VGG16's 33.68%) at roughly 1/11th ResNet18's training time. Separately worth flagging: precision/recall/F1 sit far below accuracy across all three models (33-55% vs. 73-76% accuracy) — a strong signal of class imbalance or systematic per-class confusion that accuracy alone hides. 
- **cells → ResNet18 if compute allows, AlexNet otherwise.** ResNet18 gives the best accuracy (97.25%/F1 97.21%), but AlexNet reaches 93.86%/F1 93.16% — still well clear of the 90% target — at ~11x less training time and ~8x less memory. Diminishing returns from VGG16/ResNet18's extra capacity here are real but modest.
- **orgs → ResNet18.** ResNet18 wins outright (92.05%/F1 90.88%) over AlexNet's already-solid 89.97%/F1 88.52%. 
- **Overall: AlexNet is the standout general-purpose model in this benchmark.** It is never more than ~2-3 points of accuracy behind the larger architectures, is outright the best model on `chest` and tied-best on `lesions`, and is consistently 8-13x cheaper on training time and memory than ResNet18 across every dataset.

## Section 2 — Green Initiative

**Run configuration**: identical to Section 1 (`BATCH_SIZE=128`, `LEARNING_RATE=0.001`, `Adam`, `CrossEntropyLoss`, `DROPOUT=0.5`, `ACTIVATION=ReLU`, `VAL_SPLIT=0.1`, `EPOCHS=20`) — only the architecture varies, isolating capacity as the sole variable between `VGG16` and `GreenVGG16` across all four datasets.

`GreenVGG16` keeps VGG16's block structure but roughly halves the channel width at every stage (32/64/128/256/256 vs. 64/128/256/512/512) and shrinks the classifier head (512→256 vs. 1024→512).

### Efficiency Verification Matrix

| Dataset | Model | Train Runtime (s) | Train Peak Mem (MB) | Inference Runtime (s) | Inference Latency (ms/sample) | Inference Peak Mem (MB) |
|---------|-------|-------------------:|----------------------:|------------------------:|--------------------------------:|---------------------------:|
| cells   | VGG16      | 130.50 | 2072.7 | 0.61 | 0.179 | 719.5 |
| cells   | GreenVGG16 | 63.23  | 880.7  | 0.31 | 0.091 | 201.3 |
| chest   | VGG16      | 48.65  | 2068.1 | 0.11 | 0.169 | 714.1 |
| chest   | GreenVGG16 | 22.56  | 877.2  | 0.05 | 0.085 | 196.1 |
| lesions | VGG16      | 76.07  | 2072.0 | 0.37 | 0.185 | 995.4 |
| lesions | GreenVGG16 | 37.21  | 880.9  | 0.18 | 0.089 | 200.8 |
| orgs    | VGG16      | 142.93 | 2068.6 | 1.38 | 0.168 | 715.7 |
| orgs    | GreenVGG16 | 68.95  | 876.2  | 0.59 | 0.072 | 196.3 |

### Accuracy comparison

| Dataset | Model | Accuracy | Precision | Recall | Macro F1 |
|---------|-------|---------:|----------:|-------:|---------:|
| cells   | VGG16      | 96.40% | 96.90% | 95.83% | 96.33% |
| cells   | GreenVGG16 | 97.14% | 97.12% | 96.66% | 96.86% |
| chest   | VGG16      | 88.14% | 91.35% | 84.44% | 86.39% |
| chest   | GreenVGG16 | 91.35% | 93.37% | 88.72% | 90.31% |
| lesions | VGG16      | 72.67% | 38.76% | 33.28% | 33.68% |
| lesions | GreenVGG16 | 74.81% | 50.63% | 41.49% | 43.34% |
| orgs    | VGG16      | 89.57% | 88.00% | 88.46% | 88.05% |
| orgs    | GreenVGG16 | 92.00% | 91.08% | 90.47% | 90.63% |

All four `GreenVGG16` runs clear their assignment-specified minimum accuracy targets (cells 90%, chest 87%, lesions 67%, orgs 83%) comfortably.

### Trade-off analysis

Across all four datasets, `GreenVGG16` relative to `VGG16`:

- **~2.1x faster training** (2.06x cells, 2.16x chest, 2.04x lesions, 2.07x orgs)
- **~2.36x less peak training memory** (a consistent ~57.6% reduction on every dataset)
- **~2.1x lower inference latency per sample**
- **~3.6-5.0x less peak inference memory** (lesions sees the largest drop, ~5.0x)

And it does this **without sacrificing accuracy — it wins on every single dataset**: +0.74pp (cells), +3.21pp (chest), +2.14pp (lesions), +2.43pp (orgs) accuracy, with macro F1 gains as large as +9.66pp on `lesions`. This goes beyond the assignment's "comparable accuracy" bar — `GreenVGG16` is strictly better on both accuracy and cost, on every dataset tested.

A plausible explanation: at only 20 epochs, VGG16's larger capacity (512-channel final stages, 1024-unit dense head) likely overfits on datasets of this size, while GreenVGG16's smaller capacity is better matched to the amount of training data available, acting as implicit regularization on top of its efficiency benefit. Here, "green" and "more accurate" are not in tension.

### Recommendation

Replace `VGG16` with `GreenVGG16` as the standard VGG-family architecture across all four datasets. There is no dataset in this comparison where `VGG16` offers any advantage — it is simultaneously worse in accuracy/F1 and more expensive on every resource axis measured (training time, training memory, inference latency, inference memory). `GreenVGG16` is the clear choice for deployment on resource-constrained diagnostic devices.
