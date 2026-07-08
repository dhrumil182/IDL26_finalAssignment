# Consolidated Benchmark Report — Operation Cyber-Histology

## Section 1 — Pipeline Reconstruction Baseline

**Run configuration** (`BATCH_SIZE=128`, `LEARNING_RATE=0.001`, `OPTIMIZER=Adam`, `LOSS=CrossEntropyLoss`, `DROPOUT=0.5`, `ACTIVATION=ReLU`, `VAL_SPLIT=0.1`, `EPOCHS=20`)

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

| Dataset | Target | AlexNet | VGG16 | ResNet18 | Pass |
|---------|-------:|--------:|------:|---------:|:---------:|
| cells   | 90% | 93.86% | 96.40% | 97.25% | Yes |
| chest   | 87% | 90.71% | 88.14% | 87.18% | Yes |
| lesions | 67% | 75.76% | 72.67% | 75.76% | Yes |
| orgs    | 83% | 89.97% | 89.57% | 92.05% | Yes |

All 12 dataset/model combinations meet their minimum test accuracy.

### Architectural recommendations

| Dataset | Recommended Model | Key Metrics | Why |
|---------|:---:|---|---|
| chest   | AlexNet  | 90.71% acc / 89.59% F1 | Best accuracy **and** cheapest (8.58s train, 399MB). VGG16/ResNet18 cost 6-13x more compute and 5-8x more memory while scoring *lower* on every metric — extra capacity actively hurts here, likely overfitting at 20 epochs on a comparatively simple/lower-resolution modality. Clearest result in the whole grid. |
| lesions | AlexNet  | 75.76% acc / 49.73% F1 | Ties ResNet18 for best accuracy and wins F1 by a wide margin (vs. 41.23% ResNet18, 33.68% VGG16) at ~1/11th ResNet18's training time. Note: precision/recall/F1 sit far below accuracy for all three models (33-55% vs. 73-76%) — a sign of class imbalance or systematic per-class confusion that accuracy alone hides. |
| cells   | ResNet18 (AlexNet if compute-constrained) | 97.25% acc / 97.21% F1 | ResNet18 wins outright, but AlexNet's 93.86%/93.16% F1 stays well clear of the 90% target at ~11x less training time and ~8x less memory — a modest but real capacity/cost trade-off. |
| orgs    | ResNet18 | 92.05% acc / 90.88% F1 | Wins outright over AlexNet's already-solid 89.97%/88.52% F1. |

**Overall: AlexNet is the standout general-purpose model in this benchmark.** It is never more than ~2-3 points of accuracy behind the larger architectures, is outright the best model on `chest` and tied-best on `lesions`, and is consistently 8-13x cheaper on training time and memory than ResNet18 across every dataset.

## Section 2 — Green Initiative

**Run configuration**: identical to Section 1 (`BATCH_SIZE=128`, `LEARNING_RATE=0.001`, `Adam`, `CrossEntropyLoss`, `DROPOUT=0.5`, `ACTIVATION=ReLU`, `VAL_SPLIT=0.1`, `EPOCHS=20`) — only the architecture varies between `VGG16` and `GreenVGG16` across all four datasets.

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

All four `GreenVGG16` runs clear their minimum accuracy targets (cells 90%, chest 87%, lesions 67%, orgs 83%) comfortably.

### Trade-off analysis

Across all four datasets, `GreenVGG16` relative to `VGG16`:

- **~2.1x faster training** (2.06x cells, 2.16x chest, 2.04x lesions, 2.07x orgs)
- **~2.36x less peak training memory** (a consistent ~57.6% reduction on every dataset)
- **~2.1x lower inference latency per sample**
- **~3.6-5.0x less peak inference memory** (lesions sees the largest drop, ~5.0x)

And it does this without sacrificing accuracy — it wins on every single dataset: +0.74pp (cells), +3.21pp (chest), +2.14pp (lesions), +2.43pp (orgs) accuracy, with macro F1 gains as large as +9.66pp on `lesions`.

### Recommendation

Replace `VGG16` with `GreenVGG16` as the standard VGG-family architecture across all four datasets. There is no dataset in this comparison where `VGG16` offers any advantage — it is simultaneously worse in accuracy/F1 and more expensive on every resource axis measured (training time, training memory, inference latency, inference memory). `GreenVGG16` is the clear choice for deployment on resource-constrained diagnostic devices.

## Section 3 — New Dataset - organs

**Run configuration**: `BATCH_SIZE=8` (reduced from Section 1/2's 128 to suit the 500-sample training set), `LEARNING_RATE=0.001`, `OPTIMIZER=Adam`, `LOSS=CrossEntropyLoss`, `DROPOUT=0.5`, `ACTIVATION=ReLU`, `VAL_SPLIT=0.1`, `EPOCHS=20`. Grid: `MODEL={AlexNet, VGG16, ResNet18} x TRANSFER_LEARNING={True, False} x AUGMENT={True, False}`.

**Transfer learning strategy**: pretrained weights are loaded by name/shape match only (so classifier heads, which differ in class count between source and target tasks, are left randomly initialized). The feature extractor is then frozen except for its deepest block/stage, which stays trainable so the most task-specific features can adapt to the new domain without risking overfitting the whole backbone on 500 samples. The optimizer uses a differential learning rate, full `LEARNING_RATE` on the classifier head, `0.1x` on the unfrozen backbone block, plus `1e-4` weight decay.

### Summary table

| Model | Transfer Learning | Augment | Accuracy | Precision | Recall | Macro F1 |
|-------|:---:|:---:|---------:|----------:|-------:|---------:|
| ResNet18 | Yes | Yes | 69.50% | 67.17% | 63.01% | 63.08% |
| ResNet18 | Yes | No  | 70.00% | 66.27% | 63.75% | 63.36% |
| ResNet18 | No  | Yes | 57.50% | 53.30% | 50.96% | 48.71% |
| ResNet18 | No  | No  | 57.00% | 58.72% | 49.93% | 48.76% |
| AlexNet  | Yes | Yes | 58.00% | 60.03% | 54.43% | 53.49% |
| AlexNet  | Yes | No  | 61.00% | 64.98% | 56.40% | 54.78% |
| AlexNet  | No  | Yes | 58.50% | 52.97% | 49.26% | 47.29% |
| AlexNet  | No  | No  | 54.50% | 47.60% | 45.29% | 42.49% |
| VGG16    | Yes | Yes | 59.00% | 51.57% | 51.37% | 48.09% |
| VGG16    | Yes | No  | 58.50% | 51.51% | 50.24% | 48.62% |
| VGG16    | No  | Yes | 40.00% | 21.81% | 30.24% | 23.05% |
| VGG16    | No  | No  | 41.00% | 19.13% | 31.59% | 22.26% |

### Efficiency

| Model | Transfer Learning | Train Runtime (s) | Train Peak Mem (MB) | Inference Latency (ms/sample) | Inference Peak Mem (MB) |
|-------|:---:|-------------------:|----------------------:|--------------------------------:|---------------------------:|
| ResNet18 | Yes | ~11-21 | ~211-212 | ~0.51 | ~206-207 |
| ResNet18 | No  | ~21-32 | ~381 | ~0.50 | ~240-241 |
| AlexNet  | Yes | ~5-16  | ~102  | ~0.20-0.21 | ~99.6 |
| AlexNet  | No  | ~6-20  | ~128  | ~0.19-0.21 | ~117.6 |
| VGG16    | Yes | ~10-21 | ~201  | ~0.37-0.39 | ~201.1 |
| VGG16    | No  | ~13-26 | ~337  | ~0.34-0.35 | ~259.9 |

Freezing most of the backbone under transfer learning also improves training efficiency. fewer trainable parameters means lower peak training memory across all three architectures (e.g. ResNet18: 381MB -> 211MB, VGG16: 337MB -> 201MB), and inference memory drops similarly since the loaded checkpoint's frozen stages are identical either way but training-time activation caching for gradients shrinks.

### Target check

The chief of medical testing set a bar of at least 40% test accuracy. All 12 configurations clear it; the lowest score in the entire grid is VGG16 trained from scratch (40.00-41.00%), while every transfer-learning configuration reaches 58-70%.

### Recommendation

Use transfer learning from the `orgs`-pretrained checkpoint as the default strategy for the `organs` dataset, with `ResNet18` as the architecture of choice. `VGG16` is a reasonable second choice only when paired with transfer learning on its own it is not viable at this sample size. For future data collection: since the accuracy gap between transfer learning and scratch training shrinks as more labeled `organs` data becomes available, the recommended path is to keep using the pretrained backbone as a warm start (rather than discarding it) and gradually unfreeze earlier stages as the `organs` training set grows.
