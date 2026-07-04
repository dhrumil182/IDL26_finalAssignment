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
