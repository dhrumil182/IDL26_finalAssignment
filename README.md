# MAI - IDL 2026 — Final Project Assignment

**Author:** <Your Name> — Enrollment: <Your Enrollment Number>

## Overview

This repository is a post-incident reconstruction of BioHealth Diagnostics' clinical triage pipeline (Operation Cyber-Histology). It trains and evaluates three CNN architectures (`AlexNet`, `VGG16`, `ResNet18`, plus the Green-Initiative `GreenVGG16`) across four histology/imaging datasets (`cells`, `chest`, `lesions`, `orgs`), with a fourth low-sample dataset (`organs`) handled via transfer learning and augmentation.

See `AUDIT_LOG.md` for the itemized list of bugs found and fixed in the recovered codebase, and `REPORT.md` for benchmark results.

## Repository layout

```
Code/
  data.py        - dataset loading, train/val/test split, augmentation
  models.py      - AlexNet, VGG16, ResNet18, GreenVGG16
  fit.py         - Trainer: training loop, evaluation, checkpointing
  pipeline.py    - shared setup (data/model/trainer), training/inference profiling, transfer learning
  train.py       - single-run training entrypoint (reads config.json)
  predict.py     - standalone test-set evaluation from a saved checkpoint
  benchmark.py   - grid runner: sweeps BENCHMARK_CONFIG.GRID and writes a results CSV
  config.json    - all experiment/runtime/benchmark configuration
  checkpoints/   - saved model weights (created at runtime)
  results/       - per-run metrics, loss curves, benchmark CSVs (created at runtime)
data/            - dataset tensors (cells.pt, chest.pt, lesions.pt, orgs.pt, organs.pt)
AUDIT_LOG.md     - incident audit log (bug -> root cause -> fix -> commit)
REPORT.md        - benchmark results and analysis
requirements.txt - Python dependencies
```

## Prerequisites

- Python 3.10+
- The dataset files from the emergency backup, extracted into `data/` at the repository root (sibling of `Code/`)

Install dependencies:

```bash
pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install `torch`/`torchvision` from the matching wheel index instead of the default PyPI build, e.g.:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Configuration

Everything is driven by `Code/config.json`:

- `BASE_CONFIG.RUN_CONFIG` - environment-level settings: data path, checkpoint path, whether to save models, random seed.
- `BASE_CONFIG.EXPERIMENT_CONFIG` - a single experiment's hyperparameters: dataset, model, batch size, learning rate, epochs, optimizer, loss, dropout, activation, validation split, and (for Section 3) `TRANSFER_LEARNING`, `PRETRAINED_MODEL_PATH`, `AUGMENT`.
- `BENCHMARK_CONFIG.GRID` - one or more config keys mapped to lists of values; `benchmark.py` runs the cartesian product of all of them. `BENCHMARK_CONFIG.TAG` names the run (used in checkpoint/result filenames).

## Usage

All commands are run from the `Code/` directory.

Train a single model/dataset combination (uses `BASE_CONFIG` from `config.json`):

```bash
python train.py
```

Evaluate a saved checkpoint on the test set without retraining:

```bash
python predict.py
```

Run every combination in `BENCHMARK_CONFIG.GRID` (writes `results/{TAG}_results.csv` plus per-run loss curves):

```bash
python benchmark.py
```

To switch which sweep runs, edit `BENCHMARK_CONFIG` in `config.json` (see `REPORT.md` for which grids produced which results in this repo's history).

## Datasets and models

| Dataset | Role |
|---------|------|
| `cells` | Section 1 core dataset |
| `chest` | Section 1 core dataset |
| `lesions` | Section 1 core dataset |
| `orgs` | Section 1 core dataset; also the transfer-learning source for `organs` |
| `organs` | Section 3 low-sample dataset (transfer learning + augmentation) |

| Model | Role |
|-------|------|
| `AlexNet` | Section 1 baseline |
| `VGG16` | Section 1 baseline; Green Initiative comparison baseline |
| `ResNet18` | Section 1 baseline; Section 3 transfer-learning backbone |
| `GreenVGG16` | Section 2 Green Initiative lightweight architecture |
