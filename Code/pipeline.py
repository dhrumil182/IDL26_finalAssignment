"""
MAI/IDL SS26 - Final assignment.
Shared pipeline setup (data, model, trainer), training/inference
profiling, and result reporting - used by train.py and predict.py
so training and evaluation can be run independently of one another.

MG 6/6/2026
"""
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from fit import Trainer


def load_feature_weights(model, checkpoint_path, device):
    """
    Load pretrained feature weights while leaving the classifier
    randomly initialized.

    This enables transfer learning even when the source and target
    datasets have different numbers of classes.
    """

    print(f"Loading pretrained weights from: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model_dict = model.state_dict()

    pretrained_dict = {
        k: v
        for k, v in checkpoint.items()
        if (
            k in model_dict
            and model_dict[k].shape == v.shape
        )
    }

    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict)

    print(f"Loaded {len(pretrained_dict)} pretrained tensors.")
    missing = len(model_dict) - len(pretrained_dict)
    print(f"Skipped {missing} tensors.")

    return model

def freeze_feature_extractor(model):
    """
    Freeze the feature extractor while leaving the classifier - and the
    deepest backbone block - trainable. Leaving the last block unfrozen
    lets the most task-specific features fine-tune to the target domain,
    without risking overfitting the whole backbone on a small dataset.
    """

    if hasattr(model, "features"):
        # AlexNet / VGG / GreenVGG
        blocks = list(model.features.children())
        for block in blocks[:-1]:
            for param in block.parameters():
                param.requires_grad = False

    else:
        # ResNet18
        modules = [
            "conv1",
            "bn1",
            "stage1",
            "stage2",
            "stage3",
        ]

        for name in modules:

            if hasattr(model, name):

                module = getattr(model, name)

                for param in module.parameters():
                    param.requires_grad = False

    return model


def build_pipeline(config):
    """Shared setup: seeds, data loaders, model, trainer
    """
    import random
    random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["SEED"])
        # cuDNN algorithm selection is not bit-reproducible by default;
        # force deterministic, non-benchmarked algorithms.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, num_channels, num_classes, input_size = get_loaders(
        data=config["DATA"],
        data_path=config["DATA_PATH"],
        batch_size=config["BATCH_SIZE"],
        val_split=config["VAL_SPLIT"],
        seed=config["SEED"],
        augment=config.get("AUGMENT", False)
    )

    model_class = getattr(models, config["MODEL"])
    model = model_class(
        in_channels=num_channels,
        num_classes=num_classes,
        drop_rate=config["DROPOUT"],
        activation_str=config["ACTIVATION"],
        input_size=input_size
    ).to(device)

    # Transfer Learning
    if config.get("TRANSFER_LEARNING", False):

        checkpoint_path = config["PRETRAINED_MODEL_PATH"]

        if not checkpoint_path:
            raise ValueError(
                "TRANSFER_LEARNING=True but PRETRAINED_MODEL_PATH is empty."
            )

        print(f"Transfer learning enabled.")
        print(f"Loading checkpoint: {checkpoint_path}")

        model = load_feature_weights(
            model,
            checkpoint_path,
            device,
        )

        freeze_feature_extractor(model)

        print("Feature extractor frozen (last stage left trainable for fine-tuning).")

    criterion_class = getattr(nn, config["LOSS"])
    criterion = criterion_class()
    optimizer_class = getattr(optim, config["OPTIMIZER"])

    if config.get("TRANSFER_LEARNING", False):
        backbone_lr_multiplier = config.get("BACKBONE_LR_MULTIPLIER", 0.1)
        head_params = list(model.classifier.parameters())
        head_param_ids = {id(p) for p in head_params}
        backbone_params = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in head_param_ids
        ]

        weight_decay = config.get("WEIGHT_DECAY", 1e-4)

        optimizer = optimizer_class(
            [
                {"params": head_params, "lr": config["LEARNING_RATE"]},
                {"params": backbone_params, "lr": config["LEARNING_RATE"] * backbone_lr_multiplier},
            ],
            weight_decay=weight_decay,
        )
    else:
        trainable_parameters = filter(
            lambda p: p.requires_grad,
            model.parameters(),
        )

        optimizer = optimizer_class(
            trainable_parameters,
            lr=config["LEARNING_RATE"]
        )
    
    trainer = Trainer(model, criterion, optimizer, device, save_model=config["SAVE_MODEL"], model_path=config["MODEL_PATH"])

    return device, train_loader, val_loader, test_loader, model, trainer


def run_training(trainer, train_loader, val_loader, epochs, device):
    """Runs Trainer.fit() while measuring wall-clock runtime and peak
    memory"""
    start = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history = trainer.fit(train_loader, val_loader, epochs=epochs)

    duration_sec = time.perf_counter() - start

    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )
    else:
        peak_memory_mb = None

    return {
        "train_runtime_sec": duration_sec,
        "train_peak_memory_mb": peak_memory_mb,
        "history": history,
    }


def run_evaluation(trainer, test_loader, device):
    """Runs Trainer.evaluate() while measuring total runtime, per-sample
    inference latency, and peak memory, on CUDA or CPU alike. Returns the
    classification metrics dict merged with profiling fields."""
    start = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    metrics = trainer.evaluate(test_loader)

    duration_sec = time.perf_counter() - start

    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )
    else:
        peak_memory_mb = None

    num_samples = len(test_loader.dataset)
    latency_ms_per_sample = (duration_sec / num_samples) * 1000

    metrics.update({
        "inference_runtime_sec": duration_sec,
        "inference_latency_ms_per_sample": latency_ms_per_sample,
        "inference_peak_memory_mb": peak_memory_mb,
    })
    return metrics


def report_metrics(config, tag, metrics):
    """Prints a metrics dict in a readable format. Does not save."""
    print("-" * 50)
    print(f"Results [{config['MODEL']} on {config['DATA']} - {tag}]")
    for key, value in metrics.items():
        if value is None:
            continue
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print("-" * 50)


def save_results(config, tag, **fields):
    """Merges `fields` into results/{dataset}_{model}_{tag}.json, creating
    or updating it as needed. Used by both train.py (train-side profiling)
    and predict.py (test metrics + inference profiling) so both halves of
    a single run end up in the same result record."""
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{config['DATA']}_{config['MODEL']}_{tag}.json"

    record = {}
    if result_path.exists():
        with open(result_path, "r") as f:
            record = json.load(f)

    record.update({"model": config["MODEL"], "dataset": config["DATA"], "tag": tag})
    record.update(fields)

    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"Results saved to {result_path}")
    return result_path
