"""
MAI/IDL SS26 - Final assignment.

Benchmark runner.

Executes every experiment defined in BENCHMARK_CONFIG using the
shared training and evaluation pipeline.

MG 6/6/2026
"""

import copy
import csv
import itertools
import json
from pathlib import Path
from train import plot_loss_curve
import torch

from pipeline import (
    build_pipeline,
    run_training,
    run_evaluation,
)

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

def load_configs():
    """
    Load the runtime, experiment and benchmark configurations.
    """

    with open("config.json", "r") as f:
        config = json.load(f)

    return (
        config["BASE_CONFIG"]["RUN_CONFIG"],
        config["BASE_CONFIG"]["EXPERIMENT_CONFIG"],
        config["BENCHMARK_CONFIG"]
    )


# ----------------------------------------------------------
# Experiment generation
# ----------------------------------------------------------

def generate_experiments(
    run_config,
    experiment_config,
    benchmark_config,
):
    """
    Generate every experiment from the benchmark grid.
    """

    grid = benchmark_config["GRID"]

    keys = list(grid.keys())

    values = [grid[key] for key in keys]

    # Base directory for checkpoints, taken from RUN_CONFIG["MODEL_PATH"]
    # (e.g. "checkpoints/best_model.pt" -> "checkpoints").
    checkpoint_dir = Path(run_config["MODEL_PATH"]).parent

    tag = benchmark_config.get("TAG", "benchmark")

    experiments = []

    for combination in itertools.product(*values):

        experiment = copy.deepcopy(experiment_config)

        for key, value in zip(keys, combination):
            experiment[key] = value

        base_model = experiment_config.get("MODEL")
        base_pretrained_path = experiment_config.get("PRETRAINED_MODEL_PATH")
        if (
            "MODEL" in grid
            and experiment.get("TRANSFER_LEARNING", False)
            and base_model
            and base_pretrained_path
        ):
            experiment["PRETRAINED_MODEL_PATH"] = base_pretrained_path.replace(
                base_model, experiment["MODEL"]
            )

        config = {
            **copy.deepcopy(run_config),
            **experiment
        }

        config["TAG"] = tag

        exp_id = "_".join(str(experiment[key]) for key in keys)
        config["MODEL_PATH"] = str(
            checkpoint_dir / f"{tag}_{exp_id}.pt"
        )

        experiments.append({
        "config": config,
        "experiment": copy.deepcopy(experiment)
        })

    return experiments

# ----------------------------------------------------------
# Print experiment string
# ----------------------------------------------------------
def print_experiment(experiment):

    print("Experiment Config:")
    print("=" * 80)

    for key in sorted(experiment):

        print(f"{key:<20}: {experiment[key]}")

    print("=" * 80)

# ----------------------------------------------------------
# Single experiment
# ----------------------------------------------------------

def run_experiment(config, experiment):
    
    print_experiment(experiment)

    (
        device,
        train_loader,
        val_loader,
        test_loader,
        model,
        trainer,
    ) = build_pipeline(config)

    train_profile = run_training(
        trainer,
        train_loader,
        val_loader,
        epochs=config["EPOCHS"],
        device=device,
    )
    history = train_profile.pop("history")

    exp_id = Path(config["MODEL_PATH"]).stem
    plot_path = Path("results") / f"{exp_id}_loss_curve.png"
    plot_loss_curve(history, plot_path)

    checkpoint = Path(config["MODEL_PATH"])

    if checkpoint.exists():

        model.load_state_dict(
            torch.load(
                checkpoint,
                map_location=device,
            )
        )

    metrics = run_evaluation(
        trainer,
        test_loader,
        device,
    )

    result = dict(experiment)

    result.update(train_profile)

    result.update(metrics)

    return result


# ----------------------------------------------------------
# Reporting
# ----------------------------------------------------------

def _format_optional_mb(value):
    """Peak memory is None when running on CPU (no CUDA stats available).
    Format consistently so the summary table stays aligned either way."""
    if value is None:
        return "N/A"
    return f"{value:.1f}"


def summarize_results(results):

    print("\n")

    print("=" * 110)

    print("BENCHMARK SUMMARY")

    print("=" * 110)

    header = (
        f"{'Dataset':<12}"
        f"{'Model':<12}"
        f"{'Accuracy':>12}"
        f"{'Macro F1':>12}"
        f"{'Train_runtime(s)':>12}"
        f"{'Infer_runtime(s)':>12}"
        f"{'TrainMem(MB)':>14}"
        f"{'InferMem(MB)':>14}"
        f"{'Infer_lat_per(ms/smp)':>14}"
    )

    print(header)

    print("-" * len(header))

    for r in results:

        train_mem = _format_optional_mb(r.get("train_peak_memory_mb"))

        infer_mem = _format_optional_mb(r.get("inference_peak_memory_mb"))

        print(

           f"{r['DATA']:<12}"

            f"{r['MODEL']:<12}"

            f"{r['accuracy']:>11.2f}%"

            f"{r['macro_f1']:>11.2f}%"

            f"{r['train_runtime_sec']:>12.2f}"

            f"{r['inference_runtime_sec']:>12.2f}"

            f"{train_mem:>14}"

            f"{infer_mem:>14}"

            f"{r['inference_latency_ms_per_sample']:>14.3f}"

        )


def save_benchmark(results, tag):

    output_dir = Path("results")

    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / f"{tag}_results.csv"

    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(csv_path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(results)

    print("\nSaved benchmark results:")

    print(csv_path)


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    run_config, experiment_config, benchmark_config = load_configs()

    experiments = generate_experiments(run_config,
        experiment_config,
        benchmark_config,
    )

    results = []

    for idx, experiment in enumerate(experiments):
        print("=" * 80)
        print(f"Running experiment {idx + 1}/{len(experiments)}")
        result = run_experiment(
        experiment["config"],
        experiment["experiment"]
        )

        results.append(result)

    summarize_results(results)

    save_benchmark(results, benchmark_config.get("TAG", "benchmark"))

    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()
