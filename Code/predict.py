"""
MAI/IDL SS26 - Final assignment.
Standalone evaluation entry point: loads a saved checkpoint and runs
test-set evaluation only, without retraining. Useful for re-running
benchmarks or generating predictions independently of train.py.

MG 6/6/2026
"""
import json
from pathlib import Path
import torch
from pipeline import build_pipeline, run_evaluation, report_metrics, save_results


def main():
    with open("config.json", "r") as f:
        cfg = json.load(f)
        run_config = cfg["BASE_CONFIG"]["RUN_CONFIG"]
        experiment_config = cfg["BASE_CONFIG"]["EXPERIMENT_CONFIG"]
        config = {
            **run_config,
            **experiment_config
        }

    device, _, _, test_loader, model, trainer = build_pipeline(config)

    checkpoint_path = Path(config["MODEL_PATH"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            f"Run train.py first, or check MODEL_PATH in config.json."
        )

    print(f"Loading checkpoint from {checkpoint_path} for evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_metrics = run_evaluation(trainer, test_loader, device)

    tag = config.get("TAG", "baseline")
    report_metrics(config, tag, test_metrics)
    save_results(config, tag, **test_metrics)


if __name__ == "__main__":
    main()
