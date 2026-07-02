"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import json
from pathlib import Path
from pipeline import build_pipeline, run_training, save_results
import matplotlib.pyplot as plt

def plot_loss_curve(history, save_path):
    """Plots training and validation loss per epoch and saves it to
    `save_path` (parent directories are created if needed)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Loss curve saved to {save_path}")

def main():
    with open("config.json", "r") as f:
        cfg = json.load(f)
        run_config = cfg["BASE_CONFIG"]["RUN_CONFIG"]
        experiment_config = cfg["BASE_CONFIG"]["EXPERIMENT_CONFIG"]
        config = {
            **run_config,
            **experiment_config
        }

    device, train_loader, val_loader, test_loader, model, trainer = build_pipeline(config)
    print(f"Training executing on device: {device}")

    train_profile = run_training(trainer, train_loader, val_loader, epochs=config["EPOCHS"], device=device)
    history = train_profile.pop("history")

    print(f"\nTraining runtime: {train_profile['train_runtime_sec']:.2f}s")
    if train_profile["train_peak_memory_mb"] is not None:
        print(f"Training peak memory: {train_profile['train_peak_memory_mb']:.1f} MB")

    tag = config.get("TAG", "baseline")
    save_results(config, tag, **train_profile)

    plot_path = Path("results") / f"{config['DATA']}_{config['MODEL']}_{tag}_loss_curve.png"
    plot_loss_curve(history, plot_path)

    print(f"\nTraining finished. Run predict.py to evaluate on the test set "
          f"(checkpoint: {config['MODEL_PATH']}).")


if __name__ == "__main__":
    main()
