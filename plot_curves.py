from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_training_curves(history: List[Dict[str, object]], save_dir: Path) -> None:
    if not history:
        return

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    val_accuracy = [item["pixel_accuracy"] for item in history]
    val_miou = [item["miou"] for item in history]

    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    plt.plot(epochs, val_loss, marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_accuracy, marker="o", label="Val Pixel Accuracy")
    plt.plot(epochs, val_miou, marker="o", label="Val mIoU")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Metrics")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_dir / "val_metrics_curve.png", dpi=200)
    plt.close()
