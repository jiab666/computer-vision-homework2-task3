import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.pet import OxfordPetSegmentation
from experiment_logger import ExperimentLogger
from losses import build_loss
from models.unet import UNet
from plot_curves import plot_training_curves
from utils import (
    compute_confusion_matrix,
    compute_iou_from_confmat,
    compute_pixel_accuracy_from_confmat,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net from scratch on Oxford-IIIT Pet")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./outputs")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--loss", type=str, choices=["ce", "dice", "combo"], default="ce")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--logger", type=str, choices=["none", "wandb", "swanlab"], default="none")
    parser.add_argument("--project", type=str, default="cv-hw2-task3")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--run-all", action="store_true")
    return parser.parse_args()


def create_dataloaders(args: argparse.Namespace) -> Dict[str, DataLoader]:
    train_dataset = OxfordPetSegmentation(
        root=args.data_root,
        split="trainval",
        image_size=args.image_size,
        download=args.download,
    )
    val_dataset = OxfordPetSegmentation(
        root=args.data_root,
        split="test",
        image_size=args.image_size,
        download=args.download,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return {"train": train_loader, "val": val_loader}


def train_one_epoch(
    model: UNet,
    loader: DataLoader,
    optimizer: Adam,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: UNet,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int = 3,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for images, masks in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1).cpu()
        confmat += compute_confusion_matrix(preds, masks.cpu(), num_classes)

    metrics = compute_iou_from_confmat(confmat)
    metrics["pixel_accuracy"] = compute_pixel_accuracy_from_confmat(confmat)
    metrics["val_loss"] = total_loss / len(loader.dataset)
    return metrics


def save_checkpoint(
    path: Path,
    model: UNet,
    optimizer: Adam,
    epoch: int,
    best_miou: float,
    history: List[Dict[str, object]],
    args: argparse.Namespace,
    loss_name: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_miou": best_miou,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "loss_name": loss_name,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    model: UNet,
    optimizer: Adam,
    device: torch.device,
) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def run_experiment(args: argparse.Namespace, loss_name: str) -> Dict[str, object]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = create_dataloaders(args)

    model = UNet(in_channels=3, num_classes=3, base_channels=args.base_channels).to(device)
    criterion = build_loss(loss_name)
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_miou = -1.0
    history: List[Dict[str, object]] = []
    save_dir = Path(args.save_dir) / loss_name
    save_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = save_dir / "best_model.pt"
    last_checkpoint_path = save_dir / "last_checkpoint.pt"
    start_epoch = 1
    run_name = args.run_name or f"{loss_name}_bs{args.batch_size}_lr{args.lr}_img{args.image_size}"
    logger = ExperimentLogger(
        backend=args.logger,
        project=args.project,
        run_name=run_name,
        save_dir=save_dir,
        config={
            "loss": loss_name,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "image_size": args.image_size,
            "base_channels": args.base_channels,
            "optimizer": "Adam",
            "dataset": "Oxford-IIIT Pet",
            "num_classes": 3,
        },
    )

    resume_path: Optional[Path] = None
    if args.resume is not None:
        resume_path = Path(args.resume)
    elif last_checkpoint_path.exists():
        resume_path = last_checkpoint_path

    if resume_path is not None and resume_path.exists():
        checkpoint = load_checkpoint(resume_path, model, optimizer, device)
        checkpoint_loss_name = checkpoint.get("loss_name")
        if checkpoint_loss_name is not None and checkpoint_loss_name != loss_name:
            raise ValueError(
                f"Checkpoint loss '{checkpoint_loss_name}' does not match current loss '{loss_name}'"
            )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_miou = float(checkpoint.get("best_miou", -1.0))
        history = list(checkpoint.get("history", []))
        print(f"Resumed {loss_name} from epoch {start_epoch - 1} using {resume_path}")

    start_time = time.time()
    try:
        if start_epoch > args.epochs:
            print(f"[{loss_name}] Training already reached epoch {args.epochs}, skipping.")

        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
            val_metrics = evaluate(model, loaders["val"], criterion, device)

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["val_loss"],
                "pixel_accuracy": val_metrics["pixel_accuracy"],
                "miou": val_metrics["miou"],
                "iou_per_class": val_metrics["iou_per_class"],
            }
            history.append(epoch_record)

            print(
                f"[{loss_name}] Epoch {epoch:02d}/{args.epochs} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['val_loss']:.4f} "
                f"pixel_acc={val_metrics['pixel_accuracy']:.4f} "
                f"miou={val_metrics['miou']:.4f}"
            )

            logger.log(
                {
                    "train/loss": train_loss,
                    "val/loss": val_metrics["val_loss"],
                    "val/pixel_accuracy": val_metrics["pixel_accuracy"],
                    "val/miou": val_metrics["miou"],
                },
                step=epoch,
            )

            if val_metrics["miou"] > best_miou:
                best_miou = val_metrics["miou"]
                save_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_miou=best_miou,
                    history=history,
                    args=args,
                    loss_name=loss_name,
                )

            save_checkpoint(
                path=last_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_miou=best_miou,
                history=history,
                args=args,
                loss_name=loss_name,
            )
    finally:
        logger.finish()

    elapsed = time.time() - start_time
    result = {
        "loss_name": loss_name,
        "best_miou": best_miou,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "image_size": args.image_size,
        "base_channels": args.base_channels,
        "elapsed_seconds": elapsed,
        "history": history,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
    }
    save_json(result, str(save_dir / "metrics.json"))
    plot_training_curves(history, save_dir)
    return result


def main() -> None:
    args = parse_args()
    loss_names = ["ce", "dice", "combo"] if args.run_all else [args.loss]

    all_results = []
    for loss_name in loss_names:
        result = run_experiment(args, loss_name)
        all_results.append(
            {
                "loss_name": result["loss_name"],
                "best_miou": result["best_miou"],
                "best_checkpoint": result["best_checkpoint"],
            }
        )

    summary = {"results": all_results}
    save_json(summary, str(Path(args.save_dir) / "summary.json"))

    print("\n=== Final Summary ===")
    for item in all_results:
        print(f"{item['loss_name']}: best_miou={item['best_miou']:.4f}")


if __name__ == "__main__":
    main()
