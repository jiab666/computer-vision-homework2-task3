import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    preds = preds.view(-1)
    targets = targets.view(-1)

    valid = (targets >= 0) & (targets < num_classes)
    preds = preds[valid]
    targets = targets[valid]

    indices = num_classes * targets + preds
    conf = torch.bincount(indices, minlength=num_classes * num_classes)
    return conf.reshape(num_classes, num_classes)


def compute_iou_from_confmat(confmat: torch.Tensor) -> Dict[str, object]:
    true_positive = torch.diag(confmat).float()
    false_positive = confmat.sum(dim=0).float() - true_positive
    false_negative = confmat.sum(dim=1).float() - true_positive

    denominator = true_positive + false_positive + false_negative
    iou_per_class = torch.where(
        denominator > 0,
        true_positive / denominator,
        torch.zeros_like(denominator),
    )
    miou = iou_per_class.mean().item()

    return {
        "miou": miou,
        "iou_per_class": iou_per_class.tolist(),
    }


def compute_pixel_accuracy_from_confmat(confmat: torch.Tensor) -> float:
    total = confmat.sum().item()
    if total == 0:
        return 0.0
    correct = torch.diag(confmat).sum().item()
    return correct / total


def save_json(data: Dict[str, object], path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
