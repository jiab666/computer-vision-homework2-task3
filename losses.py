from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: Optional[int] = None) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        valid_mask = torch.ones_like(targets, dtype=torch.bool)
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index

        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0

        one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1)

        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        cardinality = torch.sum(probs + one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def build_loss(loss_name: str) -> nn.Module:
    if loss_name == "ce":
        return nn.CrossEntropyLoss()
    if loss_name == "dice":
        return DiceLoss()
    if loss_name == "combo":
        return CombinedLoss()
    raise ValueError(f"Unsupported loss: {loss_name}")
