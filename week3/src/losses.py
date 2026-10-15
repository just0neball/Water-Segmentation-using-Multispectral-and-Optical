"""BCE + Dice loss.

`dice_mode="batch"` pools all pixels of the batch into one Dice score (this is
what produced the reported Week 2 numbers). `dice_mode="per_image"` averages a
Dice score per image. Check which variant the Week 1 training used and keep both
weeks identical, otherwise the comparison is not like-for-like.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, mode: str = "batch"):
        super().__init__()
        if mode not in {"batch", "per_image"}:
            raise ValueError("mode must be 'batch' or 'per_image'")
        self.smooth = smooth
        self.mode = mode

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if self.mode == "batch":
            probs, targets = probs.reshape(-1), targets.reshape(-1)
            inter = (probs * targets).sum()
            dice = (2.0 * inter + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
            return 1.0 - dice

        probs, targets = probs.flatten(1), targets.flatten(1)
        inter = (probs * targets).sum(dim=1)
        dice = (2.0 * inter + self.smooth) / (probs.sum(1) + targets.sum(1) + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 smooth: float = 1.0, dice_mode: str = "batch"):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss(smooth=smooth, mode=dice_mode)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        return self.bce_weight * bce + self.dice_weight * self.dice(logits, targets)
