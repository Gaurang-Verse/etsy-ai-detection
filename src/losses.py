"""Loss functions. Ported from the original notebook's FocalLoss (cell 16)."""
import torch
import torch.nn as nn

from src.config import LossConfig


class FocalLoss(nn.Module):
    """Binary Focal Loss.

    gamma=2.0 — standard for detection; punishes confident wrong predictions.
    alpha=0.25 — slight down-weight of the majority class.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * bce
        return loss.mean()


def build_loss(cfg: LossConfig) -> nn.Module:
    if cfg.name == "focal":
        return FocalLoss(gamma=cfg.gamma, alpha=cfg.alpha)
    if cfg.name == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown loss: {cfg.name}")
