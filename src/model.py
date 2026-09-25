"""
Model construction: EVA-02 backbone via timm, plus the differential
learning-rate optimizer setup (head vs. backbone) and the cosine scheduler.

Ported from notebook cells 14 and 16.
"""
import timm
import torch

from src.config import ModelConfig, TrainConfig


def build_model(cfg: ModelConfig, device: str) -> torch.nn.Module:
    model = timm.create_model(
        cfg.backbone,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded {cfg.backbone} — {total_params:.1f}M parameters — input {cfg.img_size}x{cfg.img_size}")
    return model


def split_param_groups(model: torch.nn.Module):
    """Split parameters into (head_params, backbone_params) for differential LR.

    Assumes the classification head is the module named `head`, matching
    timm's EVA-02 implementation.
    """
    head_params = list(model.head.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("head")]
    return head_params, backbone_params


def build_optimizer_and_scheduler(model: torch.nn.Module, cfg: TrainConfig):
    head_params, backbone_params = split_param_groups(model)

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": cfg.lr_head},
            {"params": backbone_params, "lr": cfg.lr_backbone},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-7
    )
    return optimizer, scheduler, backbone_params
