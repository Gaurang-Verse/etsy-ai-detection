"""
Config-driven hyperparameters.

Instead of hardcoding paths/hyperparameters inline (as in the original
notebook), everything lives in a YAML file under configs/. This lets the
exact same code run:
  - locally on a MacBook (CPU/MPS) with a tiny debug config, or
  - on Colab T4 with the full training config,
by swapping --config, not editing source.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class DataConfig:
    base_path: str = "/content/drive/MyDrive/[External] DCU 2026 ML challenge - external"
    zip_name: str = "genai_image_challenge.zip"
    images_dir: str = "/content/images"
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    image_id_col: str = "image_id"
    label_col: str = "ground_truth"
    val_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelConfig:
    backbone: str = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"
    img_size: int = 448
    num_classes: int = 1
    pretrained: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 10
    lr_head: float = 1e-3
    lr_backbone: float = 2e-5
    weight_decay: float = 0.01
    warmup_epochs: int = 1
    num_workers: int = 0
    amp: bool = True
    device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"


@dataclass
class LossConfig:
    name: str = "focal"  # "focal" | "bce"
    gamma: float = 2.0
    alpha: float = 0.25


@dataclass
class AugConfig:
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.2
    brightness_contrast_p: float = 0.4
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    affine_p: float = 0.5
    affine_scale_min: float = 0.9
    affine_scale_max: float = 1.1
    affine_rotate_deg: float = 15.0
    image_compression_p: float = 0.5
    jpeg_quality_min: int = 60
    jpeg_quality_max: int = 99
    gauss_noise_p: float = 0.4
    gauss_noise_var_min: float = 10.0
    gauss_noise_var_max: float = 50.0
    cutout_enabled: bool = False
    cutout_num_holes: int = 8
    cutout_max_h_size: int = 32
    cutout_max_w_size: int = 32


@dataclass
class OutputConfig:
    checkpoint_path: str = "checkpoints/best_eva02_checkpoint.pth"
    submission_path: str = "outputs/submission_eva02.csv"
    metadata_path: str = "outputs/experiment_summary_eva02.json"
    figures_dir: str = "outputs/figures"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str]) -> Config:
    """Load a Config from YAML, falling back to defaults for anything unset."""
    default = Config().to_dict()
    if path:
        with open(path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        merged = _merge(default, user_cfg)
    else:
        merged = default

    return Config(
        data=DataConfig(**merged["data"]),
        model=ModelConfig(**merged["model"]),
        train=TrainConfig(**merged["train"]),
        loss=LossConfig(**merged["loss"]),
        aug=AugConfig(**merged["aug"]),
        output=OutputConfig(**merged["output"]),
    )


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
