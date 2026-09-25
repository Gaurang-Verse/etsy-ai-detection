"""
Data pipeline: missing-file filtering, the EtsyDataset class, and the
Albumentations transform builders.

Ported from the original notebook's cells 8 and 12, unchanged in logic —
only parameterised via Config so it runs identically on CPU (local, small
debug subset) and GPU (Colab, full dataset).
"""
import os

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from src.config import AugConfig, DataConfig, ModelConfig


def filter_missing_images(df: pd.DataFrame, img_dir: str, image_id_col: str = "image_id") -> pd.DataFrame:
    """Drop rows whose image file doesn't exist on disk.

    Prevents the DataLoader from ever trying to read a missing file, which
    otherwise crashes training mid-epoch.
    """
    exists = df[image_id_col].apply(lambda x: os.path.exists(os.path.join(img_dir, x)))
    missing = int((~exists).sum())
    if missing > 0:
        print(f"Removed {missing} missing images from dataset.")
    return df[exists].reset_index(drop=True)


def build_transforms(model_cfg: ModelConfig, aug_cfg: AugConfig):
    """Return (train_transform, val_transform)."""
    img_size = model_cfg.img_size

    train_ops = [
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=aug_cfg.horizontal_flip_p),
        A.VerticalFlip(p=aug_cfg.vertical_flip_p),
        A.RandomBrightnessContrast(
            brightness_limit=aug_cfg.brightness_limit,
            contrast_limit=aug_cfg.contrast_limit,
            p=aug_cfg.brightness_contrast_p,
        ),
        A.Affine(
            scale=(aug_cfg.affine_scale_min, aug_cfg.affine_scale_max),
            translate_percent=0.05,
            rotate=(-aug_cfg.affine_rotate_deg, aug_cfg.affine_rotate_deg),
            p=aug_cfg.affine_p,
        ),
        A.ImageCompression(
            quality_lower=aug_cfg.jpeg_quality_min,
            quality_upper=aug_cfg.jpeg_quality_max,
            p=aug_cfg.image_compression_p,
        ),
        A.GaussNoise(
            var_limit=(aug_cfg.gauss_noise_var_min, aug_cfg.gauss_noise_var_max),
            p=aug_cfg.gauss_noise_p,
        ),
    ]

    if aug_cfg.cutout_enabled:
        train_ops.append(
            A.CoarseDropout(
                max_holes=aug_cfg.cutout_num_holes,
                max_height=aug_cfg.cutout_max_h_size,
                max_width=aug_cfg.cutout_max_w_size,
                p=0.5,
            )
        )

    train_ops += [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    train_transform = A.Compose(train_ops)

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform, val_transform


class EtsyDataset(Dataset):
    """Loads an image by id and returns (tensor, label) or (tensor, image_id) for test."""

    def __init__(self, df: pd.DataFrame, image_dir: str, transform=None,
                 is_test: bool = False, image_id_col: str = "image_id",
                 label_col: str = "ground_truth", img_size: int = 448):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        self.image_id_col = image_id_col
        self.label_col = label_col
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx][self.image_id_col]
        img_path = os.path.join(self.image_dir, img_name)
        if not os.path.exists(img_path):
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        if self.is_test:
            return image, img_name

        label = self.df.iloc[idx][self.label_col]
        return image, torch.tensor(label, dtype=torch.float32)
