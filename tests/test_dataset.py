import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.dataset import EtsyDataset, filter_missing_images


@pytest.fixture
def tmp_image_dir(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for name in ["a.jpg", "b.jpg"]:
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(img_dir / name)
    return str(img_dir)


def test_filter_missing_images_drops_absent_files(tmp_image_dir):
    df = pd.DataFrame({
        "image_id": ["a.jpg", "b.jpg", "missing.jpg"],
        "ground_truth": [0, 1, 0],
    })
    filtered = filter_missing_images(df, tmp_image_dir)
    assert set(filtered["image_id"]) == {"a.jpg", "b.jpg"}
    assert len(filtered) == 2


def test_etsy_dataset_returns_tensor_and_label(tmp_image_dir):
    df = pd.DataFrame({"image_id": ["a.jpg"], "ground_truth": [1]})
    dataset = EtsyDataset(df, tmp_image_dir, transform=None, img_size=16)
    image, label = dataset[0]
    assert image.shape == (16, 16, 3)
    assert float(label.item()) == 1.0


def test_etsy_dataset_missing_file_returns_black_image(tmp_path):
    df = pd.DataFrame({"image_id": ["ghost.jpg"], "ground_truth": [0]})
    dataset = EtsyDataset(df, str(tmp_path), transform=None, img_size=16)
    image, _ = dataset[0]
    assert image.sum() == 0
