"""
One-off data preparation: unzip the challenge images archive and detect the
image folder. Run this once per Colab session before training.

Usage (in Colab, after mounting Drive):
    python -m src.prepare_data --config configs/eva02_base.yaml
"""
import argparse
import os
import zipfile

from src.config import load_config


def resolve_image_folder(images_dir: str) -> str:
    """If the extracted archive has a single nested folder, use that as the
    image folder (matches the original notebook's auto-detect logic)."""
    if not os.path.isdir(images_dir):
        return images_dir
    subdirs = [
        os.path.join(images_dir, d)
        for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d))
    ]
    return subdirs[0] if len(subdirs) == 1 else images_dir


def prepare(cfg) -> str:
    zip_path = os.path.join(cfg.data.base_path, cfg.data.zip_name)
    images_dir = cfg.data.images_dir

    if not os.path.exists(images_dir):
        print("Extracting images...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(images_dir)
        print(f"Done — extracted to {images_dir}")
    else:
        print(f"Already extracted: {images_dir}")

    image_folder = resolve_image_folder(images_dir)
    print(f"Images found in: {image_folder}")
    print(f"Total images: {len(os.listdir(image_folder))}")
    return image_folder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    prepare(cfg)
