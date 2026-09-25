"""
Inference entrypoint — loads a trained checkpoint and scores images.

Usage:
    python -m src.infer --config configs/eva02_base.yaml --checkpoint checkpoints/best_eva02_checkpoint.pth

On a MacBook without a GPU, point --config at a debug config and run this
against a handful of local images to sanity-check the model loads and
produces predictions; full test-set inference still belongs on Colab.
"""
import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config, resolve_device
from src.dataset import EtsyDataset, build_transforms
from src.model import build_model


def run_inference(cfg, checkpoint_path: str):
    device = resolve_device(cfg.train.device)
    print(f"Using device: {device}")

    model = build_model(cfg.model, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    _, val_transform = build_transforms(cfg.model, cfg.aug)

    test_csv = os.path.join(cfg.data.base_path, cfg.data.test_csv)
    test_df = pd.read_csv(test_csv)

    test_dataset = EtsyDataset(
        test_df, cfg.data.images_dir, transform=val_transform, is_test=True,
        image_id_col=cfg.data.image_id_col, img_size=cfg.model.img_size,
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    predictions, image_ids = [], []
    use_amp = cfg.train.amp and device == "cuda"

    print("Predicting test images...")
    with torch.no_grad():
        for images, img_names in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            predictions.extend(preds)
            image_ids.extend(img_names)

    submission_df = pd.DataFrame({"image_id": image_ids, "prediction": predictions})
    os.makedirs(os.path.dirname(cfg.output.submission_path) or ".", exist_ok=True)
    submission_df.to_csv(cfg.output.submission_path, index=False)

    print(f"\nPredictions saved to: {cfg.output.submission_path}")
    print(f"Total predictions: {len(predictions)}")
    print(f"AI-Generated: {sum(predictions)} | Authentic: {len(predictions) - sum(predictions)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_inference(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
