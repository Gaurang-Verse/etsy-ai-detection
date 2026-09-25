"""
Training entrypoint.

Usage:
    python -m src.train --config configs/eva02_base.yaml
    python -m src.train --config configs/local_debug.yaml   # CPU smoke test
"""
import argparse
import datetime
import json
import os

import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config, load_config, resolve_device
from src.dataset import EtsyDataset, build_transforms, filter_missing_images
from src.losses import build_loss
from src.model import build_model, build_optimizer_and_scheduler


def prepare_dataframes(cfg: Config):
    data_cfg = cfg.data
    train_csv = os.path.join(data_cfg.base_path, data_cfg.train_csv)
    test_csv = os.path.join(data_cfg.base_path, data_cfg.test_csv)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
    print("Label distribution:")
    print(train_df[data_cfg.label_col].value_counts())

    from src.prepare_data import resolve_image_folder
    image_folder = resolve_image_folder(data_cfg.images_dir)
    train_df = filter_missing_images(train_df, image_folder, data_cfg.image_id_col)
    print(f"Final training set size: {len(train_df)} images")

    train_data, val_data = train_test_split(
        train_df,
        test_size=data_cfg.val_size,
        random_state=data_cfg.random_state,
        stratify=train_df[data_cfg.label_col],
    )
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")
    return train_data, val_data, test_df, image_folder


def build_loaders(cfg: Config, train_data, val_data, image_folder):
    train_transform, val_transform = build_transforms(cfg.model, cfg.aug)

    train_dataset = EtsyDataset(
        train_data, image_folder, transform=train_transform,
        image_id_col=cfg.data.image_id_col, label_col=cfg.data.label_col,
        img_size=cfg.model.img_size,
    )
    val_dataset = EtsyDataset(
        val_data, image_folder, transform=val_transform,
        image_id_col=cfg.data.image_id_col, label_col=cfg.data.label_col,
        img_size=cfg.model.img_size,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=True,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader


def train(cfg: Config):
    device = resolve_device(cfg.train.device)
    print(f"Using device: {device}")

    train_data, val_data, test_df, image_folder = prepare_dataframes(cfg)
    train_loader, val_loader = build_loaders(cfg, train_data, val_data, image_folder)

    model = build_model(cfg.model, device)
    criterion = build_loss(cfg.loss)
    optimizer, scheduler, backbone_params = build_optimizer_and_scheduler(model, cfg.train)

    use_amp = cfg.train.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(os.path.dirname(cfg.output.checkpoint_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(cfg.output.submission_path) or ".", exist_ok=True)
    os.makedirs(cfg.output.figures_dir, exist_ok=True)

    best_val_f1 = 0.0
    best_epoch = 0
    epochs = cfg.train.epochs
    warmup_epochs = cfg.train.warmup_epochs

    print("=" * 65)
    print(f"Training {cfg.model.backbone} for {epochs} epochs @ {cfg.model.img_size}x{cfg.model.img_size}")
    print(f"Batch size: {cfg.train.batch_size} | Device: {device}")
    print(f"Warm-up: backbone FROZEN for first {warmup_epochs} epoch(s)")
    print("=" * 65)

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            for p in backbone_params:
                p.requires_grad_(False)
            if epoch == 0:
                print("Backbone FROZEN — warming up classification head only")
        elif epoch == warmup_epochs:
            for p in backbone_params:
                p.requires_grad_(True)
            print("Backbone UNFROZEN — full fine-tuning begins")

        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Train]")

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = train_loss / max(len(train_loader), 1)

        val_f1, val_preds, val_labels_list, raw_probs = evaluate(model, val_loader, device, use_amp)

        n_ones = sum(val_preds)
        n_total = len(val_preds)
        prob_min, prob_max = (min(raw_probs), max(raw_probs)) if raw_probs else (0.0, 0.0)
        diag = f"preds: {n_ones}/{n_total} AI | prob range [{prob_min:.3f}, {prob_max:.3f}]"

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), cfg.output.checkpoint_path)
            marker = " * BEST SAVED"
        else:
            marker = ""

        print(f"\nEpoch {epoch+1:02d} | Train Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}{marker}")
        print(f"   {diag}\n")

    print("=" * 65)
    print(f"Done! Best Val F1: {best_val_f1:.4f} at Epoch {best_epoch}")
    print(f"Checkpoint: {cfg.output.checkpoint_path}")
    print("=" * 65)

    if not os.path.exists(cfg.output.checkpoint_path):
        print(
            "\nNo checkpoint was saved - Val F1 never improved on the initial "
            "0.0 baseline (expected on a tiny/synthetic smoke-test dataset). "
            "Skipping final evaluation and metadata save.\n"
        )
        return model, test_df, image_folder, device, use_amp

    final_evaluation(cfg, model, val_loader, device, use_amp, best_epoch)
    save_experiment_summary(cfg, best_epoch, best_val_f1)

    return model, test_df, image_folder, device, use_amp


def evaluate(model, loader, device, use_amp):
    model.eval()
    preds, labels_list, raw_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            raw_probs.extend(probs.tolist())
            preds.extend((probs > 0.5).astype(int))
            labels_list.extend(labels.numpy().astype(int))
    val_f1 = f1_score(labels_list, preds, zero_division=0)
    return val_f1, preds, labels_list, raw_probs


def final_evaluation(cfg: Config, model, val_loader, device, use_amp, best_epoch):
    model.load_state_dict(torch.load(cfg.output.checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded best checkpoint from epoch {best_epoch}")

    val_f1, val_preds, val_labels_list, _ = evaluate(model, val_loader, device, use_amp)

    print("\n" + "=" * 60)
    print(classification_report(val_labels_list, val_preds, target_names=["Authentic", "AI-Generated"]))
    print(f"Final Validation F1 Score: {val_f1:.4f}")
    print("TARGET ACHIEVED: F1 > 0.90!" if val_f1 >= 0.90 else f"(Need {0.90 - val_f1:.4f} more to reach 0.90 target)")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt

        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(val_labels_list, val_preds),
            display_labels=["Authentic", "AI-Generated"],
        )
        disp.plot(colorbar=False)
        plt.title(f"Validation Confusion Matrix | F1={val_f1:.4f}")
        plt.tight_layout()
        fig_path = os.path.join(cfg.output.figures_dir, "confusion_matrix.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved confusion matrix to {fig_path}")
    except Exception as e:
        print(f"Could not save confusion matrix figure: {e}")


def save_experiment_summary(cfg: Config, best_epoch: int, best_val_f1: float):
    metadata = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": cfg.model.backbone,
        "image_size": f"{cfg.model.img_size}x{cfg.model.img_size}",
        "total_epochs_run": cfg.train.epochs,
        "best_epoch": best_epoch,
        "batch_size": cfg.train.batch_size,
        "optimizer": "AdamW",
        "lr_head": cfg.train.lr_head,
        "lr_backbone": cfg.train.lr_backbone,
        "weight_decay": cfg.train.weight_decay,
        "loss_function": f"{cfg.loss.name} (gamma={cfg.loss.gamma}, alpha={cfg.loss.alpha})",
        "scheduler": f"CosineAnnealingLR (T_max={cfg.train.epochs})",
        "final_metrics": {
            "best_validation_f1": round(best_val_f1, 4),
            "target_achieved_gt_0.90": best_val_f1 >= 0.90,
        },
    }
    with open(cfg.output.metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to: {cfg.output.metadata_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train EVA-02 AI-image detector")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
