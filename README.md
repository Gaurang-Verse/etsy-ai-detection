# Etsy AI-Generated Image Detection

Binary classifier that separates AI-generated product photos from authentic
ones, built for the DCU 2026 Machine Learning Challenge (Etsy-sponsored
track). Upgrades a `ConvNeXt-Small` + BCE baseline to an **EVA-02 Vision
Transformer + Focal Loss** system, reaching **F1 = 0.9144** on a held-out
960-image validation set (precision/recall ≈ 0.92 on both classes).

Full write-up: [`report/etsy_eva02_report.pdf`](report/etsy_eva02_report.pdf).

## Why this approach

| Component | Baseline | This system | Why |
|---|---|---|---|
| Architecture | ConvNeXt-Small (CNN, 224×224) | EVA-02 base (ViT, 448×448) | Global self-attention reasons about the whole image at once — catches structural anomalies (e.g. implausible hand anatomy) that a CNN's local receptive field builds up only indirectly, across many layers. |
| Loss | Binary Cross-Entropy | Focal Loss (γ=2.0, α=0.25) | Down-weights easy examples (obvious cartoons) so gradient signal concentrates on hard, photorealistic fakes. |
| Augmentation | Flip + brightness | + JPEG compression simulation + Gaussian noise | Corrupts the exact low-level pixel artefacts a naive model would memorise, forcing reliance on structural cues instead. |

## Results

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Authentic (0) | 0.92 | 0.92 | 0.92 | 497 |
| AI-Generated (1) | 0.92 | 0.91 | 0.91 | 463 |
| **Weighted avg** | **0.92** | **0.92** | **0.92** | 960 |

Validation F1 climbed from 0.6820 (warm-up epoch, backbone frozen) to 0.9144
over 10 epochs, with the largest single jump (+0.11) occurring the moment
the backbone unfroze — see `report/` for the full epoch-by-epoch table.

## Repo structure

```
src/            Core library: config, dataset, model, losses, train, infer
configs/        YAML configs — swap files instead of editing code
notebooks/      Thin Colab runner notebook (calls into src/)
tests/          CPU-only unit tests (loss math, dataset filtering)
outputs/        Generated metrics, figures, submissions (gitignored)
checkpoints/    Model weights (gitignored — see Model Weights below)
report/         Original technical report (PDF)
```

## How to run

This project is split across two environments because EVA-02-base at 448px
needs a CUDA GPU that a MacBook doesn't have.

### Locally (MacBook, CPU/MPS) — dev loop only, not real training

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/                                          # unit tests, no GPU needed
python -m src.train --config configs/local_debug.yaml  # 1-epoch smoke test on a tiny local sample
```

Use this to catch bugs in the data pipeline, config, or CLI before spending
Colab GPU time. `local_debug.yaml` uses the smaller `eva02_small_patch14_336`
backbone and a handful of images — its metrics are meaningless, it's a
smoke test only.

### On Colab (T4/A100) — real training

1. Open `notebooks/colab_train.ipynb` in Colab, set a T4/A100 runtime.
2. Update `REPO_URL` in the clone cell to this repo.
3. Run all cells — it installs dependencies, mounts Drive, unzips the
   challenge data, and calls `src.train` / `src.infer` with
   `configs/eva02_base.yaml`.
4. Checkpoints, submission CSV, and the experiment summary JSON are written
   to Google Drive (paths set in the config's `output:` section).

### Back on the MacBook — write-up and packaging

Pull the checkpoint's metrics/figures back down, update `outputs/figures/`
and the README results table, and commit.

## Model weights

Checkpoints are not committed (~330MB+). Either re-run training from
`configs/eva02_base.yaml`, or download a released checkpoint from the
repo's Releases page (if published).

## Ablations / future work

- `configs/eva02_base_extended.yaml` — 15 epochs + Cutout regularisation
  (per the report's Future Work section).
- Test-Time Augmentation (average predictions over flipped/rotated test
  images) — not yet implemented; tracked as an open item.
- `eva02_small_patch14_336` swap for lower-latency deployment.

## Attribution

Built by Gaurang Kumbhar and Aditya Dhayagude for the DCU 2026 ML
Challenge. See `report/` for the full contribution breakdown (architecture
research, loss engineering, and data pipeline vs. augmentation strategy,
training optimisation, and documentation).

## License

MIT — see `LICENSE`.
