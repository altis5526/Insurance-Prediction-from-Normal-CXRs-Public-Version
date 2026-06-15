# Embedding Extraction & PCA Visualization

Two-step pipeline for inspecting the penultimate-layer representations learned by the insurance-prediction models (DenseNet121, MedMamba, Swin Transformer).

```
embedding_pca/
├── extract_embeddings.py   # step 1: run trained models on the test set, dump embeddings to .npz
├── plot_pca.py             # step 2: load .npz, render 2D PCA comparison PDF
├── plot_pca.ipynb          # notebook variant of step 2 for interactive tweaking
├── RawImageDataset.py      # local dataset class with configurable image root
├── checkpoints/            # model checkpoints — gitignored (too large for GitHub)
│   ├── MedGemma_MIMIC_densenet.pt
│   ├── MedGemma_MIMIC_swinTF.pt
│   └── MedGemma_MIMIC_mamba.pt
├── csv/                    # train/val/test splits
│   ├── insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv
│   ├── insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv
│   └── insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv
└── embeddings/             # output .npz files — gitignored
```

Model definitions (`model.py`, `swintransformer.py`, `MedMamba/`) are imported from the parent project folder via `sys.path` patching — no manual path setup needed.

## Step 1 — Extract embeddings

With all defaults (runs all 3 models, uses local checkpoints and test CSV):

```bash
conda activate base_model_insurance
cd Insurance-Project-Journal-version-
python embedding_pca/extract_embeddings.py
```

All arguments are optional — defaults point to the repo-local `checkpoints/` and `csv/` directories:

| Argument | Default | Description |
|---|---|---|
| `--test-csv` | `csv/insurance_dataset_..._test_medgemmaChecked.csv` | Test split CSV |
| `--densenet-ckpt` | `checkpoints/MedGemma_MIMIC_densenet.pt` | DenseNet121 checkpoint |
| `--swin-ckpt` | `checkpoints/MedGemma_MIMIC_swinTF.pt` | Swin Transformer checkpoint |
| `--medmamba-ckpt` | `checkpoints/MedGemma_MIMIC_mamba.pt` | MedMamba checkpoint |
| `--img-root` | `/datasets/mimic/mimic-cxr-jpg/2.0.0/files/` | MIMIC-CXR-JPG image root |
| `--output-dir` | `embeddings/all_models_test_set_128dim/` | Where to write `.npz` files |
| `--batch-size` | `32` | |
| `--num-workers` | `8` | |
| `--resize` | `448` | Input image size |
| `--seed` | `123` | |
| `--models` | `densenet121 swin_transformer medmamba` | Run a subset of models |

To run a single model or override the image root:

```bash
python embedding_pca/extract_embeddings.py --models densenet121

python embedding_pca/extract_embeddings.py --img-root /your/custom/mimic/path/
```

Each output `.npz` contains: `embeddings`, `labels`, per-class `metrics` (AUC, Precision, Recall, F1, Accuracy), and optional metadata (`img_ids`, `ages`, `genders`, `races`).

## Step 2 — Plot PCA

```bash
python embedding_pca/plot_pca.py
```

Produces `pca_comparison_3models.pdf` — a 1×3 figure with KDE density shadows and scatter points for each model. Override paths with `--embeddings-dir` and `--output`.

For interactive exploration open `plot_pca.ipynb` instead.

## Requirements

Conda env: `base_model_insurance` (torch 2.1.1+cu118, torchmetrics, tqdm, scikit-learn, matplotlib, seaborn).
