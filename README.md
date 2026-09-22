
# Predicting-Insurance-Type-from-Normal-Chest-Xrays

**Pretrained Weights:** [InsurancePrediction/insurance_paper_weights](https://huggingface.co/InsurancePrediction/insurance_paper_weights)

---

## Introduction
This is the official repository of the paper "AI Hidden Features: Predicting Health Insurance Type from Normal Chest X-ray Images".

## Abstract
Due to the broad use of artificial intelligence in healthcare settings, one should be more cautious about the spurious correlation that has been learned by the model. In this study, we focused on the health insurance type feature, which is highly correlated to patients' socioeconomic status, hidden in the chest X-ray images. We demonstrated that common deep vision models are able to learn insurance type information unperceivable to humans in the subtle textures of the medical images. The result serves as a calling to re-examine the current trained chest X-ray classifiers and ensure that they treat economically different populations equally.

---

## Repository Structure

```
├── run_exp0.py …  run_exp4-2.py        # per-experiment training dispatchers (loop configs, call train scripts)
├── train_insurance_fullimgsize_*.py    # underlying train/test scripts (densenet / mamba / swintransformer)
│                                        #   variants: _addDemo_new, _high_pass, _low_pass,
│                                        #   _mask_area, _mask_mostarea, _random, _CheXpert
├── model.py, swintransformer.py        # model definitions
├── dataloader.py, MyLoader_new.py,     # dataset / dataloader helpers
│   RawImageDataset.py
├── low_pass_func.py                    # frequency-filtering utilities
│
├── bootstrap_evaluate.py               # shared bootstrap engine (inference once → full-test metrics
│                                        #   + patient-level bootstrap CIs + per-sample predictions)
├── run_exp0_bootstrap.py,              # bootstrap wrappers (per experiment)
│   run_exp0-2_bootstrap.py,
│   run_exp1_bootstrap.py,
│   run_exp{2,3,4}_bootstrap_medgemma.py
├── run_exp{0,0-2,1,2,3,4,5}_bootstrap.sh   # local launchers (plain `bash`, run the wrappers above)
├── statistical_testing_DeLong.py       # patient-clustered DeLong test of AUC vs chance (0.5)
├── statistical_testing_DeLong_ml.py    # same DeLong test for the tabular ML baselines
├── compare_auc_DeLong.py               # paired patient-clustered DeLong between two models
├── influence_correlation.py            # per-patient influence correlation between models
├── run_exp5_subgroup_training*.sh,     # subgroup training + statistical-test launchers
│   run_exp5_statistical_test*.sh
│
├── ml_analysis_predefined_with_raw_age.ipynb  # tabular ML baselines (raw age/race/sex → insurance)
├── gradcam.py, extract_embeddings.py   # interpretability helpers
│
├── MIMIC_split.pickle, CheXpert_split.pkl   # released train/val/test patient-id splits
│
├── slurm/                              # SLURM scripts for MedMamba-S re-runs / fixes
│   ├── mg_exp{0,1,2,3,4}_bootstrap_s_fix*.sh
│   ├── mg_exp0_mamba_array.sh, mg_exp0_mamba_s_fix.sh
│   └── medgemma/                       # MedGemma-dataset training + bootstrap arrays
│       ├── mg_exp{1,2,3,4}_*.sh
│       └── checklist.txt               # MedGemma run log / status
│
├── MedGemma/                           # MedGemma-based "normal CXR" filtering pipeline
├── MedMamba/                           # MedMamba model + Grad-CAM (vendored)
├── embedding_pca/                      # embedding extraction + PCA visualization
├── bootstrap_results/                  # bootstrap + prediction CSVs (exp0 … exp5),
│   ├── statistical tests/              #   DeLong / comparison / influence-correlation results
│   └── old_results_before20260911/     #   archived image-level bootstrap results (superseded)
├── bootstrap_results_ml/               # tabular-ML bootstrap + prediction outputs
└── influence_correlation/              # influence correlation, CNN × ML-model pairings
```

## Installation

### Environment
`requirements.txt` is a conda **explicit package spec** (platform `linux-64`, Python 3.8). Recreate the environment with conda (not `pip`):
```bash
conda create --name jama_insurance --file requirements.txt
conda activate jama_insurance
```
Key dependencies: `torch==2.1.1`, `mamba-ssm==1.0.1` + `causal-conv1d==1.4.0` (MedMamba; **CUDA required**), `timm`, `transformers==4.44.0`, `grad-cam`, `tfrecord`, `lion-pytorch`, `wandb`.

### Rebuild the dataset
Due to the privacy policies of both MIMIC and CheXpert datasets, we are not allowed to provide our parsed dataset, but we provide our train/val/test patient ids (`MIMIC_split.pickle` and `CheXpert_split.pkl`) in both MIMIC and CheXpert datasets to replicate our results.

#### MIMIC
Make sure you include column names: `dicom_id`, `subject_id_x`, `study_id`, `new_insurance_type`, `gender`, `anchor_age`, `race` in your parsed csv. You could find all those corresponding data on the MIMIC official website as long as you signed up for the privacy agreement.

#### CheXpert
Make sure you include at least these keys in your parsed tfrecord file: `jpg_bytes` for CXR, `insurance_type`, `age`, `sex`, `race`. You could find all those corresponding data on the CheXpert official website as long as you signed up for the privacy agreement.

---

## Training

Each experiment has a top-level dispatcher (`run_exp*.py`) that loops over the relevant configurations and calls the underlying `train_insurance_fullimgsize_*` scripts for **both training and testing**. Trained weights are written under `{root_dir}/{weight_dir}/`. The training seed is fixed to **123**.

**Common arguments** (shared by the dispatchers):

| Arg | Meaning |
|-----|---------|
| `--train_path` | Train CSV (MIMIC) / tfrecord (CheXpert) |
| `--val_path` | Validation CSV / tfrecord |
| `--test_path` | Test CSV / tfrecord |
| `--experiment_name` | Free-form name, embedded in weight filenames |
| `--weight_dir` | Sub-directory (under `root_dir`) where weights are saved |
| `--root_dir` | Root directory for weights (default `/home/sebasmos/orcd/pool/code/JAMA_codes/`) |

### Exp0 — Baseline (full 448×448 image, no extras)

Trains and tests one of densenet / mamba / swinTF on the full image. `run_exp0.py` runs train + test internally for each seed.

```bash
# MIMIC
python run_exp0.py --dataset MIMIC --model densenet \
    --train_path TRAIN_PATH --val_path VAL_PATH --test_path TEST_PATH \
    --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR

# CheXpert
python run_exp0.py --dataset CheXpert --model swinTF \
    --train_path TRAIN_PATH --val_path VAL_PATH --test_path TEST_PATH \
    --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR
```
`--model` accepts `densenet`, `mamba`, or `swinTF`.

### Exp1 — Patch localization (3×3 grid on 448×448)

Each dispatcher loops over patches 1–9 (left-to-right, top-to-bottom) and runs train + test per patch. `--method remove` masks one patch out; `--method keep` trains on one patch only.

```bash
# MedMamba (run_exp1-1.py)
python run_exp1-1.py --method remove \
    --train_path TRAIN_PATH --val_path VAL_PATH --test_path TEST_PATH \
    --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR

# DenseNet / SwinTF (run_exp1-2.py)
python run_exp1-2.py --method keep \
    --train_path TRAIN_PATH --val_path VAL_PATH --test_path TEST_PATH \
    --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR
```
(`run_exp1.py` is an equivalent single-model patch dispatcher and also takes `--method`.)

### Exp2 — Resolution (image downscale)

Images are resized **down** to a target resolution then **up** to 448×448 (two-step resize to test information loss). `run_exp2.py` handles all models via `--model` (`mamba`, `mamba_t`, `mamba_s`, `densenet`, `swinTF`). Optional `--resolutions 4,7,14` restricts which resolutions to train; `--pretrained_path` supplies a `.pth` for MedMamba-S init.

```bash
python run_exp2.py --model mamba \
    --train_path TRAIN_PATH --val_path VAL_PATH --test_path TEST_PATH \
    --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR
```
Resolutions: 2, 4, 7, 14, 28, 56, 112, 224 (swinTF skips 2).

### Exp3 — Demographics (addDemo models)

Trains models with demographic features over 7 combos. One dispatcher per model: `run_exp3.py` (mamba), `run_exp3-1.py` (densenet), `run_exp3-2.py` (swinTF).

```bash
python run_exp3-1.py \
    --train_path TRAIN_PATH --val_path VAL_PATH --test_path TEST_PATH \
    --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR
```
Demo combos: `sex`, `age`, `race`, `sexage`, `sexrace`, `agerace`, `sexagerace`.

### Exp4 — Frequency filtering (high/low pass)

Trains models on frequency-filtered images over 8 cutoff frequencies. `--pass_type high|low`. One dispatcher per model: `run_exp4.py` (mamba), `run_exp4-1.py` (swinTF, also takes `--frequencies`), `run_exp4-2.py` (densenet).

```bash
python run_exp4-2.py --pass_type high \
    --train_path TRAIN_PATH --val_path VAL_PATH --test_path TEST_PATH \
    --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR
```
Frequencies: 1, 5, 10, 25, 50, 100, 200, 400 Hz.

### Exp5 — Subgroup training & statistical tests

`run_exp5_subgroup_training.sh` trains models on isolated demographic subgroups (`Only_female_*`, `Only_male_*`, `Only_white_*`, `Only_black_*`, `Only_young_*`, `Only_middle_*`, `Only_old_*` CSVs). CheXpert variant: `run_exp5_subgroup_training_CheXpert.sh`.

Comparisons between two models now go through [`compare_auc_DeLong.py`](#paired-comparison-of-two-models-compare_auc_delongpy), which is a paired, patient-clustered test on the `*_predictions.csv` files:

```bash
python compare_auc_DeLong.py \
    --set_a bootstrap_results/exp0/MIMIC_densenet_exp0_densenet_Rand123_predictions.csv \
    --set_b "bootstrap_results/exp5/MIMIC_densenet_exp5_*OnlyFemale*_predictions.csv" \
    --label_a "all patients" --label_b "female-only"
```

### Tabular ML baselines

Health-insurance prediction from age/race/sex using classical ML (Decision Tree, Random Forest, KNN, XGBoost, CatBoost). See [`ml_analysis_predefined_with_raw_age.ipynb`](ml_analysis_predefined_with_raw_age.ipynb).

This notebook replaces the earlier `ml_analysis_predefined_val.ipynb`. It uses **raw (continuous) `anchor_age`** instead of 3 age bins. With binned age, gender × age × race gave only 18 distinct feature rows, so every model reduced to the same contingency table. Two more changes: race is now one-hot encoded, and KNN has a `StandardScaler` inside its `Pipeline`. Evaluation uses the same patient-level bootstrap as `bootstrap_evaluate.py` (1000 iterations). The notebook writes per-sample predictions, full-test metrics and bootstrap CSVs to `bootstrap_results_ml_rawage/`. The results in [`bootstrap_results_ml/`](bootstrap_results_ml/) come from the earlier binned-age models (see `statistical_testing_DeLong_ml.py` below).

---

## Bootstrap Evaluation (Confidence Intervals)

Produces point estimates and bootstrap confidence intervals (overall + demographic subgroups) for trained models. `bootstrap_evaluate.py` loads a model and runs inference **once**. It then:
1. saves per-sample predictions (logits, probabilities, labels, patient id, demographics),
2. computes metrics on the **entire test set** with no resampling (the point estimate to report),
3. resamples the predictions `--n_bootstrap` times to get 95% CIs for AUC, Precision, Recall, F1 and Accuracy.

**Patient-level (cluster) bootstrap.** By default, whole **patients** are resampled with replacement, and all of a drawn patient's images enter the resample together (`--bootstrap_unit patient`). Images from the same patient are correlated, so resampling individual images would understate the uncertainty. Patient ids come from `subject_id_x` in the MIMIC test CSV, or from the CheXpert loader (`return_patient=True`). If patient ids are missing, the script falls back to image-level resampling and prints a warning. `--bootstrap_unit image` restores the legacy behaviour.

**Sample size.** `--sample_size` defaults to the full test-set size N, so the CI describes the test set you actually evaluated. A smaller value gives an m-out-of-n bootstrap, with CIs about √(N/m) wider; the script prints a warning when this happens.

Subgroups evaluated: Overall, Gender (Male/Female), Age (<40 / 40–50 / 50–65), Race (White/Black/Other). Subgroups with fewer than 10 samples are skipped in the full-test metrics.

### Files

| File | Purpose |
|------|---------|
| `bootstrap_evaluate.py` | Shared engine. Supports base / addDemo models, patch masking, and frequency filtering. Key args: `--dataset`, `--model`, `--test_path`, `--weight_path`, `--model_variant {base,addDemo}`, `--demo_labels`, `--preprocessing`, `--patch_idx`, `--filter_diameter`, `--resize`, `--n_bootstrap` (1000), `--sample_size` (default: full test set N), `--bootstrap_unit {patient,image}` (patient), `--no_save_predictions`, `--seed` (42), `--num_workers` (4), `--batch_size` (32), `--output_dir`, `--experiment_name`. |
| `run_exp0_bootstrap.py` | Wrapper for exp0 (baseline full-image models, MIMIC + CheXpert). |
| `run_exp0-2_bootstrap.py` | Wrapper for exp0-2 (perturbed-label control). |
| `run_exp1_bootstrap.py` | Wrapper for exp1 (patch remove / keep), loops over patches 1–9. |
| `run_exp2_bootstrap_medgemma.py` | Wrapper for exp2 (resolution), loops over resolutions. |
| `run_exp3_bootstrap_medgemma.py` | Wrapper for exp3/3-1/3-2 (demographics addDemo), loops over 7 combos. |
| `run_exp4_bootstrap_medgemma.py` | Wrapper for exp4/4-1/4-2 (frequency), loops over 8 frequencies. |

The wrapper `.sh` launchers in the repo root run these locally (plain `bash`, **not** `sbatch`):

```bash
bash run_exp0_bootstrap.sh      # baseline (6 model × dataset combos)
bash run_exp0-2_bootstrap.sh    # perturbed-label control (patient-level, see below)
bash run_exp1_bootstrap.sh      # patches (6 model/method combos × 9 patches)
bash run_exp2_bootstrap.sh      # resolution
bash run_exp3_bootstrap.sh      # demographics
bash run_exp4_bootstrap.sh      # frequency (high + low)
bash run_exp5_bootstrap.sh      # subgroup (Only_female / male / white / old …)
```

> The `run_exp{2,3,4}_bootstrap.sh` launchers call the `*_bootstrap_medgemma.py` wrappers (the MedGemma-refined dataset is the current default test set). Edit the `--test_path`/`--weight_dir` inside each `.sh` to point at your local data and weights.

> **Wrapper defaults:** the `run_exp*_bootstrap*.py` wrappers default to `--n_bootstrap 1000` and leave `--sample_size` unset, so `bootstrap_evaluate.py` resamples the full test set. `--sample_size` is only forwarded when you pass it explicitly.

### SLURM (MedMamba-S re-runs)

MedMamba-S checkpoints are re-run on the cluster via SLURM array jobs under `slurm/`:

```bash
sbatch slurm/mg_exp0_bootstrap_s_fix.sh
sbatch slurm/mg_exp1_bootstrap_s_fix.sh
sbatch slurm/mg_exp2_bootstrap_s_fix.sh
sbatch slurm/mg_exp3_bootstrap_s_fix.sh
sbatch slurm/mg_exp4_bootstrap_s_fix_high.sh
sbatch slurm/mg_exp4_bootstrap_s_fix_low.sh
```

### Direct single-checkpoint examples

```bash
# Baseline
python run_exp0_bootstrap.py \
    --dataset MIMIC --model densenet \
    --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
    --experiment_name exp0_densenet --weight_dir exp0/MedGemma_MIMIC_densenet.pt \
    --n_bootstrap 1000 --seed 42 --train_seed 123

# Single patch (remove patch 1, mamba)
python bootstrap_evaluate.py \
    --dataset MIMIC --model mamba --preprocessing mask_remove --patch_idx 1 \
    --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
    --weight_path WEIGHTS/exp1-1/mamba_remove/patch1.pt \
    --n_bootstrap 1000 --seed 42 \
    --output_dir bootstrap_results/exp1 --experiment_name MIMIC_mamba_remove_patch1

# Single resolution (28, mamba)
python bootstrap_evaluate.py \
    --dataset MIMIC --model mamba --resize 28 \
    --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
    --weight_path WEIGHTS/exp2/mamba_28.pt \
    --n_bootstrap 1000 --seed 42 \
    --output_dir bootstrap_results/exp2 --experiment_name MIMIC_mamba_res28

# Single demo combo (sex+age, densenet)
python bootstrap_evaluate.py \
    --dataset MIMIC --model densenet --model_variant addDemo --demo_labels sex age \
    --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
    --weight_path WEIGHTS/exp3-1/sexage.pt \
    --n_bootstrap 1000 --seed 42 \
    --output_dir bootstrap_results/exp3 --experiment_name MIMIC_densenet_sexage

# Single frequency (5 Hz high-pass, densenet)
python bootstrap_evaluate.py \
    --dataset MIMIC --model densenet --preprocessing high_pass --filter_diameter 5 \
    --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
    --weight_path WEIGHTS/exp4-2/highpass/5Hz.pt \
    --n_bootstrap 1000 --seed 42 \
    --output_dir bootstrap_results/exp4 --experiment_name MIMIC_densenet_5HighPass
```

### Output

Each checkpoint produces four files in its `--output_dir`:
- `<name>_predictions.csv`: per-sample `sample_id`, `patient_id`, `true_label`, `pred_label`, `correct`, `logit_*`, `prob_*`, and demographic codes/names. This is the input to the DeLong tests below. Skip it with `--no_save_predictions`.
- `<name>_full_test_metrics.csv`: metrics on the entire test set (no resampling), overall and per subgroup, with `N`
- `<name>_bootstrap_iterations.csv`: per-iteration metrics (one row per bootstrap iteration × subgroup)
- `<name>_bootstrap_summary.csv`: bootstrap mean, std and 95% CI for AUC, Precision, Recall, F1 and Accuracy

Results are organized under [`bootstrap_results/`](bootstrap_results/). Each prefix below is followed by `_Rand123_{predictions,full_test_metrics,bootstrap_iterations,bootstrap_summary}.csv`:

| Directory | Experiment | Naming convention |
|-----------|------------|-------------------|
| [`exp0/`](bootstrap_results/exp0/) | Baseline | `{dataset}_{model}_exp0_…_Rand123_*.csv` |
| [`exp0-2/`](bootstrap_results/exp0-2/) | Perturbed-label control, labels permuted per image | `{dataset}_{model}_exp0-2_…_Rand123_*.csv` |
| [`exp0-2_patientlevelperturbed/`](bootstrap_results/exp0-2_patientlevelperturbed/) | Perturbed-label control, labels permuted per patient (current; what `run_exp0-2_bootstrap.sh` produces) | `{dataset}_{model}_exp0-2_…_patientlevelperturbed_Rand123_*.csv` |
| [`exp1/`](bootstrap_results/exp1/) | Patches | `MIMIC_{model}_{keep,remove}_patch{1-9}_Rand123_bootstrap_*.csv` |
| [`exp2/`](bootstrap_results/exp2/) | Resolution | `MIMIC_medgemma_{model}_res{N}_Rand123_bootstrap_*.csv` |
| [`exp3/`](bootstrap_results/exp3/) | Demographics | `MIMIC_medgemma_{model}_{combo}_Rand123_bootstrap_*.csv` |
| [`exp4/`](bootstrap_results/exp4/) | Frequency | `MIMIC_medgemma_{model}_{freq}{HighPass,LowPass}_Rand123_bootstrap_*.csv` |
| [`exp5/`](bootstrap_results/exp5/) | Subgroup | `{dataset}_{model}_exp5_…Only{Group}_*_Rand123_*.csv` |
| [`statistical tests/`](<bootstrap_results/statistical tests/>) | DeLong / comparisons | see [Statistical Testing](#statistical-testing) |
| [`old_results_before20260911/`](bootstrap_results/old_results_before20260911/) | Archive | Earlier image-level bootstrap (20 × 1000 samples), kept for reference only |

> Filenames may also carry pretrained-variant tags (e.g. `PretrainedB_swinTF`, `pretrained_mamba`) depending on the checkpoint used. See each directory for the exact files present.

---

## Statistical Testing

All tests use the per-sample `*_predictions.csv` files. AUCs are computed on the **entire test set**. Standard errors are **cluster-adjusted by patient**: one patient can contribute up to 15 CXRs, and all of them share the same insurance label, so the patient (≈1224 in MIMIC test) is the independent unit rather than the image (1965).

### DeLong test vs. chance: `statistical_testing_DeLong.py`

Tests H0: AUC = 0.5 for every predictions file. The variance comes from the DeLong placement values: each patient's contributions are summed, and the variance is taken across patients (the cluster-robust form, Obuchowski 1997). With one image per patient it reduces to classical DeLong; `--self_test` checks this. The script reports raw p-values, Benjamini–Hochberg q-values (within each experiment and across all tests), Bonferroni p-values, 95% CIs, and the design effect.

```bash
# All experiments under bootstrap_results/
python statistical_testing_DeLong.py \
    --results_dir bootstrap_results \
    --experiments exp0 exp1 exp2 exp3 exp4 exp5 \
    --output "bootstrap_results/statistical tests/delong_vs_chance.csv"

# Specific files; --show_naive also prints the unclustered SE
python statistical_testing_DeLong.py \
    --predictions bootstrap_results/exp1/MIMIC_densenet_keep_patch5_Rand123_predictions.csv --show_naive
```
Other flags: `--baseline` (0.5), `--alpha` (0.05), `--score_col` (`prob_1`), `--label_col`, `--cluster_col` (`patient_id`), and `--ignore_clusters`.

### Tabular ML baselines: `statistical_testing_DeLong_ml.py`

Rebuilds the binned-age ML pipeline and refits each model with its selected hyperparameters. It then writes `<Model>_predictions.csv` to `bootstrap_results_ml/` in the same schema as `bootstrap_evaluate.py` and runs the same clustered DeLong test. The statistics are imported from `statistical_testing_DeLong.py`. `--verify` checks that the refits reproduce the notebook AUCs.

```bash
python statistical_testing_DeLong_ml.py --show_naive
```

### Paired comparison of two models: `compare_auc_DeLong.py`

DeLong used for its original purpose: comparing two correlated ROC curves. Both models see the same images, so the per-image contributions are differenced first, then summed within patient and varied across patients. One expression therefore handles both the pairing and the clustering. Only images present in both files are used, and the intersection is reported. Every `--set_a` × `--set_b` pair is tested, with BH q-values across the pairs.

The CNNs encode Private as class 1 while the ML models encode Public as class 1. The script detects this by joining on `sample_id`, then realigns by flipping both labels and scores, which leaves each AUC unchanged. `--positive_class` names the convention for the report; `--no_auto_polarity` fails instead of realigning.

```bash
python compare_auc_DeLong.py \
    --set_a bootstrap_results_ml/*_predictions.csv \
    --set_b "bootstrap_results/exp0/MIMIC_*_predictions.csv" \
    --label_a "Demographics" --label_b "CXR image" \
    --output "bootstrap_results/statistical tests/delong_ml_vs_exp0.csv"
```

### Influence correlation: `influence_correlation.py`

For two models scored on the same patients, this script correlates each patient's AUC influence term h_k. The result, r, shows whether the models succeed on the *same* patients. It also sets the power of a paired AUC comparison: Var(A−B) = Var(A) + Var(B) − 2r·SE_A·SE_B. Label polarity is aligned automatically, as in `compare_auc_DeLong.py`. By default it compares demographics only (raw-age XGBoost), image only (exp0 DenseNet) and image + demographics (exp3 DenseNet `sexagerace`); the raw-age file comes from `bootstrap_results_ml_rawage/`, which is created by running the raw-age notebook.

```bash
python influence_correlation.py \
    --model "Demographics=bootstrap_results_ml_rawage/XGBoost_predictions.csv" \
    --model "Image only=bootstrap_results/exp0/MIMIC_densenet_exp0_densenet_Rand123_predictions.csv" \
    --model "Image+demo=bootstrap_results/exp3/MIMIC_medgemma_densenet_sexagerace_Rand123_predictions.csv" \
    --output influence_correlation.csv
```
[`influence_correlation/`](influence_correlation/) contains one file per CNN × ML-model pairing: `influence_correlation_{densenet,mamba,swinTF}_{CB,DT,KNN,RF,XGB}.csv`.

### Result files (`bootstrap_results/statistical tests/`)

| File | Content |
|------|---------|
| `delong_vs_chance.csv` | Clustered DeLong vs. AUC 0.5 for exp0–exp5 |
| `delong0-2patientlevelperturbed_vs_chance.csv` | Same, for the patient-level perturbed-label control |
| `delong_ml_vs_exp0.csv` | `compare_auc_DeLong.py`: binned-age ML models vs. exp0 CNNs |
| `delong_rawage_vs_exp0.csv` | `compare_auc_DeLong.py`: raw-age ML models vs. exp0 CNNs |
| `influence_correlation_agerace.csv` | Influence correlation: demographics vs. image-only vs. image + demographics |
| `[Chexpert_]comparison_{model}_{female,male,old,white,pertubed}.csv` | Earlier bootstrap-iteration comparisons (exp5 subgroups, perturbed control) |

---

## MedGemma Experiments

The MedGemma pipeline ([`MedGemma/`](MedGemma/)) uses the MedGemma vision-language model to filter MIMIC-CXR down to **normal CXRs only** (the `mimic-cxr-gemma` dataset), then repeats experiments 2, 3, and 4 on that refined set. Experiments 0 and 0-2 are handled separately. See [`slurm/medgemma/checklist.txt`](slurm/medgemma/checklist.txt) for the run log and storage policy.

- **Refined dataset:** `insurance_dataset_8_1_1_PMMthree_{train,val,test}_medgemmaChecked.csv`
- **Weights:** `insurance_paper_weights/exp{1,1-1,1-2,2,2-1,2-2,3,3-1,3-2,4,4-1,4-2}_medgemma/`
- **MedMamba-S re-runs:** the bootstrap wrappers read the mamba checkpoints from the `_s_fix` directories (`exp1-1_medgemma_s_fix`, `exp2_medgemma_s_fix`, `exp3_medgemma_s_fix`, `exp4_medgemma_s_fix`); densenet and swinTF read the plain `_medgemma` directories.

### Training (SLURM)

Per-model training/array scripts live in [`slurm/medgemma/`](slurm/medgemma/):

| Exp | Model | Script(s) |
|-----|-------|-----------|
| exp1 | mamba (keep/remove) | `mg_exp1_mamba_{keep,remove}_array.sh`, `…_s_fix_array.sh` |
| exp2 | mamba-t / mamba-s / densenet / swinTF | `mg_exp2_mamba_t.sh`, `mg_exp2_mamba_s.sh`, `mg_exp2_mamba_s_fix.sh`, `mg_exp2_densenet.sh`, `mg_exp2_swinTF.sh` |
| exp3 | mamba | `mg_exp3_mamba_array.sh`, `mg_exp3_mamba_s_fix_array.sh` |
| exp4 | mamba (high/low) | `mg_exp4_mamba_{high,low}_array.sh`, `…_s_fix_array.sh` |

Submit with `sbatch slurm/medgemma/<script>.sh`. (densenet/swinTF for exp2–4 reuse the same `run_exp*.py` dispatchers as the originals, pointed at the `medgemmaChecked` CSVs.)

### Bootstrap (MedGemma)

The root `run_exp{2,3,4}_bootstrap.sh` launchers already target the MedGemma test set via the `*_bootstrap_medgemma.py` wrappers:

```bash
# Exp2 (resolution)
python run_exp2_bootstrap_medgemma.py --model mamba \
    --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
    --n_bootstrap 1000 --seed 42

# Exp3 (demographics)
python run_exp3_bootstrap_medgemma.py --model densenet \
    --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
    --n_bootstrap 1000 --seed 42 --train_seed 123

# Exp4 (frequency)
python run_exp4_bootstrap_medgemma.py --model swinTF --direction highpass \
    --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
    --n_bootstrap 1000 --seed 42 --train_seed 123
```

By default these wrappers write to `bootstrap_results/medgemma_exp{2,3,4}/` (created on first run); the root launchers as shipped point at `bootstrap_results/exp{2,3,4}/`.

### Note on MedMamba

[`MedMamba/MedMamba.py`](MedMamba/MedMamba.py) includes both `VSSM_DoubleLinear` (base) and `VSSM_Double_addDemothen2` (demographics addDemo variant with a deeper classification head). MedMamba requires CUDA (the `selective_scan` kernel only runs on GPU).

---

## Interpretability & Analysis

- **Embeddings / PCA** — [`embedding_pca/`](embedding_pca/): `extract_embeddings.py` dumps penultimate-layer embeddings to `.npz`, then `plot_pca.py` (or `plot_pca.ipynb`) renders the 2D PCA comparison across DenseNet / MedMamba / SwinTF. See [`embedding_pca/README.md`](embedding_pca/README.md).
- **Grad-CAM** — [`gradcam.py`](gradcam.py) and [`MedMamba/grad_cam/`](MedMamba/grad_cam/) for class-activation maps.

## Pretrained Weights

All pretrained checkpoints are available on Hugging Face: **[InsurancePrediction/insurance_paper_weights](https://huggingface.co/InsurancePrediction/insurance_paper_weights)** (seed = 123).

```bash
# Download all weights
git lfs install
git clone https://huggingface.co/InsurancePrediction/insurance_paper_weights
```
```python
# Download a single experiment
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="InsurancePrediction/insurance_paper_weights",
    allow_patterns="exp3-1/*",
    local_dir="./weights",
)
```

### Checkpoint Table

| Experiment | Description | Model | Config | # Checkpoints | Training script | HF weight path |
|------------|-------------|-------|--------|---------------|-----------------|----------------|
| **exp0** | Baseline (full image) | densenet, mamba, swinTF | MIMIC + CheXpert | 6 | `run_exp0.py` | `exp0/{dataset}/{model}.pt` |
| **exp0-2** | Perturbed labels (control) | densenet, mamba, swinTF | MIMIC + CheXpert | 6 | `run_exp0-2.py` | `exp0-2_medgemma/{model}_patientlevelperturbed[_chexpert].pt` |
| **exp1-1** | Patch keep/remove | mamba | 9 patches × 2 | 18 | `run_exp1-1.py` | `exp1-1/mamba_{keep,remove}/patch{1-9}.pt` |
| **exp1-2** | Patch keep/remove | densenet | 9 patches × 2 | 18 | `run_exp1-2.py` | `exp1-2/densenet_{keep,remove}/patch{1-9}.pt` |
| **exp1-2** | Patch keep/remove | swinTF | 9 patches × 2 | 18 | `run_exp1-2.py` | `exp1-2/swinTF_{keep,remove}/patch{1-9}.pt` |
| **exp2** | Resolution | mamba | 8 resolutions | 8 | `run_exp2.py` | `exp2/mamba_{N}.pt` |
| **exp2-1** | Resolution | densenet | 8 resolutions | 8 | `run_exp2.py` | `exp2-1/densenet_{N}.pt` |
| **exp2-2** | Resolution | swinTF | 7 resolutions | 7 | `run_exp2.py` | `exp2-2/swinTF_{N}.pt` |
| **exp3** | Demographics (addDemo) | mamba | 7 demo combos | 7 | `run_exp3.py` | `exp3/{combo}.pt` |
| **exp3-1** | Demographics (addDemo) | densenet | 7 demo combos | 7 | `run_exp3-1.py` | `exp3-1/{combo}.pt` |
| **exp3-2** | Demographics (addDemo) | swinTF | 7 demo combos | 7 | `run_exp3-2.py` | `exp3-2/{combo}.pt` |
| **exp4** | Freq filtering (HP+LP) | mamba | 8 freq × 2 | 16 | `run_exp4.py` | `exp4/{highpass,lowpass}/{F}Hz.pt` |
| **exp4-1** | Freq filtering (HP+LP) | swinTF | 8 freq × 2 | 16 | `run_exp4-1.py` | `exp4-1/{highpass,lowpass}/{F}Hz.pt` |
| **exp4-2** | Freq filtering (HP+LP) | densenet | 8 freq × 2 | 16 | `run_exp4-2.py` | `exp4-2/{highpass,lowpass}/{F}Hz.pt` |

The MedGemma-refined variants live in parallel `exp{2,2-1,2-2,3,3-1,3-2,4,4-1,4-2}_medgemma/` directories on the same HF repo.

- **Demo combos** (exp3): `sex`, `age`, `race`, `sexage`, `sexrace`, `agerace`, `sexagerace`
- **Frequencies** (exp4): 1, 5, 10, 25, 50, 100, 200, 400 Hz
- **Resolutions** (exp2): 2, 4, 7, 14, 28, 56, 112, 224
- **Patches** (exp1): 1–9, a 3×3 grid on 448×448 images (left-to-right, top-to-bottom)

## Code References
MedMamba: [Link](https://github.com/YubiaoYue/MedMamba)

SwinTransformer V2: [Link](https://github.com/ChristophReich1996/Swin-Transformer-V2)
