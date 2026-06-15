
# Predicting-Insurance-Type-from-Normal-Chest-Xrays

**Pretrained Weights:** [InsurancePrediction/insurance_paper_weights](https://huggingface.co/InsurancePrediction/insurance_paper_weights)

---


## Introduction
This is the official repository of the ML4H 2025 finding paper (under submission): "The Unawareness of AI Looking for Health Insurance Type from Normal Chest X-ray Images".

## Abstract
Due to the broad use of artificial intelligence in healthcare settings, one should be more cautious about the spurious correlation that has been learned by the model. In this study, we focused on the health insurance type feature, which is highly correlated to patients' socioeconomic status, hidden in the chest X-ray images. We demonstrated that common deep vision models are able to learn insurance type information unperceivable to humans in the subtle textures of the medical images. The result serves as a calling to re-examine the current trained chest X-ray classifiers and ensure that they treat economically different populations equally.

## Installation

### Environment
```
pip install requirements.txt
```
### Rebuild the dataset
Due to the privacy policies of both MIMIC and CheXpert datasets, we are not allowed to provide our parsed dataset, but we provide our train/val/test patient ids ("MIMIC_split.pickle" and "CheXpert_split.pkl") in both MIMIC and CheXpert datasets to replicate our reults.

#### MIMIC
Make sure you include column names: "dicom_id", "subject_id_x", "study_id", "new_insurance_type", "gender", "anchor_age", "race" in your parsed csv. You could find all those corresponding data in the MIMIC official website as long as you signed up for the privacy agreement.

#### CheXpert
Make sure you include at least these keys in your parsed tfrecord file: "jpg_bytes" for CXR, "insurance_type", "age", "sex", "race". You could find all those corresponding data in the CheXpert official website as long as you signed up for the privacy agreement.

### Experiment 1: Health Insurance Prediction from CXRs
#### Training
**MIMIC**
```
python run_exp1.py --dataset MIMIC --mode train --train_path TRAIN_PATH --val_path VAL_PATH --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR
```

**CheXpert**
```
python run_exp1.py --dataset CheXpert --mode train --train_path TRAIN_PATH --val_path VAL_PATH --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR
```

#### Testing
**MIMIC**
```
python run_exp1.py --dataset MIMIC --mode test --val_path TEST_PATH --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR
```

**CheXpert**
```
python run_exp1.py --dataset CheXpert --mode test --val_path VAL_PATH --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR
```
train_path: The train csv/tfrecord dataset file location
val_path: The test csv/tfrecord dataset file location
experiment_name: Name your experiment as you wish
weight_dir: The directory where you saved your weights

### Experiment 2: Localization of insurance information on Xray - Patch-based training

Each runner loops over patches 1-9 internally (3x3 grid on 448x448 images, left-to-right, top-to-bottom) and runs train + test sequentially for each patch.

#### exp1-1: MedMamba Remove-One-Patch
```
python run_exp1-1.py --method remove \
    --train_path TRAIN_PATH --val_path VAL_PATH --test_path TEST_PATH \
    --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR
```

#### exp1-2: DenseNet/SwinTF Patch Experiments

**Remove-One-Patch**
```
python run_exp1-2.py --method remove \
    --train_path TRAIN_PATH --val_path VAL_PATH --test_path TEST_PATH \
    --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR
```

**Keep-One-Patch**
```
python run_exp1-2.py --method keep \
    --train_path TRAIN_PATH --val_path VAL_PATH --test_path TEST_PATH \
    --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR
```

train_path: The train csv dataset file location
val_path: The validation csv dataset file location
test_path: The test csv dataset file location
experiment_name: Name your experiment (used in weight filenames)
weight_dir: Subdirectory name under root_dir where weights are saved (default root_dir: `/home/sebasmos/orcd/pool/code/insurance_paper_weights/`)

### Experiment 3: Experiments on Demographic Mediators
#### Health insurance type prediction performance across multiple machine learning methods given the combination of age, race, and sex attributes.
Refer to ml_analysis.ipynb file

#### DenseNet121 trained on isolated White people
**Train**
```
python train_insurance_fullimgsize_densenet.py --mode train --method keep --train_path TRAIN_PATH --val_path VAL_PATH --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR --idx INDEX
```

**Test**
```
python train_insurance_fullimgsize_densenet.py --mode test --method keep --train_path TRAIN_PATH --val_path VAL_PATH --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR --idx INDEX
```
train_path: The train csv dataset file location (Only with White people)
val_path: The test csv dataset file location (Only with White people)
experiment_name: Name your experiment as you wish
weight_dir: The directory where you saved your weights

## Pretrained Weights

All 245 pretrained checkpoints (153 original + 92 MedGemma) are available on Hugging Face:

**[InsurancePrediction/insurance_paper_weights](https://huggingface.co/InsurancePrediction/insurance_paper_weights)**

```bash
# Download all weights
git lfs install
git clone https://huggingface.co/InsurancePrediction/insurance_paper_weights

# Download a single experiment
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="InsurancePrediction/insurance_paper_weights",
    allow_patterns="exp3-1/*",
    local_dir="./weights"
)
```

See the [HF README](https://huggingface.co/InsurancePrediction/insurance_paper_weights) for the full checkpoint table, loading examples, and path mappings.

## Bootstrap Evaluation (Confidence Intervals)

Produces bootstrap confidence intervals (overall + demographic subgroups) for trained models. Runs inference once, then resamples predictions to compute 95% CIs for AUC, Precision, Recall, F1, and Accuracy.

### Files

| File | Purpose |
|------|---------|
| `bootstrap_evaluate.py` | Shared bootstrap engine — loads model, runs inference once, resamples N times for CIs (overall + subgroups). Supports base models, addDemo models, patch masking, and frequency filtering. Args: `--num_workers` (default 4), `--batch_size` (default 32). |
| `run_exp0_bootstrap.py` | Wrapper for exp0 (baseline full-image models) |
| `run_exp1_bootstrap.py` | Wrapper for exp1-1/1-2 (patch-based: remove-one-patch / keep-one-patch) |
| `run_exp2_bootstrap.py` | Wrapper for exp2/2-1/2-2 (resolution experiments, loops over resolutions) |
| `run_exp3_bootstrap.py` | Wrapper for exp3/3-1/3-2 (demographics addDemo models) |
| `run_exp4_bootstrap.py` | Wrapper for exp4/4-1/4-2 (frequency filtering high/low pass) |
| `slurm/exp0_bootstrap.sh` | SLURM array job: 6 tasks (3 models × 2 datasets) |
| `slurm/exp1_bootstrap.sh` | SLURM array job: 6 tasks (mamba remove, mamba keep, densenet keep, densenet remove, swinTF keep, swinTF remove) × 9 patches each |
| `slurm/exp2_bootstrap.sh` | SLURM array job: 3 tasks (mamba, densenet, swinTF) × 7-8 resolutions each |
| `slurm/exp3_bootstrap.sh` | SLURM array job: 3 tasks (mamba, densenet, swinTF) × 7 demo combos each |
| `slurm/exp4_bootstrap.sh` | SLURM array job: 6 tasks (3 models × 2 directions) × 8 frequencies each |
| `slurm/run_all_bootstrap.sh` | **Full production run**: submits SLURM jobs (n_bootstrap=20, sample_size=1000) |
| `slurm/test_all_bootstrap.sh` | Quick smoke test: 1 real checkpoint per experiment type (n_bootstrap=2, sample_size=10) |

### Shared Defaults

All SLURM scripts share these configurable parameters:

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_PATH` (MIMIC) | `/home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv` | Path to MIMIC test CSV |
| `TEST_PATH` (CheXpert) | `/orcd/pool/006/lceli_shared/data/ChestXray/Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord` | Path to CheXpert test tfrecord |
| `ROOT_DIR` | `/home/sebasmos/orcd/pool/code/insurance_paper_weights/` | Root directory where trained weights live (HF repo) |
| `N_BOOTSTRAP` | `20` | Number of bootstrap iterations |
| `SAMPLE_SIZE` | `1000` | Samples per bootstrap iteration |
| `BOOTSTRAP_SEED` | `42` | Seed for reproducible resampling |
| `TRAIN_SEED` | `123` | Seed used during model training (for weight path construction) |

### Environment Setup (for all local runs)

```bash
cd /home/sebasmos/orcd/pool/code/Insurance-Project-Journal-version-
module load miniforge/24.3.0-0
module load cuda/12.4.0
conda activate jama_insurance
```

---

### Exp0: Baseline (full image, no extras)

**Note:** Exp0 weights were trained by Chi-Yu and are available in the HF repo at `insurance_paper_weights/exp0/{MIMIC,CheXpert}/{densenet,mamba,swinTF}.pt`. The bootstrap SLURM script points directly at these HF weight paths.

**SLURM (all 6 combos: 3 models × 2 datasets)**
```bash
sbatch slurm/exp0_bootstrap.sh
```

**Local (single combo)**
```bash
python run_exp0_bootstrap.py \
    --dataset MIMIC --model densenet \
    --test_path /home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv \
    --experiment_name exp0_baseline --weight_dir weights/exp0 \
    --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123
```

---

### Exp1: Patch-based (remove-one-patch / keep-one-patch)

Evaluates patch localization experiments. Each combo loops over patches 1-9.

| Array task | Model | Method | Training script | # Checkpoints |
|------------|-------|--------|-----------------|---------------|
| 1 | mamba | remove | exp1-1 | 9 |
| 2 | densenet | keep | exp1-2 | 9 |
| 3 | swinTF | keep | exp1-2 | 9 |
| 4 | swinTF | remove | exp1-2 | 9 |
| 5 | mamba | keep | exp1-1 | 9 |
| 6 | densenet | remove | exp1-2 | 9 |

**SLURM (all 6 combos in parallel, 9 patches each)**
```bash
sbatch slurm/exp1_bootstrap.sh
```
Logs: `slurm/bootstrap_exp1/`. Output: `bootstrap_results/exp1/`.

**Local (single model/method)**
```bash
# mamba remove-one-patch (exp1-1)
python run_exp1_bootstrap.py \
    --model mamba --method remove \
    --test_path /home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv \
    --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123

# densenet keep-one-patch (exp1-2)
python run_exp1_bootstrap.py \
    --model densenet --method keep \
    --test_path /home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv \
    --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123

# swinTF keep-one-patch (exp1-2)
python run_exp1_bootstrap.py \
    --model swinTF --method keep \
    --test_path /home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv \
    --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123

# swinTF remove-one-patch (exp1-2)
python run_exp1_bootstrap.py \
    --model swinTF --method remove \
    --test_path /home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv \
    --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123
```

**Direct (single patch)**
```bash
python bootstrap_evaluate.py \
    --dataset MIMIC --model mamba \
    --preprocessing mask_remove --patch_idx 1 \
    --test_path /home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv \
    --weight_path /home/sebasmos/orcd/pool/code/insurance_paper_weights/exp1-1/mamba_remove/patch1.pt \
    --n_bootstrap 20 --sample_size 1000 --seed 42 \
    --output_dir bootstrap_results/exp1 --experiment_name MIMIC_mamba_remove_patch1
```

---

### Exp2: Resolution (image downscale experiments)

Evaluates models trained on images downscaled to different resolutions. Images are resized DOWN to the target resolution then UP to 448×448 before feeding to the model (two-step resize to test information loss).

| Array task | Model | Resolutions | # Checkpoints |
|------------|-------|-------------|---------------|
| 1 | mamba | 2, 4, 7, 14, 28, 56, 112, 224 | 8 |
| 2 | densenet | 2, 4, 7, 14, 28, 56, 112, 224 | 8 |
| 3 | swinTF | 4, 7, 14, 28, 56, 112, 224 | 7 |

**SLURM (all 3 models in parallel)**
```bash
sbatch slurm/exp2_bootstrap.sh
```
Logs: `slurm/bootstrap_exp2/`. Output: `bootstrap_results/exp2/`.

**Local (single model, all resolutions)**
```bash
python run_exp2_bootstrap.py \
    --model mamba \
    --test_path /home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv \
    --output_dir bootstrap_results/exp2 \
    --n_bootstrap 20 --sample_size 1000 --seed 42
```

**Direct (single resolution)**
```bash
python bootstrap_evaluate.py \
    --dataset MIMIC --model mamba \
    --resize 28 \
    --test_path /home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv \
    --weight_path /home/sebasmos/orcd/pool/code/insurance_paper_weights/exp2/mamba_28.pt \
    --n_bootstrap 20 --sample_size 1000 --seed 42 \
    --output_dir bootstrap_results/exp2 --experiment_name MIMIC_mamba_res28_Rand123
```

---

### Exp3: Demographics (addDemo models)

Evaluates models trained with demographic features (sex, age, race combinations).

| Array task | Model | Training script | # Checkpoints (7 combos) |
|------------|-------|-----------------|--------------------------|
| 1 | mamba | exp3 | 7 |
| 2 | densenet | exp3-1 | 7 |
| 3 | swinTF | exp3-2 | 7 |

Demo combos: `sex`, `age`, `race`, `sexage`, `sexrace`, `agerace`, `sexagerace`

**SLURM (all 3 models in parallel, 7 combos each)**
```bash
sbatch slurm/exp3_bootstrap.sh
```
Logs: `slurm/bootstrap_exp3/`. Output: `bootstrap_results/exp3/`.

**Local (single model, all 7 combos)**
```bash
# mamba (exp3)
python run_exp3_bootstrap.py \
    --model mamba \
    --test_path /home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv \
    --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123

# densenet (exp3-1)
python run_exp3_bootstrap.py \
    --model densenet \
    --test_path /home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv \
    --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123

# swinTF (exp3-2)
python run_exp3_bootstrap.py \
    --model swinTF \
    --test_path /home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv \
    --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123
```

**Direct (single combo)**
```bash
python bootstrap_evaluate.py \
    --dataset MIMIC --model densenet \
    --model_variant addDemo --demo_labels sex age \
    --test_path /home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv \
    --weight_path /home/sebasmos/orcd/pool/code/insurance_paper_weights/exp3-1/sexage.pt \
    --n_bootstrap 20 --sample_size 1000 --seed 42 \
    --output_dir bootstrap_results/exp3 --experiment_name MIMIC_densenet_sexage
```

---

### Exp4: Frequency Filtering (high/low pass)

Evaluates models trained on frequency-filtered images.

| Array task | Model | Direction | Training script | # Checkpoints (8 freqs) |
|------------|-------|-----------|-----------------|-------------------------|
| 1 | mamba | highpass | exp4 | 8 |
| 2 | mamba | lowpass | exp4 | 8 |
| 3 | swinTF | highpass | exp4-1 | 8 |
| 4 | swinTF | lowpass | exp4-1 | 8 |
| 5 | densenet | highpass | exp4-2 | 8 |
| 6 | densenet | lowpass | exp4-2 | 8 |

Frequencies: 1, 5, 10, 25, 50, 100, 200, 400 Hz

**SLURM (all 6 combos in parallel, 8 frequencies each)**
```bash
sbatch slurm/exp4_bootstrap.sh
```
Logs: `slurm/bootstrap_exp4/`. Output: `bootstrap_results/exp4/`.

**Local (single model + direction, all 8 frequencies)**
```bash
# densenet highpass (exp4-2)
python run_exp4_bootstrap.py \
    --model densenet --direction highpass \
    --test_path /home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv \
    --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123

# mamba lowpass (exp4)
python run_exp4_bootstrap.py \
    --model mamba --direction lowpass \
    --test_path /home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv \
    --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123

# swinTF highpass (exp4-1)
python run_exp4_bootstrap.py \
    --model swinTF --direction highpass \
    --test_path /home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv \
    --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123
```

**Direct (single frequency)**
```bash
python bootstrap_evaluate.py \
    --dataset MIMIC --model densenet \
    --preprocessing high_pass --filter_diameter 5 \
    --test_path /home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv \
    --weight_path /home/sebasmos/orcd/pool/code/insurance_paper_weights/exp4-2/highpass/5Hz.pt \
    --n_bootstrap 20 --sample_size 1000 --seed 42 \
    --output_dir bootstrap_results/exp4 --experiment_name MIMIC_densenet_5HighPass
```

---

### Run ALL bootstrap experiments via SLURM (full production run)

Single command to submit all 153 checkpoints with `n_bootstrap=20, sample_size=1000`:

```bash
cd /home/sebasmos/orcd/pool/code/Insurance-Project-Journal-version-
bash slurm/run_all_bootstrap.sh
```

Or preview without submitting:
```bash
bash slurm/run_all_bootstrap.sh --dry-run
```

This submits 3 SLURM array jobs (14 array tasks total), covering all experiments that have trained weights. See the [Checkpoint Table](#checkpoint-table) for full details.

### Quick smoke test (verify pipeline before full run)

```bash
sbatch slurm/test_all_bootstrap.sh
```

Runs 1 real checkpoint per experiment type with reduced bootstrap (`n_bootstrap=2, sample_size=10`). Uses real test data (3628 samples) and real trained weights. Logs to `slurm/test_all_bootstrap_<jobid>.log`.

### Output

Each checkpoint produces two files:
- `<name>_bootstrap_iterations.csv` — per-iteration metrics (N=20 rows × subgroups)
- `<name>_bootstrap_summary.csv` — mean, std, 95% CI for AUC, Precision, Recall, F1, Accuracy

Subgroups evaluated: Overall, Gender (Male/Female), Age (<40/40-50/50-65), Race (White/Black/Other).

#### `bootstrap_results/exp0/` — Baseline experiments (exp0)

Naming: `{dataset}_{model}_exp0_baseline_Rand123_bootstrap_{iterations,summary}.csv`

| Dataset | Model | Summary |
|---------|-------|---------|
| MIMIC | mamba | [`MIMIC_mamba_exp0_baseline_Rand123`](bootstrap_results/exp0/MIMIC_mamba_exp0_baseline_Rand123_bootstrap_summary.csv) |
| MIMIC | densenet | [`MIMIC_densenet_exp0_baseline_Rand123`](bootstrap_results/exp0/MIMIC_densenet_exp0_baseline_Rand123_bootstrap_summary.csv) |
| MIMIC | swinTF | [`MIMIC_swinTF_exp0_baseline_Rand123`](bootstrap_results/exp0/MIMIC_swinTF_exp0_baseline_Rand123_bootstrap_summary.csv) |
| CheXpert | mamba | [`CheXpert_mamba_exp0_baseline_Rand123`](bootstrap_results/exp0/CheXpert_mamba_exp0_baseline_Rand123_bootstrap_summary.csv) |
| CheXpert | densenet | [`CheXpert_densenet_exp0_baseline_Rand123`](bootstrap_results/exp0/CheXpert_densenet_exp0_baseline_Rand123_bootstrap_summary.csv) |
| CheXpert | swinTF | [`CheXpert_swinTF_exp0_baseline_Rand123`](bootstrap_results/exp0/CheXpert_swinTF_exp0_baseline_Rand123_bootstrap_summary.csv) |

**Total: 6 evaluations → 12 files**

#### `bootstrap_results/exp0-2/` — Random init control (exp0-2)

| Model | Summary |
|-------|---------|
| swinTF (random) | [`MIMIC_swinTF_random_Rand123`](bootstrap_results/exp0-2/MIMIC_swinTF_random_Rand123_bootstrap_summary.csv) |

**Total: 1 evaluation → 2 files**

#### `bootstrap_results/exp1/` — Patch experiments (exp1-1 + exp1-2)

Naming: `MIMIC_{model}_{method}_patch{N}_Rand123_bootstrap_{iterations,summary}.csv`

| Experiment | Model | Method | Files (patch 1-9) |
|------------|-------|--------|--------------------|
| exp1-1 | mamba | remove | [`MIMIC_mamba_remove_patch1_Rand123`](bootstrap_results/exp1/MIMIC_mamba_remove_patch1_Rand123_bootstrap_summary.csv) ... [`patch9`](bootstrap_results/exp1/MIMIC_mamba_remove_patch9_Rand123_bootstrap_summary.csv) |
| exp1-1 | mamba | keep | [`MIMIC_mamba_keep_patch1_Rand123`](bootstrap_results/exp1/MIMIC_mamba_keep_patch1_Rand123_bootstrap_summary.csv) ... [`patch9`](bootstrap_results/exp1/MIMIC_mamba_keep_patch9_Rand123_bootstrap_summary.csv) |
| exp1-2 | densenet | keep | [`MIMIC_densenet_keep_patch1_Rand123`](bootstrap_results/exp1/MIMIC_densenet_keep_patch1_Rand123_bootstrap_summary.csv) ... [`patch9`](bootstrap_results/exp1/MIMIC_densenet_keep_patch9_Rand123_bootstrap_summary.csv) |
| exp1-2 | densenet | remove | [`MIMIC_densenet_remove_patch1_Rand123`](bootstrap_results/exp1/MIMIC_densenet_remove_patch1_Rand123_bootstrap_summary.csv) ... [`patch9`](bootstrap_results/exp1/MIMIC_densenet_remove_patch9_Rand123_bootstrap_summary.csv) |
| exp1-2 | swinTF | keep | [`MIMIC_swinTF_keep_patch1_Rand123`](bootstrap_results/exp1/MIMIC_swinTF_keep_patch1_Rand123_bootstrap_summary.csv) ... [`patch9`](bootstrap_results/exp1/MIMIC_swinTF_keep_patch9_Rand123_bootstrap_summary.csv) |
| exp1-2 | swinTF | remove | [`MIMIC_swinTF_remove_patch1_Rand123`](bootstrap_results/exp1/MIMIC_swinTF_remove_patch1_Rand123_bootstrap_summary.csv) ... [`patch9`](bootstrap_results/exp1/MIMIC_swinTF_remove_patch9_Rand123_bootstrap_summary.csv) |

**Total: 54 evaluations → 108 files** (9 patches × 6 model/method combos × 2 CSVs)

#### `bootstrap_results/exp2/` — Resolution experiments (exp2 + exp2-1 + exp2-2)

Naming: `MIMIC_{model}_res{N}_Rand123_bootstrap_{iterations,summary}.csv`

Images are resized DOWN to the target resolution then UP to 448×448 (the model's native input size), testing information loss at different resolutions.

| Experiment | Model | Resolutions | Files |
|------------|-------|-------------|-------|
| exp2 | mamba | 2, 4, 7, 14, 28, 56, 112, 224 | [`res2`](bootstrap_results/exp2/MIMIC_mamba_res2_Rand123_bootstrap_summary.csv), [`res4`](bootstrap_results/exp2/MIMIC_mamba_res4_Rand123_bootstrap_summary.csv), [`res7`](bootstrap_results/exp2/MIMIC_mamba_res7_Rand123_bootstrap_summary.csv), [`res14`](bootstrap_results/exp2/MIMIC_mamba_res14_Rand123_bootstrap_summary.csv), [`res28`](bootstrap_results/exp2/MIMIC_mamba_res28_Rand123_bootstrap_summary.csv), [`res56`](bootstrap_results/exp2/MIMIC_mamba_res56_Rand123_bootstrap_summary.csv), [`res112`](bootstrap_results/exp2/MIMIC_mamba_res112_Rand123_bootstrap_summary.csv), [`res224`](bootstrap_results/exp2/MIMIC_mamba_res224_Rand123_bootstrap_summary.csv) |
| exp2-1 | densenet | 2, 4, 7, 14, 28, 56, 112, 224 | [`res2`](bootstrap_results/exp2/MIMIC_densenet_res2_Rand123_bootstrap_summary.csv), [`res4`](bootstrap_results/exp2/MIMIC_densenet_res4_Rand123_bootstrap_summary.csv), [`res7`](bootstrap_results/exp2/MIMIC_densenet_res7_Rand123_bootstrap_summary.csv), [`res14`](bootstrap_results/exp2/MIMIC_densenet_res14_Rand123_bootstrap_summary.csv), [`res28`](bootstrap_results/exp2/MIMIC_densenet_res28_Rand123_bootstrap_summary.csv), [`res56`](bootstrap_results/exp2/MIMIC_densenet_res56_Rand123_bootstrap_summary.csv), [`res112`](bootstrap_results/exp2/MIMIC_densenet_res112_Rand123_bootstrap_summary.csv), [`res224`](bootstrap_results/exp2/MIMIC_densenet_res224_Rand123_bootstrap_summary.csv) |
| exp2-2 | swinTF | 4, 7, 14, 28, 56, 112, 224 | [`res4`](bootstrap_results/exp2/MIMIC_swinTF_res4_Rand123_bootstrap_summary.csv), [`res7`](bootstrap_results/exp2/MIMIC_swinTF_res7_Rand123_bootstrap_summary.csv), [`res14`](bootstrap_results/exp2/MIMIC_swinTF_res14_Rand123_bootstrap_summary.csv), [`res28`](bootstrap_results/exp2/MIMIC_swinTF_res28_Rand123_bootstrap_summary.csv), [`res56`](bootstrap_results/exp2/MIMIC_swinTF_res56_Rand123_bootstrap_summary.csv), [`res112`](bootstrap_results/exp2/MIMIC_swinTF_res112_Rand123_bootstrap_summary.csv), [`res224`](bootstrap_results/exp2/MIMIC_swinTF_res224_Rand123_bootstrap_summary.csv) |

**Total: 23 evaluations → 46 files** (8+8+7 resolutions × 2 CSVs)

#### `bootstrap_results/exp3/` — Demographics experiments (exp3 + exp3-1 + exp3-2)

Naming: `MIMIC_{model}_{combo}_Rand123_bootstrap_{iterations,summary}.csv`

Demo combos: `sex`, `age`, `race`, `sexage`, `sexrace`, `agerace`, `sexagerace`

| Experiment | Model | Files |
|------------|-------|-------|
| exp3 | mamba | [`MIMIC_mamba_sex_Rand123`](bootstrap_results/exp3/MIMIC_mamba_sex_Rand123_bootstrap_summary.csv), [`age`](bootstrap_results/exp3/MIMIC_mamba_age_Rand123_bootstrap_summary.csv), [`race`](bootstrap_results/exp3/MIMIC_mamba_race_Rand123_bootstrap_summary.csv), [`sexage`](bootstrap_results/exp3/MIMIC_mamba_sexage_Rand123_bootstrap_summary.csv), [`sexrace`](bootstrap_results/exp3/MIMIC_mamba_sexrace_Rand123_bootstrap_summary.csv), [`agerace`](bootstrap_results/exp3/MIMIC_mamba_agerace_Rand123_bootstrap_summary.csv), [`sexagerace`](bootstrap_results/exp3/MIMIC_mamba_sexagerace_Rand123_bootstrap_summary.csv) |
| exp3-1 | densenet | [`MIMIC_densenet_sex_Rand123`](bootstrap_results/exp3/MIMIC_densenet_sex_Rand123_bootstrap_summary.csv), [`age`](bootstrap_results/exp3/MIMIC_densenet_age_Rand123_bootstrap_summary.csv), [`race`](bootstrap_results/exp3/MIMIC_densenet_race_Rand123_bootstrap_summary.csv), [`sexage`](bootstrap_results/exp3/MIMIC_densenet_sexage_Rand123_bootstrap_summary.csv), [`sexrace`](bootstrap_results/exp3/MIMIC_densenet_sexrace_Rand123_bootstrap_summary.csv), [`agerace`](bootstrap_results/exp3/MIMIC_densenet_agerace_Rand123_bootstrap_summary.csv), [`sexagerace`](bootstrap_results/exp3/MIMIC_densenet_sexagerace_Rand123_bootstrap_summary.csv) |
| exp3-2 | swinTF | [`MIMIC_swinTF_sex_Rand123`](bootstrap_results/exp3/MIMIC_swinTF_sex_Rand123_bootstrap_summary.csv), [`age`](bootstrap_results/exp3/MIMIC_swinTF_age_Rand123_bootstrap_summary.csv), [`race`](bootstrap_results/exp3/MIMIC_swinTF_race_Rand123_bootstrap_summary.csv), [`sexage`](bootstrap_results/exp3/MIMIC_swinTF_sexage_Rand123_bootstrap_summary.csv), [`sexrace`](bootstrap_results/exp3/MIMIC_swinTF_sexrace_Rand123_bootstrap_summary.csv), [`agerace`](bootstrap_results/exp3/MIMIC_swinTF_agerace_Rand123_bootstrap_summary.csv), [`sexagerace`](bootstrap_results/exp3/MIMIC_swinTF_sexagerace_Rand123_bootstrap_summary.csv) |

**Total: 21 checkpoints → 42 files** (7 combos × 3 models × 2 CSVs)

#### `bootstrap_results/exp4/` — Frequency filtering experiments (exp4 + exp4-1 + exp4-2)

Naming: `MIMIC_{model}_{freq}{HighPass|LowPass}_Rand123_bootstrap_{iterations,summary}.csv`

Frequencies: 1, 5, 10, 25, 50, 100, 200, 400 Hz

| Experiment | Model | Direction | Files |
|------------|-------|-----------|-------|
| exp4 | mamba | highpass | [`MIMIC_mamba_1HighPass_Rand123`](bootstrap_results/exp4/MIMIC_mamba_1HighPass_Rand123_bootstrap_summary.csv), [`5`](bootstrap_results/exp4/MIMIC_mamba_5HighPass_Rand123_bootstrap_summary.csv), [`10`](bootstrap_results/exp4/MIMIC_mamba_10HighPass_Rand123_bootstrap_summary.csv), [`25`](bootstrap_results/exp4/MIMIC_mamba_25HighPass_Rand123_bootstrap_summary.csv), [`50`](bootstrap_results/exp4/MIMIC_mamba_50HighPass_Rand123_bootstrap_summary.csv), [`100`](bootstrap_results/exp4/MIMIC_mamba_100HighPass_Rand123_bootstrap_summary.csv), [`200`](bootstrap_results/exp4/MIMIC_mamba_200HighPass_Rand123_bootstrap_summary.csv), [`400`](bootstrap_results/exp4/MIMIC_mamba_400HighPass_Rand123_bootstrap_summary.csv) |
| exp4 | mamba | lowpass | [`MIMIC_mamba_1LowPass_Rand123`](bootstrap_results/exp4/MIMIC_mamba_1LowPass_Rand123_bootstrap_summary.csv), [`5`](bootstrap_results/exp4/MIMIC_mamba_5LowPass_Rand123_bootstrap_summary.csv), [`10`](bootstrap_results/exp4/MIMIC_mamba_10LowPass_Rand123_bootstrap_summary.csv), [`25`](bootstrap_results/exp4/MIMIC_mamba_25LowPass_Rand123_bootstrap_summary.csv), [`50`](bootstrap_results/exp4/MIMIC_mamba_50LowPass_Rand123_bootstrap_summary.csv), [`100`](bootstrap_results/exp4/MIMIC_mamba_100LowPass_Rand123_bootstrap_summary.csv), [`200`](bootstrap_results/exp4/MIMIC_mamba_200LowPass_Rand123_bootstrap_summary.csv), [`400`](bootstrap_results/exp4/MIMIC_mamba_400LowPass_Rand123_bootstrap_summary.csv) |
| exp4-1 | swinTF | highpass | [`MIMIC_swinTF_1HighPass_Rand123`](bootstrap_results/exp4/MIMIC_swinTF_1HighPass_Rand123_bootstrap_summary.csv), [`5`](bootstrap_results/exp4/MIMIC_swinTF_5HighPass_Rand123_bootstrap_summary.csv), [`10`](bootstrap_results/exp4/MIMIC_swinTF_10HighPass_Rand123_bootstrap_summary.csv), [`25`](bootstrap_results/exp4/MIMIC_swinTF_25HighPass_Rand123_bootstrap_summary.csv), [`50`](bootstrap_results/exp4/MIMIC_swinTF_50HighPass_Rand123_bootstrap_summary.csv), [`100`](bootstrap_results/exp4/MIMIC_swinTF_100HighPass_Rand123_bootstrap_summary.csv), [`200`](bootstrap_results/exp4/MIMIC_swinTF_200HighPass_Rand123_bootstrap_summary.csv), [`400`](bootstrap_results/exp4/MIMIC_swinTF_400HighPass_Rand123_bootstrap_summary.csv) |
| exp4-1 | swinTF | lowpass | [`MIMIC_swinTF_1LowPass_Rand123`](bootstrap_results/exp4/MIMIC_swinTF_1LowPass_Rand123_bootstrap_summary.csv), [`5`](bootstrap_results/exp4/MIMIC_swinTF_5LowPass_Rand123_bootstrap_summary.csv), [`10`](bootstrap_results/exp4/MIMIC_swinTF_10LowPass_Rand123_bootstrap_summary.csv), [`25`](bootstrap_results/exp4/MIMIC_swinTF_25LowPass_Rand123_bootstrap_summary.csv), [`50`](bootstrap_results/exp4/MIMIC_swinTF_50LowPass_Rand123_bootstrap_summary.csv), [`100`](bootstrap_results/exp4/MIMIC_swinTF_100LowPass_Rand123_bootstrap_summary.csv), [`200`](bootstrap_results/exp4/MIMIC_swinTF_200LowPass_Rand123_bootstrap_summary.csv), [`400`](bootstrap_results/exp4/MIMIC_swinTF_400LowPass_Rand123_bootstrap_summary.csv) |
| exp4-2 | densenet | highpass | [`MIMIC_densenet_1HighPass_Rand123`](bootstrap_results/exp4/MIMIC_densenet_1HighPass_Rand123_bootstrap_summary.csv), [`5`](bootstrap_results/exp4/MIMIC_densenet_5HighPass_Rand123_bootstrap_summary.csv), [`10`](bootstrap_results/exp4/MIMIC_densenet_10HighPass_Rand123_bootstrap_summary.csv), [`25`](bootstrap_results/exp4/MIMIC_densenet_25HighPass_Rand123_bootstrap_summary.csv), [`50`](bootstrap_results/exp4/MIMIC_densenet_50HighPass_Rand123_bootstrap_summary.csv), [`100`](bootstrap_results/exp4/MIMIC_densenet_100HighPass_Rand123_bootstrap_summary.csv), [`200`](bootstrap_results/exp4/MIMIC_densenet_200HighPass_Rand123_bootstrap_summary.csv), [`400`](bootstrap_results/exp4/MIMIC_densenet_400HighPass_Rand123_bootstrap_summary.csv) |
| exp4-2 | densenet | lowpass | [`MIMIC_densenet_1LowPass_Rand123`](bootstrap_results/exp4/MIMIC_densenet_1LowPass_Rand123_bootstrap_summary.csv), [`5`](bootstrap_results/exp4/MIMIC_densenet_5LowPass_Rand123_bootstrap_summary.csv), [`10`](bootstrap_results/exp4/MIMIC_densenet_10LowPass_Rand123_bootstrap_summary.csv), [`25`](bootstrap_results/exp4/MIMIC_densenet_25LowPass_Rand123_bootstrap_summary.csv), [`50`](bootstrap_results/exp4/MIMIC_densenet_50LowPass_Rand123_bootstrap_summary.csv), [`100`](bootstrap_results/exp4/MIMIC_densenet_100LowPass_Rand123_bootstrap_summary.csv), [`200`](bootstrap_results/exp4/MIMIC_densenet_200LowPass_Rand123_bootstrap_summary.csv), [`400`](bootstrap_results/exp4/MIMIC_densenet_400LowPass_Rand123_bootstrap_summary.csv) |

**Total: 48 checkpoints → 96 files** (8 frequencies × 2 directions × 3 models × 2 CSVs)

#### Grand total (original): 153 evaluations → 306 result files

| Experiment | Evaluations | Files |
|------------|-------------|-------|
| exp0 (baseline) | 6 | 12 |
| exp0-2 (random init) | 1 | 2 |
| exp1 (patches) | 54 | 108 |
| exp2 (resolution) | 23 | 46 |
| exp3 (demographics) | 21 | 42 |
| exp4 (frequency) | 48 | 96 |
| **Original Total** | **153** | **306** |
| MedGemma exp2 (resolution) | 23 | 46 |
| MedGemma exp3 (demographics) | 21 | 42 |
| MedGemma exp4 (frequency) | 48 | 96 |
| **MedGemma Total** | **92** | **184** |
| **Grand Total** | **245** | **490** |

### Note on MedMamba

`MedMamba/MedMamba.py` includes both `VSSM_DoubleLinear` (base) and `VSSM_Double_addDemothen2` (demographics addDemo variant with deeper classification head). MedMamba requires CUDA (the selective_scan kernel only runs on GPU).

---

## MedGemma Experiments

Repeats experiments 2, 3, and 4 on the **MedGemma-refined dataset** (`mimic-cxr-gemma`), which contains only normal CXRs filtered by MedGemma. Experiments 0 and 0-2 are handled separately by Chi-Yu.

**Dataset:** `/orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_{train,val,test}_medgemmaChecked.csv`

**Weights:** `insurance_paper_weights/exp{2,3,3-1,3-2,4,4-1,4-2}_medgemma/`

**Bootstrap results:** `bootstrap_results/medgemma_exp{2,3,4}/`

### MedGemma Training

All training scripts are in `slurm/medgemma/`. Submit all training jobs at once:

```bash
bash slurm/medgemma/mg_run_all_training.sh
```

#### MedGemma Exp2: Resolution (image downscale)

Images are resized DOWN to the target resolution then UP to 448×448 (two-step resize to test information loss).

| Exp | Model | Resolutions | Training script | SLURM script |
|-----|-------|-------------|-----------------|--------------|
| exp2 | mamba | 2, 4, 7, 14, 28, 56, 112, 224 | `run_exp2.py` | `slurm/medgemma/mg_exp2_mamba.sh` |
| exp2-1 | densenet | 2, 4, 7, 14, 28, 56, 112, 224 | `run_exp2.py` | `slurm/medgemma/mg_exp2-1_densenet.sh` |
| exp2-2 | swinTF | 4, 7, 14, 28, 56, 112, 224 | `run_exp2.py` | `slurm/medgemma/mg_exp2-2_swinTF.sh` |

Weights: `insurance_paper_weights/exp2_medgemma/mamba_{N}.pt`, `exp2-1_medgemma/densenet_{N}.pt`, `exp2-2_medgemma/swinTF_{N}.pt`

#### MedGemma Exp3: Demographics (addDemo models)

| Exp | Model | Training script | SLURM script |
|-----|-------|-----------------|--------------|
| exp3 | mamba | `run_exp3.py` | `slurm/medgemma/mg_exp3_mamba.sh` |
| exp3-1 | densenet | `run_exp3-1.py` | `slurm/medgemma/mg_exp3-1_densenet.sh` |
| exp3-2 | swinTF | `run_exp3-2.py` | `slurm/medgemma/mg_exp3-2_swinTF.sh` |

Demo combos: `sex`, `age`, `race`, `sexage`, `sexrace`, `agerace`, `sexagerace`

Weights: `insurance_paper_weights/exp3[-1|-2]_medgemma/{combo}.pt`

#### MedGemma Exp4: Frequency Filtering (high/low pass)

| Exp | Model | Direction | SLURM script |
|-----|-------|-----------|--------------|
| exp4 | mamba | highpass | `slurm/medgemma/mg_exp4_mamba_high.sh` |
| exp4 | mamba | lowpass | `slurm/medgemma/mg_exp4_mamba_low.sh` |
| exp4-1 | swinTF | highpass | `slurm/medgemma/mg_exp4-1_swinTF_high.sh` |
| exp4-1 | swinTF | lowpass | `slurm/medgemma/mg_exp4-1_swinTF_low.sh` |
| exp4-2 | densenet | highpass | `slurm/medgemma/mg_exp4-2_densenet_high.sh` |
| exp4-2 | densenet | lowpass | `slurm/medgemma/mg_exp4-2_densenet_low.sh` |

Frequencies: 1, 5, 10, 25, 50, 100, 200, 400 Hz

Weights: `insurance_paper_weights/exp4[-1|-2]_medgemma/{highpass,lowpass}/{freq}Hz.pt`

### MedGemma Bootstrap Evaluation

Submit all bootstrap jobs at once:

```bash
bash slurm/medgemma/mg_run_all_bootstrap.sh
```

| File | Purpose |
|------|---------|
| `run_exp2_bootstrap_medgemma.py` | Wrapper for MedGemma exp2 (resolution, mamba only) |
| `run_exp3_bootstrap_medgemma.py` | Wrapper for MedGemma exp3/3-1/3-2 (demographics, 3 models) |
| `run_exp4_bootstrap_medgemma.py` | Wrapper for MedGemma exp4/4-1/4-2 (frequency, 3 models × 2 directions) |

**Local examples:**

```bash
# Exp2 bootstrap (mamba, all 8 resolutions)
python run_exp2_bootstrap_medgemma.py \
    --model mamba \
    --test_path /orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
    --n_bootstrap 20 --sample_size 1000 --seed 42

# Exp3 bootstrap (densenet, all 7 combos)
python run_exp3_bootstrap_medgemma.py \
    --model densenet \
    --test_path /orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
    --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123

# Exp4 bootstrap (swinTF highpass, all 8 frequencies)
python run_exp4_bootstrap_medgemma.py \
    --model swinTF --direction highpass \
    --test_path /orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
    --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123
```

### MedGemma Bootstrap Results

#### `bootstrap_results/medgemma_exp2/` — Resolution (MedGemma)

Naming: `MIMIC_medgemma_{model}_res{N}_Rand123_bootstrap_{iterations,summary}.csv`

| Experiment | Model | Resolutions | Files |
|------------|-------|-------------|-------|
| exp2 | mamba | 2, 4, 7, 14, 28, 56, 112, 224 | [`res2`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_mamba_res2_Rand123_bootstrap_summary.csv), [`res4`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_mamba_res4_Rand123_bootstrap_summary.csv), [`res7`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_mamba_res7_Rand123_bootstrap_summary.csv), [`res14`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_mamba_res14_Rand123_bootstrap_summary.csv), [`res28`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_mamba_res28_Rand123_bootstrap_summary.csv), [`res56`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_mamba_res56_Rand123_bootstrap_summary.csv), [`res112`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_mamba_res112_Rand123_bootstrap_summary.csv), [`res224`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_mamba_res224_Rand123_bootstrap_summary.csv) |
| exp2-1 | densenet | 2, 4, 7, 14, 28, 56, 112, 224 | [`res2`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_densenet_res2_Rand123_bootstrap_summary.csv), [`res4`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_densenet_res4_Rand123_bootstrap_summary.csv), [`res7`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_densenet_res7_Rand123_bootstrap_summary.csv), [`res14`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_densenet_res14_Rand123_bootstrap_summary.csv), [`res28`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_densenet_res28_Rand123_bootstrap_summary.csv), [`res56`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_densenet_res56_Rand123_bootstrap_summary.csv), [`res112`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_densenet_res112_Rand123_bootstrap_summary.csv), [`res224`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_densenet_res224_Rand123_bootstrap_summary.csv) |
| exp2-2 | swinTF | 4, 7, 14, 28, 56, 112, 224 | [`res4`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_swinTF_res4_Rand123_bootstrap_summary.csv), [`res7`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_swinTF_res7_Rand123_bootstrap_summary.csv), [`res14`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_swinTF_res14_Rand123_bootstrap_summary.csv), [`res28`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_swinTF_res28_Rand123_bootstrap_summary.csv), [`res56`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_swinTF_res56_Rand123_bootstrap_summary.csv), [`res112`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_swinTF_res112_Rand123_bootstrap_summary.csv), [`res224`](bootstrap_results/medgemma_exp2/MIMIC_medgemma_swinTF_res224_Rand123_bootstrap_summary.csv) |

**Total: 23 evaluations → 46 files**

#### `bootstrap_results/medgemma_exp3/` — Demographics (MedGemma)

Naming: `MIMIC_medgemma_{model}_{combo}_Rand123_bootstrap_{iterations,summary}.csv`

Demo combos: `sex`, `age`, `race`, `sexage`, `sexrace`, `agerace`, `sexagerace`

| Experiment | Model | Files |
|------------|-------|-------|
| exp3 | mamba | [`sex`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_mamba_sex_Rand123_bootstrap_summary.csv), [`age`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_mamba_age_Rand123_bootstrap_summary.csv), [`race`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_mamba_race_Rand123_bootstrap_summary.csv), [`sexage`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_mamba_sexage_Rand123_bootstrap_summary.csv), [`sexrace`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_mamba_sexrace_Rand123_bootstrap_summary.csv), [`agerace`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_mamba_agerace_Rand123_bootstrap_summary.csv), [`sexagerace`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_mamba_sexagerace_Rand123_bootstrap_summary.csv) |
| exp3-1 | densenet | [`sex`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_densenet_sex_Rand123_bootstrap_summary.csv), [`age`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_densenet_age_Rand123_bootstrap_summary.csv), [`race`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_densenet_race_Rand123_bootstrap_summary.csv), [`sexage`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_densenet_sexage_Rand123_bootstrap_summary.csv), [`sexrace`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_densenet_sexrace_Rand123_bootstrap_summary.csv), [`agerace`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_densenet_agerace_Rand123_bootstrap_summary.csv), [`sexagerace`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_densenet_sexagerace_Rand123_bootstrap_summary.csv) |
| exp3-2 | swinTF | [`sex`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_swinTF_sex_Rand123_bootstrap_summary.csv), [`age`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_swinTF_age_Rand123_bootstrap_summary.csv), [`race`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_swinTF_race_Rand123_bootstrap_summary.csv), [`sexage`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_swinTF_sexage_Rand123_bootstrap_summary.csv), [`sexrace`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_swinTF_sexrace_Rand123_bootstrap_summary.csv), [`agerace`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_swinTF_agerace_Rand123_bootstrap_summary.csv), [`sexagerace`](bootstrap_results/medgemma_exp3/MIMIC_medgemma_swinTF_sexagerace_Rand123_bootstrap_summary.csv) |

**Total: 21 evaluations → 42 files** (7 combos × 3 models × 2 CSVs)

#### `bootstrap_results/medgemma_exp4/` — Frequency filtering (MedGemma)

Naming: `MIMIC_medgemma_{model}_{freq}{HighPass|LowPass}_Rand123_bootstrap_{iterations,summary}.csv`

Frequencies: 1, 5, 10, 25, 50, 100, 200, 400 Hz

| Experiment | Model | Direction | Files |
|------------|-------|-----------|-------|
| exp4 | mamba | highpass | [`1`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_mamba_1HighPass_Rand123_bootstrap_summary.csv), [`5`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_mamba_5HighPass_Rand123_bootstrap_summary.csv), [`10`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_mamba_10HighPass_Rand123_bootstrap_summary.csv), [`25`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_mamba_25HighPass_Rand123_bootstrap_summary.csv), [`50`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_mamba_50HighPass_Rand123_bootstrap_summary.csv), [`100`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_mamba_100HighPass_Rand123_bootstrap_summary.csv), [`200`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_mamba_200HighPass_Rand123_bootstrap_summary.csv), [`400`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_mamba_400HighPass_Rand123_bootstrap_summary.csv) |
| exp4 | mamba | lowpass | [`1`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_mamba_1LowPass_Rand123_bootstrap_summary.csv), [`5`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_mamba_5LowPass_Rand123_bootstrap_summary.csv), [`10`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_mamba_10LowPass_Rand123_bootstrap_summary.csv), [`25`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_mamba_25LowPass_Rand123_bootstrap_summary.csv), [`50`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_mamba_50LowPass_Rand123_bootstrap_summary.csv), [`100`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_mamba_100LowPass_Rand123_bootstrap_summary.csv), [`200`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_mamba_200LowPass_Rand123_bootstrap_summary.csv), [`400`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_mamba_400LowPass_Rand123_bootstrap_summary.csv) |
| exp4-1 | swinTF | highpass | [`1`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_swinTF_1HighPass_Rand123_bootstrap_summary.csv), [`5`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_swinTF_5HighPass_Rand123_bootstrap_summary.csv), [`10`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_swinTF_10HighPass_Rand123_bootstrap_summary.csv), [`25`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_swinTF_25HighPass_Rand123_bootstrap_summary.csv), [`50`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_swinTF_50HighPass_Rand123_bootstrap_summary.csv), [`100`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_swinTF_100HighPass_Rand123_bootstrap_summary.csv), [`200`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_swinTF_200HighPass_Rand123_bootstrap_summary.csv), [`400`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_swinTF_400HighPass_Rand123_bootstrap_summary.csv) |
| exp4-1 | swinTF | lowpass | [`1`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_swinTF_1LowPass_Rand123_bootstrap_summary.csv), [`5`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_swinTF_5LowPass_Rand123_bootstrap_summary.csv), [`10`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_swinTF_10LowPass_Rand123_bootstrap_summary.csv), [`25`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_swinTF_25LowPass_Rand123_bootstrap_summary.csv), [`50`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_swinTF_50LowPass_Rand123_bootstrap_summary.csv), [`100`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_swinTF_100LowPass_Rand123_bootstrap_summary.csv), [`200`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_swinTF_200LowPass_Rand123_bootstrap_summary.csv), [`400`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_swinTF_400LowPass_Rand123_bootstrap_summary.csv) |
| exp4-2 | densenet | highpass | [`1`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_densenet_1HighPass_Rand123_bootstrap_summary.csv), [`5`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_densenet_5HighPass_Rand123_bootstrap_summary.csv), [`10`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_densenet_10HighPass_Rand123_bootstrap_summary.csv), [`25`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_densenet_25HighPass_Rand123_bootstrap_summary.csv), [`50`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_densenet_50HighPass_Rand123_bootstrap_summary.csv), [`100`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_densenet_100HighPass_Rand123_bootstrap_summary.csv), [`200`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_densenet_200HighPass_Rand123_bootstrap_summary.csv), [`400`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_densenet_400HighPass_Rand123_bootstrap_summary.csv) |
| exp4-2 | densenet | lowpass | [`1`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_densenet_1LowPass_Rand123_bootstrap_summary.csv), [`5`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_densenet_5LowPass_Rand123_bootstrap_summary.csv), [`10`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_densenet_10LowPass_Rand123_bootstrap_summary.csv), [`25`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_densenet_25LowPass_Rand123_bootstrap_summary.csv), [`50`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_densenet_50LowPass_Rand123_bootstrap_summary.csv), [`100`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_densenet_100LowPass_Rand123_bootstrap_summary.csv), [`200`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_densenet_200LowPass_Rand123_bootstrap_summary.csv), [`400`](bootstrap_results/medgemma_exp4/MIMIC_medgemma_densenet_400LowPass_Rand123_bootstrap_summary.csv) |

**Total: 48 evaluations → 96 files** (8 frequencies × 2 directions × 3 models × 2 CSVs)

#### MedGemma grand total: 92 evaluations → 184 result files

### MedGemma Checkpoint Table

All MedGemma pretrained weights are available on HuggingFace: **[InsurancePrediction/insurance_paper_weights](https://huggingface.co/InsurancePrediction/insurance_paper_weights)**

| Experiment | Description | Model | Config | # Checkpoints | Training script | Bootstrap script | HF weight path |
|------------|-------------|-------|--------|---------------|-----------------|------------------|----------------|
| **exp2** | Resolution | mamba | 8 resolutions | 8 | `run_exp2.py` | `mg_exp2_bootstrap_mamba.sh` | `exp2_medgemma/mamba_{N}.pt` |
| **exp2-1** | Resolution | densenet | 8 resolutions | 8 | `run_exp2.py` | `mg_exp2_bootstrap_densenet.sh` | `exp2-1_medgemma/densenet_{N}.pt` |
| **exp2-2** | Resolution | swinTF | 7 resolutions | 7 | `run_exp2.py` | `mg_exp2_bootstrap_swinTF.sh` | `exp2-2_medgemma/swinTF_{N}.pt` |
| **exp3** | Demographics (addDemo) | mamba | 7 demo combos | 7 | `run_exp3.py` | `mg_exp3_bootstrap_mamba.sh` | `exp3_medgemma/{combo}.pt` |
| **exp3-1** | Demographics (addDemo) | densenet | 7 demo combos | 7 | `run_exp3-1.py` | `mg_exp3_bootstrap_densenet.sh` | `exp3-1_medgemma/{combo}.pt` |
| **exp3-2** | Demographics (addDemo) | swinTF | 7 demo combos | 7 | `run_exp3-2.py` | `mg_exp3_bootstrap_swinTF.sh` | `exp3-2_medgemma/{combo}.pt` |
| **exp4** | Freq filtering (HP+LP) | mamba | 8 freq × 2 | 16 | `run_exp4.py` | `mg_exp4_bootstrap_mamba_{high,low}.sh` | `exp4_medgemma/{highpass,lowpass}/{F}Hz.pt` |
| **exp4-1** | Freq filtering (HP+LP) | swinTF | 8 freq × 2 | 16 | `run_exp4-1.py` | `mg_exp4_bootstrap_swinTF_{high,low}.sh` | `exp4-1_medgemma/{highpass,lowpass}/{F}Hz.pt` |
| **exp4-2** | Freq filtering (HP+LP) | densenet | 8 freq × 2 | 16 | `run_exp4-2.py` | `mg_exp4_bootstrap_densenet_{high,low}.sh` | `exp4-2_medgemma/{highpass,lowpass}/{F}Hz.pt` |

**Total: 92 MedGemma checkpoints** (23 exp2 + 21 exp3 + 48 exp4)

---

---

## MedMamba-T and MedMamba-S Comparison

Investigating whether pretrained variants of MedMamba can improve upon the base MedMamba (`depths=[2,2,4,2]`, trained from scratch) and other models on the insurance prediction task.

- **MedMamba-T**: same Tiny architecture as base MedMamba (`depths=[2,2,4,2]`), but initialized from PneumoniaMNIST pretrained weights (`MedMamba_PneumoniaMNIST.pth`)
- **MedMamba-S [2,2,9,2]**: off-by-one bug — all original S runs used `depths=[2,2,9,2]` instead of the canonical `[2,2,8,2]`. Results kept for reference. Pretrained weight loading was `strict=False` with shape filter, so blocks 0–7 of stage 3 loaded correctly and block 8 was random.
- **MedMamba-S [2,2,8,2]**: corrected Small variant matching the original MedMamba paper (`depths=[2,2,8,2]`), re-run from the same `MedMamba_S_PneumoniaMNIST.pth` pretrained weights. Both columns shown for comparison.

**Pretrained weights (PneumoniaMNIST init):**
```
/orcd/pool/006/lceli_shared/weights/insurance_paper_weights/MedMamba_pretrained_weight/MedMamba_PneumoniaMNIST.pth    # T (Tiny pretrained)
/orcd/pool/006/lceli_shared/weights/insurance_paper_weights/MedMamba_pretrained_weight/MedMamba_S_PneumoniaMNIST.pth  # S (Small pretrained)
```

**To run MedMamba-T resolution experiments (exp2, MedGemma dataset):**
```bash
python run_exp2.py \
    --model mamba_t \
    --train_path /orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv \
    --val_path   /orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv \
    --test_path  /orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
    --experiment_name mg_mamba_t_resize \
    --weight_dir mg_exp2/mamba_t \
    --root_dir /home/sebasmos/orcd/scratch/JAMA_codes_medgemma/ \
    --pretrained_path /orcd/pool/006/lceli_shared/weights/insurance_paper_weights/MedMamba_pretrained_weight/MedMamba_PneumoniaMNIST.pth
```

**To run MedMamba-S resolution experiments (exp2, MedGemma dataset):**
```bash
python run_exp2.py \
    --model mamba_s \
    --train_path /orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv \
    --val_path   /orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv \
    --test_path  /orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
    --experiment_name mg_mamba_s_resize \
    --weight_dir mg_exp2/mamba_s \
    --root_dir /home/sebasmos/orcd/scratch/JAMA_codes_medgemma/ \
    --pretrained_path /orcd/pool/006/lceli_shared/weights/insurance_paper_weights/MedMamba_pretrained_weight/MedMamba_S_PneumoniaMNIST.pth
```

### Exp0 — Baseline (full-image, no extras)

AUC mean [95% CI] on test split. MedMamba (scratch), DenseNet, SwinTF use bootstrap CI. MedMamba-T/S are point estimates while training is in progress.

| Dataset | MedMamba (scratch) | MedMamba-T | MedMamba-S [2,2,9,2] | MedMamba-S [2,2,8,2] | DenseNet | SwinTF |
|---------|-------------------|-----------|---------------------|---------------------|---------|--------|
| MIMIC | 0.666 [0.640, 0.698] | 0.662 | 0.665 | 0.623 [0.601, 0.646] | **0.702 [0.677, 0.727]** | 0.618 [0.593, 0.655] |
| CheXpert | 0.618 [0.580, 0.652] | TBD | TBD | TBD | **0.685 [0.645, 0.719]** | 0.599 [0.551, 0.635] |

### Exp2 — Resolution (MedGemma dataset, MIMIC)

AUC (test, point estimate) at each resolution. Images resized down then up to 448×448.

| Resolution | MedMamba (scratch) | MedMamba-T | MedMamba-S [2,2,9,2] | MedMamba-S [2,2,8,2] | DenseNet | SwinTF |
|------------|-------------------|-----------|---------------------|---------------------|---------|--------|
| 2 | 0.554 | 0.558 | 0.555 | 0.562 [0.538, 0.586] | 0.554 | N/A |
| 4 | 0.605 | 0.584 | 0.598 | 0.603 [0.586, 0.624] | 0.601 | 0.558 |
| 7 | 0.606 | 0.605 | 0.616 | 0.625 [0.593, 0.645] | 0.604 | 0.595 |
| 14 | 0.623 | 0.645 | 0.635 | 0.635 [0.605, 0.661] | **0.638** | 0.602 |
| 28 | 0.637 | 0.634 | 0.639 | 0.635 [0.609, 0.658] | **0.647** | 0.632 |
| 56 | 0.626 | **0.652** | 0.646 | 0.627 [0.599, 0.652] | 0.637 | 0.632 |
| 112 | 0.609 | 0.656 | 0.658 | 0.640 [0.616, 0.665] | **0.664** | 0.623 |
| 224 | 0.610 | 0.646 | 0.659 | 0.648 [0.622, 0.674] | **0.680** | 0.616 |

> Fill in MedMamba-T and MedMamba-S columns as results complete. Bold = best per row.

### Exp3: Demographics (addDemo) — MedMamba-T / MedMamba-S

| Demo Combo | MedMamba (base) | DenseNet121 | SwinTF | MedMamba-T | MedMamba-S [2,2,9,2] | MedMamba-S [2,2,8,2] |
|------------|-----------------|-------------|--------|------------|---------------------|---------------------|
| sex | TBD | TBD | TBD | 0.638 | 0.650 | 0.644 [0.614, 0.673] |
| age | TBD | TBD | TBD | 0.641 | 0.643 | 0.638 [0.608, 0.664] |
| race | TBD | TBD | TBD | 0.655 | 0.647 | 0.664 [0.644, 0.694] |
| sexage | TBD | TBD | TBD | 0.639 | 0.640 | 0.639 [0.617, 0.664] |
| sexrace | TBD | TBD | TBD | 0.653 | 0.650 | 0.652 [0.632, 0.679] |
| agerace | TBD | TBD | TBD | 0.652 | 0.654 | 0.651 [0.621, 0.682] |
| sexagerace | TBD | TBD | TBD | 0.656 | 0.650 | 0.651 [0.623, 0.680] |

### Exp1: Patch-based — MedMamba-T / MedMamba-S

Test AUC per patch (3×3 grid on 448×448, left-to-right top-to-bottom). Remove = mask one patch out; Keep = train on one patch only.

#### Remove-One-Patch

| Patch | MedMamba-T | MedMamba-S [2,2,9,2] | MedMamba-S [2,2,8,2] |
|-------|-----------|---------------------|---------------------|
| 1 | 0.634 | 0.641 | 0.648 [0.615, 0.676] |
| 2 | 0.643 | 0.633 | 0.627 [0.599, 0.655] |
| 3 | 0.630 | 0.641 | 0.630 [0.603, 0.654] |
| 4 | **0.650** | 0.636 | 0.626 [0.602, 0.655] |
| 5 | 0.616 | 0.638 | 0.622 [0.597, 0.652] |
| 6 | 0.626 | **0.652** | 0.646 [0.616, 0.669] |
| 7 | 0.644 | 0.636 | 0.636 [0.612, 0.667] |
| 8 | 0.645 | 0.649 | 0.634 [0.612, 0.654] |
| 9 | 0.633 | 0.649 | 0.623 [0.590, 0.650] |

#### Keep-One-Patch

> Fixed CUDA device bug (set_device(1)→set_device(0)) on Apr 27; resubmitted as jobs 12629673 (T) and 12629674 (S).

| Patch | MedMamba-T | MedMamba-S [2,2,9,2] | MedMamba-S [2,2,8,2] |
|-------|-----------|---------------------|---------------------|
| 1 | 0.604 | 0.615 | 0.602 [0.578, 0.622] |
| 2 | 0.585 | 0.598 | 0.598 [0.571, 0.621] |
| 3 | 0.595 | 0.601 | 0.605 [0.566, 0.633] |
| 4 | 0.594 | 0.600 | 0.595 [0.571, 0.618] |
| 5 | 0.612 | 0.611 | 0.606 [0.578, 0.626] |
| 6 | 0.599 | 0.586 | 0.609 [0.581, 0.628] |
| 7 | 0.596 | 0.592 | 0.593 [0.574, 0.612] |
| 8 | 0.574 | 0.586 | 0.585 [0.563, 0.606] |
| 9 | 0.607 | 0.603 | 0.601 [0.576, 0.627] |

### Exp4: Frequency Filtering — MedMamba-T / MedMamba-S

Test AUC at each cutoff frequency.

#### Highpass

| Freq (Hz) | MedMamba-T | MedMamba-S [2,2,9,2] | MedMamba-S [2,2,8,2] |
|-----------|-----------|---------------------|---------------------|
| 1 | 0.636 | **0.635** | 0.625 [0.599, 0.647] |
| 5 | 0.618 | 0.607 | 0.608 [0.585, 0.627] |
| 10 | 0.611 | 0.614 | 0.613 [0.575, 0.648] |
| 25 | 0.630 | 0.613 | 0.604 [0.571, 0.633] |
| 50 | 0.621 | 0.610 | 0.612 [0.582, 0.647] |
| 100 | 0.620 | 0.621 | 0.587 [0.557, 0.621] |
| 200 | 0.615 | 0.604 | 0.603 [0.575, 0.625] |
| 400 | 0.598 | 0.594 | 0.600 [0.570, 0.630] |

#### Lowpass

| Freq (Hz) | MedMamba-T | MedMamba-S [2,2,9,2] | MedMamba-S [2,2,8,2] |
|-----------|-----------|---------------------|---------------------|
| 1 | 0.607 | 0.604 | 0.608 [0.583, 0.635] |
| 5 | 0.608 | 0.609 | 0.602 [0.580, 0.631] |
| 10 | 0.627 | 0.619 | 0.613 [0.585, 0.647] |
| 25 | **0.646** | 0.636 | 0.646 [0.623, 0.663] |
| 50 | 0.649 | 0.629 | 0.640 [0.613, 0.664] |
| 100 | **0.655** | 0.658 | 0.652 [0.629, 0.679] |
| 200 | 0.626 | 0.659 | 0.653 [0.630, 0.680] |
| 400 | 0.630 | 0.659 | 0.656 [0.631, 0.685] |

---

## Code References
MedMamba: [Link](https://github.com/YubiaoYue/MedMamba)

SwinTransformer V2: [Link](https://github.com/ChristophReich1996/Swin-Transformer-V2)

## Checkpoint Table

All pretrained weights are available on [HuggingFace](https://huggingface.co/InsurancePrediction/insurance_paper_weights). Seed=123.

All weights are available in the HF repo at `/home/sebasmos/orcd/pool/code/insurance_paper_weights/` (cloned from [HuggingFace](https://huggingface.co/InsurancePrediction/insurance_paper_weights)). Bootstrap code points directly to HF paths.

| Experiment | Description | Model | Config | # Checkpoints | Training script | Bootstrap script | HF weight path |
|------------|-------------|-------|--------|---------------|-----------------|------------------|----------------|
| **exp0** | Baseline (full image) | densenet, mamba, swinTF | MIMIC + CheXpert | 6 | `run_exp0.py` | `exp0_bootstrap.sh` | `exp0/{dataset}/{model}.pt` |
| **exp0-2** | Random init | swinTF | — | 1 | — | `bootstrap_evaluate.py` | `exp0-2/swinTF_random.pt` |
| **exp1-1** | Patch keep/remove | mamba | 9 patches × 2 | 18 | `run_exp1-1.py` | `exp1_bootstrap.sh` | `exp1-1/mamba_{keep,remove}/patch{1-9}.pt` |
| **exp1-2** | Patch keep/remove | densenet | 9 patches × 2 | 18 | `run_exp1-2.py` | `exp1_bootstrap.sh` | `exp1-2/densenet_{keep,remove}/patch{1-9}.pt` |
| **exp1-2** | Patch keep/remove | swinTF | 9 patches × 2 | 18 | `run_exp1-2.py` | `exp1_bootstrap.sh` | `exp1-2/swinTF_{keep,remove}/patch{1-9}.pt` |
| **exp2** | Resolution | mamba | 8 resolutions | 8 | — | `exp2_bootstrap.sh` | `exp2/mamba_{N}.pt` |
| **exp2-1** | Resolution | densenet | 8 resolutions | 8 | — | `exp2_bootstrap.sh` | `exp2-1/densenet_{N}.pt` |
| **exp2-2** | Resolution | swinTF | 7 resolutions | 7 | — | `exp2_bootstrap.sh` | `exp2-2/swinTF_{N}.pt` |
| **exp3** | Demographics (addDemo) | mamba | 7 demo combos | 7 | `run_exp3.py` | `exp3_bootstrap.sh` | `exp3/{combo}.pt` |
| **exp3-1** | Demographics (addDemo) | densenet | 7 demo combos | 7 | `run_exp3-1.py` | `exp3_bootstrap.sh` | `exp3-1/{combo}.pt` |
| **exp3-2** | Demographics (addDemo) | swinTF | 7 demo combos | 7 | `run_exp3-2.py` | `exp3_bootstrap.sh` | `exp3-2/{combo}.pt` |
| **exp4** | Freq filtering (HP+LP) | mamba | 8 freq × 2 | 16 | `run_exp4.py` | `exp4_bootstrap.sh` | `exp4/{highpass,lowpass}/{F}Hz.pt` |
| **exp4-1** | Freq filtering (HP+LP) | swinTF | 8 freq × 2 | 16 | `run_exp4-1.py` | `exp4_bootstrap.sh` | `exp4-1/{highpass,lowpass}/{F}Hz.pt` |
| **exp4-2** | Freq filtering (HP+LP) | densenet | 8 freq × 2 | 16 | `run_exp4-2.py` | `exp4_bootstrap.sh` | `exp4-2/{highpass,lowpass}/{F}Hz.pt` |

**Total: 153 original + 92 MedGemma = 245 unique checkpoints**

- **Demo combos** (exp3): `sex`, `age`, `race`, `sexage`, `sexrace`, `agerace`, `sexagerace`
- **Frequencies** (exp4): 1, 5, 10, 25, 50, 100, 200, 400 Hz
- **Resolutions** (exp2): 2, 4, 7, 14, 28, 56, 112, 224
- **Patches** (exp1): 1-9, corresponding to a 3x3 grid on 448x448 images (left-to-right, top-to-bottom)

