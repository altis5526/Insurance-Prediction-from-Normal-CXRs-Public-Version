#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=12:00:00
#SBATCH --job-name=mg_e0_s_fix
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

trap 'echo "Signal received at $(date)"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1

# MedMamba-S with corrected depths=[2,2,8,2] (fix for off-by-one bug where [2,2,9,2] was used).
# Exp0 MIMIC baseline. Original [2,2,9,2] run saved under mg_exp0/mamba_s.

### Environment Setup
cd /home/sebasmos/orcd/pool/code/Insurance-Project-Journal-version-
module load miniforge/24.3.0-0
conda activate jama_insurance
export PYTHONNOUSERSITE=1

PT=/orcd/pool/006/lceli_shared/weights/insurance_paper_weights/MedMamba_pretrained_weight/MedMamba_S_PneumoniaMNIST.pth

### Paths (MIMIC exp0)
TRAIN=/home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_train_no_support_devices.csv
VAL=/home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_val_no_support_devices.csv
TEST=/home/sebasmos/orcd/pool/code/JAMA_codes/insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv
ROOT=/home/sebasmos/orcd/scratch/JAMA_codes_exp0/
HF_WEIGHTS=/orcd/pool/006/lceli_shared/weights/insurance_paper_weights

### Logging
LOG_DIR="slurm/mg_exp0_mamba_s_fix"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/exp0_mimic_${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Variant: mamba_s_fix (depths=[2,2,8,2]), Dataset: MIMIC (exp0 baseline)"
echo "========================================="

### Train
python train_insurance_fullimgsize_mamba.py --mode train \
    --train_path $TRAIN --val_path $VAL \
    --experiment_name mg_exp0_mamba_s_fix_MIMIC \
    --weight_dir mg_exp0/mamba_s_fix_MIMIC \
    --seed 123 \
    --root_dir $ROOT \
    --model_variant mamba_s \
    --pretrained_path $PT

### Test
python train_insurance_fullimgsize_mamba.py --mode test \
    --train_path $TRAIN --val_path $TEST \
    --experiment_name mg_exp0_mamba_s_fix_MIMIC \
    --weight_dir mg_exp0/mamba_s_fix_MIMIC \
    --seed 123 \
    --root_dir $ROOT \
    --model_variant mamba_s \
    --pretrained_path $PT

### Collect weight
mkdir -p $HF_WEIGHTS/exp0/MIMIC
SRC=$ROOT/mg_exp0/mamba_s_fix_MIMIC/Rand123/Rand123_mg_exp0_mamba_s_fix_MIMIC_model_aucbest.pt
DST=$HF_WEIGHTS/exp0/MIMIC/mamba_s_fix.pt
if [ -f "$SRC" ]; then cp "$SRC" "$DST" && echo "Collected: $DST"; fi

echo "========================================="
echo "Completed at: $(date)"
echo "========================================="
