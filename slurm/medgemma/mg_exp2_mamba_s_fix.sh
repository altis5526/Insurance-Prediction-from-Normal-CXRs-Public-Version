#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=mg_exp2_mamba_s_fix
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

trap 'echo "Signal received at $(date)"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1

# MedMamba-S with corrected depths=[2,2,8,2] (fix for off-by-one bug where [2,2,9,2] was used).
# Exp2 resolution sweep (8 resolutions via run_exp2.py). Original [2,2,9,2] run saved under mg_exp2/mamba_s.

### Environment Setup
cd /home/sebasmos/orcd/pool/code/Insurance-Project-Journal-version-
module load miniforge/24.3.0-0
conda activate jama_insurance
export PYTHONNOUSERSITE=1

### Paths
TRAIN=/orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv
VAL=/orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv
TEST=/orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv
ROOT=/home/sebasmos/orcd/scratch/JAMA_codes_medgemma/
HF_WEIGHTS=/home/sebasmos/orcd/pool/code/insurance_paper_weights

### Logging
EXPERIMENT_NAME="mg_exp2_mamba_s_fix"
LOG_DIR="slurm/medgemma/${EXPERIMENT_NAME}"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_${SLURM_JOB_ID}.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Attempt: ${SLURM_RESTART_COUNT:-0}"
echo "Variant: mamba_s_fix (depths=[2,2,8,2])"
echo "========================================="

### Run Experiment 2 - MedMamba-S fix (8 resolutions)
python run_exp2.py \
    --train_path $TRAIN --val_path $VAL --test_path $TEST \
    --experiment_name mg_mamba_s_fix_resize \
    --weight_dir mg_exp2/mamba_s_fix \
    --model mamba_s \
    --pretrained_path /orcd/pool/006/lceli_shared/weights/insurance_paper_weights/MedMamba_pretrained_weight/MedMamba_S_PneumoniaMNIST.pth \
    --root_dir $ROOT

### Collect weights to flat HF structure
echo "Collecting weights to HF structure..."
mkdir -p $HF_WEIGHTS/exp2_medgemma_s_fix
for res in 2 4 7 14 28 56 112 224; do
    SRC=$ROOT/${res}_mg_exp2/mamba_s_fix/Rand123/Rand123_${res}_mg_mamba_s_fix_resize_model_aucbest.pt
    DST=$HF_WEIGHTS/exp2_medgemma_s_fix/mamba_s_fix_${res}.pt
    if [ -f "$SRC" ]; then cp "$SRC" "$DST" && echo "Collected: $DST"; fi
done

echo "========================================="
echo "Completed at: $(date)"
echo "========================================="
