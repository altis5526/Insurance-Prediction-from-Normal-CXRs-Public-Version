#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue
#SBATCH --array=1-9
#SBATCH --signal=B:USR1@120

trap 'echo "Signal received at $(date)"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1

# MedMamba-S with corrected depths=[2,2,8,2] (fix for off-by-one bug where [2,2,9,2] was used).
# Exp1 remove-one-patch. Submit with: sbatch --job-name=mg_e1r_s_fix slurm/medgemma/mg_exp1_mamba_remove_s_fix_array.sh

### Environment Setup
cd /home/sebasmos/orcd/pool/code/Insurance-Project-Journal-version-
module load miniforge/24.3.0-0
conda activate jama_insurance
export PYTHONNOUSERSITE=1

### Patch index from array task
PATCH=$SLURM_ARRAY_TASK_ID

PT=/orcd/pool/006/lceli_shared/weights/insurance_paper_weights/MedMamba_pretrained_weight/MedMamba_S_PneumoniaMNIST.pth

### Paths
TRAIN=/orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv
VAL=/orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv
TEST=/orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv
ROOT=/home/sebasmos/orcd/scratch/JAMA_codes_medgemma/
HF_WEIGHTS=/orcd/pool/006/lceli_shared/weights/insurance_paper_weights

### Logging
LOG_DIR="slurm/medgemma/mg_exp1_mamba_s_fix_remove"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/patch${PATCH}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Variant: mamba_s_fix (depths=[2,2,8,2]), Method: remove, Patch: ${PATCH}"
echo "========================================="

### Train
python train_insurance_fullimgsize_mamba_mask_area.py --mode train \
    --train_path $TRAIN --val_path $VAL \
    --experiment_name mg_mamba_s_fix_remove_patch${PATCH} \
    --weight_dir mg_exp1/mamba_s_fix/remove \
    --seed 123 --patch_idx $PATCH \
    --root_dir $ROOT \
    --model_variant mamba_s \
    --pretrained_path $PT

### Test
python train_insurance_fullimgsize_mamba_mask_area.py --mode test \
    --train_path $TRAIN --val_path $TEST \
    --experiment_name mg_mamba_s_fix_remove_patch${PATCH} \
    --weight_dir mg_exp1/mamba_s_fix/remove \
    --seed 123 --patch_idx $PATCH \
    --root_dir $ROOT \
    --model_variant mamba_s \
    --pretrained_path $PT

### Collect weight
mkdir -p $HF_WEIGHTS/exp1-1_medgemma_s_fix
SRC=$ROOT/mg_exp1/mamba_s_fix/remove/Rand123/Rand123_patchidx${PATCH}_mg_mamba_s_fix_remove_patch${PATCH}_model_aucbest.pt
DST=$HF_WEIGHTS/exp1-1_medgemma_s_fix/remove_patch${PATCH}.pt
if [ -f "$SRC" ]; then cp "$SRC" "$DST" && echo "Collected: $DST"; fi

echo "========================================="
echo "Completed at: $(date)"
echo "========================================="
