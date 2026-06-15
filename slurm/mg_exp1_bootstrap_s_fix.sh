#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -G 1
#SBATCH --time=06:00:00
#SBATCH --job-name=mg_e1_bs_sfix
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

trap 'echo "Signal received at $(date)"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1

cd /home/sebasmos/orcd/pool/code/Insurance-Project-Journal-version-
module load miniforge/24.3.0-0
conda activate jama_insurance
export PYTHONNOUSERSITE=1

LOG_DIR="slurm/mg_bootstrap_s_fix"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/exp1_${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

TEST_PATH="/orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv"

echo "========================================="
echo "Bootstrap Evaluation - Exp1 mamba_s_fix [MedGemma]"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Attempt: ${SLURM_RESTART_COUNT:-0}"
echo "Log file: $LOG_FILE"
echo "========================================="

echo "--- Remove-One-Patch ---"
python -u run_exp1_bootstrap_medgemma_s_fix.py \
    --method remove \
    --test_path "$TEST_PATH" \
    --n_bootstrap 20 \
    --sample_size 1000 \
    --seed 42

REMOVE_EXIT=$?

echo "--- Keep-One-Patch ---"
python -u run_exp1_bootstrap_medgemma_s_fix.py \
    --method keep \
    --test_path "$TEST_PATH" \
    --n_bootstrap 20 \
    --sample_size 1000 \
    --seed 42

KEEP_EXIT=$?

if [ $REMOVE_EXIT -eq 0 ] && [ $KEEP_EXIT -eq 0 ]; then
    echo "========================================="
    echo "COMPLETED: exp1 mamba_s_fix (remove+keep) at $(date)"
    echo "========================================="
else
    echo "FAILED: remove_exit=$REMOVE_EXIT keep_exit=$KEEP_EXIT"
    exit 1
fi
