#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -G 1
#SBATCH --time=06:00:00
#SBATCH --job-name=mg_e4l_bs_sfix
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

#########################
### SIGNAL HANDLING   ###
#########################
trap 'echo "Signal received at $(date)"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1

### Environment Setup
cd /home/sebasmos/orcd/pool/code/Insurance-Project-Journal-version-
module load miniforge/24.3.0-0
conda activate jama_insurance
export PYTHONNOUSERSITE=1

### Logging
LOG_DIR="slurm/mg_bootstrap_s_fix"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/exp4_low_${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

### Experiment configuration
TEST_PATH="/orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv"

### Logging header
echo "========================================="
echo "Bootstrap Evaluation - Exp4 LowPass mamba_s_fix [MedGemma]"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Attempt: ${SLURM_RESTART_COUNT:-0}"
echo "Log file: $LOG_FILE"
echo "========================================="

### Run bootstrap evaluation (loops over frequencies internally)
python -u run_exp4_bootstrap_medgemma_s_fix.py \
    --direction lowpass \
    --test_path "$TEST_PATH" \
    --n_bootstrap 20 \
    --sample_size 1000 \
    --seed 42 \
    --train_seed 123

PYTHON_EXIT=$?
if [ $PYTHON_EXIT -eq 0 ]; then
    echo "========================================="
    echo "COMPLETED: exp4 lowpass mamba_s_fix at $(date)"
    echo "========================================="
else
    echo "FAILED with exit code $PYTHON_EXIT"
    exit 1
fi
