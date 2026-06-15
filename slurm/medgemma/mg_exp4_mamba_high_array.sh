#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue
#SBATCH --array=0-7
#SBATCH --signal=B:USR1@120

trap 'echo "Signal received at $(date)"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1

### Args: VARIANT (t or s) passed via --export
# Submit with: sbatch --export=VARIANT=t --job-name=mg_e4h_t slurm/medgemma/mg_exp4_mamba_high_array.sh
#              sbatch --export=VARIANT=s --job-name=mg_e4h_s slurm/medgemma/mg_exp4_mamba_high_array.sh

### Environment Setup
cd /home/sebasmos/orcd/pool/code/Insurance-Project-Journal-version-
module load miniforge/24.3.0-0
conda activate jama_insurance
export PYTHONNOUSERSITE=1

### Frequency from array index
FREQUENCIES=(1 5 10 25 50 100 200 400)
FREQ=${FREQUENCIES[$SLURM_ARRAY_TASK_ID]}

### Pretrained path
if [ "$VARIANT" = "t" ]; then
    PT=/orcd/pool/006/lceli_shared/weights/insurance_paper_weights/MedMamba_pretrained_weight/MedMamba_PneumoniaMNIST.pth
else
    PT=/orcd/pool/006/lceli_shared/weights/insurance_paper_weights/MedMamba_pretrained_weight/MedMamba_S_PneumoniaMNIST.pth
fi

### Paths
TRAIN=/orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv
VAL=/orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv
TEST=/orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv
ROOT=/home/sebasmos/orcd/scratch/JAMA_codes_medgemma/
HF_WEIGHTS=/orcd/pool/006/lceli_shared/weights/insurance_paper_weights

### Logging
LOG_DIR="slurm/medgemma/mg_exp4_mamba_${VARIANT}_high"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/freq${FREQ}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Variant: mamba_${VARIANT}, Direction: highpass, Frequency: ${FREQ}"
echo "========================================="

### Train
python train_insurance_fullimgsize_mamba_high_pass.py --mode train \
    --train_path $TRAIN --val_path $VAL \
    --experiment_name ${FREQ}HighPass_mg_mamba_${VARIANT}_highpass \
    --weight_dir ${FREQ}HighPass_mg_exp4/mamba_${VARIANT}/highpass \
    --seed 123 --high_pass_diameter $FREQ \
    --root_dir $ROOT \
    --model_variant mamba_${VARIANT} \
    --pretrained_path $PT

### Test
python train_insurance_fullimgsize_mamba_high_pass.py --mode test \
    --train_path $TRAIN --val_path $TEST \
    --experiment_name ${FREQ}HighPass_mg_mamba_${VARIANT}_highpass \
    --weight_dir ${FREQ}HighPass_mg_exp4/mamba_${VARIANT}/highpass \
    --seed 123 --high_pass_diameter $FREQ \
    --root_dir $ROOT \
    --model_variant mamba_${VARIANT} \
    --pretrained_path $PT

### Collect weight
mkdir -p $HF_WEIGHTS/exp4_medgemma_${VARIANT}/highpass
SRC=$ROOT/${FREQ}HighPass_mg_exp4/mamba_${VARIANT}/highpass/Rand123/Rand123_${FREQ}HighPass_mg_mamba_${VARIANT}_highpass_model_aucbest.pt
DST=$HF_WEIGHTS/exp4_medgemma_${VARIANT}/highpass/${FREQ}Hz.pt
if [ -f "$SRC" ]; then cp "$SRC" "$DST" && echo "Collected: $DST"; fi

echo "========================================="
echo "Completed at: $(date)"
echo "========================================="
