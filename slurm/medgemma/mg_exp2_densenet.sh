#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=mg_exp2-1_dn
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

trap 'echo "Signal received at $(date)"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1

cd /home/sebasmos/orcd/pool/code/Insurance-Project-Journal-version-
module load miniforge/24.3.0-0
conda activate jama_insurance
export PYTHONNOUSERSITE=1

TRAIN=/orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv
VAL=/orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv
TEST=/orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv
ROOT=/home/sebasmos/orcd/scratch/JAMA_codes_medgemma/
HF_WEIGHTS=/orcd/pool/006/lceli_shared/weights/insurance_paper_weights

EXPERIMENT_NAME="mg_exp2_densenet"
LOG_DIR="slurm/medgemma/${EXPERIMENT_NAME}"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_${SLURM_JOB_ID}.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Exp2-1: DenseNet resolution (8 resolutions, MedGemma)"
echo "========================================="

python run_exp2.py \
    --train_path $TRAIN --val_path $VAL --test_path $TEST \
    --experiment_name mg_densenet_resize \
    --weight_dir mg_exp2/densenet \
    --model densenet \
    --root_dir $ROOT

### Collect weights
echo "Collecting weights..."
mkdir -p $HF_WEIGHTS/exp2-1_medgemma
for res in 2 4 7 14 28 56 112 224; do
    SRC=$ROOT/${res}_mg_exp2/densenet/Rand123/Rand123_${res}_mg_densenet_resize_model_aucbest.pt
    DST=$HF_WEIGHTS/exp2-1_medgemma/densenet_${res}.pt
    if [ -f "$SRC" ]; then cp "$SRC" "$DST" && echo "Collected: $DST"; fi
done

echo "========================================="
echo "Completed at: $(date)"
echo "========================================="
