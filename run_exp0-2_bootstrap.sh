"""
Bootstrap validation wrapper for Experiment 0 (baseline models).

Mirrors run_exp0.py structure. Calls bootstrap_evaluate.py for each
model/dataset combination using the correct weight paths.

Usage:
    python run_exp0_bootstrap.py \
        --dataset MIMIC --model densenet \
        --test_path insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv \
        --experiment_name exp0_name \
        --weight_dir weights/exp0
"""

#!/bin/bash
python run_exp0-2_bootstrap.py \
        --dataset MIMIC --model densenet \
        --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --experiment_name exp0-2_medgemma \
        --weight_dir exp0-2_medgemma/densenet_perturbed.pt

python run_exp0-2_bootstrap.py \
        --dataset MIMIC --model mamba \
        --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --experiment_name exp0-2_mamba_pretrained \
        --weight_dir exp0-2_medgemma/mamba_pretrained_perturbed.pt

python run_exp0-2_bootstrap.py \
        --dataset MIMIC --model swinTF \
        --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --experiment_name exp0-2_medgemma_pretrained \
        --weight_dir exp0-2_medgemma/swinTF_pretrained_perturbed.pt

python run_exp0_bootstrap.py \
        --dataset CheXpert --model swinTF \
        --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_test_insurance_randomlabel.tfrecord \
        --experiment_name exp0-2 \
        --weight_dir exp0-2_medgemma/swinTF_pretrained_perturbed_chexpert.pt \

python run_exp0_bootstrap.py \
        --dataset CheXpert --model densenet \
        --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_test_insurance_randomlabel.tfrecord \
        --experiment_name exp0-2 \
        --weight_dir exp0-2_medgemma/densenet_perturbed_chexpert.pt \

python run_exp0_bootstrap.py \
        --dataset CheXpert --model mamba \
        --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_test_insurance_randomlabel.tfrecord \
        --experiment_name exp0-2 \
        --weight_dir exp0-2_medgemma/mamba_pretrained_perturbed_chexpert.pt \


