#!/bin/bash

python run_exp0-2.py --dataset MIMIC \
                --model densenet \
                --train_path insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv \
                --val_path insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv \
                --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
                --experiment_name PerturbedLabel_Densenet_MedgemmaChecked_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_DoubleLinear_20260604 \
                --weight_dir PerturbedLabel_Densenet_MedgemmaChecked_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_DoubleLinear_20260604

python run_exp0-2.py --dataset MIMIC \
                --model mamba \
                --train_path insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv \
                --val_path insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv \
                --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
                --experiment_name PerturbedLabel_PretrainedMamba_MedgemmaChecked_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_DoubleLinear_20260604 \
                --weight_dir PerturbedLabel_PretrainedMamba_MedgemmaChecked_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_DoubleLinear_20260604

python run_exp0-2.py --dataset MIMIC \
                --model swinTF \
                --train_path insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv \
                --val_path insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv \
                --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
                --experiment_name PerturbedLabel_PretrainedSwinTFB_MedgemmaChecked_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_DoubleLinear_20260604 \
                --weight_dir PerturbedLabel_PretrainedSwinTFB_MedgemmaChecked_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_DoubleLinear_20260604


python run_exp0.py \
    --dataset CheXpert \
    --model densenet \
    --train_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_train_insurance_randomlabel.tfrecord \
    --val_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_val_insurance_randomlabel.tfrecord \
    --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_test_insurance_randomlabel.tfrecord \
    --experiment_name PerturbedLabel_Densenet_CheXpert_20260605 \
    --weight_dir PerturbedLabel_Densenet_CheXpert_20260605

python run_exp0.py \
    --dataset CheXpert \
    --model mamba \
    --train_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_train_insurance_randomlabel.tfrecord \
    --val_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_val_insurance_randomlabel.tfrecord \
    --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_test_insurance_randomlabel.tfrecord \
    --experiment_name PerturbedLabel_PretrainedMamba_CheXpert_20260605 \
    --weight_dir PerturbedLabel_PretrainedMamba_CheXpert_20260605

python run_exp0.py \
    --dataset CheXpert \
    --model swinTF \
    --train_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_train_insurance_randomlabel.tfrecord \
    --val_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_val_insurance_randomlabel.tfrecord \
    --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_test_insurance_randomlabel.tfrecord \
    --experiment_name PerturbedLabel_PretrainedSwinTF_CheXpert_20260605 \
    --weight_dir PerturbedLabel_PretrainedSwinTF_CheXpert_20260605