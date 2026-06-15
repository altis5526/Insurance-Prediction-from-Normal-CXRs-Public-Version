#!/bin/bash

python run_exp0.py --dataset MIMIC --model densenet --train_path insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv --val_path insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv --experiment_name NAME --weight_dir NAME

python run_exp0.py --dataset MIMIC --model swinTF --train_path insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv --val_path insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv --experiment_name  NAME --weight_dir NAME

python run_exp0.py --dataset MIMIC --model mamba --train_path insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv --val_path insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv --experiment_name NAME --weight_dir NAME

python run_exp0.py --dataset CheXpert --model densenet --train_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord --val_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord --experiment_name NAME  --weight_dir NAME

python run_exp0.py --dataset CheXpert --model swinTF --train_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord --val_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord --experiment_name  NAME --weight_dir NAME

python run_exp0.py --dataset CheXpert --model mamba --train_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord --val_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord --experiment_name  NAME --weight_dir NAME



