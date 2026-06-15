#!/bin/bash

# python run_exp0.py \
#     --dataset CheXpert \
#     --model densenet \
#     --train_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyWhite_Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord \
#     --val_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyWhite_Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord \
#     --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyWhite_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#     --experiment_name OnlyWhite_densenet_CheXpert \
#     --weight_dir OnlyWhite_densenet_CheXpert

python run_exp0.py \
    --dataset CheXpert \
    --model mamba \
    --train_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyWhite_Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord \
    --val_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyWhite_Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord \
    --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyWhite_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
    --experiment_name OnlyWhite_pretrained_mamba_CheXpert \
    --weight_dir OnlyWhite_pretrained_mamba_CheXpert

# python run_exp0.py \
#     --dataset CheXpert \
#     --model swinTF \
#     --train_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyWhite_Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord \
#     --val_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyWhite_Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord \
#     --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyWhite_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#     --experiment_name OnlyWhite_pretrained_swinTFB_CheXpert \
#     --weight_dir OnlyWhite_pretrained_swinTFB_CheXpert

# python run_exp0.py \
#     --dataset CheXpert \
#     --model densenet \
#     --train_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyMale_Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord \
#     --val_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyMale_Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord \
#     --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyMale_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#     --experiment_name OnlyMale_densenet_CheXpert \
#     --weight_dir OnlyMale_densenet_CheXpert

python run_exp0.py \
    --dataset CheXpert \
    --model mamba \
    --train_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyMale_Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord \
    --val_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyMale_Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord \
    --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyMale_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
    --experiment_name OnlyMale_pretrained_mamba_CheXpert \
    --weight_dir OnlyMale_pretrained_mamba_CheXpert


# python run_exp0.py \
#     --dataset CheXpert \
#     --model swinTF \
#     --train_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyMale_Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord \
#     --val_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyMale_Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord \
#     --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyMale_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#     --experiment_name OnlyMale_pretrained_swinTFB_CheXpert \
#     --weight_dir OnlyMale_pretrained_swinTFB_CheXpert

# python run_exp0.py \
#     --dataset CheXpert \
#     --model densenet \
#     --train_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyFemale_Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord \
#     --val_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyFemale_Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord \
#     --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyFemale_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#     --experiment_name OnlyFemale_densenet_CheXpert \
#     --weight_dir OnlyFemale_densenet_CheXpert

python run_exp0.py \
    --dataset CheXpert \
    --model mamba \
    --train_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyFemale_Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord \
    --val_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyFemale_Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord \
    --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyFemale_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
    --experiment_name OnlyFemale_pretrained_mamba_CheXpert \
    --weight_dir OnlyFemale_pretrained_mamba_CheXpert

# python run_exp0.py \
#     --dataset CheXpert \
#     --model swinTF \
#     --train_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyFemale_Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord \
#     --val_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyFemale_Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord \
#     --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyFemale_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#     --experiment_name OnlyFemale_pretrained_swinTFB_CheXpert \
#     --weight_dir OnlyFemale_pretrained_swinTFB_CheXpert

# python run_exp0.py \
#     --dataset CheXpert \
#     --model densenet \
#     --train_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyOld_Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord \
#     --val_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyOld_Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord \
#     --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyOld_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#     --experiment_name OnlyOld_densenet_CheXpert \
#     --weight_dir OnlyOld_densenet_CheXpert

python run_exp0.py \
    --dataset CheXpert \
    --model mamba \
    --train_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyOld_Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord \
    --val_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyOld_Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord \
    --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyOld_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
    --experiment_name OnlyOld_pretrained_mamba_CheXpert \
    --weight_dir OnlyOld_pretrained_mamba_CheXpert

# python run_exp0.py \
#     --dataset CheXpert \
#     --model swinTF \
#     --train_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyOld_Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord \
#     --val_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyOld_Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord \
#     --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyOld_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#     --experiment_name OnlyOld_pretrained_swinTFB_CheXpert \
#     --weight_dir OnlyOld_pretrained_swinTFB_CheXpert