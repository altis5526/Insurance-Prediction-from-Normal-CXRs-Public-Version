#!/bin/bash

#!/bin/bash
# python run_exp0_bootstrap.py \
#         --dataset MIMIC --model densenet \
#         --test_path Only_female_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
#         --experiment_name exp5_OnlyFemale_densenet \
#         --weight_dir exp5_medgemma/OnlyFemale_densenet/OnlyFemale_densenet.pt

python run_exp0_bootstrap.py \
        --dataset MIMIC --model mamba \
        --test_path Only_female_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --experiment_name exp5_OnlyFemale_mamba \
        --weight_dir exp5_medgemma/OnlyFemale_pretrained_mamba/OnlyFemale_pretrained_mamba.pt

# python run_exp0_bootstrap.py \
#         --dataset MIMIC --model swinTF \
#         --test_path Only_female_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
#         --experiment_name exp5_OnlyFemale_pretrained_swinTFB \
#         --weight_dir exp5_medgemma/OnlyFemale_pretrained_swinTFB/OnlyFemale_pretrained_swinTFB.pt

# python run_exp0_bootstrap.py \
#         --dataset MIMIC --model densenet \
#         --test_path Only_male_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
#         --experiment_name exp5_OnlyMale_densenet \
#         --weight_dir exp5_medgemma/OnlyMale_densenet/OnlyMale_densenet.pt

python run_exp0_bootstrap.py \
        --dataset MIMIC --model mamba \
        --test_path Only_male_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --experiment_name exp5_OnlyMale_mamba \
        --weight_dir exp5_medgemma/OnlyMale_pretrained_mamba/OnlyMale_pretrained_mamba.pt

# python run_exp0_bootstrap.py \
#         --dataset MIMIC --model swinTF \
#         --test_path Only_male_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
#         --experiment_name exp5_OnlyMale_pretrained_swinTFB \
#         --weight_dir exp5_medgemma/OnlyMale_pretrained_swinTFB/OnlyMale_pretrained_swinTFB.pt

# python run_exp0_bootstrap.py \
#         --dataset MIMIC --model densenet \
#         --test_path Only_white_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
#         --experiment_name exp5_OnlyWhite_densenet \
#         --weight_dir exp5_medgemma/OnlyWhite_densenet/OnlyWhite_densenet.pt

python run_exp0_bootstrap.py \
        --dataset MIMIC --model mamba \
        --test_path Only_white_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --experiment_name exp5_OnlyWhite_mamba \
        --weight_dir exp5_medgemma/OnlyWhite_pretrained_mamba/OnlyWhite_pretrained_mamba.pt

# python run_exp0_bootstrap.py \
#         --dataset MIMIC --model swinTF \
#         --test_path Only_white_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
#         --experiment_name exp5_OnlyWhite_pretrained_swinTFB \
#         --weight_dir exp5_medgemma/OnlyWhite_pretrained_swinTFB/OnlyWhite_pretrained_swinTFB.pt

# python run_exp0_bootstrap.py \
#         --dataset MIMIC --model densenet \
#         --test_path Only_old_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
#         --experiment_name exp5_OnlyOld_densenet \
#         --weight_dir exp5_medgemma/OnlyOld_densenet/OnlyOld_densenet.pt

python run_exp0_bootstrap.py \
        --dataset MIMIC --model mamba \
        --test_path Only_old_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --experiment_name exp5_OnlyOld_mamba \
        --weight_dir exp5_medgemma/OnlyOld_pretrained_mamba/OnlyOld_pretrained_mamba.pt

# python run_exp0_bootstrap.py \
#         --dataset MIMIC --model swinTF \
#         --test_path Only_old_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
#         --experiment_name exp5_OnlyOld_pretrained_swinTFB \
#         --weight_dir exp5_medgemma/OnlyOld_pretrained_swinTFB/OnlyOld_pretrained_swinTFB.pt


# python run_exp0_bootstrap.py \
#         --dataset CheXpert --model densenet \
#         --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyWhite_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#         --experiment_name exp5_CheXpert_OnlyWhite_densenet \
#         --weight_dir exp5_medgemma/OnlyWhite_densenet_CheXpert/OnlyWhite_densenet_CheXpert.pt


# python run_exp0_bootstrap.py \
#         --dataset CheXpert --model mamba \
#         --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyWhite_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#         --experiment_name exp5_CheXpert_OnlyWhite_mamba \
#         --weight_dir exp5_medgemma/OnlyWhite_pretrained_mamba_CheXpert/OnlyWhite_pretrained_mamba_CheXpert.pt

# python run_exp0_bootstrap.py \
#         --dataset CheXpert --model swinTF \
#         --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyWhite_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#         --experiment_name exp5_OnlyWhite_pretrained_swinTFB_CheXpert \
#         --weight_dir exp5_medgemma/OnlyWhite_pretrained_swinTFB_CheXpert/OnlyWhite_pretrained_swinTFB_CheXpert.pt


# python run_exp0_bootstrap.py \
#         --dataset CheXpert --model densenet \
#         --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyMale_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#         --experiment_name exp5_CheXpert_OnlyMale_densenet \
#         --weight_dir exp5_medgemma/OnlyMale_densenet_CheXpert/OnlyMale_densenet_CheXpert.pt


# python run_exp0_bootstrap.py \
#         --dataset CheXpert --model mamba \
#         --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyMale_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#         --experiment_name exp5_CheXpert_OnlyMale_mamba \
#         --weight_dir exp5_medgemma/OnlyMale_pretrained_mamba_CheXpert/OnlyMale_pretrained_mamba_CheXpert.pt


# python run_exp0_bootstrap.py \
#         --dataset CheXpert --model swinTF \
#         --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyMale_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#         --experiment_name exp5_OnlyMale_pretrained_swinTFB_CheXpert \
#         --weight_dir exp5_medgemma/OnlyMale_pretrained_swinTFB_CheXpert/OnlyMale_pretrained_swinTFB_CheXpert.pt


# python run_exp0_bootstrap.py \
#         --dataset CheXpert --model densenet \
#         --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyFemale_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#         --experiment_name exp5_CheXpert_OnlyFemale_densenet \
#         --weight_dir exp5_medgemma/OnlyFemale_densenet_CheXpert/OnlyFemale_densenet_CheXpert.pt


# python run_exp0_bootstrap.py \
#         --dataset CheXpert --model mamba \
#         --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyFemale_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#         --experiment_name exp5_CheXpert_OnlyFemale_mamba \
#         --weight_dir exp5_medgemma/OnlyFemale_pretrained_mamba_CheXpert/OnlyFemale_pretrained_mamba_CheXpert.pt


# python run_exp0_bootstrap.py \
#         --dataset CheXpert --model swinTF \
#         --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyFemale_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#         --experiment_name exp5_OnlyFemale_pretrained_swinTFB_CheXpert \
#         --weight_dir exp5_medgemma/OnlyFemale_pretrained_swinTFB_CheXpert/OnlyFemale_pretrained_swinTFB_CheXpert.pt

# python run_exp0_bootstrap.py \
#         --dataset CheXpert --model densenet \
#         --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyOld_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#         --experiment_name exp5_CheXpert_OnlyOld_densenet \
#         --weight_dir exp5_medgemma/OnlyOld_densenet_CheXpert/OnlyOld_densenet_CheXpert.pt

# python run_exp0_bootstrap.py \
#         --dataset CheXpert --model mamba \
#         --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyOld_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#         --experiment_name exp5_CheXpert_OnlyOld_mamba \
#         --weight_dir exp5_medgemma/OnlyOld_pretrained_mamba_CheXpert/OnlyOld_pretrained_mamba_CheXpert.pt


# python run_exp0_bootstrap.py \
#         --dataset CheXpert --model swinTF \
#         --test_path /mnt/new_usb/jupyter-altis5526/MedGemmaChecked_OnlyOld_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord \
#         --experiment_name exp5_OnlyOld_pretrained_swinTFB_CheXpert \
#         --weight_dir exp5_medgemma/OnlyOld_pretrained_swinTFB_CheXpert/OnlyOld_pretrained_swinTFB_CheXpert.pt



