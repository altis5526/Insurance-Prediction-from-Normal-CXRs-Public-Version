#!/bin/bash

# python run_exp0.py --dataset MIMIC \
#                 --model densenet \
#                 --train_path Only_white_insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv \
#                 --val_path Only_white_insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv \
#                 --test_path Only_white_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
#                 --experiment_name MedgemmaChecked_OnlyWhite_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_densenet_DoubleLinear_Lion5e-6_20260329 \
#                 --weight_dir MedgemmaChecked_OnlyWhite_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_densenet_DoubleLinear_Lion5e-6_20260329


# python run_exp0.py --dataset MIMIC \
#                 --model densenet \
#                 --train_path Only_male_insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv \
#                 --val_path Only_male_insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv \
#                 --test_path Only_male_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
#                 --experiment_name MedgemmaChecked_OnlyMale_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_densenet_DoubleLinear_Lion5e-6_20260329 \
#                 --weight_dir MedgemmaChecked_OnlyMale_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_densenet_DoubleLinear_Lion5e-6_20260329


# python run_exp0.py --dataset MIMIC \
#                 --model densenet \
#                 --train_path Only_female_insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv \
#                 --val_path Only_female_insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv \
#                 --test_path Only_female_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
#                 --experiment_name MedgemmaChecked_OnlyFemale_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_densenet_DoubleLinear_Lion5e-6_20260329 \
#                 --weight_dir MedgemmaChecked_OnlyFemale_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_densenet_DoubleLinear_Lion5e-6_20260329


# python run_exp0.py --dataset MIMIC \
#                 --model densenet \
#                 --train_path Only_old_insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv \
#                 --val_path Only_old_insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv \
#                 --test_path Only_old_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
#                 --experiment_name MedgemmaChecked_OnlyOld_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_densenet_DoubleLinear_Lion5e-6_20260329 \
#                 --weight_dir MedgemmaChecked_OnlyOld_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_densenet_DoubleLinear_Lion5e-6_20260329


python run_exp0.py --dataset MIMIC \
                --model mamba \
                --train_path Only_white_insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv \
                --val_path Only_white_insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv \
                --test_path Only_white_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
                --experiment_name PretrainedMamba_MedgemmaChecked_OnlyWhite_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_mamba_DoubleLinear_Lion1e-5_20260425 \
                --weight_dir PretrainedMamba_MedgemmaChecked_OnlyWhite_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_mamba_DoubleLinear_Lion1e-5_20260425


python run_exp0.py --dataset MIMIC \
                --model mamba \
                --train_path Only_male_insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv \
                --val_path Only_male_insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv \
                --test_path Only_male_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
                --experiment_name PretrainedMamba_MedgemmaChecked_OnlyMale_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_mamba_DoubleLinear_Lion1e-5_20260425 \
                --weight_dir PretrainedMamba_MedgemmaChecked_OnlyMale_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_mamba_DoubleLinear_Lion1e-5_20260425


python run_exp0.py --dataset MIMIC \
                --model mamba \
                --train_path Only_female_insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv \
                --val_path Only_female_insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv \
                --test_path Only_female_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
                --experiment_name PretrainedMamba_MedgemmaChecked_OnlyFemale_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_mamba_DoubleLinear_Lion1e-5_20260425 \
                --weight_dir PretrainedMamba_MedgemmaChecked_OnlyFemale_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_mamba_DoubleLinear_Lion1e-5_20260425


python run_exp0.py --dataset MIMIC \
                --model mamba \
                --train_path Only_old_insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv \
                --val_path Only_old_insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv \
                --test_path Only_old_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
                --experiment_name PretrainedMamba_MedgemmaChecked_OnlyOld_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_mamba_DoubleLinear_Lion1e-5_20260425 \
                --weight_dir PretrainedMamba_MedgemmaChecked_OnlyOld_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_mamba_DoubleLinear_Lion1e-5_20260425


# python run_exp0.py --dataset MIMIC \
#                 --model swinTF \
#                 --train_path Only_white_insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv \
#                 --val_path Only_white_insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv \
#                 --test_path Only_white_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
#                 --experiment_name MedgemmaChecked_OnlyWhite_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_pretrained_swinTFB_DoubleLinear_Lion1e-5_20260413 \
#                 --weight_dir MedgemmaChecked_OnlyWhite_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_pretrained_swinTFB_DoubleLinear_Lion1e-5_20260413


# python run_exp0.py --dataset MIMIC \
#                 --model swinTF \
#                 --train_path Only_male_insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv \
#                 --val_path Only_male_insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv \
#                 --test_path Only_male_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
#                 --experiment_name MedgemmaChecked_OnlyMale_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_pretrained_swinTFB_DoubleLinear_Lion1e-5_20260413 \
#                 --weight_dir MedgemmaChecked_OnlyMale_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_pretrained_swinTFB_DoubleLinear_Lion1e-5_20260413


# python run_exp0.py --dataset MIMIC \
#                 --model swinTF \
#                 --train_path Only_female_insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv \
#                 --val_path Only_female_insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv \
#                 --test_path Only_female_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
#                 --experiment_name MedgemmaChecked_OnlyFemale_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_pretrained_swinTFB_DoubleLinear_Lion1e-5_20260413 \
#                 --weight_dir MedgemmaChecked_OnlyFemale_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_pretrained_swinTFB_DoubleLinear_Lion1e-5_20260413


# python run_exp0.py --dataset MIMIC \
#                 --model swinTF \
#                 --train_path Only_old_insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv \
#                 --val_path Only_old_insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv \
#                 --test_path Only_old_insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
#                 --experiment_name MedgemmaChecked_OnlyOld_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_pretrained_swinTFB_DoubleLinear_Lion1e-5_20260413 \
#                 --weight_dir MedgemmaChecked_OnlyOld_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_pretrained_swinTFB_DoubleLinear_Lion1e-5_20260413