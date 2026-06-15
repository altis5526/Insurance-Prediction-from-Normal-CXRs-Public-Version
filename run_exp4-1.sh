#!/bin/bash

# python run_exp4-1.py --train_path insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv --val_path insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv --experiment_name PretrainedB_swinTF --weight_dir PretrainedB_swinTF --pass_type high

python run_exp4-1.py --train_path insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv --val_path insurance_dataset_8_1_1_PMMthree_val_medgemmaChecked.csv --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv --experiment_name PretrainedB_LowPass_swinTF --weight_dir PretrainedB_swinTF --pass_type low




