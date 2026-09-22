#!/bin/bash

python run_exp4_bootstrap_medgemma.py \
        --model swinTF --direction highpass \
        --test_path ./insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --n_bootstrap 1000 --seed 42 --train_seed 123

python run_exp4_bootstrap_medgemma.py \
        --model swinTF --direction lowpass \
        --test_path ./insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --n_bootstrap 1000 --seed 42 --train_seed 123
