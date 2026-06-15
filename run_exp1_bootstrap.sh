#!/bin/bash

python run_exp1_bootstrap.py \
        --model mamba --method remove \
        --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123
python run_exp1_bootstrap.py \
        --model mamba --method keep \
        --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123
python run_exp1_bootstrap.py \
        --model densenet --method remove \
        --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123
python run_exp1_bootstrap.py \
        --model densenet --method keep \
        --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123
python run_exp1_bootstrap.py \
        --model swinTF --method remove \
        --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123
python run_exp1_bootstrap.py \
        --model swinTF --method keep \
        --test_path insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123



