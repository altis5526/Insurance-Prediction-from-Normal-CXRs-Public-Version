#!/bin/bash

python run_exp3_bootstrap_medgemma.py \
        --model swinTF \
        --test_path ./insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123




