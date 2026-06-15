#!/bin/bash

python run_exp2_bootstrap_medgemma.py \
        --model swinTF \
        --test_path ./insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --n_bootstrap 20 --sample_size 1000 --seed 42



