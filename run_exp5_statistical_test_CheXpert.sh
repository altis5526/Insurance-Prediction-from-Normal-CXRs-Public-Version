#!/bin/bash

python statistical_testing.py \
        --exp1 bootstrap_results/exp0-2/CheXpert_densenet_exp0-2_Rand123_bootstrap_iterations.csv \
        --exp2 bootstrap_results/exp5/CheXpert/CheXpert_densenet_exp5_CheXpert_OnlyFemale_densenet_Rand123_bootstrap_iterations.csv \
        --exp1_name “densenet” --exp2_name “denset_female” \
        --output ./bootstrap_results/chexpert_comparison_densenet_female.csv


python statistical_testing.py \
        --exp1 bootstrap_results/exp0-2/CheXpert_densenet_exp0-2_Rand123_bootstrap_iterations.csv \
        --exp2 bootstrap_results/exp5/CheXpert/CheXpert_densenet_exp5_CheXpert_OnlyMale_densenet_Rand123_bootstrap_iterations.csv \
        --exp1_name “densenet” --exp2_name “denset_male” \
        --output ./bootstrap_results/chexpert_comparison_densenet_male.csv


python statistical_testing.py \
        --exp1 bootstrap_results/exp0-2/CheXpert_densenet_exp0-2_Rand123_bootstrap_iterations.csv \
        --exp2 bootstrap_results/exp5/CheXpert/CheXpert_densenet_exp5_CheXpert_OnlyOld_densenet_Rand123_bootstrap_iterations.csv \
        --exp1_name “densenet” --exp2_name “denset_old” \
        --output ./bootstrap_results/chexpert_comparison_densenet_old.csv


python statistical_testing.py \
        --exp1 bootstrap_results/exp0-2/CheXpert_densenet_exp0-2_Rand123_bootstrap_iterations.csv \
        --exp2 bootstrap_results/exp5/CheXpert/CheXpert_densenet_exp5_CheXpert_OnlyWhite_densenet_Rand123_bootstrap_iterations.csv \
        --exp1_name “densenet” --exp2_name “denset_white” \
        --output ./bootstrap_results/chexpert_comparison_densenet_white.csv

python statistical_testing.py \
        --exp1 bootstrap_results/exp0-2/CheXpert_mamba_exp0-2_Rand123_bootstrap_iterations.csv \
        --exp2 bootstrap_results/exp5/CheXpert/CheXpert_mamba_exp5_CheXpert_OnlyFemale_pretrained_mamba_Rand123_bootstrap_iterations.csv \
        --exp1_name “mamba” --exp2_name “mamba_female” \
        --output ./bootstrap_results/chexpert_comparison_mamba_female.csv

python statistical_testing.py \
        --exp1 bootstrap_results/exp0-2/CheXpert_mamba_exp0-2_Rand123_bootstrap_iterations.csv \
        --exp2 bootstrap_results/exp5/CheXpert/CheXpert_mamba_exp5_CheXpert_OnlyMale_pretrained_mamba_Rand123_bootstrap_iterations.csv \
        --exp1_name “mamba” --exp2_name “mamba_male” \
        --output ./bootstrap_results/chexpert_comparison_mamba_male.csv


python statistical_testing.py \
        --exp1 bootstrap_results/exp0-2/CheXpert_mamba_exp0-2_Rand123_bootstrap_iterations.csv \
        --exp2 bootstrap_results/exp5/CheXpert/CheXpert_mamba_exp5_CheXpert_OnlyOld_pretrained_mamba_Rand123_bootstrap_iterations.csv \
        --exp1_name “mamba” --exp2_name “mamba_old” \
        --output ./bootstrap_results/chexpert_comparison_mamba_old.csv

python statistical_testing.py \
        --exp1 bootstrap_results/exp0-2/CheXpert_mamba_exp0-2_Rand123_bootstrap_iterations.csv \
        --exp2 bootstrap_results/exp5/CheXpert/CheXpert_mamba_exp5_CheXpert_OnlyWhite_pretrained_mamba_Rand123_bootstrap_iterations.csv \
        --exp1_name “mamba” --exp2_name “mamba_white” \
        --output ./bootstrap_results/chexpert_comparison_mamba_white.csv


python statistical_testing.py \
        --exp1 bootstrap_results/exp0-2/CheXpert_swinTF_exp0-2_Rand123_bootstrap_iterations.csv \
        --exp2 bootstrap_results/exp5/CheXpert/CheXpert_swinTF_exp5_OnlyFemale_pretrained_swinTFB_CheXpert_Rand123_bootstrap_iterations.csv \
        --exp1_name “swinTF” --exp2_name “swinTF_female” \
        --output ./bootstrap_results/chexpert_comparison_swinTF_female.csv


python statistical_testing.py \
        --exp1 bootstrap_results/exp0-2/CheXpert_swinTF_exp0-2_Rand123_bootstrap_iterations.csv \
        --exp2 bootstrap_results/exp5/CheXpert/CheXpert_swinTF_exp5_OnlyMale_pretrained_swinTFB_CheXpert_Rand123_bootstrap_iterations.csv \
        --exp1_name “swinTF” --exp2_name “swinTF_male” \
        --output ./bootstrap_results/chexpert_comparison_swinTF_male.csv


python statistical_testing.py \
        --exp1 bootstrap_results/exp0-2/CheXpert_swinTF_exp0-2_Rand123_bootstrap_iterations.csv \
        --exp2 bootstrap_results/exp5/CheXpert/CheXpert_swinTF_exp5_OnlyOld_pretrained_swinTFB_CheXpert_Rand123_bootstrap_iterations.csv \
        --exp1_name “swinTF” --exp2_name “swinTF_old” \
        --output ./bootstrap_results/chexpert_comparison_swinTF_old.csv
        

python statistical_testing.py \
        --exp1 bootstrap_results/exp0-2/CheXpert_swinTF_exp0-2_Rand123_bootstrap_iterations.csv \
        --exp2 bootstrap_results/exp5/CheXpert/CheXpert_swinTF_exp5_OnlyWhite_pretrained_swinTFB_CheXpert_Rand123_bootstrap_iterations.csv \
        --exp1_name “swinTF” --exp2_name “swinTF_white” \
        --output ./bootstrap_results/chexpert_comparison_swinTF_white.csv