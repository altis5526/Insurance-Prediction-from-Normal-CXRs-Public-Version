"""
Bootstrap validation wrapper for Experiment 4 (frequency filtering) — MedGemma dataset.

Covers:
  exp4_medgemma:   mamba   (high_pass + low_pass)
  exp4-1_medgemma: swinTF  (high_pass + low_pass)
  exp4-2_medgemma: densenet (high_pass + low_pass)

Loops over 8 frequencies and calls bootstrap_evaluate.py with the
correct --preprocessing and --filter_diameter.

Usage:
    python run_exp4_bootstrap_medgemma.py \
        --model densenet --direction highpass \
        --test_path /orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123
"""

import subprocess
import os
import argparse

FREQUENCIES = [1, 5, 10, 25, 50, 100, 200, 400]

HF_WEIGHTS = "/mnt/new_usb/jupyter-altis5526/insurance_paper_weights"

# MedGemma experiment subdirectory per model
EXP_SUBDIR = {
    "mamba": "exp4_medgemma",
    "swinTF": "exp4-1_medgemma_pretrainedB",
    "densenet": "exp4-2_medgemma",
}

# Direction label in filenames/dirs
DIRECTION_LABEL = {
    "highpass": "HighPass",
    "lowpass": "LowPass",
}

# Preprocessing mode for bootstrap_evaluate.py
DIRECTION_PREPROCESSING = {
    "highpass": "high_pass",
    "lowpass": "low_pass",
}


def build_weight_path(model, direction, freq, seed, root_dir):
    """Construct weight path using MedGemma HF naming convention.

    HF path: insurance_paper_weights/exp4[-1|-2]_medgemma/{direction}/{freq}Hz.pt
    """
    if model == "swinTF":
        return os.path.join(
        HF_WEIGHTS, EXP_SUBDIR[model], direction, f"{freq}{DIRECTION_LABEL[direction]}_PretrainedB_swinTF.pt"
    )
        
    else:
        return os.path.join(
        HF_WEIGHTS, EXP_SUBDIR[model], direction, f"{freq}Hz.pt"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap validation for Experiment 4 MedGemma (frequency filtering)"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=["mamba", "densenet", "swinTF"])
    parser.add_argument("--direction", type=str, required=True,
                        choices=["highpass", "lowpass"])
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--root_dir", type=str,
                        default="/mnt/new_usb/jupyter-altis5526/insurance_paper_weights/")
    parser.add_argument("--n_bootstrap", type=int, default=20)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_seed", type=int, default=123)
    parser.add_argument("--output_dir", type=str, default="bootstrap_results/medgemma_exp4")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--frequencies", type=str, default=None,
                        help="Comma-separated list of frequencies to run (e.g. 2,3,4). Defaults to all.")
    args = parser.parse_args()

    seed = args.train_seed
    preprocessing = DIRECTION_PREPROCESSING[args.direction]
    dir_label = DIRECTION_LABEL[args.direction]
    frequencies = [int(f) for f in args.frequencies.split(",")] if args.frequencies else FREQUENCIES

    failed = []

    for freq in frequencies:
        weight_path = build_weight_path(
            args.model, args.direction, freq, seed, args.root_dir
        )

        if not os.path.exists(weight_path):
            print(f"WARNING: Checkpoint not found: {weight_path} — skipping")
            failed.append(freq)
            continue
            
        if args.model == "swinTF":
            experiment_name = (
            f"MIMIC_medgemma_PretrainedB_{args.model}_{freq}{dir_label}_Rand{seed}"
        )
        
        else:
            experiment_name = (
            f"MIMIC_medgemma_{args.model}_{freq}{dir_label}_Rand{seed}"
        )

        cmd = (
            f"python bootstrap_evaluate.py"
            f" --dataset MIMIC"
            f" --model {args.model}"
            f" --test_path {args.test_path}"
            f" --weight_path {weight_path}"
            f" --preprocessing {preprocessing}"
            f" --filter_diameter {freq}"
            f" --n_bootstrap {args.n_bootstrap}"
            f" --sample_size {args.sample_size}"
            f" --seed {args.seed}"
            f" --output_dir {args.output_dir}"
            f" --experiment_name {experiment_name}"
            f" --num_workers {args.num_workers}"
            f" --batch_size {args.batch_size}"
        )

        print(f"\n{'=' * 70}")
        print(f"Frequency: {freq} Hz {dir_label}")
        print(f"Model: {args.model} [MedGemma]")
        print(f"Weight: {weight_path}")
        print(f"{'=' * 70}")

        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"FAILED: {freq} Hz exit code {result.returncode}")
            failed.append(freq)

    print(f"\n{'=' * 70}")
    if failed:
        print(f"DONE with errors. Failed frequencies: {failed}")
        exit(1)
    else:
        print(f"ALL 8 frequencies completed successfully for {args.model} {args.direction} [MedGemma].")
