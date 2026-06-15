"""
Bootstrap validation wrapper for Experiment 0 (baseline models).

Mirrors run_exp0.py structure. Calls bootstrap_evaluate.py for each
model/dataset combination using the correct weight paths.

Usage:
    python run_exp0_bootstrap.py \
        --dataset MIMIC --model densenet \
        --test_path insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv \
        --experiment_name exp0_name \
        --weight_dir weights/exp0
"""

import subprocess
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap validation for Experiment 0"
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["MIMIC", "CheXpert"])
    parser.add_argument("--model", type=str, required=True, choices=["densenet", "mamba", "swinTF"])
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--weight_dir", type=str, required=True)
    parser.add_argument("--root_dir", type=str, default="/mnt/new_usb/jupyter-altis5526/insurance_paper_weights")
    parser.add_argument("--n_bootstrap", type=int, default=20)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_seed", type=int, default=123,
                        help="Seed used during training (for weight path construction)")
    parser.add_argument("--output_dir", type=str, default="bootstrap_results")
    args = parser.parse_args()

    seed = args.train_seed

    # Construct weight path matching training script conventions
    train_wandb_name = f"Rand{seed}_{args.experiment_name}"

    # if args.dataset == "MIMIC":
    #     weight_dir = os.path.join(
    #         args.root_dir, args.weight_dir, f"Rand{seed}"
    #     )
    #     weight_filename = f"{train_wandb_name}_model_aucbest.pt"
    # elif args.dataset == "CheXpert":
    #     weight_dir = os.path.join(args.weight_dir, f"Rand{seed}")
    #     weight_filename = f"{train_wandb_name}_model_best.pt"

    # weight_path = os.path.join(weight_dir, weight_filename)

    
    weight_dir = os.path.join(
        args.root_dir, args.weight_dir
    )
    weight_path = weight_dir

    if not os.path.exists(weight_path):
        print(f"ERROR: Checkpoint not found: {weight_path}")
        print("Please ensure training has completed before running bootstrap.")
        exit(1)

    # Construct experiment name for output files
    output_experiment_name = (
        f"{args.dataset}_{args.model}_{args.experiment_name}_Rand{seed}"
    )

    cmd = (
        f"python bootstrap_evaluate.py"
        f" --dataset {args.dataset}"
        f" --model {args.model}"
        f" --test_path {args.test_path}"
        f" --weight_path {weight_path}"
        f" --n_bootstrap {args.n_bootstrap}"
        f" --sample_size {args.sample_size}"
        f" --seed {args.seed}"
        f" --output_dir {args.output_dir}"
        f" --experiment_name {output_experiment_name}"
    )

    print(f"{'=' * 70}")
    print(f"Bootstrap Evaluation: {args.dataset} / {args.model}")
    print(f"Weight path: {weight_path}")
    print(f"{'=' * 70}")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"Bootstrap evaluation failed with exit code {result.returncode}")
        exit(result.returncode)
