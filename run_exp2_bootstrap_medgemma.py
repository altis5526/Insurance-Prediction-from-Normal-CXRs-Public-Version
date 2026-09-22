"""
Bootstrap validation wrapper for Experiment 2 (resolution) -- MedGemma dataset.

Covers:
  exp2_medgemma:   mamba    at 8 resolutions (2, 4, 7, 14, 28, 56, 112, 224)
  exp2-1_medgemma: densenet at 8 resolutions (2, 4, 7, 14, 28, 56, 112, 224)
  exp2-2_medgemma: swinTF   at 7 resolutions (4, 7, 14, 28, 56, 112, 224)

Loops over resolutions and calls bootstrap_evaluate.py with --resize.

Usage:
    python run_exp2_bootstrap_medgemma.py \
        --model mamba \
        --test_path /orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --n_bootstrap 1000 --seed 42
"""

import subprocess
import os
import argparse

HF_WEIGHTS = "/mnt/new_usb/jupyter-altis5526/insurance_paper_weights"

WEIGHT_CONFIGS = {
    "mamba": {
        "weight_dir": os.path.join(HF_WEIGHTS, "exp2_medgemma_s_fix"),
        "filename_template": "mamba_{res}.pt",
        "resolutions": [2, 4, 7, 14, 28, 56, 112, 224],
    },
    "densenet": {
        "weight_dir": os.path.join(HF_WEIGHTS, "exp2-1_medgemma"),
        "filename_template": "densenet_{res}.pt",
        "resolutions": [2, 4, 7, 14, 28, 56, 112, 224],
    },
    "swinTF": {
        "weight_dir": os.path.join(HF_WEIGHTS, "exp2-2_medgemma"),
        "filename_template": "PretrainedSwinTFB_{res}.pt",
        "resolutions": [2, 4, 7, 14, 28, 56, 112, 224],
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap validation for Experiment 2 MedGemma (resolution)"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=["mamba", "densenet", "swinTF"])
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Resample size; defaults to the full test set size N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="bootstrap_results/medgemma_exp2")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resolutions", type=str, default=None,
                        help="Comma-separated list of resolutions to run (e.g. 56,112). Defaults to all.")
    args = parser.parse_args()

    config = WEIGHT_CONFIGS[args.model]
    resolutions = [int(r) for r in args.resolutions.split(",")] if args.resolutions else config["resolutions"]
    failed = []

    for res in resolutions:
        weight_filename = config["filename_template"].format(res=res)
        weight_path = os.path.join(config["weight_dir"], weight_filename)

        if not os.path.exists(weight_path):
            print(f"WARNING: Checkpoint not found: {weight_path} -- skipping")
            failed.append(res)
            continue

        if args.model == "swinTF":
            experiment_name = f"MIMIC_medgemma_PretrainedB_{args.model}_res{res}_Rand123"

        else:
            experiment_name = f"MIMIC_medgemma_{args.model}_res{res}_Rand123"

        cmd = (
            f"python bootstrap_evaluate.py"
            f" --dataset MIMIC"
            f" --model {args.model}"
            f" --test_path {args.test_path}"
            f" --weight_path {weight_path}"
            f" --resize {res}"
            f" --n_bootstrap {args.n_bootstrap}"
            f"{f' --sample_size {args.sample_size}' if args.sample_size is not None else ''}"
            f" --seed {args.seed}"
            f" --output_dir {args.output_dir}"
            f" --experiment_name {experiment_name}"
            f" --num_workers {args.num_workers}"
            f" --batch_size {args.batch_size}"
        )

        print(f"\n{'=' * 70}")
        print(f"Resolution {res}: {args.model} [MedGemma]")
        print(f"Weight: {weight_path}")
        print(f"{'=' * 70}")

        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"FAILED: resolution {res} exit code {result.returncode}")
            failed.append(res)

    print(f"\n{'=' * 70}")
    if failed:
        print(f"DONE with errors. Failed resolutions: {failed}")
        exit(1)
    else:
        print(f"ALL {len(config['resolutions'])} resolutions completed successfully for {args.model} [MedGemma].")
