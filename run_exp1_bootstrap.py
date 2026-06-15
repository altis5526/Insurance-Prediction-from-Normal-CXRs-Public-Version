"""
Bootstrap validation wrapper for Experiment 1 (patch-based models).

Covers:
  exp1-1: mamba remove-one-patch (9 patches), mamba keep-one-patch (9 patches)
  exp1-2: densenet keep/remove (9 each), swinTF keep/remove (9 each)

Loops over patch indices 1-9 and calls bootstrap_evaluate.py with the
correct weight path and --preprocessing flag.

Usage:
    python run_exp1_bootstrap.py \
        --model mamba --method remove \
        --test_path insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv \
        --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123
"""

import subprocess
import os
import argparse

HF_WEIGHTS = "/mnt/new_usb/jupyter-altis5526/insurance_paper_weights"

# All weight paths use HF repo with clean naming: patch{idx}.pt
WEIGHT_CONFIGS = {
    ("mamba", "remove"): {
        "absolute_dir": f"{HF_WEIGHTS}/exp1-1_medgemma/mamba_remove",
        "filename_template": "patch{idx}.pt",
    },
    ("mamba", "keep"): {
        "absolute_dir": f"{HF_WEIGHTS}/exp1-1_medgemma/mamba_keep",
        "filename_template": "patch{idx}.pt",
    },
    ("densenet", "keep"): {
        "absolute_dir": f"{HF_WEIGHTS}/exp1_medgemma/densenet_keep",
        "filename_template": "patch{idx}.pt",
    },
    ("densenet", "remove"): {
        "absolute_dir": f"{HF_WEIGHTS}/exp1_medgemma/densenet_remove",
        "filename_template": "patch{idx}.pt",
    },
    ("swinTF", "keep"): {
        "absolute_dir": f"{HF_WEIGHTS}/exp1-2_medgemma/swinTFB_keep",
        "filename_template": "SwinTFB_patch{idx}.pt",
    },
    ("swinTF", "remove"): {
        "absolute_dir": f"{HF_WEIGHTS}/exp1-2_medgemma/swinTFB_remove",
        "filename_template": "SwinTFB_patch{idx}.pt",
    },
}



PREPROCESSING_MAP = {
    "remove": "mask_remove",
    "keep": "mask_keep",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap validation for Experiment 1 (patch-based)"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=["mamba", "densenet", "swinTF"])
    parser.add_argument("--method", type=str, required=True,
                        choices=["remove", "keep"])
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--root_dir", type=str,
                        default="/mnt/new_usb/jupyter-altis5526/new_insurancetype_weight/MedGemma_checked/")
    parser.add_argument("--n_bootstrap", type=int, default=20)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_seed", type=int, default=123)
    parser.add_argument("--output_dir", type=str, default="bootstrap_results/exp1")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    key = (args.model, args.method)
    if key not in WEIGHT_CONFIGS:
        print(f"ERROR: No weight config for model={args.model}, method={args.method}")
        print(f"Valid combos: {list(WEIGHT_CONFIGS.keys())}")
        exit(1)

    config = WEIGHT_CONFIGS[key]
    preprocessing = PREPROCESSING_MAP[args.method]
    seed = args.train_seed

    failed = []

    for patch_idx in range(1, 10):
        weight_dir = config["absolute_dir"]
        weight_filename = config["filename_template"].format(idx=patch_idx)
        weight_path = os.path.join(weight_dir, weight_filename)

        if not os.path.exists(weight_path):
            print(f"WARNING: Checkpoint not found: {weight_path} — skipping")
            failed.append(patch_idx)
            continue

        experiment_name = (
            f"MIMIC_{args.model}_{args.method}_patch{patch_idx}_Rand{seed}"
        )

        cmd = (
            f"python bootstrap_evaluate.py"
            f" --dataset MIMIC"
            f" --model {args.model}"
            f" --test_path {args.test_path}"
            f" --weight_path {weight_path}"
            f" --preprocessing {preprocessing}"
            f" --patch_idx {patch_idx}"
            f" --n_bootstrap {args.n_bootstrap}"
            f" --sample_size {args.sample_size}"
            f" --seed {args.seed}"
            f" --output_dir {args.output_dir}"
            f" --experiment_name {experiment_name}"
            f" --num_workers {args.num_workers}"
            f" --batch_size {args.batch_size}"
        )

        print(f"\n{'=' * 70}")
        print(f"Patch {patch_idx}/9: {args.model} {args.method}")
        print(f"Weight: {weight_path}")
        print(f"{'=' * 70}")

        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"FAILED: patch {patch_idx} exit code {result.returncode}")
            failed.append(patch_idx)

    print(f"\n{'=' * 70}")
    if failed:
        print(f"DONE with errors. Failed patches: {failed}")
        exit(1)
    else:
        print("ALL 9 patches completed successfully.")
