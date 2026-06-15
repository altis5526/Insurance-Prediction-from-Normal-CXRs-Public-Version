import subprocess
import os
import argparse


def is_experiment_completed(root_dir, weight_dir, experiment_name, seed):
    """Check if experiment already completed by looking at the STATUS file."""
    status_path = os.path.join(
        root_dir, weight_dir, f'Rand{seed}',
        f'Rand{seed}_{experiment_name}_STATUS.txt'
    )
    if os.path.exists(status_path):
        with open(status_path, 'r') as f:
            content = f.read()
        return 'STATUS: COMPLETED' in content or 'STATUS: STOPPED_EARLY' in content
    # Fallback: if no STATUS file, check weight file (backward compat)
    weight_path = os.path.join(
        root_dir, weight_dir, f'Rand{seed}',
        f'Rand{seed}_{experiment_name}_model_aucbest.pt'
    )
    return os.path.exists(weight_path)


TRAIN_SCRIPTS = {
    "mamba": "train_insurance_fullimgsize_mamba.py",
    "mamba_t": "train_insurance_fullimgsize_mamba.py",
    "mamba_s": "train_insurance_fullimgsize_mamba.py",
    "densenet": "train_insurance_fullimgsize_densenet.py",
    "swinTF": "train_insurance_fullimgsize_swintransformer_csv.py",
}

MAMBA_VARIANTS = {
    "mamba": "mamba",
    "mamba_t": "mamba_t",
    "mamba_s": "mamba_s",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", help='', type=str)
    parser.add_argument("--val_path", help='', type=str)
    parser.add_argument("--test_path", help='', type=str)
    parser.add_argument("--experiment_name", help='', type=str)
    parser.add_argument("--weight_dir", help='', type=str)
    parser.add_argument("--model", help='Model: mamba, mamba_t, mamba_s, densenet, or swinTF', type=str, default="mamba",
                        choices=["mamba", "mamba_t", "mamba_s", "densenet", "swinTF"])
    parser.add_argument("--root_dir", help='', type=str, default="/home/sebasmos/orcd/pool/code/JAMA_codes/")
    parser.add_argument("--resolutions", help='Comma-separated list of resolutions to train (e.g. 4,7,14)', type=str, default=None)
    parser.add_argument("--pretrained_path", help='Path to pretrained .pth for MedMamba-S init (optional)', type=str, default=None)
    args = parser.parse_args()

    train_script = TRAIN_SCRIPTS[args.model]
    seeds = [123]
    sizes = [224, 112, 56, 28, 14, 7, 4, 2]

    print(f"Model: {args.model} -> {train_script}")
    print(f"Resolutions: {sizes}")

    for seed in seeds:
        for size in sizes:
            size_experiment_name = f'{size}_{args.experiment_name}'
            size_weight_dir = f'{size}_{args.weight_dir}'

            if is_experiment_completed(args.root_dir, size_weight_dir, size_experiment_name, seed):
                print(f"\n{'='*50}")
                print(f"SKIPPING (already completed): Rand{seed}_{size_experiment_name}")
                print(f"{'='*50}\n")
                continue

            print(f"\n{'='*70}")
            print(f"Starting training: Rand{seed}_{size_experiment_name}")
            print(f"{'='*70}\n")
            variant_flags = ""
            if args.model in MAMBA_VARIANTS:
                variant_flags += f' --model_variant {MAMBA_VARIANTS[args.model]}'
            if args.pretrained_path:
                variant_flags += f' --pretrained_path {args.pretrained_path}'

            train_result = subprocess.run(
                f'python {train_script} --mode train '
                f'--train_path {args.train_path} --val_path {args.val_path} '
                f'--experiment_name {size_experiment_name} --weight_dir {size_weight_dir} '
                f'--seed {seed} --resize {size} --root_dir {args.root_dir}{variant_flags}',
                shell=True
            )

            if train_result.returncode == 0:
                print(f"\n{'='*70}")
                print(f"Starting testing: Rand{seed}_{size_experiment_name}")
                print(f"{'='*70}\n")
                subprocess.run(
                    f'python {train_script} --mode test '
                    f'--train_path {args.train_path} --val_path {args.test_path} '
                    f'--experiment_name {size_experiment_name} --weight_dir {size_weight_dir} '
                    f'--seed {seed} --resize {size} --root_dir {args.root_dir}{variant_flags}',
                    shell=True
                )
            else:
                print(f"\n{'='*70}")
                print(f"WARNING: Training failed or was interrupted for Rand{seed}_{size_experiment_name}")
                print(f"Skipping test phase. Fix issues and re-run.")
                print(f"{'='*70}\n")
