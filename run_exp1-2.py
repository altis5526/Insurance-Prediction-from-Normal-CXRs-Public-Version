import subprocess
import os
import argparse


def is_experiment_completed(root_dir, weight_dir, experiment_name, seed, patch_idx):
    """Check if experiment is already completed (both training AND testing) by reading STATUS.txt file"""
    exp_name = f'Rand{seed}_patchidx{patch_idx}_{experiment_name}'
    status_file = os.path.join(root_dir, weight_dir, f'Rand{seed}', f'{exp_name}_STATUS.txt')
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            content = f.read()
            # Must have completed training AND have test results
            training_complete = ('STATUS: COMPLETED' in content or 'STATUS: STOPPED_EARLY' in content)
            testing_complete = 'PAPER_REPORTABLE_AUC:' in content or 'TEST_AUC:' in content
            if training_complete and testing_complete:
                return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help='', type=str)
    parser.add_argument("--train_path", help='', type=str)
    parser.add_argument("--val_path", help='', type=str)
    parser.add_argument("--test_path", help='', type=str)
    parser.add_argument("--experiment_name", help='', type=str)
    parser.add_argument("--weight_dir", help='', type=str)
    parser.add_argument("--root_dir", help='', type=str, default="/home/sebasmos/orcd/pool/code/JAMA_codes/")
    args = parser.parse_args()

    seeds = [123]

    if args.method == "remove":
        for seed in seeds:
            for idx in range(1, 10):
                # Check if experiment already completed
                if is_experiment_completed(args.root_dir, args.weight_dir, args.experiment_name, seed, idx):
                    print(f"\n{'='*50}")
                    print(f"SKIPPING (already completed): Rand{seed}_patchidx{idx}_{args.experiment_name}")
                    print(f"{'='*50}\n")
                    continue

                # Run training
                print(f"\n{'='*70}")
                print(f"Starting training: Rand{seed}_patchidx{idx}_{args.experiment_name}")
                print(f"{'='*70}\n")
                train_result = subprocess.run(
                    f'python train_insurance_fullimgsize_swintransformer_mask_area.py --mode train '
                    f'--train_path {args.train_path} --val_path {args.val_path} '
                    f'--experiment_name {args.experiment_name} --weight_dir {args.weight_dir} '
                    f'--seed {seed} --patch_idx {idx} --root_dir {args.root_dir}',
                    shell=True
                )

                # Only run testing if training succeeded (exit code 0)
                if train_result.returncode == 0:
                    print(f"\n{'='*70}")
                    print(f"Starting testing: Rand{seed}_patchidx{idx}_{args.experiment_name}")
                    print(f"{'='*70}\n")
                    subprocess.run(
                        f'python train_insurance_fullimgsize_swintransformer_mask_area.py --mode test '
                        f'--train_path {args.train_path} --val_path {args.test_path} '
                        f'--experiment_name {args.experiment_name} --weight_dir {args.weight_dir} '
                        f'--seed {seed} --patch_idx {idx} --root_dir {args.root_dir}',
                        shell=True
                    )
                else:
                    print(f"\n{'='*70}")
                    print(f"WARNING: Training failed or was interrupted for Rand{seed}_patchidx{idx}_{args.experiment_name}")
                    print(f"Skipping test phase. Fix issues and re-run.")
                    print(f"{'='*70}\n")

    if args.method == "keep":
        for seed in seeds:
            for idx in range(1, 10):
                # Check if experiment already completed
                if is_experiment_completed(args.root_dir, args.weight_dir, args.experiment_name, seed, idx):
                    print(f"\n{'='*50}")
                    print(f"SKIPPING (already completed): Rand{seed}_patchidx{idx}_{args.experiment_name}")
                    print(f"{'='*50}\n")
                    continue

                # Run training
                print(f"\n{'='*70}")
                print(f"Starting training: Rand{seed}_patchidx{idx}_{args.experiment_name}")
                print(f"{'='*70}\n")
                train_result = subprocess.run(
                    f'python train_insurance_fullimgsize_swintransformer_mask_mostarea.py --mode train '
                    f'--train_path {args.train_path} --val_path {args.val_path} '
                    f'--experiment_name {args.experiment_name} --weight_dir {args.weight_dir} '
                    f'--seed {seed} --patch_idx {idx} --root_dir {args.root_dir}',
                    shell=True
                )

                # Only run testing if training succeeded (exit code 0)
                if train_result.returncode == 0:
                    print(f"\n{'='*70}")
                    print(f"Starting testing: Rand{seed}_patchidx{idx}_{args.experiment_name}")
                    print(f"{'='*70}\n")
                    subprocess.run(
                        f'python train_insurance_fullimgsize_swintransformer_mask_mostarea.py --mode test '
                        f'--train_path {args.train_path} --val_path {args.test_path} '
                        f'--experiment_name {args.experiment_name} --weight_dir {args.weight_dir} '
                        f'--seed {seed} --patch_idx {idx} --root_dir {args.root_dir}',
                        shell=True
                    )
                else:
                    print(f"\n{'='*70}")
                    print(f"WARNING: Training failed or was interrupted for Rand{seed}_patchidx{idx}_{args.experiment_name}")
                    print(f"Skipping test phase. Fix issues and re-run.")
                    print(f"{'='*70}\n")
