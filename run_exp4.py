# High-pass filter (trains 8 models: 1, 5, 10, 25, 50, 100, 200, 400 Hz)
# python run_exp4.py --train_path insurance_dataset_8_1_1_PMMthree_train_no_support_devices.csv --val_path insurance_dataset_8_1_1_PMMthree_val_no_support_devices.csv --test_path insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv --experiment_name mamba_highpass --weight_dir weights/mamba/highpass --pass_type high

# Low-pass filter (trains 8 models: 1, 5, 10, 25, 50, 100, 200, 400 Hz)
# python run_exp4.py --train_path insurance_dataset_8_1_1_PMMthree_train_no_support_devices.csv --val_path insurance_dataset_8_1_1_PMMthree_val_no_support_devices.csv --test_path insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv --experiment_name mamba_lowpass --weight_dir weights/mamba/lowpass --pass_type low

import subprocess
import os
import argparse


def is_experiment_completed(root_dir, weight_dir, experiment_name, seed):
    """Check if experiment is already completed (both training AND testing) by reading STATUS.txt file"""
    exp_name = f'Rand{seed}_{experiment_name}'
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
    parser.add_argument("--train_path", help='', type=str)
    parser.add_argument("--val_path", help='', type=str)
    parser.add_argument("--test_path", help='', type=str)
    parser.add_argument("--experiment_name", help='', type=str)
    parser.add_argument("--weight_dir", help='', type=str)
    parser.add_argument("--pass_type", help='high or low', type=str)
    parser.add_argument("--root_dir", help='', type=str, default="/home/sebasmos/orcd/pool/code/JAMA_codes/")
    args = parser.parse_args()

    seeds = [123]
    frequencies = [1, 5, 10, 25, 50, 100, 200, 400]

    for seed in seeds:
        if args.pass_type == "high":
            for frequency in frequencies:
                freq_experiment_name = f'{frequency}HighPass_{args.experiment_name}'
                freq_weight_dir = f'{frequency}HighPass_{args.weight_dir}'

                # Check if experiment already completed
                if is_experiment_completed(args.root_dir, freq_weight_dir, freq_experiment_name, seed):
                    print(f"\n{'='*50}")
                    print(f"SKIPPING (already completed): Rand{seed}_{freq_experiment_name}")
                    print(f"{'='*50}\n")
                    continue

                # Run training
                print(f"\n{'='*70}")
                print(f"Starting training: Rand{seed}_{freq_experiment_name}")
                print(f"{'='*70}\n")
                train_result = subprocess.run(
                    f'python train_insurance_fullimgsize_mamba_high_pass.py --mode train '
                    f'--train_path {args.train_path} --val_path {args.val_path} '
                    f'--experiment_name {freq_experiment_name} --weight_dir {freq_weight_dir} '
                    f'--seed {seed} --high_pass_diameter {frequency} --root_dir {args.root_dir}',
                    shell=True
                )

                # Only run testing if training succeeded (exit code 0)
                if train_result.returncode == 0:
                    print(f"\n{'='*70}")
                    print(f"Starting testing: Rand{seed}_{freq_experiment_name}")
                    print(f"{'='*70}\n")
                    subprocess.run(
                        f'python train_insurance_fullimgsize_mamba_high_pass.py --mode test '
                        f'--train_path {args.train_path} --val_path {args.test_path} '
                        f'--experiment_name {freq_experiment_name} --weight_dir {freq_weight_dir} '
                        f'--seed {seed} --high_pass_diameter {frequency} --root_dir {args.root_dir}',
                        shell=True
                    )
                else:
                    print(f"\n{'='*70}")
                    print(f"WARNING: Training failed or was interrupted for Rand{seed}_{freq_experiment_name}")
                    print(f"Skipping test phase. Fix issues and re-run.")
                    print(f"{'='*70}\n")

        if args.pass_type == "low":
            for frequency in frequencies:
                freq_experiment_name = f'{frequency}LowPass_{args.experiment_name}'
                freq_weight_dir = f'{frequency}LowPass_{args.weight_dir}'

                # Check if experiment already completed
                if is_experiment_completed(args.root_dir, freq_weight_dir, freq_experiment_name, seed):
                    print(f"\n{'='*50}")
                    print(f"SKIPPING (already completed): Rand{seed}_{freq_experiment_name}")
                    print(f"{'='*50}\n")
                    continue

                # Run training
                print(f"\n{'='*70}")
                print(f"Starting training: Rand{seed}_{freq_experiment_name}")
                print(f"{'='*70}\n")
                train_result = subprocess.run(
                    f'python train_insurance_fullimgsize_mamba_low_pass.py --mode train '
                    f'--train_path {args.train_path} --val_path {args.val_path} '
                    f'--experiment_name {freq_experiment_name} --weight_dir {freq_weight_dir} '
                    f'--seed {seed} --low_pass_diameter {frequency} --root_dir {args.root_dir}',
                    shell=True
                )

                # Only run testing if training succeeded (exit code 0)
                if train_result.returncode == 0:
                    print(f"\n{'='*70}")
                    print(f"Starting testing: Rand{seed}_{freq_experiment_name}")
                    print(f"{'='*70}\n")
                    subprocess.run(
                        f'python train_insurance_fullimgsize_mamba_low_pass.py --mode test '
                        f'--train_path {args.train_path} --val_path {args.test_path} '
                        f'--experiment_name {freq_experiment_name} --weight_dir {freq_weight_dir} '
                        f'--seed {seed} --low_pass_diameter {frequency} --root_dir {args.root_dir}',
                        shell=True
                    )
                else:
                    print(f"\n{'='*70}")
                    print(f"WARNING: Training failed or was interrupted for Rand{seed}_{freq_experiment_name}")
                    print(f"Skipping test phase. Fix issues and re-run.")
                    print(f"{'='*70}\n")
