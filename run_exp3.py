import subprocess
import os
import argparse
from itertools import combinations

# frequency

def is_experiment_completed(root_dir, weight_dir, experiment_name, seed):
    """Check if experiment is already completed (both training AND testing) by reading STATUS.txt file"""
    status_file = os.path.join(root_dir, weight_dir, f'Rand{seed}', f'Rand{seed}_{experiment_name}_STATUS.txt')
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
    parser.add_argument("--root_dir", help='', type=str, default="/home/sebasmos/orcd/pool/code/JAMA_codes/")
    args = parser.parse_args()

    seeds = [123]
    demo = ["sex", "age", "race"]

    for seed in seeds:
        for demo_input in combinations(demo, 1):
            exp_name = demo_input[0] + '_' + args.experiment_name
            w_dir = demo_input[0] + '_' + args.weight_dir

            if is_experiment_completed(args.root_dir, w_dir, exp_name, seed):
                print(f"\n{'='*50}")
                print(f"SKIPPING (already completed): {exp_name}")
                print(f"{'='*50}\n")
            else:
                subprocess.run(f"python train_insurance_fullimgsize_mamba_addDemo_new.py --mode train --train_path {args.train_path} --val_path {args.val_path} --experiment_name {exp_name} --weight_dir {w_dir} --seed {seed} --demo_labels {demo_input[0]} --root_dir {args.root_dir}", shell=True)
                subprocess.run(f"python train_insurance_fullimgsize_mamba_addDemo_new.py --mode test --train_path {args.train_path} --val_path {args.test_path} --experiment_name {exp_name} --weight_dir {w_dir} --seed {seed} --demo_labels {demo_input[0]} --root_dir {args.root_dir}", shell=True)

        for demo_input in combinations(demo, 2):
            exp_name = demo_input[0] + demo_input[1] + '_' + args.experiment_name
            w_dir = demo_input[0] + demo_input[1] + '_' + args.weight_dir

            if is_experiment_completed(args.root_dir, w_dir, exp_name, seed):
                print(f"\n{'='*50}")
                print(f"SKIPPING (already completed): {exp_name}")
                print(f"{'='*50}\n")
            else:
                # Run training
                train_result = subprocess.run(f"python train_insurance_fullimgsize_mamba_addDemo_new.py --mode train --train_path {args.train_path} --val_path {args.val_path} --experiment_name {exp_name} --weight_dir {w_dir} --seed {seed} --demo_labels {demo_input[0]} {demo_input[1]} --root_dir {args.root_dir}", shell=True)

                # Only run testing if training succeeded (exit code 0)
                if train_result.returncode == 0:
                    subprocess.run(f"python train_insurance_fullimgsize_mamba_addDemo_new.py --mode test --train_path {args.train_path} --val_path {args.test_path} --experiment_name {exp_name} --weight_dir {w_dir} --seed {seed} --demo_labels {demo_input[0]} {demo_input[1]} --root_dir {args.root_dir}", shell=True)
                else:
                    print(f"\n{'='*70}")
                    print(f"WARNING: Training failed or was interrupted for {exp_name}")
                    print(f"Skipping test phase. Fix issues and re-run.")
                    print(f"{'='*70}\n")

        for demo_input in combinations(demo, 3):
            exp_name = demo_input[0] + demo_input[1] + demo_input[2] + '_' + args.experiment_name
            w_dir = demo_input[0] + demo_input[1] + demo_input[2] + '_' + args.weight_dir

            if is_experiment_completed(args.root_dir, w_dir, exp_name, seed):
                print(f"\n{'='*50}")
                print(f"SKIPPING (already completed): {exp_name}")
                print(f"{'='*50}\n")
            else:
                # Run training
                train_result = subprocess.run(f"python train_insurance_fullimgsize_mamba_addDemo_new.py --mode train --train_path {args.train_path} --val_path {args.val_path} --experiment_name {exp_name} --weight_dir {w_dir} --seed {seed} --demo_labels {demo_input[0]} {demo_input[1]} {demo_input[2]} --root_dir {args.root_dir}", shell=True)

                # Only run testing if training succeeded (exit code 0)
                if train_result.returncode == 0:
                    subprocess.run(f"python train_insurance_fullimgsize_mamba_addDemo_new.py --mode test --train_path {args.train_path} --val_path {args.test_path} --experiment_name {exp_name} --weight_dir {w_dir} --seed {seed} --demo_labels {demo_input[0]} {demo_input[1]} {demo_input[2]} --root_dir {args.root_dir}", shell=True)
                else:
                    print(f"\n{'='*70}")
                    print(f"WARNING: Training failed or was interrupted for {exp_name}")
                    print(f"Skipping test phase. Fix issues and re-run.")
                    print(f"{'='*70}\n")
