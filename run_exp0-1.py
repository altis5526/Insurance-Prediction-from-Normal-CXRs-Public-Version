import subprocess
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help='', type=str)
    parser.add_argument("--model", help='', type=str)
    parser.add_argument("--train_path", help='', type=str)
    parser.add_argument("--val_path", help='', type=str)
    parser.add_argument("--test_path", help='', type=str)
    parser.add_argument("--experiment_name", help='', type=str)
    parser.add_argument("--weight_dir", help='', type=str)
    parser.add_argument("--subgroup_type", help='', type=str)
    
    args = parser.parse_args()

    seeds = ["123"]

    if args.subgroup_type == "sex":
        demo_type = ["Male", "Female"]

    elif args.subgroup_type == "age":
        demo_type = ["Young", "Middle", "Old"]

    elif args.subgroup_type == "race":
        demo_type = ["White", "Black", "Race_Others"]

    elif args.subgroup_type == "all":
        demo_type = ["all"]

    if args.dataset == "MIMIC":
        if args.model == "densenet":
            for seed in seeds:
                for demo in demo_type:
                    subprocess.run(f'python train_insurance_fullimgsize_densenet.py --mode test --train_path {args.train_path} --val_path {args.test_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed} --subgroup {demo}', shell=True)

        elif args.model == "mamba":
            for seed in seeds:
                for demo in demo_type:
                    subprocess.run(f'python train_insurance_fullimgsize_mamba.py --mode test --train_path {args.train_path} --val_path {args.test_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed} --subgroup {demo}', shell=True)


        elif args.model == "swinTF":
            for seed in seeds:
                for demo in demo_type:
                    subprocess.run(f'python train_insurance_fullimgsize_swintransformer_csv.py --mode test --train_path {args.train_path} --val_path {args.test_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed} --subgroup {demo}', shell=True)


    if args.dataset == "CheXpert":
        if args.model == "densenet":
            for seed in seeds:
                for demo in demo_type:
                    subprocess.run(f'python train_insurance_fullimgsize_densenet_CheXpert.py --mode test --train_path {args.train_path} --val_path {args.test_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed} --subgroup {demo}', shell=True)

        elif args.model == "swinTF":
            for seed in seeds:
                for demo in demo_type:
                    subprocess.run(f'python train_insurance_fullimgsize_swintransformer_CheXpert.py --mode test --train_path {args.train_path} --val_path {args.test_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed} --subgroup {demo}', shell=True)

        elif args.model == "mamba":
            for seed in seeds:
                for demo in demo_type:
                    subprocess.run(f'python train_insurance_fullimgsize_mamba_CheXpert.py --mode test --train_path {args.train_path} --val_path {args.test_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed} --subgroup {demo}', shell=True)


        