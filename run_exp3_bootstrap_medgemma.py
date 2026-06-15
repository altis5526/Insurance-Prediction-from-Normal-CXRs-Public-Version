"""
Bootstrap validation wrapper for Experiment 3 (demographics models) — MedGemma dataset.

Covers:
  exp3_medgemma:   mamba   + addDemothen2
  exp3-1_medgemma: densenet + addDemothen2
  exp3-2_medgemma: swinTF  + addDemothen2

Loops over all 7 demographic combinations and calls bootstrap_evaluate.py
with --model_variant addDemo and the appropriate --demo_labels.

Usage:
    python run_exp3_bootstrap_medgemma.py \
        --model mamba \
        --test_path /orcd/pool/006/lceli_shared/data/mimic-cxr-gemma/insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv \
        --n_bootstrap 20 --sample_size 1000 --seed 42 --train_seed 123
"""

import subprocess
import os
import argparse

# All 7 demographic combos, with their corresponding demo_labels
DEMO_COMBOS = {
    "sex":          ["sex"],
    "age":          ["age"],
    "race":         ["race"],
    "sexage":       ["sex", "age"],
    "sexrace":      ["sex", "race"],
    "agerace":      ["age", "race"],
    "sexagerace":   ["sex", "age", "race"],
}

HF_WEIGHTS = "/mnt/new_usb/jupyter-altis5526/insurance_paper_weights"

# MedGemma experiment subdirectory per model
EXP_SUBDIR = {
    "mamba": "exp3_medgemma",
    "densenet": "exp3-1_medgemma",
    "swinTF": "exp3-2_medgemma_pretrainedB",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap validation for Experiment 3 MedGemma (demographics)"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=["mamba", "densenet", "swinTF"])
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--root_dir", type=str,
                        default="/mnt/new_usb/jupyter-altis5526/insurance_paper_weights")
    parser.add_argument("--n_bootstrap", type=int, default=20)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_seed", type=int, default=123)
    parser.add_argument("--output_dir", type=str, default="bootstrap_results/medgemma_exp3")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    seed = args.train_seed

    failed = []

    for combo_name, demo_labels in DEMO_COMBOS.items():
        # MedGemma weight path: insurance_paper_weights/exp3[-1|-2]_medgemma/{combo}.pt
        if args.model == "swinTF":
            weight_path = os.path.join(
                HF_WEIGHTS, EXP_SUBDIR[args.model], f"{combo_name}_PretrainedB_swinTF.pt"
            )
            
        else:
            weight_path = os.path.join(
                HF_WEIGHTS, EXP_SUBDIR[args.model], f"{combo_name}.pt"
            )

        if not os.path.exists(weight_path):
            print(f"WARNING: Checkpoint not found: {weight_path} — skipping")
            failed.append(combo_name)
            continue

        if args.model == "swinTF":
            experiment_name = (
                f"MIMIC_medgemma_PretrainedB{args.model}_{combo_name}_Rand{seed}"
            )

        else: 
            experiment_name = (
                f"MIMIC_medgemma_{args.model}_{combo_name}_Rand{seed}"
            )

        demo_labels_str = " ".join(demo_labels)

        cmd = (
            f"python bootstrap_evaluate.py"
            f" --dataset MIMIC"
            f" --model {args.model}"
            f" --model_variant addDemo"
            f" --demo_labels {demo_labels_str}"
            f" --test_path {args.test_path}"
            f" --weight_path {weight_path}"
            f" --n_bootstrap {args.n_bootstrap}"
            f" --sample_size {args.sample_size}"
            f" --seed {args.seed}"
            f" --output_dir {args.output_dir}"
            f" --experiment_name {experiment_name}"
            f" --num_workers {args.num_workers}"
            f" --batch_size {args.batch_size}"
        )

        print(f"\n{'=' * 70}")
        print(f"Demo combo: {combo_name} -> labels: {demo_labels}")
        print(f"Model: {args.model} (addDemo) [MedGemma]")
        print(f"Weight: {weight_path}")
        print(f"{'=' * 70}")

        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"FAILED: {combo_name} exit code {result.returncode}")
            failed.append(combo_name)

    print(f"\n{'=' * 70}")
    if failed:
        print(f"DONE with errors. Failed combos: {failed}")
        exit(1)
    else:
        print(f"ALL 7 demo combos completed successfully for {args.model} [MedGemma].")
