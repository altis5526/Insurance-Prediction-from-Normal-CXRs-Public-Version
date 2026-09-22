"""
Bootstrap evaluation for insurance classification experiments.

Runs model inference on the full test set once, reports metrics on the entire
test set, then bootstrap-resamples predictions to compute confidence intervals.

Bootstrapping defaults to resampling PATIENTS with replacement (cluster
bootstrap), not images: images from the same patient are correlated, so
image-level resampling understates the true uncertainty. All images of a drawn
patient enter the resample together. --sample_size defaults to the full test
set size N, so the CI describes the study you actually ran; passing a smaller
value gives an m-out-of-n bootstrap whose CI is wider by roughly sqrt(N/m).

Outputs written to --output_dir (prefixed with --experiment_name):
- <name>_predictions.csv          per-sample logits, probabilities, labels, demographics
- <name>_full_test_metrics.csv    metrics on the entire test set (overall + subgroups)
- <name>_bootstrap_iterations.csv per-bootstrap-iteration metrics
- <name>_bootstrap_summary.csv    bootstrap mean/std/95% CI

Supports:
- Base models (image-only): densenet, mamba, swinTF
- Demographics models (image + demo): densenet, mamba, swinTF with addDemothen2
- Preprocessing: none, mask_remove, mask_keep, high_pass, low_pass

Usage:
    # Base model (exp0, exp1-1, exp1-2, exp4)
    python bootstrap_evaluate.py \
        --dataset MIMIC --model densenet \
        --test_path test.csv --weight_path checkpoint.pt

    # Demographics model (exp3)
    python bootstrap_evaluate.py \
        --dataset MIMIC --model densenet --model_variant addDemo \
        --demo_labels sex age \
        --test_path test.csv --weight_path checkpoint.pt

    # Frequency-filtered (exp4)
    python bootstrap_evaluate.py \
        --dataset MIMIC --model densenet \
        --preprocessing high_pass --filter_diameter 5 \
        --test_path test.csv --weight_path checkpoint.pt

    # Patch-masked (exp1-1, exp1-2)
    python bootstrap_evaluate.py \
        --dataset MIMIC --model mamba \
        --preprocessing mask_remove --patch_idx 1 \
        --test_path test.csv --weight_path checkpoint.pt
"""

import torch
import numpy as np
import os
import random
import argparse
import csv
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)

# Model imports — base
from model import DenseNetWithDoubleLinear, DenseNetWithDoubleLinear_addDemothen2
from swintransformer import (
    SwinTDoubleLinear,
    SwinTDoubleLinear_CheXpert,
    SwinTDoubleLinear_addDemothen2,
)

try:
    from MedMamba.MedMamba import VSSM_DoubleLinear as medmamba
except ImportError:
    medmamba = None

try:
    from MedMamba.MedMamba import VSSM_Double_addDemothen2 as medmamba_demo
except ImportError:
    medmamba_demo = None

# Dataset imports
from RawImageDataset import (
    MIMIC_raw,
    MIMIC_raw_ICD_mask_image,
    MIMIC_raw_ICD_mask_mostimage,
    MIMIC_raw_high_pass,
    MIMIC_raw_low_pass,
    MIMIC_raw_random_label,
)
from MyLoader_new import CheXpertLoader


# Patch index -> (starting_point, height, width) mapping for 3x3 grid on 448x448
PATCH_GRID = {
    1: ([0, 0], 150, 150),
    2: ([0, 150], 150, 150),
    3: ([0, 300], 150, 148),
    4: ([150, 0], 150, 150),
    5: ([150, 150], 150, 150),
    6: ([150, 300], 150, 148),
    7: ([300, 0], 148, 150),
    8: ([300, 150], 148, 150),
    9: ([300, 300], 148, 148),
}

# Demographics size lookup
DEMO_SIZE = {"sex": 2, "age": 3, "race": 3}
DEMO_INDEX = {"sex": 0, "age": 1, "race": 2}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_model(model_name, dataset_name, num_classes, device, model_variant="base", demo_size=0):
    """Instantiate model architecture matching training configuration."""
    if model_variant == "addDemo":
        if model_name == "densenet":
            model = DenseNetWithDoubleLinear_addDemothen2(
                num_classes=num_classes, dropout_prob=0, demo_size=demo_size
            )
        elif model_name == "mamba":
            if medmamba_demo is None:
                raise ImportError("VSSM_addDemothen2 not found in MedMamba.MedMamba.")
                
            model = medmamba_demo(depths=[2, 2, 8, 2], dims=[96, 192, 384, 768], num_classes=num_classes, demo_size=demo_size)
        elif model_name == "swinTF":
            model = SwinTDoubleLinear_addDemothen2(
                use_checkpoint=True, num_classes=num_classes, demo_size=demo_size
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    else:
        if model_name == "densenet":
            model = DenseNetWithDoubleLinear(num_classes=num_classes, dropout_prob=0)
        elif model_name == "mamba":
            if medmamba is None:
                raise ImportError("VSSM_DoubleLinear not found in MedMamba.MedMamba.")
            model = medmamba(depths=[2, 2, 8, 2], dims=[96, 192, 384, 768], num_classes=num_classes)
        elif model_name == "swinTF":
            if dataset_name == "CheXpert":
                model = SwinTDoubleLinear_CheXpert(
                    use_checkpoint=True, num_classes=num_classes
                )
            else:
                model = SwinTDoubleLinear(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    model.to(device)
    return model


def _extract_demographics(batch):
    """Extract and convert demographics from batch to scalar indices."""
    gender = batch["gender"]
    age = batch["age"]
    race = batch["race"]

    if isinstance(gender, torch.Tensor) and gender.dim() == 2:
        gender_idx = torch.argmax(gender, dim=1)
    else:
        gender_idx = gender

    if isinstance(age, torch.Tensor) and age.dim() == 2:
        age_idx = torch.argmax(age, dim=1)
    else:
        age_idx = age

    if isinstance(race, torch.Tensor) and race.dim() == 2:
        race_idx = torch.argmax(race, dim=1)
    else:
        race_idx = race

    return gender_idx, age_idx, race_idx


def _build_demo_tensor(batch, demo_labels, device):
    """Build concatenated demographics tensor for addDemo models."""
    gender = batch["gender"].to(torch.float32)
    age = batch["age"].to(torch.float32)
    race = batch["race"].to(torch.float32)

    # Ensure gender is 2D one-hot
    if gender.dim() == 1:
        g_onehot = torch.zeros(gender.size(0), 2)
        g_onehot.scatter_(1, gender.long().unsqueeze(1), 1)
        gender = g_onehot

    demo_list = [gender, age, race]
    selected = [demo_list[DEMO_INDEX[label]] for label in demo_labels]
    demo_tensor = torch.cat(selected, dim=1).to(device)
    return demo_tensor


def collect_predictions_mimic(model, dataloader, device, model_variant="base", demo_labels=None):
    """Run inference on MIMIC dataset, collect predictions + demographics + ids."""
    model.eval()
    all_preds = []
    all_labels = []
    all_gender = []
    all_age = []
    all_race = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["full_img"].to(device)
            labels = batch["insurance"].to(device).squeeze(-1)

            if model_variant == "addDemo" and demo_labels:
                demos = _build_demo_tensor(batch, demo_labels, device)
                output = model(imgs, demos)
            else:
                output = model(imgs)

            all_preds.append(output.cpu())
            all_labels.append(labels.cpu())

            gender_idx, age_idx, race_idx = _extract_demographics(batch)
            all_gender.append(gender_idx)
            all_age.append(age_idx)
            all_race.append(race_idx)

            img_id = batch.get("img_id")
            if img_id is not None:
                all_ids.extend([str(x) for x in img_id])

    return (
        torch.cat(all_preds),
        torch.cat(all_labels),
        torch.cat(all_gender).long(),
        torch.cat(all_age).long(),
        torch.cat(all_race).long(),
        all_ids,
    )


def collect_predictions_chexpert(model, dataloader, device):
    """Run inference on CheXpert dataset, collect predictions + demographics + patients."""
    model.eval()
    all_preds = []
    all_labels = []
    all_gender = []
    all_age = []
    all_race = []
    all_patients = []

    with torch.no_grad():
        for batch in dataloader:
            # loader yields (imgs, labels, age, sex, race[, patient]) depending on return_patient
            if len(batch) == 6:
                imgs, labels, age, sex, race, patient = batch
                all_patients.extend([str(int(x)) for x in patient])
            else:
                imgs, labels, age, sex, race = batch
            imgs = imgs.to(device)
            labels = labels.to(device).squeeze(-1)

            output = model(imgs)

            all_preds.append(output.cpu())
            all_labels.append(labels.cpu())

            all_gender.append(sex.long())

            age_binned = torch.zeros_like(age, dtype=torch.long)
            age_binned[(age >= 40) & (age < 50)] = 1
            age_binned[(age >= 50)] = 2
            all_age.append(age_binned)

            race_remapped = torch.zeros_like(race, dtype=torch.long)
            race_remapped[race == 1] = 0  # White -> 0
            race_remapped[race == 0] = 1  # Black -> 1
            race_remapped[race == 2] = 2  # Other -> 2
            all_race.append(race_remapped)

    preds_cat = torch.cat(all_preds)
    labels_cat = torch.cat(all_labels)
    gender_cat = torch.cat(all_gender).long().squeeze()
    age_cat = torch.cat(all_age).long().squeeze()
    race_cat = torch.cat(all_race).long().squeeze()
    # CheXpert loader yields no per-sample identifier; fall back to row order
    ids = [str(i) for i in range(len(preds_cat))]
    patients = all_patients if len(all_patients) == len(preds_cat) else None
    return preds_cat, labels_cat, gender_cat, age_cat, race_cat, ids, patients


def compute_metrics(preds, labels, num_classes, device):
    """Compute AUC, Precision, Recall, F1, Accuracy from predictions and labels."""
    preds = preds.to(device)
    labels = labels.to(device)

    _, target_idx = torch.max(labels, 1)

    auc_metric = MulticlassAUROC(
        num_classes=num_classes, average="macro", thresholds=None
    ).to(device)
    precision_metric = MulticlassPrecision(
        num_classes=num_classes, average="macro"
    ).to(device)
    recall_metric = MulticlassRecall(
        num_classes=num_classes, average="macro"
    ).to(device)
    f1_metric = MulticlassF1Score(
        num_classes=num_classes, average="macro"
    ).to(device)

    _, pred_idx = torch.max(preds, 1)
    acc = (pred_idx == target_idx).float().mean().item()

    try:
        auc = auc_metric(preds, target_idx).item()
    except Exception:
        auc = float("nan")

    precision = precision_metric(preds, target_idx).item()
    recall = recall_metric(preds, target_idx).item()
    f1 = f1_metric(preds, target_idx).item()

    return {
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": acc,
    }


# Subgroup definitions (consistent label order across MIMIC/CheXpert)
SUBGROUPS = {
    "Gender": {0: "Male", 1: "Female"},
    "Age": {0: "<40", 1: "40-50", 2: "50-65"},
    "Race": {0: "White", 1: "Black", 2: "Other"},
}


def evaluate_full_dataset(
    all_preds,
    all_labels,
    all_gender,
    all_age,
    all_race,
    num_classes,
    device,
):
    """Compute metrics on the entire test set (no resampling), overall + per subgroup."""
    results = []

    metrics = compute_metrics(all_preds, all_labels, num_classes, device)
    metrics["group"] = "Overall"
    metrics["N"] = int(len(all_preds))
    results.append(metrics)

    demo_tensors = {
        "Gender": all_gender,
        "Age": all_age,
        "Race": all_race,
    }
    for group_name, label_map in SUBGROUPS.items():
        demo = demo_tensors[group_name]
        for val, val_name in label_map.items():
            mask = demo == val
            n = int(mask.sum())
            if n < 10:
                continue
            sub_metrics = compute_metrics(
                all_preds[mask],
                all_labels[mask],
                num_classes,
                device,
            )
            sub_metrics["group"] = f"{group_name}: {val_name}"
            sub_metrics["N"] = n
            results.append(sub_metrics)

    return results


def print_full_results(full_results):
    """Print formatted metrics table for the entire test set."""
    print(f"\n{'=' * 80}")
    print("Full Test Set Results (no resampling)")
    print(f"{'=' * 80}")

    metrics = ["AUC", "Precision", "Recall", "F1", "Accuracy"]
    header = f"{'Group':<22} | {'N':>7} | " + " | ".join(f"{m:>9}" for m in metrics)
    print(header)
    print("-" * len(header))
    for row in full_results:
        values = " | ".join(f"{row[m]:>9.4f}" for m in metrics)
        print(f"{row['group']:<22} | {row['N']:>7} | {values}")


def save_full_results(full_results, output_dir, experiment_name):
    """Save whole-test-set metrics to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{experiment_name}_full_test_metrics.csv")
    fieldnames = ["group", "N", "AUC", "Precision", "Recall", "F1", "Accuracy"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in full_results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nFull test set metrics saved to: {path}")
    return path


def save_predictions(
    all_preds,
    all_labels,
    all_gender,
    all_age,
    all_race,
    all_ids,
    output_dir,
    experiment_name,
    all_patients=None,
):
    """Save per-sample predictions (logits, probabilities, labels, demographics) to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{experiment_name}_predictions.csv")

    preds = all_preds.detach().cpu().float()
    probs = torch.softmax(preds, dim=1)
    pred_idx = torch.argmax(preds, dim=1)
    _, target_idx = torch.max(all_labels.detach().cpu(), 1)

    gender = all_gender.detach().cpu()
    age = all_age.detach().cpu()
    race = all_race.detach().cpu()

    num_classes = preds.shape[1]
    fieldnames = (
        ["sample_id", "patient_id", "true_label", "pred_label", "correct"]
        + [f"logit_{c}" for c in range(num_classes)]
        + [f"prob_{c}" for c in range(num_classes)]
        + ["gender", "age", "race", "gender_name", "age_name", "race_name"]
    )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(preds)):
            g = int(gender[i])
            a = int(age[i])
            r = int(race[i])
            row = {
                "sample_id": all_ids[i] if i < len(all_ids) else i,
                "patient_id": all_patients[i] if all_patients is not None else "",
                "true_label": int(target_idx[i]),
                "pred_label": int(pred_idx[i]),
                "correct": int(pred_idx[i] == target_idx[i]),
                "gender": g,
                "age": a,
                "race": r,
                "gender_name": SUBGROUPS["Gender"].get(g, "Unknown"),
                "age_name": SUBGROUPS["Age"].get(a, "Unknown"),
                "race_name": SUBGROUPS["Race"].get(r, "Unknown"),
            }
            for c in range(num_classes):
                row[f"logit_{c}"] = float(preds[i, c])
                row[f"prob_{c}"] = float(probs[i, c])
            writer.writerow(row)

    print(f"Per-sample predictions saved to: {path}")
    return path


def load_patient_ids(test_path, all_ids):
    """Map each collected sample id (dicom_id) to its patient id from the test CSV.

    Returns None if the CSV has no usable patient column or any sample is missing,
    so the caller can fall back to image-level resampling instead of guessing.
    """
    try:
        with open(test_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [r for r in reader if r]
    except OSError as e:
        print(f"WARNING: could not read {test_path} for patient ids ({e})")
        return None

    subj_col = None
    for name in ("subject_id_x", "subject_id", "subject_idx"):
        if name in header:
            subj_col = header.index(name)
            break
    if subj_col is None:
        # RawImageDataset builds the image path from column 1, which is the subject id
        subj_col = 1
    if subj_col >= len(header):
        print("WARNING: no patient id column found in test CSV")
        return None

    mapping = {row[0]: row[subj_col] for row in rows}
    patients = [mapping.get(sample_id) for sample_id in all_ids]
    n_missing = sum(1 for p in patients if p is None)
    if n_missing:
        print(
            f"WARNING: {n_missing}/{len(patients)} samples had no patient id in "
            f"{test_path}; falling back to image-level resampling"
        )
        return None
    return patients


def build_clusters(cluster_labels):
    """Group sample positions by cluster label, preserving first-seen order."""
    members = {}
    order = []
    for pos, label in enumerate(cluster_labels):
        if label not in members:
            members[label] = []
            order.append(label)
        members[label].append(pos)
    return [np.array(members[label], dtype=np.int64) for label in order]


def bootstrap_evaluate(
    all_preds,
    all_labels,
    all_gender,
    all_age,
    all_race,
    num_classes,
    device,
    n_bootstrap,
    sample_size,
    seed,
    clusters=None,
):
    """Run bootstrap resampling and compute metrics per iteration.

    If `clusters` is given (a list of index arrays, one per patient), whole
    patients are resampled with replacement and all their images come along —
    the cluster bootstrap, which respects the correlation between images of the
    same patient. Otherwise individual images are resampled.
    """
    N = len(all_preds)
    rng = np.random.RandomState(seed)

    n_draw = None
    if clusters is not None:
        n_clusters = len(clusters)
        # Draw the observed number of patients; scale down proportionally if the
        # caller asked for a smaller-than-full resample.
        frac = min(sample_size, N) / N
        n_draw = max(1, int(round(n_clusters * frac)))

    all_results = []

    for i in range(n_bootstrap):
        if clusters is not None:
            picked = rng.choice(len(clusters), size=n_draw, replace=True)
            indices = np.concatenate([clusters[c] for c in picked])
        else:
            indices = rng.choice(N, size=min(sample_size, N), replace=True)
        indices_t = torch.tensor(indices)

        preds_sample = all_preds[indices_t]
        labels_sample = all_labels[indices_t]
        gender_sample = all_gender[indices_t]
        age_sample = all_age[indices_t]
        race_sample = all_race[indices_t]

        # Overall metrics
        metrics = compute_metrics(preds_sample, labels_sample, num_classes, device)
        metrics["group"] = "Overall"
        metrics["iteration"] = i
        all_results.append(metrics)

        # Per-subgroup metrics
        demo_tensors = {
            "Gender": gender_sample,
            "Age": age_sample,
            "Race": race_sample,
        }
        for group_name, label_map in SUBGROUPS.items():
            demo = demo_tensors[group_name]
            for val, val_name in label_map.items():
                mask = demo == val
                if mask.sum() < 10:
                    continue
                sub_metrics = compute_metrics(
                    preds_sample[mask],
                    labels_sample[mask],
                    num_classes,
                    device,
                )
                sub_metrics["group"] = f"{group_name}: {val_name}"
                sub_metrics["iteration"] = i
                all_results.append(sub_metrics)

    return all_results


def summarize_results(all_results):
    """Compute mean, std, 95% CI per group from bootstrap iterations."""
    from collections import defaultdict

    groups = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        g = r["group"]
        for metric in ["AUC", "Precision", "Recall", "F1", "Accuracy"]:
            groups[g][metric].append(r[metric])

    summary = []
    for group_name, metrics_dict in groups.items():
        row = {"group": group_name}
        for metric, values in metrics_dict.items():
            arr = np.array(values)
            arr_clean = arr[~np.isnan(arr)]
            if len(arr_clean) == 0:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
                row[f"{metric}_ci_lower"] = float("nan")
                row[f"{metric}_ci_upper"] = float("nan")
            else:
                row[f"{metric}_mean"] = np.mean(arr_clean)
                row[f"{metric}_std"] = np.std(arr_clean)
                row[f"{metric}_ci_lower"] = np.percentile(arr_clean, 2.5)
                row[f"{metric}_ci_upper"] = np.percentile(arr_clean, 97.5)
        summary.append(row)

    return summary


def print_summary(summary, n_bootstrap, sample_size, unit="image", n_clusters=None):
    """Print formatted summary table."""
    print(f"\n{'=' * 80}")
    print(f"Bootstrap Validation Results (N={n_bootstrap}, sample_size={sample_size})")
    if unit == "patient":
        print(f"Resampling unit: patient ({n_clusters} patients drawn per iteration)")
    else:
        print("Resampling unit: image")
    print(f"{'=' * 80}")

    metrics = ["AUC", "Precision", "Recall", "F1", "Accuracy"]

    group_order = ["Overall"]
    other_groups = sorted(
        [s["group"] for s in summary if s["group"] != "Overall"]
    )
    group_order.extend(other_groups)

    for group_name in group_order:
        row = next((s for s in summary if s["group"] == group_name), None)
        if row is None:
            continue
        print(f"\n--- {group_name} ---")
        print(f"{'Metric':<12} | {'Mean':>8} | {'Std':>8} | {'95% CI':>22}")
        print(f"{'-' * 12}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 22}")
        for m in metrics:
            mean = row.get(f"{m}_mean", float("nan"))
            std = row.get(f"{m}_std", float("nan"))
            ci_lo = row.get(f"{m}_ci_lower", float("nan"))
            ci_hi = row.get(f"{m}_ci_upper", float("nan"))
            print(
                f"{m:<12} | {mean:>8.4f} | {std:>8.4f} | [{ci_lo:.4f}, {ci_hi:.4f}]"
            )


def save_results(all_results, summary, output_dir, experiment_name):
    """Save per-iteration and summary results to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    iter_path = os.path.join(
        output_dir, f"{experiment_name}_bootstrap_iterations.csv"
    )
    fieldnames = ["group", "iteration", "AUC", "Precision", "Recall", "F1", "Accuracy"]
    with open(iter_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nPer-iteration results saved to: {iter_path}")

    summary_path = os.path.join(
        output_dir, f"{experiment_name}_bootstrap_summary.csv"
    )
    metrics = ["AUC", "Precision", "Recall", "F1", "Accuracy"]
    summary_fields = ["group"]
    for m in metrics:
        summary_fields.extend(
            [f"{m}_mean", f"{m}_std", f"{m}_ci_lower", f"{m}_ci_upper"]
        )
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: row.get(k, "") for k in summary_fields})
    print(f"Summary results saved to: {summary_path}")


def build_mimic_dataset(test_path, preprocessing, patch_idx, filter_diameter, resize=448, random_label=False):
    """Build the appropriate MIMIC dataset based on preprocessing type."""
    if random_label:
        return MIMIC_raw_random_label(test_path, resize=resize, transform=False)
    if preprocessing == "mask_remove":
        start, h, w = PATCH_GRID[patch_idx]
        return MIMIC_raw_ICD_mask_image(test_path, start, h, w, transform=False)
    elif preprocessing == "mask_keep":
        start, h, w = PATCH_GRID[patch_idx]
        return MIMIC_raw_ICD_mask_mostimage(test_path, start, h, w, transform=False)
    elif preprocessing == "high_pass":
        return MIMIC_raw_high_pass(test_path, resize=448, diameter=filter_diameter, transform=False)
    elif preprocessing == "low_pass":
        return MIMIC_raw_low_pass(test_path, resize=448, diameter=filter_diameter, transform=False)
    else:
        upscale_to = 448 if resize != 448 else None
        return MIMIC_raw(test_path, resize=resize, transform=False, upscale_to=upscale_to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap evaluation for insurance classification"
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["MIMIC", "CheXpert"])
    parser.add_argument("--model", type=str, required=True, choices=["densenet", "mamba", "swinTF"])
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Resample size; defaults to the full test set size N. "
                             "Smaller values give an m-out-of-n bootstrap with wider CIs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="bootstrap_results")
    parser.add_argument("--experiment_name", type=str, default="bootstrap")
    # Experiment variant args
    parser.add_argument("--model_variant", type=str, default="base", choices=["base", "addDemo"])
    parser.add_argument("--demo_labels", nargs="+", type=str, default=None,
                        choices=["sex", "age", "race"],
                        help="Demographics to pass to addDemo model (e.g. --demo_labels sex age)")
    parser.add_argument("--preprocessing", type=str, default="none",
                        choices=["none", "mask_remove", "mask_keep", "high_pass", "low_pass"])
    parser.add_argument("--patch_idx", type=int, default=None, choices=range(1, 10),
                        help="Patch index 1-9 for mask experiments")
    parser.add_argument("--filter_diameter", type=int, default=50,
                        help="Frequency filter diameter for high/low pass experiments")
    parser.add_argument("--resize", type=int, default=448,
                        help="Image resize resolution (default 448, for exp2 resolution experiments)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader num_workers (0 for main-process loading)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--random_label", type=bool, default=False)
    parser.add_argument("--no_save_predictions", action="store_true",
                        help="Skip writing the per-sample predictions CSV")
    parser.add_argument("--bootstrap_unit", type=str, default="patient",
                        choices=["patient", "image"],
                        help="Resample whole patients (default, correct for repeated "
                             "images per patient) or individual images (legacy)")
    args = parser.parse_args()

    # Validation
    if args.model_variant == "addDemo" and not args.demo_labels:
        parser.error("--demo_labels required when --model_variant is addDemo")
    if args.preprocessing in ("mask_remove", "mask_keep") and args.patch_idx is None:
        parser.error("--patch_idx required for mask preprocessing")

    num_classes = 2
    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    set_seed(args.seed)

    # Compute demo_size for addDemo models
    demo_size = 0
    if args.model_variant == "addDemo":
        demo_size = sum(DEMO_SIZE[d] for d in args.demo_labels)

    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model} (variant={args.model_variant})")
    if args.model_variant == "addDemo":
        print(f"Demo labels: {args.demo_labels} (demo_size={demo_size})")
    if args.preprocessing != "none":
        extra = ""
        if args.patch_idx:
            extra += f" patch_idx={args.patch_idx}"
        if args.preprocessing in ("high_pass", "low_pass"):
            extra += f" diameter={args.filter_diameter}"
        print(f"Preprocessing: {args.preprocessing}{extra}")
    print(f"Weight path: {args.weight_path}")
    ss_desc = "full test set" if args.sample_size is None else args.sample_size
    print(f"Bootstrap: N={args.n_bootstrap}, sample_size={ss_desc}, "
          f"unit={args.bootstrap_unit}, seed={args.seed}")
    print(f"Device: {device}")

    # Load model and weights
    model = load_model(
        args.model, args.dataset, num_classes, device,
        model_variant=args.model_variant, demo_size=demo_size
    )
    checkpoint = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from: {args.weight_path}")

    # Load test data and run full inference
    g = torch.Generator()
    g.manual_seed(0)

    if args.dataset == "MIMIC":
        if args.resize != 448:
            print(f"Resolution experiment: resize to {args.resize} then upscale to 448")
        test_dataset = build_mimic_dataset(
            args.test_path, args.preprocessing, args.patch_idx, args.filter_diameter,
            resize=args.resize,
            random_label = args.random_label
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            worker_init_fn=seed_worker,
            num_workers=args.num_workers,
            shuffle=False,
            generator=g,
        )
        print(f"Test set size: {len(test_dataset)}")
        print("Running full inference on test set...")
        all_preds, all_labels, all_gender, all_age, all_race, all_ids = (
            collect_predictions_mimic(
                model, test_loader, device,
                model_variant=args.model_variant,
                demo_labels=args.demo_labels,
            )
        )
        all_patients = load_patient_ids(args.test_path, all_ids)
    elif args.dataset == "CheXpert":
        test_loader = CheXpertLoader(
            args.test_path, None, batch_size, num_workers=1, shuffle=False,
            return_patient=True,
        )
        print("Running full inference on CheXpert test set...")
        all_preds, all_labels, all_gender, all_age, all_race, all_ids, all_patients = (
            collect_predictions_chexpert(model, test_loader, device)
        )

    print(f"Collected {len(all_preds)} predictions")

    N = len(all_preds)
    if args.sample_size is None:
        args.sample_size = N
        print(f"sample_size not given; using the full test set (N={N})")
    elif args.sample_size < N:
        print(
            f"WARNING: sample_size={args.sample_size} < N={N}: this is an "
            f"m-out-of-n bootstrap, CIs are ~{(N / args.sample_size) ** 0.5:.2f}x "
            f"wider than for the full test set"
        )

    # Patient-level (cluster) resampling unless asked otherwise or ids unavailable
    clusters = None
    if args.bootstrap_unit == "patient":
        if all_patients is None:
            print(
                "WARNING: no patient ids available for this dataset; "
                "falling back to image-level resampling"
            )
        else:
            clusters = build_clusters(all_patients)
            sizes = np.array([len(c) for c in clusters])
            print(
                f"Patient-level bootstrap: {len(clusters)} patients, "
                f"{N} images, max {sizes.max()} images/patient, "
                f"{int((sizes > 1).sum())} patients with >1 image"
            )

    # Save every prediction from the full test set
    if not args.no_save_predictions:
        save_predictions(
            all_preds,
            all_labels,
            all_gender,
            all_age,
            all_race,
            all_ids,
            args.output_dir,
            args.experiment_name,
            all_patients=all_patients,
        )

    # Metrics on the entire test set (point estimates, no resampling)
    full_results = evaluate_full_dataset(
        all_preds,
        all_labels,
        all_gender,
        all_age,
        all_race,
        num_classes,
        device,
    )
    print_full_results(full_results)
    save_full_results(full_results, args.output_dir, args.experiment_name)

    # Run bootstrap
    print(f"\nRunning {args.n_bootstrap} bootstrap iterations...")
    all_results = bootstrap_evaluate(
        all_preds,
        all_labels,
        all_gender,
        all_age,
        all_race,
        num_classes,
        device,
        args.n_bootstrap,
        args.sample_size,
        args.seed,
        clusters=clusters,
    )

    # Summarize and output
    summary = summarize_results(all_results)
    print_summary(
        summary,
        args.n_bootstrap,
        args.sample_size,
        unit="patient" if clusters is not None else "image",
        n_clusters=len(clusters) if clusters is not None else None,
    )
    save_results(all_results, summary, args.output_dir, args.experiment_name)
