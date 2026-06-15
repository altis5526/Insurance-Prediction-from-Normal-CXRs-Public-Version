"""Extract penultimate-layer embeddings from trained DenseNet / Swin / MedMamba
checkpoints for the insurance-prediction task and save them as .npz files
consumed by plot_pca.py / plot_pca.ipynb.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm import tqdm

import importlib.util

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent

# Parent project for model definitions (model.py, swintransformer.py, MedMamba/).
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from model import DenseNetWithDoubleLinear  # noqa: E402
from swintransformer import SwinTDoubleLinear  # noqa: E402
from MedMamba.MedMamba import VSSM_DoubleLinear as medmamba  # noqa: E402

# Force-load the local RawImageDataset.py (correct /datasets/mimic/ image root).
_spec = importlib.util.spec_from_file_location("RawImageDataset", _HERE / "RawImageDataset.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MIMIC_raw = _mod.MIMIC_raw


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


class FeatureExtractor:
    def __init__(self):
        self.features = None

    def hook_fn_output(self, module, input, output):
        self.features = output.detach()

    def hook_fn_input(self, module, input, output):
        self.features = input[0].detach()


def load_clean_state_dict(model, checkpoint_path, device):
    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model


def calculate_metrics(logits_tensor, labels_tensor, num_classes=2):
    probs = torch.softmax(logits_tensor, dim=1)
    targets = labels_tensor.squeeze(-1)
    if targets.ndim > 1:
        _, target_indices = torch.max(targets, 1)
    else:
        target_indices = targets.long()

    metric_args = {"num_classes": num_classes, "average": "none"}
    metrics = {
        "AUC": MulticlassAUROC(num_classes=num_classes, average="none", thresholds=None),
        "Precision": MulticlassPrecision(**metric_args),
        "Recall": MulticlassRecall(**metric_args),
        "F1": MulticlassF1Score(**metric_args),
        "Accuracy": MulticlassAccuracy(**metric_args),
    }
    return {name: fn(probs, target_indices).cpu().numpy() for name, fn in metrics.items()}


def run_extraction_for_model(model_name, model_instance, checkpoint_path, data_loader, device, output_dir):
    print(f"\n{'=' * 20} Processing Model: {model_name} {'=' * 20}")

    model_instance = load_clean_state_dict(model_instance, checkpoint_path, device)
    model_instance.to(device)
    model_instance.eval()

    extractor = FeatureExtractor()
    handle = None

    if hasattr(model_instance, "linear_probe1"):
        print(f"[{model_name}] Found 'linear_probe1'. Hooking output.")
        handle = model_instance.linear_probe1.register_forward_hook(extractor.hook_fn_output)
    elif hasattr(model_instance, "head"):
        if isinstance(model_instance.head, nn.Sequential):
            print(f"[{model_name}] 'head' is Sequential. Hooking output of head[0].")
            handle = model_instance.head[0].register_forward_hook(extractor.hook_fn_output)
        elif isinstance(model_instance.head, nn.Linear):
            print(f"[{model_name}] 'head' is Linear. Hooking INPUT to head.")
            handle = model_instance.head.register_forward_hook(extractor.hook_fn_input)
        else:
            handle = model_instance.head.register_forward_hook(extractor.hook_fn_input)
    else:
        raise AttributeError(f"Model {model_name} has neither 'linear_probe1' nor 'head'.")

    all_embeddings, all_labels_numpy = [], []
    all_logits_tensor, all_labels_tensor = [], []
    all_metadata = {"img_ids": [], "ages": [], "genders": [], "races": []}

    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"{model_name} Inference"):
            imgs = batch["full_img"].to(device)
            labels = batch["insurance"].to(device)
            logits = model_instance(imgs)

            if extractor.features is None:
                raise RuntimeError("Hook failed to capture features.")
            embeddings = extractor.features

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels_numpy.append(labels.cpu().numpy())
            all_logits_tensor.append(logits.cpu())
            all_labels_tensor.append(labels.cpu())

            if "img_id" in batch:
                all_metadata["img_ids"].extend(batch["img_id"])
            if "age" in batch:
                all_metadata["ages"].append(batch["age"].numpy())
            if "gender" in batch:
                all_metadata["genders"].append(batch["gender"].numpy())
            if "race" in batch:
                all_metadata["races"].append(batch["race"].numpy())

    if handle:
        handle.remove()

    print(f"Calculating metrics for {model_name}...")
    preds_cat = torch.cat(all_logits_tensor, dim=0)
    targets_cat = torch.cat(all_labels_tensor, dim=0)
    scores = calculate_metrics(preds_cat, targets_cat)

    print(f"--- Results for {model_name} (Per Class) ---")
    num_classes_found = len(next(iter(scores.values())))
    header = f"{'Metric':<15}" + "".join([f" | Class {i:<8}" for i in range(num_classes_found)]) + " | MACRO AVG"
    print(header)
    print("-" * len(header))
    for k, v in scores.items():
        macro_avg = np.mean(v)
        values_str = "".join([f" | {val:.4f}    " for val in v])
        print(f"{k:<15}{values_str} | {macro_avg:.4f}")

    embeddings_array = np.concatenate(all_embeddings, axis=0)
    labels_array = np.concatenate(all_labels_numpy, axis=0)
    save_path = os.path.join(output_dir, f"{model_name}_embeddings.npz")

    save_dict = {"embeddings": embeddings_array, "labels": labels_array, "metrics": scores}
    if all_metadata["img_ids"]:
        save_dict["img_ids"] = np.array(all_metadata["img_ids"])
    if all_metadata["ages"]:
        save_dict["ages"] = np.concatenate(all_metadata["ages"], axis=0)
    if all_metadata["genders"]:
        save_dict["genders"] = np.concatenate(all_metadata["genders"], axis=0)
    if all_metadata["races"]:
        save_dict["races"] = np.concatenate(all_metadata["races"], axis=0)

    np.savez_compressed(save_path, **save_dict)
    print(f"Saved embeddings to: {save_path}")

    del model_instance, preds_cat, targets_cat, all_embeddings, all_logits_tensor
    torch.cuda.empty_cache()


def parse_args():
    _here = Path(__file__).resolve().parent
    default_out = _here / "embeddings" / "all_models_test_set_128dim"
    default_csv = _here / "csv" / "insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv"
    default_densenet = _here / "checkpoints" / "MedGemma_MIMIC_densenet.pt"
    default_swin = _here / "checkpoints" / "MedGemma_MIMIC_swinTF.pt"
    default_mamba = _here / "checkpoints" / "MedGemma_MIMIC_mamba.pt"
    p = argparse.ArgumentParser(description="Extract penultimate-layer embeddings for DenseNet/Swin/MedMamba.")
    p.add_argument("--test-csv", default=str(default_csv), help="Path to the test CSV consumed by MIMIC_raw.")
    p.add_argument("--densenet-ckpt", default=str(default_densenet), help="DenseNet checkpoint (.pt).")
    p.add_argument("--swin-ckpt", default=str(default_swin), help="Swin Transformer checkpoint (.pt).")
    p.add_argument("--medmamba-ckpt", default=str(default_mamba), help="MedMamba checkpoint (.pt).")
    p.add_argument("--output-dir", default=str(default_out), help=f"Where to write *_embeddings.npz (default: {default_out}).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--resize", type=int, default=448)
    p.add_argument("--img-root", default="/datasets/mimic/mimic-cxr-jpg/2.0.0/files/",
                   help="Root directory of MIMIC-CXR-JPG images.")
    p.add_argument("--models", nargs="+", default=["densenet121", "swin_transformer", "medmamba"],
                   choices=["densenet121", "swin_transformer", "medmamba"],
                   help="Subset of models to process.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    print("Loading Dataset...")
    test_dataset = MIMIC_raw(args.test_csv, resize=args.resize, transform=True, img_root=args.img_root)
    g = torch.Generator()
    g.manual_seed(args.seed)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    print(f"Total Test Samples: {len(test_dataset)}")

    all_configs = {
        "densenet121": {
            "name": "densenet121",
            "model": DenseNetWithDoubleLinear(num_classes=args.num_classes, dropout_prob=0),
            "path": args.densenet_ckpt,
        },
        "swin_transformer": {
            "name": "swin_transformer",
            "model": SwinTDoubleLinear(use_checkpoint=False, num_classes=args.num_classes),
            "path": args.swin_ckpt,
        },
        "medmamba": {
            "name": "medmamba",
            "model": medmamba(depths=[2, 2, 8, 2], dims=[96, 192, 384, 768], num_classes=args.num_classes),
            "path": args.medmamba_ckpt,
        },
    }
    model_configs = [all_configs[m] for m in args.models]

    for config in model_configs:
        try:
            run_extraction_for_model(
                model_name=config["name"],
                model_instance=config["model"],
                checkpoint_path=config["path"],
                data_loader=test_loader,
                device=device,
                output_dir=args.output_dir,
            )
        except Exception as e:
            print(f"Error processing {config['name']}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll models processed successfully.")


if __name__ == "__main__":
    main()
