import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import random
from tqdm import tqdm
from torchmetrics.classification import (
    MulticlassAUROC, 
    MulticlassPrecision, 
    MulticlassRecall, 
    MulticlassF1Score, 
    MulticlassAccuracy
)

# ============================================================================
# 1. Model Imports
# ============================================================================
# Ensure these files are in your python path or current directory
from model import DenseNetWithDoubleLinear 
from swintransformer import SwinTDoubleLinear 
from MedMamba.MedMamba import VSSM_DoubleLinear as medmamba
from RawImageDataset import MIMIC_raw

# ============================================================================
# 2. Helper Classes & Functions
# ============================================================================

def set_seed(seed):
    """Restores the seeding logic from your training script."""
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
    """Class to handle the forward hook for embedding extraction."""
    def __init__(self):
        self.features = None

    def hook_fn_output(self, module, input, output):
        """Capture layer output."""
        self.features = output.detach()

    def hook_fn_input(self, module, input, output):
        """Capture layer input (tuple, so we take index 0)."""
        self.features = input[0].detach()

def load_clean_state_dict(model, checkpoint_path, device):
    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix if present (common in DataParallel training)
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    return model

def calculate_metrics(logits_tensor, labels_tensor, num_classes=2):
    # Apply Softmax to convert Logits -> Probabilities
    probs = torch.softmax(logits_tensor, dim=1)
    
    targets = labels_tensor.squeeze(-1)
    if targets.ndim > 1:
        _, target_indices = torch.max(targets, 1)
    else:
        target_indices = targets.long()

    # We use average='none' to get per-class scores, 
    # but we will also calculate the macro average for verification.
    metric_args = {'num_classes': num_classes, 'average': "none"}
    
    metrics = {
        "AUC": MulticlassAUROC(num_classes=num_classes, average="none", thresholds=None),
        "Precision": MulticlassPrecision(**metric_args),
        "Recall": MulticlassRecall(**metric_args),
        "F1": MulticlassF1Score(**metric_args),
        "Accuracy": MulticlassAccuracy(**metric_args)
    }

    results = {}
    for name, metric_fn in metrics.items():
        # .cpu().numpy() converts the result to numpy array
        results[name] = metric_fn(probs, target_indices).cpu().numpy()
        
    return results

# ============================================================================
# 3. Core Extraction Engine
# ============================================================================

def run_extraction_for_model(
    model_name,
    model_instance,
    checkpoint_path,
    data_loader,
    device,
    output_dir
):
    print(f"\n{'='*20} Processing Model: {model_name} {'='*20}")
    
    # 1. Load Weights
    model_instance = load_clean_state_dict(model_instance, checkpoint_path, device)
    model_instance.to(device)
    model_instance.eval()

    # 2. Register Hook (Dynamic Logic)
    extractor = FeatureExtractor()
    handle = None
    
    if hasattr(model_instance, 'linear_probe1'):
        print(f"[{model_name}] Found 'linear_probe1'. Hooking output.")
        handle = model_instance.linear_probe1.register_forward_hook(extractor.hook_fn_output)
        
    elif hasattr(model_instance, 'head'):
        print(f"[{model_name}] 'linear_probe1' not found. Found 'head'.")
        if isinstance(model_instance.head, nn.Sequential):
            print(f"[{model_name}] 'head' is Sequential. Hooking output of head[0].")
            handle = model_instance.head[0].register_forward_hook(extractor.hook_fn_output)
        elif isinstance(model_instance.head, nn.Linear):
            print(f"[{model_name}] 'head' is Linear. Hooking INPUT to head.")
            handle = model_instance.head.register_forward_hook(extractor.hook_fn_input)
        else:
            print(f"[{model_name}] Hooking INPUT to 'head' layer.")
            handle = model_instance.head.register_forward_hook(extractor.hook_fn_input) 
    else:
        raise AttributeError(f"Model {model_name} has neither 'linear_probe1' nor 'head'.")

    # 3. Inference Loop
    all_embeddings = []
    all_labels_numpy = []
    all_logits_tensor = []
    all_labels_tensor = []
    
    all_metadata = {'img_ids': [], 'ages': [], 'genders': [], 'races': []}

    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"{model_name} Inference"):
            imgs = batch["full_img"].to(device)
            labels = batch["insurance"].to(device)

            logits = model_instance(imgs)
            
            # Capture embedding from hook
            if extractor.features is not None:
                embeddings = extractor.features
            else:
                raise RuntimeError("Hook failed to capture features.")

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels_numpy.append(labels.cpu().numpy())
            all_logits_tensor.append(logits.cpu())
            all_labels_tensor.append(labels.cpu())

            if "img_id" in batch: all_metadata['img_ids'].extend(batch["img_id"])
            if "age" in batch: all_metadata['ages'].append(batch["age"].numpy())
            if "gender" in batch: all_metadata['genders'].append(batch["gender"].numpy())
            if "race" in batch: all_metadata['races'].append(batch["race"].numpy())

    if handle:
        handle.remove()

    # 4. Calculate Metrics
    print(f"Calculating metrics for {model_name}...")
    preds_cat = torch.cat(all_logits_tensor, dim=0)
    targets_cat = torch.cat(all_labels_tensor, dim=0)
    
    scores = calculate_metrics(preds_cat, targets_cat)
    
    # Print Results
    print(f"--- Results for {model_name} (Per Class) ---")
    num_classes_found = len(next(iter(scores.values())))
    
    # Header
    header = f"{'Metric':<15}" + "".join([f" | Class {i:<8}" for i in range(num_classes_found)]) + " | MACRO AVG"
    print(header)
    print("-" * len(header))

    for k, v in scores.items():
        # Calculate macro average manually for display to match your first script
        macro_avg = np.mean(v)
        values_str = "".join([f" | {val:.4f}    " for val in v])
        print(f"{k:<15}{values_str} | {macro_avg:.4f}")

    # 5. Save to NPZ
    embeddings_array = np.concatenate(all_embeddings, axis=0)
    labels_array = np.concatenate(all_labels_numpy, axis=0)
    
    save_path = os.path.join(output_dir, f"{model_name}_embeddings.npz")
    
    save_dict = {
        'embeddings': embeddings_array,
        'labels': labels_array,
        'metrics': scores 
    }
    
    if all_metadata['img_ids']: save_dict['img_ids'] = np.array(all_metadata['img_ids'])
    if all_metadata['ages']: save_dict['ages'] = np.concatenate(all_metadata['ages'], axis=0)
    if all_metadata['genders']: save_dict['genders'] = np.concatenate(all_metadata['genders'], axis=0)
    if all_metadata['races']: save_dict['races'] = np.concatenate(all_metadata['races'], axis=0)

    np.savez_compressed(save_path, **save_dict)
    print(f"Saved embeddings to: {save_path}")
    
    # Cleanup
    del model_instance, preds_cat, targets_cat, all_embeddings, all_logits_tensor
    torch.cuda.empty_cache()

# ============================================================================
# 4. Main Execution
# ============================================================================

if __name__ == "__main__":
    
    # 1. Shared Parameters
    TEST_CSV_PATH = "./insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv"
    OUTPUT_DIR = "/mnt/new_usb/jupyter-altis5526/embeddings/all_models_test_set_128dim"
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    NUM_CLASSES = 2
    SEED = 123
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set Seed
    set_seed(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Dataset Statistics
    # *** FIX APPLIED HERE: Added resize=448 and removed transform=False ***
    print("Loading Dataset...")
    test_dataset = MIMIC_raw(TEST_CSV_PATH, resize=448, transform=True)
    
    g = torch.Generator()
    g.manual_seed(SEED)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    print(f"Total Test Samples: {len(test_dataset)}")
    
    # 3. Define Models
    model_configs = [
        {
            "name": "densenet121",
            "model": DenseNetWithDoubleLinear(num_classes=NUM_CLASSES, dropout_prob=0),
            "path": "/mnt/new_usb/jupyter-altis5526/new_insurancetype_weight/ValAug_TestNoAug_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_densenet_DoubleLinear_Lion1e-5_20251120/Rand123/Rand123_ValAug_TestNoAug_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_densenet_DoubleLinear_Lion1e-5_20251120_model_aucbest.pt"
        },
        {
            "name": "swin_transformer",
            "model": SwinTDoubleLinear(use_checkpoint=False, num_classes=NUM_CLASSES),
            "path": "/mnt/new_usb/jupyter-altis5526/new_insurancetype_weight/Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_swinTF_DoubleLinear_Lion1e-5_20251025/Rand123/Rand123_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_swinTF_DoubleLinear_Lion1e-5_20251025_model_aucbest.pt"
        },
        {
            "name": "medmamba",
            "model": medmamba(num_classes=NUM_CLASSES),
            "path": "/mnt/new_usb/jupyter-altis5526/new_insurancetype_weight/Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_mamba_DoubleLinear_Lion1e-5_20251025/Rand123/Rand123_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_mamba_DoubleLinear_Lion1e-5_20251025_model_aucbest.pt" 
        }
    ]

    # 4. Execution Loop
    for config in model_configs:
        try:
            run_extraction_for_model(
                model_name=config["name"],
                model_instance=config["model"],
                checkpoint_path=config["path"],
                data_loader=test_loader,
                device=device,
                output_dir=OUTPUT_DIR
            )
        except Exception as e:
            print(f"Error processing {config['name']}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\nAll models processed successfully.")