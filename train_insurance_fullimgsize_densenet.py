import torch
from model import *
import numpy as np
import os
import random
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import torchvision.models as models
from torchmetrics.classification import MultilabelAveragePrecision, MulticlassAUROC, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import wandb
from torch.optim.lr_scheduler import ExponentialLR
from model import DenseNetClassification, DenseNetWithDoubleLinear
from lion_pytorch import Lion
from RawImageDataset import MIMIC_raw
import argparse

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
    
    
def evaluate(model, val_loader, num_classes):
    model.eval()
    test_running_loss = 0.0
    test_total = 0
    correct = 0
    
    with torch.no_grad():
        record_target_label = torch.zeros(1, num_classes).to(device)
        record_predict_label = torch.zeros(1, num_classes).to(device)
        
        for batch in val_loader:
            test_imgs = batch["full_img"]
            test_labels = batch["insurance"]
            age = batch["age"]
            gender = batch["gender"]
            race = batch["race"]
            test_imgs = test_imgs.to(device)
            test_labels = test_labels.to(device)
            test_labels = test_labels.squeeze(-1)

            # _, age_label = torch.max(age, 1)
            # _, race_label = torch.max(race, 1)
            # filter = (age_label == 2)
            # filter = (race_label == 2)
            # filter = (gender == 1)
            # test_imgs = test_imgs[filter.flatten()]
            # test_labels = test_labels[filter.flatten()]

            if test_imgs.size(0) == 0:
                continue
            
            test_output = model(test_imgs)
            loss = criterion(test_output, test_labels)
            loss = loss.mean()
            test_running_loss += loss.item() * test_imgs.size(0)
            test_total += test_imgs.size(0)
            
            record_target_label = torch.cat((record_target_label, test_labels), 0)
            record_predict_label = torch.cat((record_predict_label, test_output), 0)
            
            _, test_label_transform = torch.max(test_labels, 1)
            _, test_output_transform = torch.max(test_output, 1)
            
            correct += (test_label_transform==test_output_transform).sum()
            
        
        record_target_label = record_target_label[1::]
        record_predict_label = record_predict_label[1::]
        
        _, one_label_target = torch.max(record_target_label, 1)
        
        auc_metric = MulticlassAUROC(num_classes=num_classes, average="macro", thresholds=None).to(device)
        precision_metric = MulticlassPrecision(num_classes=num_classes, average="macro").to(device)
        recall_metric = MulticlassRecall(num_classes=num_classes, average="macro").to(device)
        f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
        
        auc = auc_metric(record_predict_label, one_label_target)
        precision = precision_metric(record_predict_label, one_label_target)
        recall = recall_metric(record_predict_label, one_label_target)
        f1 = f1_metric(record_predict_label, one_label_target)
        acc = correct / test_total
        
        
        
    return auc, precision, recall, f1, acc, test_running_loss, test_total
            

if __name__ == "__main__":
    ## python train_insurance_fullimgsize_densenet.py --train_path insurance_dataset_8_1_1_PMMthree_train_no_support_devices.csv --val_path insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv --experiment_name Rand789_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_densenet121_SingleLinear_Lion1e-5_20251024 --weight_dir /mnt/new_usb/jupyter-altis5526/new_insurancetype_weight/Rand789_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_densenet121_SingleLinear_Lion1e-5_20251024 --mode test --seed 789
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", help='', type=str)
    parser.add_argument("--val_path", help='', type=str)
    parser.add_argument("--experiment_name", help='', type=str)
    parser.add_argument("--weight_dir", help='', type=str)
    parser.add_argument("--mode", help='', type=str)
    parser.add_argument("--seed", help='', type=int)
    parser.add_argument("--resize", help='', type=int, default=448)
    parser.add_argument("--root_dir", help='', type=str, default="/mnt/new_usb/jupyter-altis5526/new_insurancetype_weight/MedGemma_checked/")
    parser.add_argument("--subgroup", help='', type=str, default="all")
    
    args = parser.parse_args()

    set_seed(args.seed)
    torch.cuda.set_device(0)

    # Set up directories
    weight_dir = os.path.join(args.root_dir, args.weight_dir, f'Rand{str(args.seed)}')
    wandb_dir = os.path.join(args.root_dir, 'wandb')

    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)

    train_path = args.train_path
    val_path = args.val_path
    train_wandb_name = f'Rand{str(args.seed)}' + '_' + args.experiment_name
    val_wandb_name = f'Test_Rand{str(args.seed)}' + '_' + args.experiment_name

    if args.mode == "train":
        training = True
    elif args.mode == "test":
        training = False
    
    epochs = 100
    batch_size = 32
    num_classes = 2
    
    opt_lr = 5e-6
    weight_decay = 0
    img_size = 448
    
    dropout_prob = 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = DenseNetWithDoubleLinear(num_classes=num_classes, dropout_prob=dropout_prob)
    encoder.to(device)
    
    g = torch.Generator()
    g.manual_seed(0)
    opt = Lion(encoder.parameters(), lr=opt_lr, weight_decay = weight_decay)
    upscale_to = 448 if args.resize != 448 else None
    train_dataset = MIMIC_raw(train_path, resize=args.resize, transform=True, upscale_to=upscale_to)
    val_dataset = MIMIC_raw(val_path, resize=args.resize, transform=False, upscale_to=upscale_to)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, worker_init_fn=seed_worker, num_workers=8, shuffle=True, generator=g, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, worker_init_fn=seed_worker, num_workers=8, shuffle=False, generator=g)
    
    criterion = nn.CrossEntropyLoss()
    
    testing_weight_path = f"{weight_dir}/{train_wandb_name}_model_aucbest.pt"
    status_file = f"{weight_dir}/{train_wandb_name}_STATUS.txt"

    if training == False:
        if not os.path.exists(testing_weight_path):
            print(f"\n{'='*50}")
            print(f"ERROR: Checkpoint not found for testing!")
            print(f"Expected path: {testing_weight_path}")
            print(f"Please run training first before testing.")
            print(f"{'='*50}\n")
            exit(1)
        encoder.load_state_dict(torch.load(testing_weight_path)["model_state_dict"])

    if training == True:
        # Check if experiment has already been completed
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status_content = f.read()
                if 'STATUS: COMPLETED' in status_content:
                    print(f"\n{'='*70}")
                    print(f"EXPERIMENT ALREADY COMPLETED: {train_wandb_name}")
                    print(f"Status file: {status_file}")
                    print(f"Skipping this experiment...")
                    print(f"{'='*70}\n")
                    exit(0)

        # Create status file to track training completion
        os.makedirs(weight_dir, exist_ok=True)
        with open(status_file, 'w') as f:
            f.write(f"STATUS: RUNNING\n")
            f.write(f"MODEL: DenseNet\n")
            f.write(f"EXPERIMENT: {train_wandb_name}\n")
            f.write(f"START_TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"TOTAL_EPOCHS: {epochs}\n")
            f.write(f"CURRENT_EPOCH: 0/{epochs}\n")

        wandb.init(
            project='insurance_classification',
            name= train_wandb_name,
            dir=wandb_dir,
            resume="allow",
            settings=wandb.Settings(start_method="fork"))
        config = wandb.config
        config.batch_size = batch_size
        config.opt_lr = opt_lr
        config.weight_decay = weight_decay
        config.dropout = dropout_prob
        config.weight_path = weight_dir
        config.image_size = img_size
        max_auc = 0
        max_acc = 0
        total = 0
        scaler = torch.cuda.amp.GradScaler()
        stop_criteria = 0
        start_epoch = 0

        # Check if resuming from previous training
        print(f"\n[DEBUG] Checking for checkpoint at: {testing_weight_path}")
        print(f"[DEBUG] File exists: {os.path.exists(testing_weight_path)}")
        if os.path.exists(testing_weight_path):
            checkpoint = torch.load(testing_weight_path)
            print(f"[DEBUG] Checkpoint loaded successfully")
            print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys())}")
            print(f"[DEBUG] 'epoch' in checkpoint: {'epoch' in checkpoint}")
            if 'epoch' in checkpoint:  # This is a checkpoint with training state
                print(f"\n{'='*50}")
                print(f"RESUMING FROM PREVIOUS TRAINING")
                print(f"{'='*50}\n")
                encoder.load_state_dict(checkpoint['model_state_dict'])
                opt.load_state_dict(checkpoint['optimizer_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                # Restore random states for reproducibility
                if 'torch_rng_state' in checkpoint:
                    torch.set_rng_state(checkpoint['torch_rng_state'])
                    torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
                    np.random.set_state(checkpoint['numpy_rng_state'])
                    random.setstate(checkpoint['python_rng_state'])
                    g.set_state(checkpoint['generator_state'])
                start_epoch = checkpoint['epoch'] + 1
                max_auc = checkpoint['max_auc']
                max_acc = checkpoint['max_acc']
                stop_criteria = checkpoint['stop_criteria']
                print(f"Resuming from epoch {start_epoch}")
                print(f"Best AUC so far: {max_auc:.4f}")
                print(f"Best ACC so far: {max_acc:.4f}")
                print(f"Checkpoint file: {testing_weight_path}\n")
            else:
                print(f"\n{'='*50}")
                print(f"STARTING NEW TRAINING")
                print(f"{'='*50}\n")
        else:
            print(f"\n{'='*50}")
            print(f"STARTING NEW TRAINING")
            print(f"{'='*50}\n")

        try:
            for epoch in range(start_epoch, epochs):
                encoder.train()
                running_loss = 0.0
                start_time = time.time()
                count = 0

                for batch in train_loader:
                    encoder.zero_grad()
                    opt.zero_grad()
                    imgs = batch["full_img"]
                    labels = batch["insurance"]
                    # dicom_id = batch["img_id"]
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    labels = labels.squeeze(-1)

                    _, target_label = torch.max(labels, 1)

                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        output = encoder(imgs)
                        loss = criterion(output, labels)

                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()


                    running_loss += loss.item() * imgs.size(0)
                    count += imgs.size(0)

                    if count != 0 and count % 2560 == 0 and total == 0:
                        print(f"epoch {epoch}: {count}/unknown finished / train loss: {running_loss / count}")

                    elif count != 0 and count % 2560 == 0 and total != 0:
                        print(f"epoch {epoch}: {count}/{total} (%.2f %%) finished / train loss: {running_loss / count}" % (count/total*100))

                total = count
                auc, precision, recall, f1, acc, test_running_loss, test_total = evaluate(encoder, val_loader, num_classes)
                # scheduler.step()
                stop_criteria += 1

                # Save checkpoint only when AUC improves (best model only)
                if auc > max_auc:
                    max_auc = auc
                    stop_criteria = 0
                    # Save best model with all training state
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': encoder.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'max_auc': max_auc,
                        'max_acc': max_acc,
                        'stop_criteria': stop_criteria,
                        'auc': auc,
                        'acc': acc,
                        # Random states for reproducibility
                        'torch_rng_state': torch.get_rng_state(),
                        'cuda_rng_state': torch.cuda.get_rng_state_all(),
                        'numpy_rng_state': np.random.get_state(),
                        'python_rng_state': random.getstate(),
                        'generator_state': g.get_state(),
                    }, testing_weight_path)
                else:
                    # No improvement, but update the epoch number in the checkpoint if it exists
                    if os.path.exists(testing_weight_path):
                        checkpoint = torch.load(testing_weight_path)
                        if 'epoch' in checkpoint:
                            checkpoint['epoch'] = epoch
                            checkpoint['stop_criteria'] = stop_criteria
                            torch.save(checkpoint, testing_weight_path)

                if acc > max_acc:
                    max_acc = acc

                if stop_criteria >= 10:
                    # Update status file with early stopping info
                    with open(status_file, 'w') as f:
                        f.write(f"STATUS: STOPPED_EARLY\n")
                        f.write(f"MODEL: DenseNet\n")
                        f.write(f"EXPERIMENT: {train_wandb_name}\n")
                        f.write(f"END_TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"REASON: Early stopping (no improvement for 10 epochs)\n")
                        f.write(f"TOTAL_EPOCHS_PLANNED: {epochs}\n")
                        f.write(f"COMPLETED_EPOCHS: {epoch+1}\n")
                        f.write(f"BEST_AUC: {max_auc:.4f}\n")
                        f.write(f"BEST_ACC: {max_acc:.4f}\n")
                        f.write(f"WEIGHT_FILE: {testing_weight_path}\n")
                    print(f"\n{'='*50}")
                    print(f"EARLY STOPPING TRIGGERED")
                    print(f"No improvement for 10 epochs")
                    print(f"Status saved to: {status_file}")
                    print(f"Weights saved to: {testing_weight_path}")
                    print(f"{'='*50}\n")
                    break

                end_time = time.time()
                duration = end_time - start_time

                print(f"epoch {epoch} / AUC: {auc} / precision: {precision} / recall: {recall} / f1: {f1} / acc: {acc} / test loss: {test_running_loss / test_total} / duration: {duration}")

                wandb.log({'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1, 'acc': acc, 'testing_loss': test_running_loss / test_total})

                # Update status file with progress
                with open(status_file, 'w') as f:
                    f.write(f"STATUS: RUNNING\n")
                    f.write(f"MODEL: DenseNet\n")
                    f.write(f"EXPERIMENT: {train_wandb_name}\n")
                    f.write(f"START_TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"TOTAL_EPOCHS_PLANNED: {epochs}\n")
                    f.write(f"CURRENT_EPOCH: {epoch+1}/{epochs}\n")
                    f.write(f"LAST_AUC: {auc:.4f}\n")
                    f.write(f"LAST_ACC: {acc:.4f}\n")
                    f.write(f"BEST_AUC: {max_auc:.4f}\n")
                    f.write(f"BEST_ACC: {max_acc:.4f}\n")
                    f.write(f"WEIGHT_FILE: {testing_weight_path}\n")

            # Training completed successfully (all epochs done)
            with open(status_file, 'w') as f:
                f.write(f"STATUS: COMPLETED\n")
                f.write(f"MODEL: DenseNet\n")
                f.write(f"EXPERIMENT: {train_wandb_name}\n")
                f.write(f"END_TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"TOTAL_EPOCHS_PLANNED: {epochs}\n")
                f.write(f"COMPLETED_EPOCHS: {epoch+1}\n")
                f.write(f"FINAL_AUC: {auc:.4f}\n")
                f.write(f"FINAL_ACC: {acc:.4f}\n")
                f.write(f"BEST_AUC: {max_auc:.4f}\n")
                f.write(f"BEST_ACC: {max_acc:.4f}\n")
                f.write(f"WEIGHT_FILE: {testing_weight_path}\n")
            print(f"\n{'='*50}")
            print(f"TRAINING COMPLETED SUCCESSFULLY!")
            print(f"All {epoch+1} epochs finished")
            print(f"Status saved to: {status_file}")
            print(f"Weights saved to: {testing_weight_path}")
            print(f"{'='*50}\n")

        except Exception as e:
            # Training failed due to error (not cluster timeout)
            with open(status_file, 'w') as f:
                f.write(f"STATUS: FAILED\n")
                f.write(f"MODEL: DenseNet\n")
                f.write(f"EXPERIMENT: {train_wandb_name}\n")
                f.write(f"ERROR_TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"COMPLETED_EPOCHS: {epoch if 'epoch' in locals() else 0}\n")
                f.write(f"ERROR: {str(e)}\n")
                f.write(f"WEIGHT_FILE: {testing_weight_path}\n")
            print(f"\n{'='*50}")
            print(f"TRAINING FAILED!")
            print(f"Error: {e}")
            print(f"Status saved to: {status_file}")
            print(f"Weights saved to: {testing_weight_path}")
            print(f"{'='*50}\n")
            raise
            
            
    if training == False:
        wandb.init(
            project='insurance_classification',
            name= val_wandb_name,
            dir=wandb_dir,
            resume="allow",
            settings=wandb.Settings(start_method="fork"))
        config = wandb.config
        config.batch_size = batch_size
        config.test_weight = testing_weight_path

        auc, precision, recall, f1, acc, test_running_loss, test_total = evaluate(encoder, val_loader, num_classes)

        print(f"AUC: {auc} / precision: {precision} / recall: {recall} / f1: {f1} / acc: {acc} / test loss: {test_running_loss / test_total}")

        wandb.log({'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1, 'acc': acc, 'testing_loss': test_running_loss / test_total})
            
    
        
        
        
                
                
                