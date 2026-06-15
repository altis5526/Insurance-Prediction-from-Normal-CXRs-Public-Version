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
from RawImageDataset import MIMIC_raw_ICD_mask_mostimage
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
    parser.add_argument("--patch_idx", help='', type=int)
    parser.add_argument("--seed", help='', type=int)
    parser.add_argument("--root_dir", help='', type=str, default="/mnt/new_usb/jupyter-altis5526/new_insurancetype_weight/MedGemma_checked/")
    
    args = parser.parse_args()

    set_seed(args.seed)
    torch.cuda.set_device(1)
    weight_dir = os.path.join(args.root_dir+args.weight_dir, f'Rand{str(args.seed)}')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    train_path = args.train_path
    val_path = args.val_path
    train_wandb_name = f'Rand{str(args.seed)}_patchidx{args.patch_idx}' + '_' + args.experiment_name
    val_wandb_name = f'Test_Rand{str(args.seed)}_patchidx{args.patch_idx}' + '_' + args.experiment_name

    if args.mode == "train":
        training = True
    elif args.mode == "test":
        training = False

    if args.patch_idx == 1:
        starting_point = [0,0]
        height = 150
        width = 150
        
    elif args.patch_idx == 2:
        starting_point = [0,150]
        height = 150
        width = 150

    elif args.patch_idx == 3:
        starting_point = [0,300]
        height = 150
        width = 148

    elif args.patch_idx == 4:
        starting_point = [150,0]
        height = 150
        width = 150

    elif args.patch_idx == 5:
        starting_point = [150,150]
        height = 150
        width = 150

    elif args.patch_idx == 6:
        starting_point = [150,300]
        height = 150
        width = 148

    elif args.patch_idx == 7:
        starting_point = [300,0]
        height = 148
        width = 150

    elif args.patch_idx == 8:
        starting_point = [300,150]
        height = 148
        width = 150

    elif args.patch_idx == 9:
        starting_point = [300,300]
        height = 148
        width = 148
    
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
    if args.mode == "train":
        train_dataset = MIMIC_raw_ICD_mask_mostimage(train_path, starting_point, height, width, transform=True)
        val_dataset = MIMIC_raw_ICD_mask_mostimage(val_path, starting_point, height, width, transform=False)

    elif args.mode == "test":
        train_dataset = MIMIC_raw_ICD_mask_mostimage(train_path, starting_point, height, width, transform=True)
        val_dataset = MIMIC_raw_ICD_mask_mostimage(val_path, starting_point, height, width, transform=False)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, worker_init_fn=seed_worker, num_workers=8, shuffle=True, generator=g, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, worker_init_fn=seed_worker, num_workers=8, shuffle=False, generator=g)
    
    criterion = nn.CrossEntropyLoss()

    testing_weight_path = f"{weight_dir}/{train_wandb_name}_model_aucbest.pt"
    status_file = f"{weight_dir}/{train_wandb_name}_STATUS.txt"

    if training == False:
        # Check if checkpoint exists
        if not os.path.exists(testing_weight_path):
            print(f"\n{'='*70}")
            print(f"ERROR: Checkpoint not found for testing!")
            print(f"Expected path: {testing_weight_path}")
            print(f"Please run training first before testing.")
            print(f"{'='*70}\n")
            exit(1)

        # Check if training is actually completed
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status_content = f.read()
                if 'STATUS: RUNNING' in status_content:
                    print(f"\n{'='*70}")
                    print(f"ERROR: Training is still RUNNING!")
                    print(f"Experiment: {train_wandb_name}")
                    print(f"Status file: {status_file}")
                    print(f"Cannot run tests on incomplete training.")
                    print(f"Please wait for training to complete first.")
                    print(f"{'='*70}\n")
                    exit(1)

        encoder.load_state_dict(torch.load(testing_weight_path)["model_state_dict"])
        
    if training == True:
        # Check if experiment has already been completed
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status_content = f.read()
                if 'STATUS: COMPLETED' in status_content or 'STATUS: STOPPED_EARLY' in status_content:
                    if 'PAPER_REPORTABLE_AUC:' in status_content or 'TEST_AUC:' in status_content:
                        print(f"\n{'='*70}")
                        print(f"EXPERIMENT ALREADY COMPLETED: {train_wandb_name}")
                        print(f"Status file: {status_file}")
                        print(f"Skipping this experiment...")
                        print(f"{'='*70}\n")
                        exit(0)

        # Create status file to track training completion
        os.makedirs(weight_dir, exist_ok=True)
        training_start_time = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(status_file, 'w') as f:
            f.write(f"STATUS: RUNNING\n")
            f.write(f"MODEL: DenseNet\n")
            f.write(f"EXPERIMENT: {train_wandb_name}\n")
            f.write(f"START_TIME: {training_start_time}\n")
            f.write(f"TOTAL_EPOCHS: {epochs}\n")
            f.write(f"CURRENT_EPOCH: 0/{epochs}\n")

        wandb.init(
            project='insurance_classification',
            name= train_wandb_name,
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

        for epoch in range(epochs):
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
            
            if auc > max_auc:
                max_auc = auc
                torch.save({
                    'model_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }, f"{weight_dir}/{train_wandb_name}_model_aucbest.pt")

                stop_criteria = 0
                
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"epoch {epoch} / AUC: {auc} / precision: {precision} / recall: {recall} / f1: {f1} / acc: {acc} / test loss: {test_running_loss / test_total} / duration: {duration}")

            wandb.log({'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1, 'acc': acc, 'testing_loss': test_running_loss / test_total})

            # Update status file with progress
            with open(status_file, 'w') as f:
                f.write(f"STATUS: RUNNING\n")
                f.write(f"MODEL: DenseNet\n")
                f.write(f"EXPERIMENT: {train_wandb_name}\n")
                f.write(f"START_TIME: {training_start_time}\n")
                f.write(f"TOTAL_EPOCHS_PLANNED: {epochs}\n")
                f.write(f"CURRENT_EPOCH: {epoch+1}/{epochs}\n")
                f.write(f"LAST_AUC: {auc:.4f}\n")
                f.write(f"LAST_ACC: {acc:.4f}\n")
                f.write(f"BEST_AUC: {max_auc:.4f}\n")
                f.write(f"BEST_ACC: {max_acc:.4f}\n")
                f.write(f"WEIGHT_FILE: {testing_weight_path}\n")

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
                break

        # If training completed all epochs without early stopping
        if stop_criteria < 10:
            with open(status_file, 'w') as f:
                f.write(f"STATUS: COMPLETED\n")
                f.write(f"MODEL: DenseNet\n")
                f.write(f"EXPERIMENT: {train_wandb_name}\n")
                f.write(f"END_TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"TOTAL_EPOCHS: {epochs}\n")
                f.write(f"BEST_AUC: {max_auc:.4f}\n")
                f.write(f"BEST_ACC: {max_acc:.4f}\n")
                f.write(f"WEIGHT_FILE: {testing_weight_path}\n")
            
    if training == False:
        wandb.init(
            project='insurance_classification',
            name= val_wandb_name,
            settings=wandb.Settings(start_method="fork"))
        config = wandb.config
        config.batch_size = batch_size
        config.test_weight = testing_weight_path

        auc, precision, recall, f1, acc, test_running_loss, test_total = evaluate(encoder, val_loader, num_classes)

        print(f"AUC: {auc} / precision: {precision} / recall: {recall} / f1: {f1} / acc: {acc} / test loss: {test_running_loss / test_total}")

        wandb.log({'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1, 'acc': acc, 'testing_loss': test_running_loss / test_total})

        # Append test results to status file
        if os.path.exists(status_file):
            with open(status_file, 'a') as f:
                f.write(f"\n--- TEST RESULTS ---\n")
                f.write(f"TEST_TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"TEST_AUC: {auc:.4f}\n")
                f.write(f"PAPER_REPORTABLE_AUC: {auc:.4f}\n")
                f.write(f"TEST_PRECISION: {precision:.4f}\n")
                f.write(f"TEST_RECALL: {recall:.4f}\n")
                f.write(f"TEST_F1: {f1:.4f}\n")
                f.write(f"TEST_ACC: {acc:.4f}\n")
            
    
        
        
        
                
                
                
