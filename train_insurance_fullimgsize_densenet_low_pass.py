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
from RawImageDataset import MIMIC_raw_low_pass
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
    parser.add_argument("--seed", help='', type=int)
    parser.add_argument("--root_dir", help='', type=str, default="/mnt/new_usb/jupyter-altis5526/new_insurancetype_weight/")
    parser.add_argument("--frequency", help='Low pass frequency/diameter', type=int)

    args = parser.parse_args()

    set_seed(args.seed)
    torch.cuda.set_device(0)
    weight_dir = os.path.join(args.root_dir+args.weight_dir, f'Rand{str(args.seed)}')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

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

    if args.mode == "train":
        train_dataset = MIMIC_raw_low_pass(train_path, resize=img_size, diameter=args.frequency, transform=True)
        val_dataset = MIMIC_raw_low_pass(val_path, resize=img_size, diameter=args.frequency, transform=False)
    elif args.mode == "test":
        train_dataset = MIMIC_raw_low_pass(train_path, resize=img_size, diameter=args.frequency, transform=True)
        val_dataset = MIMIC_raw_low_pass(val_path, resize=img_size, diameter=args.frequency, transform=False)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, worker_init_fn=seed_worker, num_workers=8, shuffle=True, generator=g, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, worker_init_fn=seed_worker, num_workers=8, shuffle=False, generator=g)
    
    criterion = nn.CrossEntropyLoss()
    
    testing_weight_path = f"{weight_dir}/{train_wandb_name}_model_aucbest.pt"

    # STATUS.txt file for tracking completion
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

        # Check if training is still running
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

        # Check for existing checkpoint to resume
        checkpoint_path = f"{weight_dir}/{train_wandb_name}_checkpoint.pt"
        start_epoch = 0
        max_auc = 0
        max_acc = 0
        stop_criteria = 0

        if os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            encoder.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            max_auc = checkpoint.get('max_auc', 0)
            stop_criteria = checkpoint.get('stop_criteria', 0)
            print(f"Resuming from epoch {start_epoch}, max_auc={max_auc}")

        wandb.init(
            project='insurance_classification',
            name=train_wandb_name,
            resume="allow",
            settings=wandb.Settings(start_method="fork"))
        config = wandb.config
        config.batch_size = batch_size
        config.opt_lr = opt_lr
        config.weight_decay = weight_decay
        config.dropout = dropout_prob
        config.weight_path = weight_dir
        config.image_size = img_size
        config.frequency = args.frequency

        # Write initial status
        with open(status_file, 'w') as f:
            f.write(f"STATUS: RUNNING\n")
            f.write(f"START_EPOCH: {start_epoch}\n")
            f.write(f"FREQUENCY: {args.frequency}\n")

        total = 0
        scaler = torch.cuda.amp.GradScaler()

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
            
            if auc > max_auc:
                max_auc = auc
                torch.save({
                    'model_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }, f"{weight_dir}/{train_wandb_name}_model_aucbest.pt")
                stop_criteria = 0

            # Save checkpoint for resuming
            torch.save({
                'epoch': epoch,
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'max_auc': max_auc,
                'stop_criteria': stop_criteria,
            }, checkpoint_path)

            end_time = time.time()
            duration = end_time - start_time

            if stop_criteria >= 10:
                # Update status file for early stopping
                with open(status_file, 'w') as f:
                    f.write(f"STATUS: STOPPED_EARLY\n")
                    f.write(f"FINAL_EPOCH: {epoch}\n")
                    f.write(f"BEST_AUC: {max_auc}\n")
                    f.write(f"FREQUENCY: {args.frequency}\n")
                break

            print(f"epoch {epoch} / AUC: {auc} / precision: {precision} / recall: {recall} / f1: {f1} / acc: {acc} / test loss: {test_running_loss / test_total} / duration: {duration}")

            wandb.log({'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1, 'acc': acc, 'testing_loss': test_running_loss / test_total})

        # Training completed all epochs (if not early stopped)
        else:
            with open(status_file, 'w') as f:
                f.write(f"STATUS: COMPLETED\n")
                f.write(f"FINAL_EPOCH: {epochs-1}\n")
                f.write(f"BEST_AUC: {max_auc}\n")
                f.write(f"FREQUENCY: {args.frequency}\n")

    if training == False:
        wandb.init(
            project='insurance_classification',
            name=val_wandb_name,
            settings=wandb.Settings(start_method="fork"))
        config = wandb.config
        config.batch_size = batch_size
        config.test_weight = testing_weight_path
        config.img_size = img_size
        config.frequency = args.frequency

        auc, precision, recall, f1, acc, test_running_loss, test_total = evaluate(encoder, val_loader, num_classes)

        print(f"AUC: {auc} / precision: {precision} / recall: {recall} / f1: {f1} / acc: {acc} / test loss: {test_running_loss / test_total}")

        wandb.log({'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1, 'acc': acc, 'testing_loss': test_running_loss / test_total})

        # Append test results to STATUS.txt
        with open(status_file, 'a') as f:
            f.write(f"\n--- TEST RESULTS ---\n")
            f.write(f"PAPER_REPORTABLE_AUC: {auc}\n")
            f.write(f"TEST_AUC: {auc}\n")
            f.write(f"TEST_PRECISION: {precision}\n")
            f.write(f"TEST_RECALL: {recall}\n")
            f.write(f"TEST_F1: {f1}\n")
            f.write(f"TEST_ACC: {acc}\n")
            
    
        
        
        
                
                
                