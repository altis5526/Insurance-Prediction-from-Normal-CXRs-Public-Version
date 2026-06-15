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
from model import DenseNetWithDoubleLinear
from lion_pytorch import Lion
from RawImageDataset import MIMIC_raw_high_pass
from MedMamba.MedMamba import VSSM_DoubleLinear as medmamba
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
            gender = batch["gender"]
            age = batch["age"]
            race = batch["race"]
            test_imgs = test_imgs.to(device)
            test_labels = test_labels.to(device)
            test_labels = test_labels.squeeze(-1)

            _, age_label = torch.max(age, 1)
            _, race_label = torch.max(race, 1)
            # filter = (race_label == 2)
            # filter = (age_label == 2)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", help='', type=str)
    parser.add_argument("--val_path", help='', type=str)
    parser.add_argument("--experiment_name", help='', type=str)
    parser.add_argument("--weight_dir", help='', type=str)
    parser.add_argument("--mode", help='', type=str)
    parser.add_argument("--seed", help='', type=int)
    parser.add_argument("--resize", help='', type=int, default=448)
    parser.add_argument("--root_dir", help='', type=str, default="/home/sebasmos/orcd/pool/code/JAMA_codes/")
    parser.add_argument("--high_pass_diameter", help='', type=int)
    parser.add_argument("--pretrained_path", help='Path to pretrained .pth weights for encoder init (optional)',
                        type=str, default="/mnt/new_usb/jupyter-altis5526/insurance_paper_weights/MedMamba_pretrained_weight/MedMamba_S_PneumoniaMNIST.pth")

    args = parser.parse_args()

    set_seed(args.seed)
    torch.cuda.set_device(0)  # Use device 0 (SLURM assigns GPU via CUDA_VISIBLE_DEVICES)
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
    opt_lr = 1e-5
    weight_decay = 0
    dropout_prob = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = medmamba(depths=[2, 2, 8, 2], dims=[96, 192, 384, 768], num_classes=num_classes)
    encoder.to(device)

    checkpoint = torch.load(args.pretrained_path, map_location=device)
    model_state = encoder.state_dict()
    filtered = {k: v for k, v in checkpoint.items() if k in model_state and v.shape == model_state[k].shape}
    missing, unexpected = encoder.load_state_dict(filtered, strict=False)
    print(f"Pretrained weights loaded: {len(filtered)}/{len(checkpoint)} keys matched. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    g = torch.Generator()
    g.manual_seed(0)
    opt = Lion(encoder.parameters(), lr=opt_lr, weight_decay = weight_decay)
    
    train_dataset = MIMIC_raw_high_pass(train_path, resize=args.resize, diameter=args.high_pass_diameter, transform=True)
    val_dataset = MIMIC_raw_high_pass(val_path, resize=args.resize, diameter=args.high_pass_diameter, transform=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, worker_init_fn=seed_worker, num_workers=8, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, worker_init_fn=seed_worker, num_workers=8, shuffle=False, generator=g)

    criterion = nn.CrossEntropyLoss()

    testing_weight_path = f"{weight_dir}/{train_wandb_name}_model_aucbest.pt"
    status_file = f"{weight_dir}/{train_wandb_name}_STATUS.txt"

    if training == False:
        # Check if checkpoint exists
        if not os.path.exists(testing_weight_path):
            print(f"\n{'='*70}")
            print(f"ERROR: No trained model found!")
            print(f"Expected checkpoint: {testing_weight_path}")
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

        # Check for checkpoint resumption (interrupted training)
        resume_checkpoint_path = f"{weight_dir}/{train_wandb_name}_checkpoint_latest.pt"
        start_epoch = 0
        max_auc = 0
        max_acc = 0
        stop_criteria = 0

        if os.path.exists(resume_checkpoint_path) and os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status_content = f.read()
                if 'STATUS: RUNNING' in status_content:
                    print(f"\n{'='*70}")
                    print(f"RESUMING INTERRUPTED TRAINING: {train_wandb_name}")
                    print(f"Loading checkpoint: {resume_checkpoint_path}")
                    checkpoint = torch.load(resume_checkpoint_path)
                    encoder.load_state_dict(checkpoint['model_state_dict'])
                    opt.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    max_auc = checkpoint['max_auc']
                    max_acc = checkpoint['max_acc']
                    stop_criteria = checkpoint['stop_criteria']
                    print(f"Resuming from epoch {start_epoch}, best AUC: {max_auc:.4f}")
                    print(f"{'='*70}\n")

        training_start_time = time.strftime('%Y-%m-%d %H:%M:%S')
        if start_epoch == 0:  # Only write fresh status if not resuming
            with open(status_file, 'w') as f:
                f.write(f"STATUS: RUNNING\n")
                f.write(f"MODEL: Mamba (High Pass Filter)\n")
                f.write(f"EXPERIMENT: {train_wandb_name}\n")
                f.write(f"START_TIME: {training_start_time}\n")
                f.write(f"HIGH_PASS_DIAMETER: {args.high_pass_diameter}\n")

        wandb.init(
            project='insurance_classification',
            name= train_wandb_name,
            resume="allow",
            settings=wandb.Settings(start_method="fork"))
        config = wandb.config
        config.batch_size = batch_size
        config.opt_lr = opt_lr
        config.weight_decay = weight_decay
        config.dropout = dropout_prob
        config.weight_path = weight_dir
        config.high_pass_diameter = args.high_pass_diameter

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

            if acc > max_acc:
                max_acc = acc
                torch.save({
                    'model_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }, f"{weight_dir}/{train_wandb_name}_model_accbest.pt")

            # Save latest checkpoint for resumption
            torch.save({
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'epoch': epoch,
                'max_auc': max_auc,
                'max_acc': max_acc,
                'stop_criteria': stop_criteria,
            }, f"{weight_dir}/{train_wandb_name}_checkpoint_latest.pt")

            end_time = time.time()
            duration = end_time - start_time

            print(f"epoch {epoch} / AUC: {auc} / precision: {precision} / recall: {recall} / f1: {f1} / acc: {acc} / test loss: {test_running_loss / test_total} / duration: {duration}")

            wandb.log({'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1, 'acc': acc, 'testing_loss': test_running_loss / test_total})

            # Update status file with progress
            with open(status_file, 'w') as f:
                f.write(f"STATUS: RUNNING\n")
                f.write(f"MODEL: Mamba (High Pass Filter)\n")
                f.write(f"EXPERIMENT: {train_wandb_name}\n")
                f.write(f"START_TIME: {training_start_time}\n")
                f.write(f"HIGH_PASS_DIAMETER: {args.high_pass_diameter}\n")
                f.write(f"CURRENT_EPOCH: {epoch+1}/{epochs}\n")
                f.write(f"BEST_VAL_AUC: {max_auc:.4f}\n")
                f.write(f"BEST_VAL_ACC: {max_acc:.4f}\n")
                f.write(f"EARLY_STOP_COUNTER: {stop_criteria}/10\n")

            if stop_criteria >= 10:
                # Update status file with early stopping info
                with open(status_file, 'w') as f:
                    f.write(f"STATUS: STOPPED_EARLY\n")
                    f.write(f"MODEL: Mamba (High Pass Filter)\n")
                    f.write(f"EXPERIMENT: {train_wandb_name}\n")
                    f.write(f"END_TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"HIGH_PASS_DIAMETER: {args.high_pass_diameter}\n")
                    f.write(f"FINAL_EPOCH: {epoch+1}\n")
                    f.write(f"BEST_VAL_AUC: {max_auc:.4f}\n")
                    f.write(f"BEST_VAL_ACC: {max_acc:.4f}\n")
                    f.write(f"REASON: No improvement for 10 epochs\n")
                print(f"\n{'='*50}")
                print(f"EARLY STOPPING TRIGGERED")
                print(f"No improvement for 10 epochs")
                print(f"Status saved to: {status_file}")
                print(f"Weights saved to: {testing_weight_path}")
                print(f"{'='*50}\n")
                break

        # Training completed successfully (all epochs done)
        if stop_criteria < 10:  # Only if not early stopped
            with open(status_file, 'w') as f:
                f.write(f"STATUS: COMPLETED\n")
                f.write(f"MODEL: Mamba (High Pass Filter)\n")
                f.write(f"EXPERIMENT: {train_wandb_name}\n")
                f.write(f"END_TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"HIGH_PASS_DIAMETER: {args.high_pass_diameter}\n")
                f.write(f"TOTAL_EPOCHS: {epochs}\n")
                f.write(f"BEST_VAL_AUC: {max_auc:.4f}\n")
                f.write(f"BEST_VAL_ACC: {max_acc:.4f}\n")
            print(f"\n{'='*50}")
            print(f"TRAINING COMPLETED SUCCESSFULLY!")
            print(f"All {epoch+1} epochs finished")
            print(f"Status saved to: {status_file}")
            print(f"Weights saved to: {testing_weight_path}")
            print(f"{'='*50}\n")

    if training == False:
        wandb.init(
            project='insurance_classification',
            name= val_wandb_name,
            resume="allow",
            settings=wandb.Settings(start_method="fork"))
        config = wandb.config
        config.batch_size = batch_size
        config.test_weight = testing_weight_path
        config.img_size = 448
        config.high_pass_diameter = args.high_pass_diameter

        auc, precision, recall, f1, acc, test_running_loss, test_total = evaluate(encoder, val_loader, num_classes)

        print(f"\n{'='*70}")
        print(f"TEST RESULTS: {val_wandb_name}")
        print(f"{'='*70}")
        print(f"TEST AUC: {auc:.4f}  *** USE THIS FOR PAPER ***")
        print(f"TEST Precision: {precision:.4f}")
        print(f"TEST Recall: {recall:.4f}")
        print(f"TEST F1: {f1:.4f}")
        print(f"TEST Accuracy: {acc:.4f}")
        print(f"TEST Loss: {test_running_loss / test_total:.4f}")
        print(f"{'='*70}\n")

        wandb.log({'test_auc': auc, 'test_precision': precision, 'test_recall': recall, 'test_f1': f1, 'test_acc': acc, 'test_loss': test_running_loss / test_total})

        # Update STATUS.txt with final test results
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status_lines = f.readlines()

            # Add test results to status file
            with open(status_file, 'w') as f:
                for line in status_lines:
                    f.write(line)
                f.write(f"\n{'='*70}\n")
                f.write(f"FINAL TEST RESULTS (Held-out Test Set)\n")
                f.write(f"{'='*70}\n")
                f.write(f"PAPER_REPORTABLE_AUC: {auc:.4f}  *** USE THIS FOR PAPER ***\n")
                f.write(f"TEST_AUC: {auc:.4f}\n")
                f.write(f"TEST_PRECISION: {precision:.4f}\n")
                f.write(f"TEST_RECALL: {recall:.4f}\n")
                f.write(f"TEST_F1: {f1:.4f}\n")
                f.write(f"TEST_ACC: {acc:.4f}\n")
                f.write(f"TEST_LOSS: {test_running_loss / test_total:.4f}\n")
                f.write(f"TEST_TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
