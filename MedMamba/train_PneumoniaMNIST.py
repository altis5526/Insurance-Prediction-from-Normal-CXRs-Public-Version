import os
import sys
import json
from datetime import datetime

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import roc_auc_score, classification_report

from MedMamba import VSSM as medmamba


class Tee:
    """Mirrors writes to both a file and the original stream."""
    def __init__(self, stream, filepath):
        self.stream = stream
        self.file = open(filepath, "w", buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()

    # Delegate attribute lookups (e.g. isatty) to the wrapped stream
    def __getattr__(self, name):
        return getattr(self.stream, name)


NPZ_PATH = "/mnt/new_usb/jupyter-altis5526/pneumoniamnist.npz"
NUM_CLASSES = 2
MODEL_NAME = "MedMamba_S_PneumoniaMNIST"


class PneumoniaMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        # images: (N, 28, 28) uint8 grayscale
        # labels: (N, 1) uint8
        self.images = images
        self.labels = labels.squeeze(1).astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def main():
    log_path = "./{}_log_{}.txt".format(MODEL_NAME, datetime.now().strftime("%Y%m%d_%H%M%S"))
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee
    print("Log file: {}".format(log_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    }

    data = np.load(NPZ_PATH)

    train_dataset = PneumoniaMNISTDataset(
        data["train_images"], data["train_labels"], transform=data_transform["train"]
    )
    val_dataset = PneumoniaMNISTDataset(
        data["val_images"], data["val_labels"], transform=data_transform["val"]
    )
    train_num = len(train_dataset)
    val_num = len(val_dataset)

    cla_dict = {0: "normal", 1: "pneumonia"}
    with open("class_indices.json", "w") as f:
        json.dump(cla_dict, f, indent=4)

    batch_size = 128
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print("Using {} dataloader workers every process".format(nw))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw
    )
    validate_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw
    )
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    net = medmamba(depths=[2, 2, 8, 2], dims=[96, 192, 384, 768], num_classes=NUM_CLASSES)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    epochs = 100
    best_acc = 0.0
    save_path = "./{}.pth".format(MODEL_NAME)
    train_steps = len(train_loader)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, batch in enumerate(train_bar):
            images, labels = batch
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(
                epoch + 1, epochs, loss
            )
        scheduler.step()

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_images, val_labels in val_bar:
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print(
            "[epoch %d] train_loss: %.3f  val_accuracy: %.3f"
            % (epoch + 1, running_loss / train_steps, val_accurate)
        )

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print("Finished Training")

    # --- Test set evaluation using best checkpoint ---
    test_dataset = PneumoniaMNISTDataset(
        data["test_images"], data["test_labels"], transform=data_transform["val"]
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=nw
    )

    net.load_state_dict(torch.load(save_path, map_location=device))
    net.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout, desc="Testing")
        for test_images, test_labels in test_bar:
            outputs = net(test_images.to(device))
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.max(outputs, dim=1)[1]
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(test_labels.numpy())

    test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    test_auc = roc_auc_score(all_labels, all_probs)
    print("\n===== Test Set Results =====")
    print("Test Accuracy: {:.4f}".format(test_acc))
    print("Test AUC:      {:.4f}".format(test_auc))
    print(classification_report(all_labels, all_preds, target_names=["normal", "pneumonia"]))

    sys.stdout = tee.stream
    tee.close()
    print("Training log saved to: {}".format(log_path))


if __name__ == "__main__":
    main()
