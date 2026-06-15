import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, classification_report

from MedMamba import VSSM as medmamba

NPZ_PATH = "/mnt/new_usb/jupyter-altis5526/pneumoniamnist.npz"
WEIGHTS_PATH = "./MedMamba_PneumoniaMNIST.pth"
RESULT_PATH = "./MedMamba_PneumoniaMNIST_test_results.csv"
NUM_CLASSES = 2


class PneumoniaMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    data = np.load(NPZ_PATH)
    test_dataset = PneumoniaMNISTDataset(data["test_images"], data["test_labels"], transform=transform)
    batch_size = 128
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)
    print("Test set size: {}".format(len(test_dataset)))

    # Base MedMamba uses default depths=[2,2,4,2]
    net = medmamba(depths=[2, 2, 4, 2], dims=[96, 192, 384, 768], num_classes=NUM_CLASSES)
    net.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    net.to(device)
    net.eval()
    print("Loaded weights from: {}".format(WEIGHTS_PATH))

    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Inference"):
            outputs = net(images.to(device))
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.max(outputs, dim=1)[1]
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    test_auc = roc_auc_score(all_labels, all_probs)

    print("\n===== Test Set Results =====")
    print("Test Accuracy: {:.4f}".format(test_acc))
    print("Test AUC:      {:.4f}".format(test_auc))
    print(classification_report(all_labels, all_preds, target_names=["normal", "pneumonia"]))

    df = pd.DataFrame({
        "label": all_labels,
        "pred": all_preds,
        "prob_pneumonia": all_probs,
    })
    df.to_csv(RESULT_PATH, index=False)
    print("Results saved to: {}".format(RESULT_PATH))


if __name__ == "__main__":
    main()
