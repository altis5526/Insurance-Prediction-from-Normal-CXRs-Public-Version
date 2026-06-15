from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import *
import torch
from RawImageDataset import MIMIC_raw
from dataloader import CombinedLoader
import numpy
import cv2
import random

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

set_seed(42)
g = torch.Generator()
g.manual_seed(0)

weight_path = "/mnt/new_usb/jupyter-altis5526/new_insurancetype_weight/MedGemma_checked/MedgemmaChecked_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_densenet_DoubleLinear_Lion5e-6_20260228/Rand123/MedGemma_MIMIC_densenet.pt"

val_path = "insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv"
model = DenseNetWithDoubleLinear(num_classes=2, dropout_prob=0)
model.load_state_dict(torch.load(weight_path)["model_state_dict"])
model.eval()
val_dataset = MIMIC_raw(val_path, transform=False)
val_loader = DataLoader(val_dataset, batch_size=1, worker_init_fn=seed_worker, num_workers=8, shuffle=False, generator=g)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_layers = [model.densenet.features[-1]]
print(model.densenet.features[-1])
i = 0
for batch in val_loader:
    test_imgs = batch["full_img"]
    test_imgs = test_imgs.to(device)
    rgb_img = cv2.imread(batch["img_dir"][0])
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (448,448))
    rgb_img = rgb_img/255.0
    # We have to specify the target we want to generate the CAM for.
    targets = [ClassifierOutputTarget(1)]
    
    # Construct the CAM object once, and then re-use it on many images.
    with GradCAM(model=model, target_layers=target_layers) as cam:
      # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        print(test_imgs.size())
        grayscale_cam = cam(input_tensor=test_imgs, targets=targets)
      # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False)
      # You can also get the model outputs without having to redo inference
        model_outputs = cam.outputs
    # print(visualization.shape)
    # print(visualization)
    image_path = batch["img_dir"][0]
    image_id = image_path.split('/')[-1].split(".")[0]
    cv2.imwrite(f'/mnt/new_usb/jupyter-altis5526/cam_images/MedGemmaChecked/private_cam_{image_id}.jpg', visualization)
    i += 1
    if i>=100:
        break
