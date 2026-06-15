import pandas as pd
import os
from PIL import Image

data_path = "/mnt/new_usb/jupyter-altis5526/CheXpert_Allfiltered_withCheXbert.csv"

data = pd.read_csv(data_path)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data = data.head(100)

for idx, row in data.iterrows():
    image_path = "/".join(row['Path'].split('/')[2:])
    image_path = os.path.join("/mnt/new_usb/jupyter-altis5526/chexpertchestxrays-u20210408/all_data/", image_path)
    img = Image.open(image_path)
    img.save(f"./save_images/CheXpert/{idx}_{row['Path'].split('/')[2]}_{row['Path'].split('/')[3]}_{row['Path'].split('/')[4]}")

image_dir = "./save_images/CheXpert"
image_list = os.listdir(image_dir)
for image in image_list:
    image_num = image.split("_")[0]
    if len(image_num) == 3:
        continue
    elif len(image_num) == 2:
        image_num = "0" + image_num
        rest_of_the_name = "_".join(image.split("_")[1:])
        os.rename(os.path.join(image_dir, image), os.path.join(image_dir, image_num + "_" + rest_of_the_name))
    elif len(image_num) == 1:
        image_num = "00" + image_num
        rest_of_the_name = "_".join(image.split("_")[1:])
        os.rename(os.path.join(image_dir, image), os.path.join(image_dir, image_num + "_" + rest_of_the_name))

image_list = os.listdir(image_dir)
df = pd.DataFrame(image_list, columns=["dicom_id"])
df.to_csv("manual_check_100CheXpert_image.csv", index=False)