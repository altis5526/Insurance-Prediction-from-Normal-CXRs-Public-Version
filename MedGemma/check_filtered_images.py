from PIL import Image
import pandas as pd
import tensorflow as tf
import cv2
import numpy as np


## MIMIC
# check_type = "train"
# dataset = pd.read_csv(f"../insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv")

# subject_ids = dataset['subject_id_x'].tolist()
# study_ids = dataset['study_id'].tolist()
# dicom_ids = dataset['dicom_id'].tolist()


# for i in range(50):
#     subject_id = subject_ids[i]
#     study_id = study_ids[i]
#     dicom_id = dicom_ids[i]
#     img_root_dir = "/mnt/new_usb/jupyter-altis5526/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
#     img_dir = f"p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
#     image_path = img_root_dir + img_dir
#     try:
#         image = Image.open(image_path)
#         image.save(f"save_images/check_filtered_images/{dicom_id}.jpg")
#     except:
#         print(f"Failed to open image: {image_path}")


## CheXpert
check_type = "train"
dataset_path = f"/mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_{check_type}_insurance.tfrecord"

dataset = tf.data.TFRecordDataset(dataset_path)
count = 0
for record in dataset:
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    features = dict(example.features.feature.items())
    decode_img = cv2.imdecode(np.frombuffer(features['jpg_bytes'].bytes_list.value[0], dtype=np.uint8), -1)
    pil_image = Image.fromarray(decode_img).convert('RGB')
    ptid = features['patient'].int64_list.value[0]
    if len(str(ptid)) == 1:
        ptid = "0000" + str(ptid)
    elif len(str(ptid)) == 2:
        ptid = "000" + str(ptid)
    elif len(str(ptid)) == 3:
        ptid = "00" + str(ptid)
    elif len(str(ptid)) == 4:
        ptid = "0" + str(ptid)
    else:
        ptid = str(ptid)
    study = features['study'].int64_list.value[0]
    image = features['image'].int64_list.value[0]
    pil_image.save(f"./save_images/check_filtered_images/CheXpert/{ptid}_study{study}.jpg")
    count += 1

    if count >= 100:
        break

