import pandas as pd
import os
from PIL import Image
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

train_insurance_file = "/mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord"
val_insurance_file = "/mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord"
test_insurance_file = "/mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord"

chexpert_train_csv = "/mnt/new_usb/jupyter-altis5526/chexpertchestxrays-u20210408/train_visualCheXbert.csv"
chexpert_valid_csv = "/mnt/new_usb/jupyter-altis5526/chexpertchestxrays-u20210408/validate/valid.csv"

original_train_insurance_dataset = tf.data.TFRecordDataset(train_insurance_file)
original_val_insurance_dataset = tf.data.TFRecordDataset(val_insurance_file)
original_test_insurance_dataset = tf.data.TFRecordDataset(test_insurance_file)

chexpert_train_data = pd.read_csv(chexpert_train_csv)
chexpert_valid_data = pd.read_csv(chexpert_valid_csv)

origin_train_id = []
origin_val_id = []
origin_test_id = []
for record in original_train_insurance_dataset:
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    features = dict(example.features.feature.items())
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
    origin_train_id.append(ptid+"_"+f"study{study}")

for record in original_val_insurance_dataset:
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    features = dict(example.features.feature.items())
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
    origin_val_id.append(ptid+"_"+f"study{study}")

for record in original_test_insurance_dataset:
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    features = dict(example.features.feature.items())
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
    origin_test_id.append(ptid+"_"+f"study{study}")

## combine three lists
origin_all_id = origin_train_id + origin_val_id + origin_test_id

## chexpert_train_data format "Path" column: CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg
chexpert_train_data["NewID"] = chexpert_train_data["Path"].str.split("/").str[2].str.split("patient").str[-1]+"_"+chexpert_train_data["Path"].str.split("/").str[3]
chexpert_valid_data["NewID"] = chexpert_valid_data["Path"].str.split("/").str[2].str.split("patient").str[-1]+"_"+chexpert_valid_data["Path"].str.split("/").str[3]

chexpert_train_data_filtered = chexpert_train_data[chexpert_train_data["NewID"].isin(origin_all_id)]
chexpert_valid_data_filtered = chexpert_valid_data[chexpert_valid_data["NewID"].isin(origin_all_id)]

chexpert_train_data_filtered = chexpert_train_data_filtered[chexpert_train_data_filtered["Frontal/Lateral"] == "Frontal"]
chexpert_valid_data_filtered = chexpert_valid_data_filtered[chexpert_valid_data_filtered["Frontal/Lateral"] == "Frontal"]
 
combined_filtered = pd.concat([chexpert_train_data_filtered, chexpert_valid_data_filtered], ignore_index=True)
print("Combined filtered data size:", len(combined_filtered))

combined_filtered.to_csv("/mnt/new_usb/jupyter-altis5526/CheXpert_Allfiltered_withCheXbert.csv", index=False)