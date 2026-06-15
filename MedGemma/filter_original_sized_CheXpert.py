import pandas as pd
import os
from PIL import Image

train_csv = "/mnt/new_usb/jupyter-altis5526/chexpertchestxrays-u20210408/train_visualCheXbert.csv"
valid_csv = "/mnt/new_usb/jupyter-altis5526/chexpertchestxrays-u20210408/validate/valid.csv"
label_path = "/mnt/new_usb/jupyter-altis5526/df_chexpert_plus_240401.csv"

use_cols = ["deid_patient_id", "insurance_type"]
label_data = pd.read_csv(label_path, usecols=use_cols)
insurance_data = list(zip(label_data['deid_patient_id'], label_data['insurance_type']))
reduced_copies_insurance_dict = dict(list(set(insurance_data)))
filtered_patient_list = []
for patient_id, insurance_type in reduced_copies_insurance_dict.items():
    if insurance_type in ['Private Insurance', 'Medicare', 'Medicaid']:
        filtered_patient_list.append(patient_id)

train_data = pd.read_csv(train_csv)
valid_data = pd.read_csv(valid_csv)

train_data_filtered = train_data[train_data['Frontal/Lateral'] == 'Frontal']
valid_data_filtered = valid_data[valid_data['Frontal/Lateral'] == 'Frontal']

print("Filtered frontal train data size:", len(train_data_filtered))
print("Filtered frontal valid data size:", len(valid_data_filtered))

train_data_filtered = train_data_filtered[train_data_filtered['No Finding'] == 1.0]
valid_data_filtered = valid_data_filtered[valid_data_filtered['No Finding'] == 1.0]

print("Filtered no finding train data size:", len(train_data_filtered))
print("Filtered no finding valid data size:", len(valid_data_filtered))

train_data_filtered = train_data_filtered[train_data_filtered['Support Devices'] == 0.0]
valid_data_filtered = valid_data_filtered[valid_data_filtered['Support Devices'] == 0.0]

print("Filtered SD train data size:", len(train_data_filtered))
print("Filtered SD valid data size:", len(valid_data_filtered))

train_data_filtered = train_data_filtered[train_data_filtered['Age'] < 65]
valid_data_filtered = valid_data_filtered[valid_data_filtered['Age'] < 65]

print("Filtered age train data size:", len(train_data_filtered))
print("Filtered age valid data size:", len(valid_data_filtered))

train_data_filtered = train_data_filtered[train_data_filtered['Path'].str.split('/').str[2].isin(filtered_patient_list)]
valid_data_filtered = valid_data_filtered[valid_data_filtered['Path'].str.split('/').str[2].isin(filtered_patient_list)]

print("Filtered insurance train data size:", len(train_data_filtered))
print("Filtered insurance valid data size:", len(valid_data_filtered))

combined_filtered = pd.concat([train_data_filtered, valid_data_filtered], ignore_index=True)
print("Combined filtered data size:", len(combined_filtered))

combined_filtered.to_csv("/mnt/new_usb/jupyter-altis5526/CheXpert_Allfiltered_withCheXbert.csv", index=False)



