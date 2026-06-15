import json
import pandas as pd

check_type = "train"

with open(f"{check_type}_response_Gemma1.5.json", "r") as f:
    data = json.load(f)

standard_data = pd.read_csv(f"../insurance_dataset_8_1_1_PMMthree_{check_type}_no_support_devices.csv")
standard_dicom_id = standard_data['dicom_id'].tolist()

failed_dcm_ids = []
Gemma_response_dcm_ids = []

for k, v in data.items():
    Gemma_response_dcm_ids.append(k)

for dcm_id in standard_dicom_id:
    if dcm_id not in Gemma_response_dcm_ids:
        failed_dcm_ids.append(dcm_id)

with open(f"{check_type}_Gemma1.5_failed_dcm_ids.txt", "w") as f:
    for dcm_id in failed_dcm_ids:
        f.write(f"{dcm_id}\n")
