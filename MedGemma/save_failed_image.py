from PIL import Image
import pandas as pd
from tqdm import tqdm

check_type = "test"
failed_txt_path = f"{check_type}_Gemma_failed_dcm_ids.txt"
dataset = pd.read_csv(f"../insurance_dataset_8_1_1_PMMthree_{check_type}_no_support_devices.csv")
with open(failed_txt_path, "r") as f:
    failed_dcm_ids = f.read().splitlines()

for failed_dcm_id in tqdm(failed_dcm_ids):
    subject_id = dataset[dataset['dicom_id'] == failed_dcm_id]['subject_id_x'].values[0]
    study_id = dataset[dataset['dicom_id'] == failed_dcm_id]['study_id'].values[0]
    img_root_dir = "/mnt/new_usb/jupyter-altis5526/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
    img_dir = f"p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{failed_dcm_id}.jpg"
    image_path = img_root_dir + img_dir
    try:
        image = Image.open(image_path)
        image.save(f"save_images/{check_type}_manual_check/{failed_dcm_id}.jpg")
    except:
        print(f"Failed to open image: {image_path}")

## train 1.0
# 5bf93b39-762874ba-653777b8-d7c7ea68-939261c1
# 5ca27ae5-1f4ab9cf-8bf74fd1-922a6d58-feefc5ce
# 6f6ccc87-4bee9e79-0e57c91b-ddfeb74d-1f0f62f0
# 7ee8243c-fb24f2bc-643156c3-519d5f61-e78b0031
# 7f6338b9-8a18d491-767e1bf7-271e66d9-9ac55a09
# 20b955ba-b3955d85-f334e5ca-a8db0e1f-d57389c5
# 76a8a4d8-5e600c83-638a6ab7-a98de374-de072804
# 225b226a-7edb722a-a7301b32-dbb866b7-583d22de
# 493c8d5b-ce142fe5-6a920201-744faa18-7d145912
# 755cb1b9-2fb16f46-5f62e078-04377e79-1045e337
# 9736ca70-ac006587-681f4ff6-805a9a31-c8f2a9ed
# 18358df9-16a24117-27b74d9b-4682756f-27f9c792
# 71316f90-8b4719ce-e380d870-8677617c-8bb0d64a
# 2003002b-1e438894-dd21dbb9-31311137-12b73519
# 82462656-4274919f-6988ae86-03d4ee38-78ee3b35
# a1442017-b4ad8c01-c5bbd5fa-83f33f3b-b20db01d
# af5a4055-1869c37c-825a8715-d21221b0-a695d690
# b7941ea0-429f1ce8-baff2c60-c281f0a3-db454ec9
# b268431a-bcd00da1-df648ac7-8bca0289-d1d34d27
# d32809d5-dda76003-96545f81-1e016561-c96b274d
# d9944862-3b24c3e3-ca2e7a22-21972439-60efb2c0
# df8f14e7-bc32984a-a4992b1f-13815edb-188de668
# ea41f2f1-b9c33d66-a8e58a72-75728d92-547e8f2d
# f0b1955c-b4f711ec-2e470e93-715e62d0-f18c44b7
# f3d773cc-4a7b6f7b-b519eec2-0f3d1352-41075c25
# fc0e628d-7754e9eb-18f4c611-ba956f73-ca4587d2

## val 1.0
# 07ebc3a9-cac1e76a-a8b68886-121f886b-ca6e5606
# 48dbe512-09de5c06-cbe1d58c-d8cce3b4-b623482b
# 6907e2de-85cc5d3e-539c829e-bc55ac40-d58ca3cc
# a3816e11-487ccd4b-5f1e1b4d-9397f7d8-a3903a73
# ab9dc926-05921eb8-6fa8fcb4-f80ac211-e5452489
# b05dbb6c-1ffd4e9c-220767f6-7c4d95ef-7068ef66
# b8b313fb-4b173cba-6880ca1c-dfca2633-0828f5fe
# f9539929-6f1f5df7-193f6fab-74a425e0-759bbde8


## test 1.0
# 5f4a29c9-b8a2933a-035cbf27-44258304-71fc12dc