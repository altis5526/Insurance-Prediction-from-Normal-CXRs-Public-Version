import json
import csv
import pandas as pd
from datasets import Dataset, Features, Sequence, Value, Image as HFImage
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import numpy as np
from tqdm.auto import tqdm
import pyarrow as pa
import re
import argparse
import ast
import logging
import os
from dotenv import load_dotenv


load_dotenv()  # This loads the variables from .env
api_token = os.getenv("API_TOKEN")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Configuration
parser = argparse.ArgumentParser()
parser.add_argument("--save_image", type=bool, default=False, help="Whether to save images")
parser.add_argument("--model_id", type=str, default="google/medgemma-1.5-4b-it", help="Model ID to use for inference")
parser.add_argument("--restart", type=bool, default=False)
parser.add_argument("--output_name", type=str, default="./Gemma_parsed_data/CheXpert_response_Gemma1.5")
args = parser.parse_args()

csv_path = "/mnt/new_usb/jupyter-altis5526/CheXpert_Allfiltered_withCheXbert.csv"
img_root_dir = "/mnt/new_usb/jupyter-altis5526/chexpertchestxrays-u20210408/all_data/"
access_token = api_token
model_id = args.model_id
batch_size = 1

# 1. Prepare Data
print("Preparing data...")
data_json = []
data = pd.read_csv(csv_path)

# Shuffle the data here so we pick usage random images, but process their prompts sequentially
# data = data.sample(frac=1, random_state=42).reset_index(drop=True)
# data = data.head(100)

# New Prompt for better negative constraint handling
PROMPT1 = (
    "Analyze this chest X-ray. Look for the following support devices: chest tubes, pigtails, pacemakers, central venous catheters, PICCs, nasogastric tubes, endotracheal tubes, surgical clips, sternal wires, spinal fusion, aortic stent, ventriculoperitoneal shunt, prosthesis, or other internal body devices.\n\nImportant: EKG leads and portable CXR labels are NOT support devices. If you see EKG leads, ignore them for the purpose of this question. Be aware of the difference between EKG leads and pacemakers. Does the image contain any of the listed support devices?\n\nStructure the answer as: Answer:{Yes or No}, Explanation:{} and output a json object."
)

PROMPT2 = (
    "Analyze this chest X-ray. First, identify if any of the following are present: Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, Pleural Effusion, Pneumonia, Pneumothorax, or Pleural Other. Conclude 'No Finding' ONLY if the image is free of all these abnormalities.\n Important: If you see support devices, ignore them for the purpose of this question.\n Is this chest X-ray labeled as “No Finding”? Structure the answer as: Answer:{No Finding or Positive Finding}, Explanation:{} and output a json object."
)

# PROMPT3 = (
#     "Analyze this chest X-ray. Check if the X-ray is a frontal or lateral view. "
#     "Is this a lateral view chest X-ray?"
#     "Structure the answer as: Answer:{Yes or No}, Explanation:{} and output a json object."
# )

# Process only first 500 for now as per original script
for idx, row in data.iterrows():
    subject_id = row['Path'].split('/')[2]
    study_id = row['Path'].split('/')[3]
    view_id = row['Path'].split('/')[4].split('.')[0]
    image_path = "/".join(row['Path'].split('/')[2:])
    image_path = os.path.join(img_root_dir, image_path)

    # Add PROMPT1
    single_image_dict_1 = {
        "question": PROMPT1,
        "image_path": image_path,
        "subject_id": subject_id,
        "study_id": study_id,
        "view_id": view_id,
        "prompt_type": "PROMPT1"
    }
    data_json.append(single_image_dict_1)

    # Add PROMPT2
    single_image_dict_2 = {
        "question": PROMPT2, 
        "image_path": image_path,
        "subject_id": subject_id,
        "study_id": study_id,
        "view_id": view_id,
        "prompt_type": "PROMPT2"
    }
    data_json.append(single_image_dict_2)

    # # Add PROMPT3
    # single_image_dict_3 = {
    #     "question": PROMPT3, 
    #     "image_path": image_path,
    #     "subject_id": subject_id,
    #     "study_id": study_id,
    #     "view_id": view_id,
    #     "prompt_type": "PROMPT3"
    # }
    # data_json.append(single_image_dict_3)

print(f"Data prepared. First item: {data_json[0]}")

# 2. Create Dataset
def transform_to_messages(example):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image_path"]},  # Store path string, not PIL Image
                    {"type": "text", "text": example["question"]}
                ]
            }
        ]
    }


def clean_text(example):
    # Adjust 'text_column' to whatever your column name is
    text = example['question']
    if isinstance(text, str):
        # This removes non-utf8 characters
        example['text_column'] = text.encode("utf-8", "ignore").decode("utf-8")
    return example

if args.restart:
    with open(f"{args.output_name}.json", "r") as f:
        reponse_dict = json.load(f)
    index_to_be_delete = []
    key_to_be_delete = []
    for i in range(len(data_json)):
        fake_dicom_id = data_json[i]["subject_id"] + "_" + data_json[i]["study_id"] + "_" + data_json[i]["view_id"]
        if fake_dicom_id in reponse_dict.keys() and len(reponse_dict[fake_dicom_id].keys()) == 2:
            index_to_be_delete.append(i)

        elif fake_dicom_id in reponse_dict.keys() and len(reponse_dict[fake_dicom_id].keys()) == 1:
            if fake_dicom_id not in key_to_be_delete:
                key_to_be_delete.append(fake_dicom_id)

    count = 0
    for index in index_to_be_delete:
        del data_json[index - count]
        count += 1

    for key in key_to_be_delete:
        del reponse_dict[key]

# Create dataset with the defined features
dataset = Dataset.from_list(data_json)
inference_ds = dataset.map(clean_text)
inference_ds = dataset.map(transform_to_messages)

# Save dataset
inference_ds.save_to_disk("inference_ds_CheXpert")
# Load dataset
inference_ds = Dataset.load_from_disk("inference_ds_CheXpert")

# 3. Load Model and Processor
print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=access_token,
)
processor = AutoProcessor.from_pretrained(model_id, token=access_token)
processor.tokenizer.padding_side = "left"

def transform(examples):
    # Iterate over the batch
    messages = []
    for q, img_path in zip(examples["question"], examples["image_path"]):
        messages.append([
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": Image.open(img_path).convert("RGB")}, 
                    {"type": "text", "text": q}
                ]
            }
        ])
    
    inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        padding=True, 
        return_dict=True,
        return_tensors="pt",
        tokenize=True
    )
    return inputs

# 2. Apply it as a "set_transform" (This doesn't rewrite the data, it processes during access)
inference_ds.set_transform(transform)

print("Preprocessing dataset...")
# inference_ds = inference_ds.map(preprocess_fn, remove_columns=inference_ds.column_names)
# inference_ds.set_format("torch")

# 4. Inference
print("Starting inference...")
data_collator = DataCollatorWithPadding(tokenizer=processor.tokenizer)
dataloader = DataLoader(inference_ds, batch_size=batch_size, collate_fn=data_collator, shuffle=False)

if args.restart == False:
    reponse_dict = {}
batch_count = 0

for i, batch in enumerate(tqdm(dataloader, desc="Inference Progress")):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    
    with torch.inference_mode():
        generated_ids = model.generate(
            **batch,
            max_new_tokens=256,
            do_sample=False
        )
    
    responses = processor.batch_decode(generated_ids, skip_special_tokens=True)
    try:
        response_text = responses[0].split("model\n")[1]
    except IndexError:
        response_text = responses[0]
    response_text = re.search(r'\{.*\}', response_text, re.DOTALL)
    if response_text:
        json_str = response_text.group(0)
            # Step 2: Parse the string into a Python dictionary
        try:
            json_object = ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            # Fallback or error handling
            print("String is not a valid Python literal either.")
    else:
        try:
            parts = re.split(r',\s*(Explanation):', responses[0].split("model\n")[1])

            # 2. Clean up the prefix and map to a dictionary
            json_object = {
                "Answer": parts[0].replace("Answer:", "").strip(),
                "Explanation": parts[2].strip() if len(parts) > 2 else parts[1].strip()
            }

        except:
            try:
                parts = re.split(r'\s*(Explanation):', responses[0].split("model\n")[1])

                json_object = {
                    "Answer": parts[0].replace("Answer:", "").strip(),
                    "Explanation": parts[2].strip() if len(parts) > 2 else parts[1].strip()
                }
            except:
                print("Could not find valid JSON in the string.")
                print("Full response:", responses[0])
                continue

    current_sample = data_json[i]
    subject_id = current_sample['subject_id']
    study_id = current_sample['study_id']
    view_id = current_sample['view_id']
    image_path = current_sample['image_path']
    prompt_type = current_sample['prompt_type']

    fake_dicom_id = str(subject_id) + '_' + str(study_id) + '_' + view_id
    if fake_dicom_id not in reponse_dict:
        reponse_dict[fake_dicom_id] = {}
    
    reponse_dict[fake_dicom_id][prompt_type] = json_object
    batch_count += 1

    if batch_count % 20 == 0:
        print(f"Saving intermediate results to {args.output_name}.json...")
        with open(f"{args.output_name}.json", "w") as f:
            json.dump(reponse_dict, f)

        with open(f"{args.output_name}.txt", "w") as f:
            for k, v in reponse_dict.items():
                f.write(f"{k}: {v}\n")


# 5. Save Results
print("Saving results...")
with open(f"{args.output_name}.json", "w") as f:
    json.dump(reponse_dict, f)

with open(f"{args.output_name}.txt", "w") as f:
    for k, v in reponse_dict.items():
        f.write(f"{k}: {v}\n")

print("Done.")
