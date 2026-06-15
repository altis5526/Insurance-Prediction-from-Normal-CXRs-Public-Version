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
import logging

# Configuration
parser = argparse.ArgumentParser()
parser.add_argument("--check_type", type=str, default="train", help="Check type (train or test)")
parser.add_argument("--save_image", type=bool, default=False, help="Whether to save images")
parser.add_argument("--model_id", type=str, default="google/medgemma-1.5-4b-it", help="Model ID to use for inference")
parser.add_argument("--num_shots", type=int, default=2, help="Number of few-shot examples")
args = parser.parse_args()

check_type = args.check_type
csv_path = f"../insurance_dataset_8_1_1_PMMthree_{check_type}_no_support_devices.csv"
img_root_dir = "/mnt/new_usb/jupyter-altis5526/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
access_token = "MY_TOKEN"
model_id = args.model_id
batch_size = 1

logging.basicConfig(
    filename='app.log',
    filemode='a', # 'a' for append, 'w' for overwrite each run
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO # Set the minimum logging level
)

# 1. Prepare Data
print("Preparing data...")
data_json = []
data = pd.read_csv(csv_path)


# Shuffle the data here so we pick usage random images, but process their prompts sequentially
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data = data.head(502)

# New Prompt for better negative constraint handling
PROMPT3 = (
    "Analyze this chest X-ray. Determine if this is a \"Frontal\" or \"Lateral\" view."

    "Note:"
    "- A Frontal view shows the lungs side-by-side."
    "- A Lateral view shows the spine and sternum in profile."
    "- Do NOT classify a Lateral view as standard PA/AP."

    "Output a JSON object: {\"view\": \"Frontal\" or \"Lateral\", \"reasoning\": \"...\"}"
)

# Load support set (always from train)
position_csv_path = "../../position.csv"
support_data = pd.read_csv(position_csv_path)
support_data = support_data.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle support data
support_examples = support_data.head(100) # Select first N examples

# Helper to get image path
def get_image_path(row):
    subject_id = row['subject_id']
    dicom_id = row['dicom_id']
    study_id = row['study_id']
    img_dir = f"p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
    return img_root_dir + img_dir

# Pre-process support messages
support_messages = []
lateral_count = 0
frontal_count = 0
for i in range(len(support_examples)):
    row = support_examples.iloc[i]
    img_path = get_image_path(row)
    
    # Ground Truth Generation
    # PROMPT3
    is_lateral = row['ViewPosition'] in ['LAT', 'LL', 'RL', 'LATERAL']
    if is_lateral and lateral_count > 0 and frontal_count == 0:
        continue
    if not is_lateral and frontal_count > 0 and lateral_count == 0:
        continue
    if lateral_count > 0 and frontal_count > 0:
        break
    
    if is_lateral:
        lateral_count += 1
    else:
        frontal_count += 1
    ans3 = "Lateral" if is_lateral else "Frontal"
    exp3 = f"The view is {ans3}."
    resp3 = json.dumps({"view": ans3, "reasoning": exp3})
    
    support_messages.append({
        "prompt_type": "PROMPT3",
        "image_path": img_path,
        "response": resp3,
        "question": PROMPT3
    })


# Process only first 500 for now as per original script
for i in range(len(data)):
    subject_id = data.iloc[i]['subject_id_x']
    dicom_id = data.iloc[i]['dicom_id']
    study_id = data.iloc[i]['study_id']
    img_dir = f"p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
    image_path = img_root_dir + img_dir

    # Add PROMPT3
    single_image_dict_3 = {
        "question": PROMPT3, 
        "image_path": image_path,
        "subject_id": subject_id,
        "dicom_id": dicom_id,
        "study_id": study_id,
        "prompt_type": "PROMPT3"
    }
    data_json.append(single_image_dict_3)

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

# Create dataset with the defined features
dataset = Dataset.from_list(data_json)
inference_ds = dataset.map(transform_to_messages)

# Save dataset
inference_ds.save_to_disk("inference_ds")
# Load dataset
inference_ds = Dataset.load_from_disk("inference_ds")

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
    messages_batch = []
    
    for q, img_path, p_type in zip(examples["question"], examples["image_path"], examples["prompt_type"]):
        
        conversation = []
        
        # Add Few-Shot Examples matching the prompt type
        for support in support_messages:
            if support["prompt_type"] == p_type:
                # User turn (Image + Question)
                conversation.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.open(support["image_path"]).convert("RGB")},
                        {"type": "text", "text": support["question"]}
                    ]
                })
                # Model turn (Answer)
                conversation.append({
                    "role": "model",
                    "content": [{"type": "text", "text": support["response"]}] # This is a string
                })
        
        # Add Current Query
        conversation.append({
            "role": "user", 
            "content": [
                {"type": "image", "image": Image.open(img_path).convert("RGB")}, 
                {"type": "text", "text": q}
            ]
        })
        
        messages_batch.append(conversation)
    
    
    inputs = processor.apply_chat_template(
        messages_batch, 
        add_generation_prompt=True, 
        padding=True, 
        return_dict=True,
        return_tensors="pt",
        tokenize=True
    )

    # inputs["pixel_values"] typically has shape (N_images, 3, H, W).
    # Since we are returning a single batch item (dataset[i]), the dataset slicing mechanism
    # interprets the first dimension (N_images) as the batch dimension and slices it, 
    # keeping only the first image and discarding the rest.
    # We must wrap it in a list so it is treated as a single item with N_images.
    if "pixel_values" in inputs:
        inputs["pixel_values"] = [inputs["pixel_values"]]
        
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

reponse_dict = {}
batch_count = 1500
for i, batch in enumerate(tqdm(dataloader, desc="Inference Progress")):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    
    # If pixel_values has an extra batch dimension (from DataCollator stacking our wrapped list), remove it.
    if "pixel_values" in batch and batch["pixel_values"].dim() == 5:
         batch["pixel_values"] = batch["pixel_values"].squeeze(0)
    
    with torch.inference_mode():
        generated_ids = model.generate(
            **batch,
            max_new_tokens=256,
            do_sample=False
        )
    
    responses = processor.batch_decode(generated_ids, skip_special_tokens=True)
    try:
        response_text = responses[0].split("model\n")[-1]
    except IndexError:
        response_text = responses[0]
    response_text = re.search(r'\{.*\}', response_text, re.DOTALL)
    if response_text:
        json_str = response_text.group(0)
        # Step 2: Parse the string into a Python dictionary
        json_object = json.loads(json_str)
    else:
        try:
            parts = re.split(r',\s*(Explanation):', responses[0].split("model\n")[-1])

            # 2. Clean up the prefix and map to a dictionary
            json_object = {
                "Answer": parts[0].replace("Answer:", "").strip(),
                "Explanation": parts[2].strip() if len(parts) > 2 else parts[1].strip()
            }

        except:
            try:
                parts = re.split(r'\s*(Explanation):', responses[0].split("model\n")[-1])

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
    dicom_id = current_sample['dicom_id']
    image_path = current_sample['image_path']
    prompt_type = current_sample['prompt_type']
    
    # Only save image once (e.g. for PROMPT1) to avoid deduplicate saving, or overwrite is fine
    if prompt_type == "PROMPT1" and args.save_image:
        image = Image.open(image_path)
        image.save(f"/mnt/new_usb/jupyter-altis5526/random_sample_MIMIC_500_for_MedGemma/{batch_count//2}_{subject_id}_{study_id}_{dicom_id}.jpg")

    if str(dicom_id) not in reponse_dict:
        reponse_dict[str(dicom_id)] = {"study_id": str(study_id), "subject_id": str(subject_id)}
    
    reponse_dict[str(dicom_id)][prompt_type] = json_object
    batch_count += 1

    if batch_count % 20 == 0:
        print(f"Saving intermediate results to {check_type}_response_500th.json...")
        with open(f"Few_shot_{check_type}_random500_response_Gemma1.5.json", "w") as f:
            json.dump(reponse_dict, f)

        with open(f"Few_shot_{check_type}_random500_response_Gemma1.5.txt", "w") as f:
            for k, v in reponse_dict.items():
                f.write(f"{k}: {v}\n")


# 5. Save Results
print("Saving results...")
with open(f"Few_shot_{check_type}_random500_response_Gemma1.5.json", "w") as f:
    json.dump(reponse_dict, f)

with open(f"Few_shot_{check_type}_random500_response_Gemma1.5.txt", "w") as f:
    for k, v in reponse_dict.items():
        f.write(f"{k}: {v}\n")

print("Done.")
