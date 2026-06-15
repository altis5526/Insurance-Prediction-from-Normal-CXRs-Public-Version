
import json
import torch
import numpy as np
import cv2
import re
import argparse
from tqdm.auto import tqdm
from PIL import Image
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText, DataCollatorWithPadding

# Configuration Defaults (can be overridden by args)
DEFAULT_MODEL_ID = "google/medgemma-4b-it"
DEFAULT_ACCESS_TOKEN = "MY_TOKEN" # As found in inference.py

# Data Description based on CheXpert_loader.py
TFRECORD_DESCRIPTION = {
    'fracture': 'int',
    'enlarged_cardiomediastinum': 'int',
    'dataset': 'int',
    'edema': 'int',
    'patient': 'int',  # maps to subject_id
    'support_devices': 'int',
    'pleural_other': 'int',
    'consolidation': 'int',
    'cardiomegaly': 'int',
    'view': 'int',
    'pneumonia': 'int',
    'airspace_opacity': 'int',
    'jpg_bytes': 'byte',
    'pleural_effusion': 'int',
    'atelectasis': 'int',
    'no_finding': 'int',
    'pneumothorax': 'int',
    'study': 'int',    # maps to study_id
    'image': 'int',    # maps to dicom_id
    'lung_lesion': 'int',
    'age': 'float',
    'race': 'byte',
    'insurance_type': 'byte',
    'sex': 'byte'
}

PROMPT = (
    "Analyze this chest X-ray. Look for the following support devices: chest tubes, pigtails, pacemakers, central venous catheters, PICCs, nasogastric tubes, endotracheal tubes, surgical clips, sternal wires, spinal fusion, or neck collar. \n\nImportant: EKG leads and portable CXR labels are NOT support devices. If you see EKG leads, ignore them for the purpose of this question. Be aware of the difference between EKG leads and pacemakers. \n\nDoes the image contain any of the listed support devices?\n\nStructure the answer as: Answer:{Yes or No}, Explanation:{} and output a json object"
)

def decode_image(features):
    """
    Decodes the image from bytes and converts to PIL Image RGB.
    """
    # cv2.imdecode might return BGR, we need RGB for transformers usually, 
    # but PIL.Image.fromarray(RGB) is expected. 
    # cv2.imdecode returns numpy array.
    nparr = np.frombuffer(features['jpg_bytes'], np.uint8)
    decode_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # -1 is IMREAD_COLOR (ish) but let's be explicit or use -1 as in loader
    # CheXpert loader uses -1: decode_img = cv2.imdecode(..., -1)
    
    # If image is None (corrupt), handle it? 
    if decode_img is None:
        raise ValueError("Could not decode image from bytes.")

    # Convert BGR to RGB (standard for PIL/HuggingFace)
    decode_img = cv2.cvtColor(decode_img, cv2.COLOR_BGR2RGB)
    
    features['pil_image'] = Image.fromarray(decode_img)
    return features


def main():
    parser = argparse.ArgumentParser(description="MedGemma Inference on TFRecord Dataset")
    parser.add_argument("--tfrecord_path", type=str, required=True, help="Path to the .tfrecord file")
    parser.add_argument("--output_json", type=str, default="tfrecord_response.json", help="Path to save output JSON")
    parser.add_argument("--output_txt", type=str, default="tfrecord_response.txt", help="Path to save output TXT")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    args = parser.parse_args()

    print(f"Loading model: {DEFAULT_MODEL_ID}...")
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            DEFAULT_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=DEFAULT_ACCESS_TOKEN,
        )
        processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_ID, token=DEFAULT_ACCESS_TOKEN)
        processor.tokenizer.padding_side = "right"
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Setting up TFRecordDataset...")
    # Initialize TFRecordDataset
    # Note: TFRecordDataset from tfrecord.torch.dataset is iterable.
    try:
        dataset = TFRecordDataset(
            args.tfrecord_path,
            None,
            TFRECORD_DESCRIPTION,
            transform=decode_image
        )
    except Exception as e:
        print(f"Error initializing TFRecordDataset: {e}")
        return

    # We need a custom collate_fn if we were using a DataLoader that batches things automatically 
    # into tensors, but for HuggingFace processing we usually want a list of PIL images and text.
    # However, to be efficient and use the model's batching, let's create a custom collator 
    # or just iterate and process.
    # Since inference.py uses batch_size=1, we can just iterate directly or use a simple loader.
    # Let's use a DataLoader to handle the worker/fetching logic if desired, but 
    # standard TFRecordDataset is an iterable dataset.
    
    # We will iterate one by one or in batches manually to prepare inputs for the model.
    # Let's iterate using DataLoader with batch_size=args.batch_size
    
    def collate_fn_custom(batch):
        # batch is a list of dictionaries (features)
        return batch

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn_custom)

    reponse_dict = {}

    print("Starting inference...")
    
    batch_count = 0

    for batch in tqdm(dataloader, desc="Inference"):
        # batch is a list of feature dicts
        
        # Prepare inputs for the batch
        batch_messages = []
        batch_meta = [] # Store IDs to map back later
        
        for item in batch:
            try:
                pil_image = item['pil_image']
    
                # Metadata
                # Note: TFRecord stores scalar ints, but sometimes they come out as 1-element arrays or tensors depending on loader
                # Checks if items are tensors or just values
                # TFRecordDataset usually yields dict of values.
                subject_id = item['patient']
                study_id = item['study']
                dicom_id = item['image']

                pil_image.save(f"./save_images/CheXpert/{subject_id[0]}_{study_id[0]}.png")
                
                batch_meta.append({
                    'subject_id': subject_id,
                    'study_id': study_id,
                    'dicom_id': dicom_id
                })

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": PROMPT}
                        ]
                    }
                ]
                batch_messages.append(messages)
            except Exception as e:
                print(f"Skipping an item due to error in prep: {e}")
                continue
        
        if not batch_messages:
            continue

        # Process inputs
        try:
            inputs = processor.apply_chat_template(
                batch_messages,
                add_generation_prompt=True,
                padding=True,
                return_dict=True,
                return_tensors="pt",
                tokenize=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )
            
            responses_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Parse responses
            for i, full_response in enumerate(responses_text):
                meta = batch_meta[i]
                
                try:
                    # Extract model output part
                    # The response often contains the prompt + "model\n" + answer
                    if "model\n" in full_response:
                        response_content = full_response.split("model\n")[1]
                    else:
                        response_content = full_response

                    # Try to find JSON
                    json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        json_object = json.loads(json_str)
                    else:
                        # Fallback parsing
                        parts = re.split(r',\s*(Explanation):', response_content)
                        if len(parts) >= 2:
                             json_object = {
                                "Answer": parts[0].replace("Answer:", "").strip(),
                                "Explanation": parts[2].strip() if len(parts) > 2 else parts[1].strip()
                            }
                        else:
                            json_object = {"error": "Could not parse response", "raw": response_content}
                
                except Exception as e:
                    print(f"Error parsing response for {meta['dicom_id']}: {e}")
                    json_object = {"error": str(e), "raw": full_response}

                # Store result
                # dicom_id -> [study_id, subject_id, json_object]
                # Ensure IDs are strings as per original script
                reponse_dict[f"{meta['subject_id']}_{meta['study_id']}_{meta['dicom_id']}"] = [json_object]
            
            batch_count += 1

        except Exception as e:
            print(f"Error during batch inference: {e}")
            continue

        # Save Results
        if batch_count % 10 == 0:
            print(f"Saving intermediate results to {args.output_json}...")
            with open(args.output_json, "w") as f:
                json.dump(reponse_dict, f)

            with open(args.output_txt, "w") as f:
                for k, v in reponse_dict.items():
                    f.write(f"{k}: {v}\n")
        
        if batch_count == 50:
            break

    # Save final results
    print(f"Saving final results to {args.output_json}...")
    with open(args.output_json, "w") as f:
        json.dump(reponse_dict, f)

    with open(args.output_txt, "w") as f:
        for k, v in reponse_dict.items():
            f.write(f"{k}: {v}\n")

    print("Done.")

if __name__ == "__main__":
    main()
