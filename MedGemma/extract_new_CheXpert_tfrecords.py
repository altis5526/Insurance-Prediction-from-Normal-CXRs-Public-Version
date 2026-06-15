import pandas as pd
import os
import json
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    json_file = "./Gemma_parsed_data/CheXpert_response_Gemma1.5.json"
    train_insurance_file = "/mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord"
    val_insurance_file = "/mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord"
    test_insurance_file = "/mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord"

    original_train_insurance_dataset = tf.data.TFRecordDataset(train_insurance_file)
    original_val_insurance_dataset = tf.data.TFRecordDataset(val_insurance_file)
    original_test_insurance_dataset = tf.data.TFRecordDataset(test_insurance_file)

    output_train_tfrecord = "/mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord"
    output_val_tfrecord = "/mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord"
    output_test_tfrecord = "/mnt/new_usb/jupyter-altis5526/MedGemmaChecked_Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord"

    with open(json_file, "r") as f:
        response_dict = json.load(f)

    responses1 = []
    responses2 = []
    explan1 = []
    explan2 = []

## Manually change this line
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
        image = features['image'].int64_list.value[0]
        data_id = f"patient{ptid}"+"_"+f"study{study}"+"_"+f"view{image}_frontal"

        if data_id == "patient49053_study1_view1_frontal":
            responses1.append(1)
            responses2.append(1)
            continue
        elif data_id == "patient59833_study1_view1_frontal":
            responses1.append(0)
            responses2.append(1)
            continue
        elif data_id == "patient45165_study1_view1_frontal":
            responses1.append(0)
            responses2.append(0)
            continue
        elif data_id == "patient46422_study2_view1_frontal":
            responses1.append(0)
            responses2.append(1)
            continue

        try:
            answer = response_dict[data_id]["PROMPT1"]["answer"]
            if answer == "Yes":
                responses1.append(1)
            elif answer == "No":
                responses1.append(0)
            else:
                print(f"Unexpected answer: {data_id} - {response_dict[data_id]['PROMPT1']['answer']}")
                responses1.append(2)

        except:
            answer = response_dict[data_id]["PROMPT1"]["Answer"]
            if answer == "Yes":
                responses1.append(1)
            elif answer == "No":
                responses1.append(0)
            else:
                print(f"Unexpected answer: {data_id} - {response_dict[data_id]['PROMPT1']['Answer']}")
                responses1.append(2)
        try:
            explan1.append(response_dict[data_id]["PROMPT1"]["Explanation"])
        except:
            explan1.append(response_dict[data_id]["PROMPT1"]["explanation"])
            
        try:
            answer = response_dict[data_id]["PROMPT2"]["Answer"]
            if answer == "No Finding":
                responses2.append(1)
            elif answer == "Positive Finding":
                responses2.append(0)
            else:
                print(f"Unexpected answer: {data_id} - {answer}")
                responses2.append(2)
        except:
            try:
                answer = response_dict[data_id]["PROMPT2"]["No Finding"]
                if answer == True:
                    responses2.append(1)
                elif answer == False:
                    responses2.append(0)
                else:
                    print(f"Unexpected answer: {data_id} - {answer}")
                    responses2.append(2)
            except:
                answer = response_dict[data_id]["PROMPT2"]["answer"]
                if answer == "No Finding":
                    responses2.append(1)
                elif answer == "Positive Finding":
                    responses2.append(0)
                else:
                    print(f"Unexpected answer: {data_id} - {answer}")
                    responses2.append(2)
        try:
            explan2.append(response_dict[data_id]["PROMPT2"]["Explanation"])
        except:
            explan2.append(response_dict[data_id]["PROMPT2"]["explanation"])

## Manually change this line
    with tf.io.TFRecordWriter(output_test_tfrecord, options=None) as writer:
        idx = 0
        ## Manually change this line
        for raw_record in original_test_insurance_dataset:
            if responses1[idx] == 0 and responses2[idx] == 1:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                features = dict(example.features.feature.items())
                example = tf.train.Example(features=tf.train.Features(feature=features))
                output_string = example.SerializeToString()
                writer.write(output_string)
                idx += 1
            else: 
                idx += 1
                continue

## Manually change this line
    filtered_test_insurance_dataset = tf.data.TFRecordDataset(output_test_tfrecord)

    count = 0
    ## Manually change this line
    for raw_record in filtered_test_insurance_dataset:
        count += 1
                
    print("Total responses1 collected:", len(responses1))
    print("Total responses2 collected:", len(responses2))

    print("Final filtered count:", count)

## train_filtered_count = 2360
## val_filtered_count = 289
## test_filtered_count = 297
    