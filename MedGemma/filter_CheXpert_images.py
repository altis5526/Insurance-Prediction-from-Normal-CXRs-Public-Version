import os
import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

train_insurance_file = "/mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord"
val_insurance_file = "/mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord"
test_insurance_file = "/mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord"

original_train_insurance_dataset = tf.data.TFRecordDataset(train_insurance_file)
original_val_insurance_dataset = tf.data.TFRecordDataset(val_insurance_file)
original_test_insurance_dataset = tf.data.TFRecordDataset(test_insurance_file)

new_Gemma_detected_file = "./Gemma_parsed_data/CheXpert_response_Gemma1.5.json"
with open(new_Gemma_detected_file, 'r') as f:
    new_Gemma_detected_result = json.load(f)

new_Gemma_detected_name_list = []
for name, result in new_Gemma_detected_result.items():
    saved_name = name.split("_")[0].split("patient")[-1]
    saved_name = int(saved_name)
    new_Gemma_detected_name_list.append(str(saved_name))  


# output_train_insurance_file = "/mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord"
# output_val_insurance_file = "/mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord"
# output_test_insurance_file = "/mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord"

origin_id_list = []
match_count = 0
for record in original_train_insurance_dataset:
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    features = dict(example.features.feature.items())
    ptid = features['patient'].int64_list.value[0]
    study = features['study'].int64_list.value[0]
    origin_id_list.append(str(ptid))
    


    # if ptid in split["train_ids"]:
    #     example = tf.train.Example(features=tf.train.Features(feature=features))
    #     output_string = example.SerializeToString()
    #     writer.write(output_string)

# with tf.io.TFRecordWriter(output_train_insurance_file, options=None) as writer:
#     for record in original_train_insurance_dataset:
#         example = tf.train.Example()
#         example.ParseFromString(record.numpy())
#         features = dict(example.features.feature.items())
        
#         ptid = features['patient'].int64_list.value[0]
#         if ptid in split["train_ids"]:
#             example = tf.train.Example(features=tf.train.Features(feature=features))
#             output_string = example.SerializeToString()
#             writer.write(output_string)

#     for record in original_test_insurance_dataset:
#         example = tf.train.Example()
#         example.ParseFromString(record.numpy())
#         features = dict(example.features.feature.items())
        
#         ptid = features['patient'].int64_list.value[0]
#         if ptid in split["train_ids"]:
#             example = tf.train.Example(features=tf.train.Features(feature=features))
#             output_string = example.SerializeToString()
#             writer.write(output_string)


# with tf.io.TFRecordWriter(output_val_insurance_file, options=None) as writer:
#     for record in original_train_insurance_dataset:
#         example = tf.train.Example()
#         example.ParseFromString(record.numpy())
#         features = dict(example.features.feature.items())
        
#         ptid = features['patient'].int64_list.value[0]
#         if ptid in split["val_ids"]:
#             example = tf.train.Example(features=tf.train.Features(feature=features))
#             output_string = example.SerializeToString()
#             writer.write(output_string)

#     for record in original_test_insurance_dataset:
#         example = tf.train.Example()
#         example.ParseFromString(record.numpy())
#         features = dict(example.features.feature.items())
        
#         ptid = features['patient'].int64_list.value[0]
#         if ptid in split["val_ids"]:
#             example = tf.train.Example(features=tf.train.Features(feature=features))
#             output_string = example.SerializeToString()
#             writer.write(output_string)


# with tf.io.TFRecordWriter(output_test_insurance_file, options=None) as writer:
#     for record in original_train_insurance_dataset:
#         example = tf.train.Example()
#         example.ParseFromString(record.numpy())
#         features = dict(example.features.feature.items())
        
#         ptid = features['patient'].int64_list.value[0]
#         if ptid in split["test_ids"]:
#             example = tf.train.Example(features=tf.train.Features(feature=features))
#             output_string = example.SerializeToString()
#             writer.write(output_string)

#     for record in original_test_insurance_dataset:
#         example = tf.train.Example()
#         example.ParseFromString(record.numpy())
#         features = dict(example.features.feature.items())
        
#         ptid = features['patient'].int64_list.value[0]
#         if ptid in split["test_ids"]:
#             example = tf.train.Example(features=tf.train.Features(feature=features))
#             output_string = example.SerializeToString()
#             writer.write(output_string)


