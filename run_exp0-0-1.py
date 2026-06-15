import subprocess
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help='', type=str)
    
    args = parser.parse_args()
    demo_label = ["race", "sex", "age"]

    if args.dataset == "MIMIC":
        for demo in demo_label:
            subprocess.run(f'python run_exp0-1.py --dataset MIMIC --model mamba --train_path insurance_dataset_8_1_1_PMMthree_train_no_support_devices.csv --val_path insurance_dataset_8_1_1_PMMthree_val_no_support_devices.csv --test_path insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv --experiment_name Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_mamba_DoubleLinear_Lion1e-5_20251025 --weight_dir Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_mamba_DoubleLinear_Lion1e-5_20251025 --subgroup {demo}', shell=True)
    
        for demo in demo_label:
            subprocess.run(f'python run_exp0-1.py --dataset MIMIC --model swinTF --train_path insurance_dataset_8_1_1_PMMthree_train_no_support_devices.csv --val_path insurance_dataset_8_1_1_PMMthree_val_no_support_devices.csv --test_path insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv --experiment_name Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_swinTF_DoubleLinear_Lion1e-5_20251025 --weight_dir Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_swinTF_DoubleLinear_Lion1e-5_20251025 --subgroup {demo}', shell=True)
    
        for demo in demo_label:
            subprocess.run(f'python run_exp0-1.py --dataset MIMIC --model densenet --train_path insurance_dataset_8_1_1_PMMthree_train_no_support_devices.csv --val_path insurance_dataset_8_1_1_PMMthree_val_no_support_devices.csv --test_path insurance_dataset_8_1_1_PMMthree_test_no_support_devices.csv --experiment_name ValAug_TestNoAug_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_densenet_DoubleLinear_Lion1e-5_20251120 --weight_dir ValAug_TestNoAug_Delete_Support_Devices_8_1_1split_BS32_PMMthree_FULLIMAGE448_densenet_DoubleLinear_Lion1e-5_20251120 --subgroup {demo}', shell=True)

    elif args.dataset == "CheXpert":
        for demo in demo_label:
            subprocess.run(f'python run_exp0-1.py --dataset CheXpert --model mamba --train_path /mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord --val_path /mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord --test_path /mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord --experiment_name ValAug_TestNoAug_CheXpert_Delete_Support_Devices_8_1_1split_BS32_PMMthree_mamba_DoubleLinear_Lion1e-6_20251122 --weight_dir ValAug_TestNoAug_CheXpert_Delete_Support_Devices_8_1_1split_BS32_PMMthree_mamba_DoubleLinear_Lion1e-6_20251122 --subgroup {demo}', shell=True)
    
        for demo in demo_label:
            subprocess.run(f'python run_exp0-1.py --dataset CheXpert --model swinTF --train_path /mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord --val_path /mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord --test_path /mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord --experiment_name ValAug_TestNoAug_CheXpert_Delete_Support_Devices_8_1_1split_BS32_PMMthree_swinTF_DoubleLinear_Lion1e-6_20251122 --weight_dir ValAug_TestNoAug_CheXpert_Delete_Support_Devices_8_1_1split_BS32_PMMthree_swinTF_DoubleLinear_Lion1e-6_20251122 --subgroup {demo}', shell=True)
    
        for demo in demo_label:
            subprocess.run(f'python run_exp0-1.py --dataset CheXpert --model densenet --train_path /mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_train_insurance.tfrecord --val_path /mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_val_insurance.tfrecord --test_path /mnt/new_usb/jupyter-altis5526/Resplit_ExcludeSD_CheXpert_test_insurance.tfrecord --experiment_name CheXpert_Delete_Support_Devices_8_1_1split_BS128_PMMthree_densenet_DoubleLinear_Lion1e-5_20251125 --weight_dir CheXpert_Delete_Support_Devices_8_1_1split_BS128_PMMthree_densenet_DoubleLinear_Lion1e-5_20251125 --subgroup {demo}', shell=True)
    