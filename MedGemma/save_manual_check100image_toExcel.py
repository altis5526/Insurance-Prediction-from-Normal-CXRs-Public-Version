import pandas as pd
import os

if __name__ == "__main__":
    image_dir = "/mnt/new_usb/jupyter-altis5526/random_sample_MIMIC_500_for_MedGemma/"
    image_list = os.listdir(image_dir)
    for image in image_list:
        image_num = image.split("_")[0]
        if len(image_num) == 3:
            continue
        elif len(image_num) == 2:
            image_num = "0" + image_num
            rest_of_the_name = "_".join(image.split("_")[1:])
            os.rename(os.path.join(image_dir, image), os.path.join(image_dir, image_num + "_" + rest_of_the_name))
        elif len(image_num) == 1:
            image_num = "00" + image_num
            rest_of_the_name = "_".join(image.split("_")[1:])
            os.rename(os.path.join(image_dir, image), os.path.join(image_dir, image_num + "_" + rest_of_the_name))

    image_list = os.listdir(image_dir)
    df = pd.DataFrame(image_list, columns=["dicom_id"])
    df.to_csv("manual_check_500image.csv", index=False)
    
