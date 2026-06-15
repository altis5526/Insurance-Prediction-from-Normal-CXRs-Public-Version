import pandas as pd
import os
import json

if __name__ == "__main__":
    json_file = "random100_CheXpert_response_Gemma1.5.json"
    ## For MIMIC
    # data_xlsx = "manual_check_500image.xlsx"
    data_csv = "manual_check_100CheXpert_image.csv"

    with open(json_file, "r") as f:
        reponse_dict = json.load(f)

    ## For MIMIC
    # data = pd.read_excel(data_xlsx)
    data = pd.read_csv(data_csv)
    data = data.sort_values(by='dicom_id', ascending=True)
    image_names = data['dicom_id'].tolist()
    dicom_ids = []
    ## For MIMIC
    # for image in image_names:
    #     dicom_id = image.split("_")[-1].split(".")[0]
    #     dicom_ids.append(dicom_id)

    for image in image_names:
        dicom_id = image.split(".")[0].split("_")[1:]
        dicom_id = "_".join(dicom_id)
        dicom_ids.append(dicom_id)

    responses1 = []
    responses2 = []
    explan1 = []
    explan2 = []
    
    idx1 = 0
    idx2 = 0
    # count response dict how many items
    print("Number of items in response dict:", len(reponse_dict))

    for dicom_id, response in reponse_dict.items():
        if str(dicom_id) == "73a7b5c1-5c07c9ca-ff925498-850bdfa0-7d50b22a":
            continue
        if str(dicom_id) == str(dicom_ids[idx1]):
            try:
                answer = response["PROMPT1"]["answer"]
                if answer == "Yes":
                    responses1.append(1)
                elif answer == "No":
                    responses1.append(0)
                else:
                    print(f"Unexpected answer: {idx1} - {response['PROMPT1']['answer']}")
            except:
                answer = response["PROMPT1"]["Answer"]
                if answer == "Yes":
                    responses1.append(1)
                elif answer == "No":
                    responses1.append(0)
                else:
                    print(f"Unexpected answer: {idx1} - {response['PROMPT1']['Answer']}")
            try:
                explan1.append(response["PROMPT1"]["Explanation"])
            except:
                explan1.append(response["PROMPT1"]["explanation"])
                
            idx1 += 1
    for dicom_id, response in reponse_dict.items(): 
        if str(dicom_id) == "73a7b5c1-5c07c9ca-ff925498-850bdfa0-7d50b22a":
            continue
        if str(dicom_id) == str(dicom_ids[idx2]):
            try:
                answer = response["PROMPT2"]["Answer"]
                if answer == "No Finding":
                    responses2.append(1)
                elif answer == "Positive Finding":
                    responses2.append(0)
                else:
                    print(f"Unexpected answer: {idx2} - {answer}")
            except:
                try:
                    answer = response["PROMPT2"]["No Finding"]
                    if answer == True:
                        responses2.append(1)
                    elif answer == False:
                        responses2.append(0)
                    else:
                        print(f"Unexpected answer: {idx2} - {answer}")
                except:
                    answer = response["PROMPT2"]["answer"]
                    if answer == "No Finding":
                        responses2.append(1)
                    elif answer == "Positive Finding":
                        responses2.append(0)
                    else:
                        print(f"Unexpected answer: {idx2} - {answer}")
            try:
                explan2.append(response["PROMPT2"]["Explanation"])
            except:
                explan2.append(response["PROMPT2"]["explanation"])
                
            idx2 += 1

        else:
            print(str(dicom_id))


    
    print("Total responses1 collected:", len(responses1))
    print("Total responses2 collected:", len(responses2))
    data["Answer1"] = responses1
    data["Answer2"] = responses2
    data["Explanation1"] = explan1
    data["Explanation2"] = explan2
    
    data.to_csv("manual_check_CheXpert_100image_Gemma1.5_with_answers.csv", index=False)
    