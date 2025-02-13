import numpy as np 
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from typing import List
from tqdm import tqdm
from store_and_retrieve import save_images_to_mongodb


def load_medical_image(file_path, size):
    try:
        image = sitk.ReadImage(file_path)
        image_array = sitk.GetArrayFromImage(image)
        image_array = np.array(image_array)


        if image_array.ndim == 3 and image_array.shape[2] <= 4:
            image_array = image_array[:, :, 0]
        elif image_array.ndim == 3 and image_array.shape[0] <= 3:
            image_array = image_array[0, :, :]

        image_array = cv2.resize(image_array, size, interpolation=cv2.INTER_AREA)
        
        image_array = np.stack([image_array] * 3, axis=2)
        return image_array
    except Exception as exception:
        raise ValueError(f"File not supported, error: {exception}")

def save_images_based_on_config(directory, config, size):
    images = []
    labels = []
    file_names_list = []
    folder_path = os.path.join(directory, config["image_folder"])

    if not os.path.exists(folder_path):
        raise ValueError(f"Directory {folder_path} does not exist")

    try:
        all_files = []
        for dir_path, _, file_names in os.walk(folder_path):
            all_files.extend([(dir_path, file_name) for file_name in file_names])

        label_dict = {}
        if config["label_format"] == "csv":
            csv_path = os.path.join(directory, config["label_file_path"])
            df = pd.read_csv(csv_path)
            label_dict = dict(zip(df["SeriesInstanceUID"], zip(df["normal"], df["abnormal"]))) #will implment this more for flexibility (not just normal and abnormal)

        for dir_path, file_name in tqdm(all_files, desc="Saving images", unit=" images"): #tqdm for visualizing the progress bar
            file_path = os.path.join(dir_path, file_name)
            try:
                if config["label_format"] == "folder":
                    is_anatomy = dir_path.endswith("anatomy")
                elif config["label_format"] == "file":
                    is_anatomy = os.path.splitext(file_name)[0].endswith("_1")
                elif config["label_format"] == "csv":
                    folder_name = os.path.basename(dir_path)
                    is_anatomy = label_dict.get(folder_name, (0, 0))[1] == 1 #check the tuple for abnormal, and if folder not in the csv, return (0, 0)
                else:
                    raise ValueError(f"Invalid label format: {config['label_format']}")
                
                image_array = load_medical_image(file_path, size)
                images.append(image_array)
                labels.append(is_anatomy)
                file_names_list.append(file_name)

            except Exception as exception:
                print(f"Skipping file {file_path}, error: {exception}")
            
        return images, file_names_list, labels
        

    except Exception as exception:
        print(f"Error when trying to load images based on config: {exception}")
        return [], [], []
    

def show_image(image_array, modality, body_part):
    image_array = image_array[:, :, 0]
    plt.imshow(image_array, cmap = "gray")
    plt.title(f"{modality.capitalize()} {body_part.capitalize()}")
    plt.show()


def process_directory(directory: str, modality: str, body_part: str, label: List, size: tuple, config: dict):
    metadata_list = []
    try:
        images_array, file_names, label = save_images_based_on_config(directory, config, size)
        save_images_to_mongodb(images_array, file_names, metadata_list, modality, body_part, label)
    except Exception as exception:
            print(f"Error when trying to save images: {exception}")