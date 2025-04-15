import numpy as np 
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from typing import List, Tuple, Dict
from tqdm import tqdm
import pickle
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

def load_images_from_folders(directory, size=(224, 224)):
    images = []
    labels = []
    
    class_folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    if not class_folders:
        raise ValueError(f"No class folders found in {directory}")
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(class_folders))}
    
    print(f"Found {len(class_folders)} classes: {class_folders}")
    print(f"Class mapping: {class_to_idx}")
    
    for class_folder in class_folders:
        class_path = os.path.join(directory, class_folder)
        class_idx = class_to_idx[class_folder]
        
        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        
        print(f"Processing class '{class_folder}' ({class_idx}) with {len(files)} images")
        
        for file_name in tqdm(files, desc=f"Loading {class_folder}", unit="files"):
            file_path = os.path.join(class_path, file_name)
            try:
                image_array = load_medical_image(file_path, size)
                
                images.append(image_array)
                labels.append(class_idx)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return images, labels, class_to_idx

def save_images_based_on_config(directory, config, size=(224, 224)):
    if "image_folder" in config:
        folder_path = os.path.join(directory, config["image_folder"])
    else:
        folder_path = directory
    
    images, labels, class_mapping = load_images_from_folders(folder_path, size)
    return images, labels

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
