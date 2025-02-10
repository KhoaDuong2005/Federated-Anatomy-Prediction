import numpy as np 
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd


def load_medical_image(file_path, size):
    try:
        image = sitk.ReadImage(file_path)
        image_array = sitk.GetArrayFromImage(image)
        image_array = np.array(image_array)
        image_array = cv2.resize(image_array, size, interpolation=cv2.INTER_AREA)

        if image_array.ndim == 3:
            image_array = image_array[:, :, 0]
    
        image_array = np.stack([image_array] * 3, axis=2)

        return image_array
    except:
        raise ValueError("File not supported")

def load_images_based_on_config(directory, config, size):
    images = []
    labels = []
    file_names = []

    if config["label_format"] == "label_folder":
        for folder_name in os.listdir(directory):
            folder_path = os.path.join(directory, folder_name)
            if os.path.isdir(folder_path):
                is_anatomy = True if folder_name.lower() == "anatomy" else False
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.isfile(file_path):
                        try:
                            image_array = load_medical_image(file_path, size)
                            images.append(image_array)
                            file_names.append(file_name)
                            labels.append(is_anatomy)
                        except ValueError as e:
                            print(f"Skipping file: {file_name}: {e}")
    return images, file_names, labels



def show_image(image_array, modality, body_part):
    plt.imshow(image_array[:,:,0], cmap = "gray")
    plt.title(f"{modality} {body_part}")
    plt.show()


