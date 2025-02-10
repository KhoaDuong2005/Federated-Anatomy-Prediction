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
    file_names_list = []
    try:
        if config["label_format"] == "label_folder":
            for dir_path, dir_names, file_names in os.walk(directory):
                for file_name in file_names:
                    file_path = os.path.join(dir_path, file_name)
                    is_anatomy = dir_path.endswith("anatomy")
                    try:
                        print(f"Loading file: {file_path}")
                        image_array = load_medical_image(file_path, size)
                        images.append(image_array)
                        file_names_list.append(file_name)
                        labels.append(is_anatomy)
                    except Exception as exception:
                        print(f"Skipping file: {file_name}: {exception}")
            return images, file_names_list, labels
        
        elif config["label_format"] == "label_file":
            folder_path = os.path.join(directory, config["image_folder"])
            for dir_path, dir_names, file_names in os.walk(folder_path):
                for file_name in file_names:
                    file_path = os.path.join(dir_path, file_name)
                    is_anatomy = os.path.splitext(file_name)[0].endswith("_1")
                    try:
                        print(f"Loading file: {file_path}")
                        image_array = load_medical_image(file_path, size)
                        images.append(image_array)
                        file_names_list.append(file_name)
                        labels.append(is_anatomy)
                    except Exception as exception:
                        print(f"Skipping file: {file_name}: {exception}")
            return images, file_names_list, labels
        
        elif config["label_format"] == "label_csv":
            folder_path = os.path.join(directory, config["image_folder"])
            csv_path = os.path.join(directory, config["label_file_path"])
            df = pd.read_csv(csv_path, header=None, names=["file_name", "label1", "label2", "label"])
            print(df)
            for dir_path, dir_names, file_names in os.walk(folder_path):
                for file_name in file_names:
                    file_path = os.path.join(folder_path, file_name)
                    try:
                        is_anatomy = df.loc[df["file_name"] == file_name, "label"].values[0]
                        image_array = load_medical_image(file_path, size)
                        images.append(image_array)
                        file_names_list.append(file_name)
                        labels.append(is_anatomy)
                    except Exception as exception:
                        print(f"Skipping file: {file_name}: {exception}")
            return images, file_names_list, labels
    except Exception as exception:
        print(f"Error when trying to load images based on config: {exception}")
    



def show_image(image_array, modality, body_part):
    plt.imshow(image_array[:,:,0], cmap = "gray")
    plt.title(f"{modality} {body_part}")
    plt.show()


