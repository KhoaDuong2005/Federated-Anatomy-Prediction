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


        if image_array.ndim == 3 and image_array.shape[2] <= 4:
            image_array = image_array[:, :, 0]
        elif image_array.ndim == 3 and image_array.shape[0] <= 3:
            image_array = image_array[0, :, :]

        image_array = cv2.resize(image_array, size, interpolation=cv2.INTER_AREA)
        
        image_array = np.stack([image_array] * 3, axis=2)
        return image_array
    except Exception as exception:
        raise ValueError(f"File not supported, error: {exception}")

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
            df = pd.read_csv(csv_path)
            label_dictionary = dict(zip(df["SeriesInstanceUID"], zip(df["normal"], df["abnormal"])))

            for dir_path, dir_names, file_names in os.walk(folder_path):
                for file_name in file_names:
                    file_path = os.path.join(dir_path, file_name)
                    folder_name = os.path.basename(os.path.dirname(file_path))
                    try:
                        is_anatomy = label_dictionary[folder_name][1] == 1
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
    image_array = image_array[:, :, 0]
    plt.imshow(image_array, cmap = "gray")
    plt.title(f"{modality} {body_part}")
    plt.show()


