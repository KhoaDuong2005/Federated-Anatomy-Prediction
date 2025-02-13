import json
import os

def create_config_file(directory, image_folder, label_format, label_file_path):
    os.chdir(directory)
    with open("config.json", "w") as file:
        config = {
            "image_folder": image_folder,
            "label_format": label_format,
            "label_file_path": label_file_path
        }
        json.dump(config, file)
    return config