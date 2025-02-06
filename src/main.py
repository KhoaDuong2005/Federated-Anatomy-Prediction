from data_loader import load_medical_image, show_image
from store_and_retrieve import save_images_to_mongodb, load_images_from_mongodb
from validators import *
import os
import argparse
import random
import matplotlib.pyplot as plt


def process_directory(directory: str, modality: str, body_part: str, is_validate: bool):
    images = []
    file_names = []
    metadata_list = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        try:
            images_array = load_medical_image(file_path)
            images.append(images_array)
            file_names.append(filename)
            metadata_list.append({"source": "dataset", "filename": filename})
        
        except Exception as exception:
            print(f"Skipping file: {filename}: {exception}")
    if images:
        save_images_to_mongodb(images, file_names, metadata_list, modality, body_part, is_validate)
    else:
        print("No valid image found in the directory")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and save all images from a directory to MongoDB")
    parser.add_argument("directory", type=str, nargs="?", default=None , help="Path to the directory containing medical images")
    parser.add_argument("--val", default=False, action="store_true", help="Specify whether the directory is for trainning or validation dataset (default = Trainning)")
    parser.add_argument("--mod", type=str, required=True, help="Specify the imaging modality (mri, ct, xray)")
    parser.add_argument("--body", type=str, required=True, help="Specify the body part (chest, brain, abdomen, leg, ...)")
    parser.add_argument("--im", type=int, help="Specify number of image print of specific database in MongoDB")

    args = parser.parse_args()

    try:
        modality = validate_modality(args.mod)
        body_part = validate_body_part(args.body)
        is_validate = args.val

        if args.directory:
            process_directory(args.directory, modality, body_part, is_validate)

        if args.im is not None:
            from store_and_retrieve import get_image_info
            images_info = get_image_info(None, modality, body_part, is_validate)
            for i, image_info in enumerate(images_info):
                if i >= args.im:
                    break
                show_image(image_info["image_array"])

    except Exception as exception:
        print(f"Error: {exception}")