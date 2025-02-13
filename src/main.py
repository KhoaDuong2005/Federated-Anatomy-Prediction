from data_loader import *
from store_and_retrieve import *
from validators import *
import os
import argparse
import random
import json
from typing import List


def process_directory(directory: str, modality: str, body_part: str, label: List, size: tuple):
    metadata_list = []
    try:
        images_array, file_names, label = save_images_based_on_config(directory, config, size)
        save_images_to_mongodb(images_array, file_names, metadata_list, modality, body_part, label)
    except Exception as exception:
            print(f"No valid image found in the directory, error: {exception}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and save all images from a directory to MongoDB")
    parser.add_argument("directory", type=str, nargs="?", default=None , help="Path to the directory containing medical images")
    parser.add_argument("--mod", type=str, required=True, help="Specify the imaging modality (mri, ct, xray)")
    parser.add_argument("--body", type=str, required=True, help="Specify the body part (chest, brain, abdomen, leg, ...)")
    parser.add_argument("--im", type=int, help="Specify number of image randomly print in a specific database in MongoDB")
    parser.add_argument("--size", type=str, default="256,256", help="Specify the size of image before saving to MongoDB (default = 256,256)")
    # parser.add_argument("--ana", action="store_true", help="Specify whether the images being saved have anatomy or not")

    args = parser.parse_args()

    try:
        modality = validate_modality(args.mod)
        body_part = validate_body_part(args.body)
        is_anatomy = None


        if args.directory:
            try:
                config = validate_client_data(args.directory)
                target_size = args.size
                target_size = tuple(map(int, target_size.split(',')))
                process_directory(args.directory, modality, body_part, is_anatomy, target_size)
                print(f"Files have been saved as {target_size}, if you want to change it, use --size x,y as a parameter (default is 256,256)")
            except Exception as exception:
                print(f"Error when going through directory: {exception}")

        if args.im is not None:
            from store_and_retrieve import get_image_info
            images_info = get_image_info(None, modality, body_part, is_anatomy)
            for i, image_info in enumerate(images_info):
                pass
            for i in range (args.im):
                random_number = random.randint(0, (len(images_info)))
                print(f"Image shape: {images_info[random_number]["image_array"].shape}")
                show_image(images_info[random_number]["image_array"], modality, body_part)

    except Exception as exception:
        print(f"Error: {exception}")


