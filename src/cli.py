from data_loader import *
from store_and_retrieve import *
from validators import *
import os
import argparse
import random
import json

def parse_arg():
    parser = argparse.ArgumentParser(description="Load and save all images from a directory to MongoDB")
    parser.add_argument("directory", type=str, nargs="?", default=None , help="Path to the directory containing medical images")
    parser.add_argument("-m", "--mod", type=str, help="Specify the imaging modality (mri, ct, xray)")
    parser.add_argument("b", "--body", type=str, help="Specify the body part (chest, brain, abdomen, leg, ...)")
    parser.add_argument("--im", type=int, help="Specify number of image randomly print in a specific database in MongoDB")
    parser.add_argument("--size", type=str, default="256,256", help="Specify the size of image before saving to MongoDB (default = 256,256)")
    parser.add_argument("--cfg", type=str, nargs=3, help="Create a config file for the client data")


    args = parser.parse_args()

    try:
        modality = validate_modality(args.mod)
        body_part = validate_body_part(args.body)
        is_anatomy = None

        if args.cfg:
            try:
                from create_config import create_config_file
                image_folder, labels_format, label_file_path = args.cfg
                create_config_file(args.directory, image_folder, labels_format, label_file_path)
                print(f"Successfully created the config file:\n")
                print(f"Image folder: {image_folder}\nLabel format: {labels_format}\nLabel file path: {label_file_path}")
            except Exception as exception:
                print(f"Error when creating the config: {exception}")

        if args.directory and not args.cfg:
            try:
                config = validate_client_data(args.directory)
                target_size = args.size
                target_size = tuple(map(int, target_size.split(',')))
                process_directory(args.directory, modality, body_part, is_anatomy, target_size, config)
            except Exception as exception:
                print(f"Error when going through directory: {exception}")

        if args.im:
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