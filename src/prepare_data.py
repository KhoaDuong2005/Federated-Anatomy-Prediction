import os
import json
import shutil
import numpy as np
import pickle
import argparse
import sys
from pathlib import Path
from data_loader import load_images_from_folders, save_images_based_on_config
from validators import validate_client_data

def prepare_client_data(source_dir, output_dirs, split_method="random"):
    print(f"Loading data from {source_dir}...")
    
    try:
        config = validate_client_data(source_dir)
        images, labels = save_images_based_on_config(
            source_dir, 
            config, 
            size=(224, 224)
        )
        print("Loaded data using config file")
    except (FileNotFoundError, ValueError):
        print("No valid config found. Loading data based on folder structure...")
        images, labels, class_mapping = load_images_from_folders(
            source_dir, 
            size=(224, 224)
        )
        
        config = {
            "label_format": "folder",
            "image_folder": "",
            "class_mapping": class_mapping
        }

    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)
        
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    num_clients = len(output_dirs)

    if split_method == "random":
        indices = np.random.permutation(len(images))
        client_indices = [np.array(split, dtype=int) for split in np.array_split(indices, num_clients)]

    elif split_method == "class_balanced":
        class_indices = {}
        for i, label in enumerate(labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)
        
        client_indices = [[] for _ in range(num_clients)]
        
        for class_label, indices in class_indices.items():
            np.random.shuffle(indices)
            splits = np.array_split(indices, num_clients)
            for i, split in enumerate(splits):
                client_indices[i].extend(split)
        
        for i in range(len(client_indices)):
            np.random.shuffle(client_indices[i])
            client_indices[i] = np.array(client_indices[i], dtype=int)
    
    else:
        raise ValueError(f"Unknown split method: {split_method}")
    
    for i, output_dir in enumerate(output_dirs):
        client_images = [images[idx] for idx in client_indices[i]]
        client_labels = [labels[idx] for idx in client_indices[i]]

        client_data = {
            "images": client_images,
            "labels": client_labels,
        }

        data_path = os.path.join(output_dir, "client_data.pkl")

        with open(data_path, "wb") as f:
            pickle.dump(client_data, f)

        print(f"Client {i+1}: Saved {len(client_images)} images to {data_path}")
        
        label_counts = {}
        for label in client_labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
            
        print(f"  - Class distribution: {label_counts}")

def main():
    parser = argparse.ArgumentParser(description="Prepare client data for Federated Learning")
    parser.add_argument("--data_dir", "--source_dir", type=str, default="D:/Docs/chest_xray", 
                      help="Path to the main data directory")
    parser.add_argument("--output_dir", type=str, default=None, 
                      help="Base output directory (if not using data_dir structure)")
    parser.add_argument("--num_clients", type=int, default=3, 
                      help="Number of clients to simulate")
    parser.add_argument("--split_method", type=str, choices=["random", "class_balanced"], 
                      default="class_balanced", 
                      help="Method to split the data")
    args = parser.parse_args()
    
    source_dir = args.data_dir
    if "train" in os.listdir(source_dir):
        source_dir = os.path.join(args.data_dir, "train")
    
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory {source_dir} does not exist!")
        sys.exit(1)
    
    if args.output_dir:
        client_data_dir = args.output_dir
    else:
        client_data_dir = os.path.join(args.data_dir, "client_data")
    
    os.makedirs(client_data_dir, exist_ok=True)
    
    client_dirs = [
        os.path.join(client_data_dir, f"client_{i+1}") 
        for i in range(args.num_clients)
    ]
    
    print(f"Preparing data for {args.num_clients} clients using {args.split_method} split...")
    prepare_client_data(source_dir, client_dirs, args.split_method)
    print("Client data preparation complete!")

if __name__ == "__main__":
    main()