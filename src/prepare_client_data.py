import os
import json
import shutil
import numpy as np
import pickle
import argparse
from pathlib import Path
from data_loader import save_images_based_on_config
from validators import validate_client_data

def prepare_client_data(source_dir, output_dirs, split_method="random"):
    config = validate_client_data(source_dir)

    print(f"Loading data from {source_dir}...")
    images, filenames, labels = save_images_based_on_config(
        source_dir, 
        config, 
        size=(224, 224)
    )

    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)

        shutil.copy(
            os.path.join(source_dir, "config.json"),
            os.path.join(output_dir, "config.json")
        )

    num_clients = len(output_dirs)

    if split_method == "random":
        indices = np.random.permutation(len(images))
        client_indices = [np.array(split, dtype=int) for split in np.array_split(indices, num_clients)]

    elif split_method == "class_balanced":
        positive_indices = [i for i, label in enumerate(labels) if label]
        negative_indices = [i for i, label in enumerate(labels) if not label]

        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)

        positive_splits = [np.array(split, dtype=int) for split in np.array_split(positive_indices, num_clients)]
        negative_splits = [np.array(split, dtype=int) for split in np.array_split(negative_indices, num_clients)]

        client_indices = [np.concatenate((pos, neg)) for pos, neg in zip(positive_splits, negative_splits)]

        for i in range(len(client_indices)):
            np.random.shuffle(client_indices[i])
    
    else:
        raise ValueError(f"Unknown split method: {split_method}")
    

    for i, output_dir in enumerate(output_dirs):
        client_images = [images[idx] for idx in client_indices[i]]
        client_labels = [labels[idx] for idx in client_indices[i]]
        client_filenames = [filenames[idx] for idx in client_indices[i]]

        client_data = {
            "images": client_images,
            "labels": client_labels,
            "filenames": client_filenames,
        }

        data_path = os.path.join(output_dir, "client_data.pkl")

        with open(data_path, "wb") as f:
            pickle.dump(client_data, f)

        print(f"Client {i+1}: Save {len(client_images)} images to {data_path}")
        print(f"  - Positive samples: {sum(client_labels)}")
        print(f"  - Negative samples: {len(client_labels) - sum(client_labels)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare client data for federated learning")
    parser.add_argument("--source_dir", type=str, required=True, help="Source data directory")
    parser.add_argument("--output_dir", type=str, default="client_data", help="Base output directory")
    parser.add_argument("--num_clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--split_method", type=str, choices=["random", "class_balanced"], 
                        default="class_balanced", help="Method to split the data")
    args = parser.parse_args()
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create client directories
    client_dirs = [
        os.path.join(args.output_dir, f"client_{i+1}") 
        for i in range(args.num_clients)
    ]
    
    # Prepare client data
    prepare_client_data(args.source_dir, client_dirs, args.split_method)

if __name__ == "__main__":
    main()