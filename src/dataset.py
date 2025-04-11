import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import pickle
from typing import List, Dict, Tuple, Any

class LocalDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_array = self.images[idx]
        label = self.labels[idx]

        image = Image.fromarray(image_array.astype(np.uint8))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
    
def load_data_from_pickle(data_path, transform=None, split_pct=0.2, batch_size=32):
    import torch.utils.data as data

    with open(data_path, "rb") as f:
        client_data = pickle.load(f)
    
    images = client_data["images"]
    labels = client_data["labels"]

    dataset = LocalDataset(images, labels, transform=transform)

    train_size = int((1 - split_pct) * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
        )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader

def create_dataset_from_config(client_dir, config=None, transform=None, batch_size=32, split_pct=0.2):
    from data_loader import save_images_based_on_config
    import torch.utils.data as data

    if config is None:
        from validators import validate_client_data
        config = validate_client_data(client_dir)

    images, filenames, labels = save_images_based_on_config(
        client_dir, 
        config, 
        size=(224, 224)
        )
    
    dataset = LocalDataset(images, labels, transform=transform)

    train_size = int((1 - split_pct) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
        )
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader
