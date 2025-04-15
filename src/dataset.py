import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
import pickle
from typing import List, Dict, Tuple, Any
from torchvision import transforms

class LocalDataset(Dataset):
    def __init__(self, images: list, labels: list, transform=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_np = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image_np)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label)
        return image, label

class NumpyDataset(Dataset):
    def __init__(self, images: list, labels: list, transform=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_np = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image_np)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label)
        return image, label

def load_data_from_pickle(data_path, transform=None, split_pct=0.2, batch_size=32):
    with open(data_path, "rb") as f:
        client_data = pickle.load(f)
    
    if "images" not in client_data or "labels" not in client_data:
        raise ValueError(f"Invalid data format in {data_path}")
    
    images = client_data["images"]
    labels = client_data["labels"]
    
    dataset = LocalDataset(images, labels, transform=transform)
    
    val_size = int(len(dataset) * split_pct)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size)
    
    return trainloader, valloader

def load_image_data(data_dir: str, target_size=(256, 256)):
    images = []
    labels = []
    label_names = sorted([directory for directory in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, directory))])
    
    label_index = {}
    for index, label_name in enumerate(label_names):
        label_index[label_name] = index

    for label_name in label_names:
        folder_path = os.path.join(data_dir, label_name)
        for file in os.listdir(folder_path):
            try:
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    image = Image.open(file_path).convert("RGB")
                    image = image.resize(target_size)
                    image_np = np.array(image)
                    images.append(image_np)
                    labels.append(label_index[label_name])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return images, labels, label_index