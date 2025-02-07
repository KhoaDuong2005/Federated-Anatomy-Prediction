import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pymongo
from store_and_retrieve import load_images_from_mongodb
from data_loader import show_image

class AnatomyPredictionDataset(Dataset):
    def __init__(self, modality: str, body_part: str, is_anatomy: str = None, is_validate: bool = False, transforms=None):
        self.transforms = transforms
        self.data = load_images_from_mongodb(modality, modality, body_part, is_anatomy, is_validate)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_info = self.data[idx]
        image = image_info["image_array"]
        label = image_info["metadata"].get("is_anatomy", None)
        label = 0 if label == "False" else 1

        return image, label

    @property
    def classes(self):
        return sorted(set([image_info["metadata"].get("label") for image_info in self.data]))




query = {}
modality = "mri"
body_part = "brain"
is_anatomy = None
is_validate = False



dataset = AnatomyPredictionDataset(modality, body_part, is_anatomy, is_validate, transforms)

        
    
