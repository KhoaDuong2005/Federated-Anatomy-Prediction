
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, List, Union
from store_and_retrieve import load_images_from_mongodb
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class MongoDBDataset(Dataset):
    def __init__(self, query: Dict = None, modality: str = None, body_part: str = None, is_anatomy: str = None, transform = None):
        super().__init__ ()
        self.data = load_images_from_mongodb(query, modality, body_part, is_anatomy)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_info = self.data[idx]
        image_array = image_info["image_array"]
        image = Image.fromarray(image_array.astype(np.uint8))
        

        if self.transform:
            image = self.transform(image)
        
        label =  1 if image_info["metadata"].get("is_anatomy", None) else 0

        return image, label

    @property
    def classes(self):
        # Use "is_anatomy" or another correct key, and filter out None values
        return sorted(set([image_info["metadata"].get("is_anatomy") for image_info in self.data if image_info["metadata"].get("is_anatomy") is not None]))


dataset = MongoDBDataset({}, "mri", "brain", None, transform=transform)

valid_pct = 0.2
num_valid = int(valid_pct * len(dataset))
num_train =  len(dataset) - num_valid
train_data, valid_data = random_split(dataset, [num_train, num_valid], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)


model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_of_features = model.fc.in_features
model.fc = nn.Linear(num_of_features, len(dataset.classes))
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)


def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs = 3):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_loss = running_loss / len(train_loader)

        #Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs}:\nLoss = {avg_loss:.4f}\nTrain Accuracy: {train_acc:.4f}\nValidate Accuracy: {val_acc:.4f}\n")
    
train_model(model, train_loader, valid_loader, criterion, optimizer, epochs = 5)





