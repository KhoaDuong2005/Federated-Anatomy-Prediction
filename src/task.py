import os
import pickle
import numpy as np
import math

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

def get_image_transforms(is_train=True):
    if not is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)


def load_image_data(data_dir: str, target_size=(256, 256)):
    images = []
    labels = []
    label_names = sorted([directory for directory in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, directory))])
    
    label_index = {}

    for index, label_name in enumerate(label_names):
        label_index[label_name] = index

    for label_name in label_names:
        folder_path = os.path.join(data_dir, label_name)
        for file in tqdm(os.listdir(folder_path), desc=f"Loading ({label_name})"):
            file_path = os.path.join(folder_path, file)
            try:
                image = Image.open(file_path).convert("RGB")
                image = image.resize(target_size)
                image_np = np.array(image)
                images.append(image_np)
                labels.append(label_index[label_name])
            except Exception as e:
                print(f"Skipping file {file_path}, error: {e}")
    return images, labels

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



def load_data(data_dir: str, cache_filename: str):
    cache_file = os.path.join(data_dir, cache_filename)
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as file:
            images, labels = pickle.load(file)
    else:
        images, labels = load_image_data(data_dir)
        with open(cache_file, "wb") as file:
            pickle.dump((images, labels), file)

    dataset = NumpyDataset(images, labels, transform=transform)
    
    valid_pct = 0.2
    num_valid = int(valid_pct * len(dataset))
    num_train = len(dataset) - num_valid

    train_dataset, valid_dataset = random_split(dataset, [num_train, num_valid],generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32)

    return train_loader, valid_loader

def load_test_data(data_dir, cache_filename):
    cache_file = os.path.join(data_dir, cache_filename)
    
    if os.path.exists(cache_file):
        print(f"Loading test data from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            test_data = pickle.load(f)
            test_images = test_data["images"]
            test_labels = test_data["labels"]
    else:
        print(f"Loading test data from directory: {data_dir}")
        test_images, test_labels, _ = load_images_from_folders(data_dir, size=(224, 224))
        
        with open(cache_file, "wb") as f:
            pickle.dump({"images": test_images, "labels": test_labels}, f)
    
    test_dataset = NumpyDataset(test_images, test_labels, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    print(f"Loaded {len(test_dataset)} test samples")
    return test_loader


def train(net, trainloader, epochs, lr, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    
    dataset_size = len(trainloader.dataset)
    if dataset_size > 0:
        T_max = max(1, math.floor(dataset_size / 32))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        scheduler = None
    
    net.train()
    running_loss = 0.0
    
    if dataset_size == 0:
        return 0.0
    
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running_loss += loss.item()
    
    avg_trainloss = running_loss / len(trainloader) / epochs if len(trainloader) > 0 else 0.0
    return avg_trainloss

def test(net, testloader, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.eval()
    
    correct = 0
    loss = 0.0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / len(testloader.dataset) if len(testloader.dataset) > 0 else 0
    avg_loss = loss / len(testloader) if len(testloader) > 0 else 0
    return avg_loss, accuracy

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    net.load_state_dict(state_dict, strict=True)