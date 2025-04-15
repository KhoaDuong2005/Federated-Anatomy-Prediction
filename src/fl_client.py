import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from dataset import load_data_from_pickle
import os
import argparse
import numpy as np

def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
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
    return model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_id = os.path.basename(client_dir).replace("client_", "")

        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        data_path = os.path.join(client_dir, "client_data.pkl")
        self.train_loader, self.val_loader = load_data_from_pickle(
            data_path, transform=transform
        )
        
        self.model = get_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001, 
            weight_decay=1e-4
        )
        
        # Store parameter names for consistent ordering
        self.param_keys = list(self.model.state_dict().keys())

    def get_parameters(self, config=None):
        params = []
        with torch.no_grad():
            for name in self.param_keys:
                param = self.model.state_dict()[name]
                # Convert to appropriate numpy type
                if param.dtype == torch.int64:
                    param_np = param.cpu().numpy().astype(np.int64)
                else:
                    param_np = param.cpu().numpy().astype(np.float32)
                
                # Ensure contiguous memory layout
                param_np = np.ascontiguousarray(param_np)
                params.append(param_np)
        return params

    def set_parameters(self, parameters):
        state_dict = {}
        
        for i, name in enumerate(self.param_keys):
            if i >= len(parameters):
                continue
                
            target_param = self.model.state_dict()[name]
            
            # Convert to numpy if needed
            if not isinstance(parameters[i], np.ndarray):
                dtype = np.int64 if target_param.dtype == torch.int64 else np.float32
                parameters[i] = np.frombuffer(parameters[i], dtype=dtype)
            
            # Create tensor with correct dtype
            if target_param.dtype == torch.int64:
                if parameters[i].size == 1:  # Scalar
                    state_dict[name] = torch.tensor(
                        int(parameters[i].item()), 
                        dtype=torch.int64,
                        device=self.device
                    )
                else:
                    state_dict[name] = torch.tensor(
                        parameters[i], 
                        dtype=torch.int64,
                        device=self.device
                    )
            else:
                state_dict[name] = torch.tensor(
                    parameters[i], 
                    dtype=target_param.dtype,
                    device=self.device
                )
                
        # Load parameters
        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        epochs = config.get("epochs", 1)
        self.set_parameters(parameters)
        
        train_accuracies = []
        train_losses = []

        for epoch in range(epochs):
            self.model.train()
            correct = 0
            total = 0
            running_loss = 0.0
            batch_count = 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                batch_count += 1

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / batch_count if batch_count > 0 else 0
            epoch_accuracy = correct / total if total > 0 else 0
            
            train_accuracies.append(epoch_accuracy)
            train_losses.append(epoch_loss)

        val_loss, val_accuracy = self.evaluate_model()
        
        return self.get_parameters(), len(self.train_loader.dataset), {
            "train_accuracy": train_accuracies[-1],
            "train_loss": train_losses[-1],
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_loss, val_accuracy = self.evaluate_model()
        
        return val_loss, len(self.val_loader.dataset), {"val_accuracy": val_accuracy}
    
    def evaluate_model(self):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                batch_loss = self.criterion(outputs, labels).item()
                total_loss += batch_loss
                batch_count += 1

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        
        return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description="Flower client")
    parser.add_argument("--client_dir", type=str, required=True)
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()
    
    client = FlowerClient(client_dir=args.client_dir)
    
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )

if __name__ == "__main__":
    main()