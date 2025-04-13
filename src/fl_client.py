import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataset import load_data_from_pickle
import os
import argparse
import numpy as np
import sys

_output_tracker = {}

def get_model():
    from torchvision import models
    
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
        print(f"\n[Client {self.client_id}] Initializing with device: {self.device}")

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

        data_path = os.path.join(client_dir, "client_data.pkl")
        self.train_loader, self.val_loader = load_data_from_pickle(
            data_path, 
            transform=transform
        )
        
        self.model = get_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001, 
            weight_decay=1e-4
        )

    def get_parameters(self, config=None):
        params = []
        with torch.no_grad():
            for name, param in self.model.state_dict().items():
                if "num_batches_tracked" in name:
                    param_np = param.cpu().detach().numpy().astype(np.int64)
                else:
                    param_np = param.cpu().detach().numpy().astype(np.float32)
                
                if not param_np.flags.c_contiguous:
                    param_np = np.ascontiguousarray(param_np)
                    
                params.append(param_np)
        
        return params

    def set_parameters(self, parameters):
        state_dict_keys = list(self.model.state_dict().keys())
        state_dict = {}
        
        for i, name in enumerate(state_dict_keys):
            if i >= len(parameters):
                state_dict[name] = self.model.state_dict()[name]
                continue
            
            param = parameters[i]
            expected_shape = self.model.state_dict()[name].shape
            
            if not isinstance(param, np.ndarray):
                if "num_batches_tracked" in name:
                    param = np.frombuffer(param, dtype=np.int64)
                else:
                    param = np.frombuffer(param, dtype=np.float32)
            
            if param.size == np.prod(expected_shape):
                if expected_shape == ():
                    scalar_value = param.item() if param.size == 1 else param[0]
                    if "num_batches_tracked" in name:
                        state_dict[name] = torch.tensor(scalar_value, dtype=torch.int64).to(self.device)
                    else:
                        tensor_dtype = self.model.state_dict()[name].dtype
                        state_dict[name] = torch.tensor(scalar_value, dtype=tensor_dtype).to(self.device)
                else:
                    reshaped_param = param.reshape(expected_shape)
                    if "num_batches_tracked" in name:
                        state_dict[name] = torch.tensor(reshaped_param, dtype=torch.int64).to(self.device)
                    else:
                        tensor_dtype = self.model.state_dict()[name].dtype
                        state_dict[name] = torch.tensor(reshaped_param, dtype=tensor_dtype).to(self.device)
            else:
                state_dict[name] = self.model.state_dict()[name]
        
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        client_id = config.get("client_id", self.client_id)
        epochs = config.get("epochs", 1)
        round_num = config.get("round_num", 0)
        
        session_key = f"{client_id}_fit_{round_num}"
        
        global _output_tracker
        if session_key in _output_tracker:
            return self.get_parameters(), 0, {}
        _output_tracker[session_key] = True
        
        self.set_parameters(parameters)
        
        all_output = []
        all_output.append(f"\n[Client {client_id}] Training for {epochs} epochs")
        
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
            
            all_output.append(f"[Client {client_id}] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        val_loss, val_accuracy, val_output = self.evaluate_model(client_id, round_num)
        all_output.extend(val_output)
        
        print('\n'.join(all_output), flush=True)
        
        return self.get_parameters(), len(self.train_loader.dataset), {
            "train_accuracy": train_accuracies[-1],
            "train_loss": train_losses[-1],
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
        }

    def evaluate(self, parameters, config):
        client_id = config.get("client_id", self.client_id)
        round_num = config.get("round_num", 0)
        
        session_key = f"{client_id}_eval_{round_num}"
        
        global _output_tracker
        if session_key in _output_tracker:
            return 0, 0, {"val_accuracy": 0}
        _output_tracker[session_key] = True
        
        self.set_parameters(parameters)
        val_loss, val_accuracy, val_output = self.evaluate_model(client_id, round_num)
        
        print('\n'.join(val_output), flush=True)
        
        return val_loss, len(self.val_loader.dataset), {"val_accuracy": val_accuracy}
    
    def evaluate_model(self, client_id, round_num=0):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        batch_count = 0
        
        output = []
        
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
        
        output.append(f"\n[Client {client_id}] Validation - Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
        output.append(f"[Client {client_id}] Dataset: {total} samples, {correct} correct")
        
        return avg_loss, accuracy, output

def main():
    parser = argparse.ArgumentParser(description="Flower client")
    parser.add_argument("--client_dir", type=str, required=True, help="Client data directory")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080", help="Server address")
    args = parser.parse_args()
    
    client = FlowerClient(client_dir=args.client_dir)
    
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )

if __name__ == "__main__":
    main()