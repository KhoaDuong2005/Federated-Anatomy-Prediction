import torch
import json
import numpy as np
import os
import pickle
from random import random
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigRecord
from src.task import Net, get_weights, set_weights, test, train
from src.dataset import LocalDataset, load_data_from_pickle
from torch.utils.data import DataLoader
from torchvision import transforms

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, context: Context):
        self.client_state = context.state
        self.net = net
        self.trainloader = trainloader  
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        if "fit_metrics" not in self.client_state.config_records:
            self.client_state.config_records["fit_metrics"] = ConfigRecord()

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        lr = config.get("lr", 0.01)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            lr,
            self.device,
        )
        fit_metrics = self.client_state.config_records["fit_metrics"]
        if "train_loss_hist" not in fit_metrics:
            fit_metrics["train_loss_hist"] = [train_loss]
        else:
            fit_metrics["train_loss_hist"].append(train_loss)
        complex_metric = {"a": 123, "b": random(), "mylist": [1, 2, 3, 4]}
        complex_metric_str = json.dumps(complex_metric)
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": float(train_loss), "my_metric": complex_metric_str},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), int(len(self.valloader.dataset)), {"accuracy": float(accuracy)}

def client_fn(context: Context):
    net = Net()
    client_id = 0
    num_clients = int(context.run_config.get("num-clients", 3))
    if hasattr(context, 'node_id'):
        if isinstance(context.node_id, int):
            client_id = context.node_id % num_clients
        elif isinstance(context.node_id, str) and "-" in context.node_id:
            try:
                client_id = int(context.node_id.split("-")[1]) - 1
            except (IndexError, ValueError):
                client_id = 0
    
    print(f"Initializing client with ID: {client_id}")
    data_dir = context.run_config.get("data_dir", "D:/Docs/chest_xray")
    data_dir = os.path.normpath(data_dir)
    parent_dir = os.path.dirname(data_dir) if "train" in data_dir else data_dir
    client_data_dir = os.path.join(parent_dir, "client_data")
    print(f"Looking for client data in: {client_data_dir}")
    os.makedirs(client_data_dir, exist_ok=True)
    client_dir = os.path.join(client_data_dir, f"client_{client_id+1}")
    if os.path.exists(client_dir) and os.path.exists(os.path.join(client_dir, "client_data.pkl")):
        print(f"Loading pre-split data for client {client_id+1} from {client_dir}")
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainloader, valloader = load_data_from_pickle(
            os.path.join(client_dir, "client_data.pkl"),
            transform=train_transform,
            split_pct=0.2,
            batch_size=32
        )
    else:
        print(f"Client data not found. Please run prepare_client_data.py first.")
        print(f"Expected path: {client_dir}")
        empty_dataset = LocalDataset([], [], None)
        trainloader = DataLoader(empty_dataset, batch_size=1)
        valloader = DataLoader(empty_dataset, batch_size=1)
    
    local_epochs = int(context.run_config.get("local-epochs", 5))
    print(f"Client {client_id+1} initialized with {len(trainloader.dataset)} training samples")
    
    return FlowerClient(net, trainloader, valloader, local_epochs, context).to_client()

app = ClientApp(client_fn=client_fn)