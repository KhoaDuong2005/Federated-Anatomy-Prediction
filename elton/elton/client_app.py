# client_app.py
import torch
from random import random
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigsRecord
from elton.task import Net, get_weights, load_data, set_weights, test, train
import json
from torchvision.transforms import Compose, ToTensor, Normalize, Grayscale, Resize

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, context: Context):
        self.client_state = context.state
        self.net = net
        self.trainloader = trainloader  
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        if "fit_metrics" not in self.client_state.configs_records:
            self.client_state.configs_records["fit_metrics"] = ConfigsRecord()

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
        fit_metrics = self.client_state.configs_records["fit_metrics"]
        if "train_loss_hist" not in fit_metrics:
            fit_metrics["train_loss_hist"] = [train_loss]
        else:
            fit_metrics["train_loss_hist"].append(train_loss)
        complex_metric = {"a": 123, "b": random(), "mylist": [1, 2, 3, 4]}
        complex_metric_str = json.dumps(complex_metric)
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss, "my_metric": complex_metric_str},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

def client_fn(context: Context):
    net = Net()
    # Read the custom data directory from the node config (or use default)
    data_dir = context.run_config["data_dir"]
    trainloader, valloader = load_data(data_dir, "processed_data.pkl")
    local_epochs = context.run_config["local-epochs"]
    return FlowerClient(net, trainloader, valloader, local_epochs, context).to_client()

app = ClientApp(client_fn=client_fn)