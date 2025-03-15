"""elton: A Flower / PyTorch app."""
import os
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from elton.task import Net, get_weights, set_weights, test, load_test_data
from torch.utils.data import DataLoader

from torchvision import transforms
def get_image_transforms():
    return transforms.Compose([
    transforms.Resize((256, 256)), ###### test #######
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

def get_evaluate_fn(testloader, device):
    """Return a callback that evaluates the global model."""
    def evaluate(server_round, parameters_ndarrays, config):
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device)
        return loss, {"cen_accuracy": accuracy}
    return evaluate

def on_fit_config(server_round: int):
    lr = 3e-4
    if server_round > 2:
        lr = 1e-4
    return {"lr": lr}

def handle_fit_metrics(metrics):
    import json
    b_values = []
    for _, m in metrics:
        my_metric_str = m["my_metric"]
        my_metric = json.loads(my_metric_str)
        b_values.append(my_metric["b"])
    return {"max_b": max(b_values)}

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {"accuracy": sum(accuracies) / total_examples}

def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Use the custom dataset for evaluation.
    data_dir = r"D:\Docs\chest_xray\test"
    cache_filename = r"test_processed_data.pkl"
    testloader = load_test_data(data_dir, cache_filename)

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=handle_fit_metrics,
        on_evaluate_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(testloader, device="cuda")
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
