import os
import torch
import json
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from src.task import Net, get_weights, set_weights, test, NumpyDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

def get_evaluate_fn(testloader, device):
    def evaluate(server_round, parameters_ndarrays, config):
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        
        if testloader is None or len(testloader.dataset) == 0:
            print(f"WARNING: Test dataset is empty. Skipping evaluation.")
            return 0.0, {"cen_accuracy": 0.0}
            
        loss, accuracy = test(net, testloader, device)
        print(f"Round {server_round} centralized evaluation: loss={loss}, accuracy={accuracy}")
        return float(loss), {"cen_accuracy": float(accuracy)}
    return evaluate

def on_fit_config(server_round: int):
    lr = 3e-4
    if server_round > 2:
        lr = 1e-4
    return {"lr": lr}

def handle_fit_metrics(metrics):
    b_values = []
    for _, m in metrics:
        if "my_metric" in m:
            try:
                my_metric = json.loads(m["my_metric"])
                if "b" in my_metric:
                    b_values.append(my_metric["b"])
            except (json.JSONDecodeError, KeyError):
                pass
    return {"max_b": max(b_values)} if b_values else {}

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {"accuracy": sum(accuracies) / total_examples} if total_examples > 0 else {"accuracy": 0.0}

def load_test_images(test_dir, batch_size=32):
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return None
        
    class_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    if not class_dirs:
        print(f"No class directories found in {test_dir}")
        return None
        
    print(f"Found {len(class_dirs)} classes: {class_dirs}")
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(class_dirs))}
    
    images = []
    labels = []
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    for class_name, idx in class_to_idx.items():
        class_dir = os.path.join(test_dir, class_name)
        print(f"Loading {class_name} (idx: {idx}) from {class_dir}")
        
        files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        print(f"Found {len(files)} images")
        
        for file_name in files[:100]:
            try:
                img_path = os.path.join(class_dir, file_name)
                img = Image.open(img_path).convert('RGB')
                images.append(np.array(img))
                labels.append(idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    if not images:
        print("No images loaded")
        return None
        
    print(f"Loaded {len(images)} test images")
    
    test_dataset = NumpyDataset(images, labels, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return test_loader

def server_fn(context: Context):
    num_rounds = int(context.run_config.get("num-server-rounds", 3))
    fraction_fit = float(context.run_config.get("fraction-fit", 1.0))
    
    num_clients = int(context.run_config.get("num-clients", 3))
    
    print(f"Starting server with {num_clients} simulated clients")
    print(f"Running for {num_rounds} rounds with fraction_fit={fraction_fit}")

    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    data_dir = context.run_config.get("data_dir", "D:/Docs/chest_xray")
    test_dir = os.path.join(data_dir, "test")
    
    print(f"Loading test data from {test_dir}")
    testloader = load_test_images(test_dir)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=num_clients,
        min_fit_clients=max(1, int(num_clients * fraction_fit)),
        min_evaluate_clients=num_clients,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=handle_fit_metrics,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(testloader, device)
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)