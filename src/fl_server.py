import flwr as fl
from typing import Dict, List, Tuple
import argparse
import os
import torch
from collections import OrderedDict
from model import get_model

def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    if not metrics:
        return {}
    
    metrics_dict = {}
    for key in metrics[0][1].keys():
        values = [num_examples * m[key] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        metrics_dict[key] = sum(values) / sum(examples) if examples else 0
    return metrics_dict

class Strategy(fl.server.strategy.FedAvg):
    def __init__(self, save_dir="fl_models", **kwargs):
        super().__init__(**kwargs)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Create reference model once for consistent structure
        self.reference_model = get_model()

    def aggregate_fit(self, server_round, results, failures):
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        if parameters_aggregated is not None:
            # Convert to list of NumPy ndarrays
            weights_list = fl.common.parameters_to_ndarrays(parameters_aggregated)
            
            # Create fresh model with correct architecture
            model = self.reference_model
            
            # Get parameter names from model
            param_keys = list(model.state_dict().keys())
            
            # Create ordered dict directly from names and weights
            state_dict = OrderedDict()
            for i, key in enumerate(param_keys):
                if i < len(weights_list):
                    # Get the original parameter for reference
                    orig_param = model.state_dict()[key]
                    
                    try:
                        # Integer buffer handling
                        if "num_batches_tracked" in key:
                            value = int(weights_list[i].item())
                            state_dict[key] = torch.tensor(value, dtype=torch.int64)
                        else:
                            # Normal parameter
                            state_dict[key] = torch.tensor(weights_list[i], dtype=orig_param.dtype)
                    except:
                        # Use original parameter if conversion fails
                        state_dict[key] = orig_param

            # Save model
            save_path = os.path.join(self.save_dir, f"model_round_{server_round}.pth")
            torch.save(state_dict, save_path)
            
            latest_path = os.path.join(self.save_dir, "latest_model.pth")
            torch.save(state_dict, latest_path)

        return parameters_aggregated, metrics_aggregated
    
def main():
    parser = argparse.ArgumentParser(description="Flower server")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min_clients", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="fl_models")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    
    strategy = Strategy(
        save_dir=args.save_dir,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        on_fit_config_fn=lambda round_num: {
            "epochs": args.epochs,
            "round_num": round_num,
        },
        on_evaluate_config_fn=lambda round_num: {
            "round_num": round_num,
        },
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average
    )
    
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

if __name__ == "__main__": 
    main()