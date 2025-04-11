import flwr as fl
from typing import Dict, List, Tuple, Optional
import argparse
import os
from pathlib import Path
import torch
import sys
import numpy as np
import struct

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

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate model weights and save the model after each round"""
        print(f"\n[Server] Round {server_round}: Aggregating model weights from {len(results)} clients")
        sys.stdout.flush()
        
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, 
            results, 
            failures
        )

        if parameters_aggregated is not None:
            # Load a fresh model
            from fl_client import get_model
            model = get_model()
            
            # Get model state dict for reference
            model_state_dict = model.state_dict()
            state_dict_keys = list(model_state_dict.keys())
            parameters_list = parameters_aggregated.tensors
            
            # Build the new state dict
            new_state_dict = {}
            
            # Process each parameter
            for i, (name, param_data) in enumerate(zip(state_dict_keys, parameters_list)):
                expected_shape = model_state_dict[name].shape
                expected_size = int(np.prod(expected_shape)) if expected_shape != () else 1
                orig_tensor_dtype = model_state_dict[name].dtype
                is_int_buffer = "num_batches_tracked" in name
                
                try:
                    # Process bytes into numpy array
                    if isinstance(param_data, bytes):
                        dtype_to_use = np.int64 if is_int_buffer else np.float32
                        np_array = np.frombuffer(param_data, dtype=dtype_to_use)
                        
                        # Handle scalar parameters or regular tensors
                        if expected_shape == ():
                            # For scalars, take the first element
                            value = int(np_array[0]) if is_int_buffer else float(np_array[0])
                            tensor_dtype = torch.int64 if is_int_buffer else orig_tensor_dtype
                            new_state_dict[name] = torch.tensor(value, dtype=tensor_dtype)
                        else:
                            # For regular tensors
                            values_needed = np_array.flatten()[:expected_size]
                            tensor_data = values_needed.reshape(expected_shape)
                            tensor_dtype = torch.int64 if is_int_buffer else orig_tensor_dtype
                            new_state_dict[name] = torch.tensor(tensor_data, dtype=tensor_dtype)
                    
                    # Process numpy arrays
                    elif isinstance(param_data, np.ndarray):
                        if expected_shape == ():
                            # For scalar tensors
                            value = int(param_data.item()) if is_int_buffer else float(param_data.item())
                            tensor_dtype = torch.int64 if is_int_buffer else orig_tensor_dtype
                            new_state_dict[name] = torch.tensor(value, dtype=tensor_dtype)
                        else:
                            # Regular tensors
                            tensor_dtype = torch.int64 if is_int_buffer else orig_tensor_dtype
                            new_state_dict[name] = torch.tensor(
                                param_data.reshape(expected_shape),
                                dtype=tensor_dtype
                            )
                except Exception as e:
                    # Fallback to original parameter if needed
                    new_state_dict[name] = model_state_dict[name]

            # Load state dict into model
            model.load_state_dict(new_state_dict, strict=True)
            print("[Server] Model parameters loaded successfully")
            
            # Save the model
            save_path = os.path.join(self.save_dir, f"model_round_{server_round}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[Server] Model saved to {save_path}")
            
            # Save as latest model too
            latest_path = os.path.join(self.save_dir, "latest_model.pth")
            torch.save(model.state_dict(), latest_path)
            sys.stdout.flush()

        return parameters_aggregated, metrics_aggregated
    
# Modify the main function to accept epochs parameter
def main():
    parser = argparse.ArgumentParser(description="Flower server")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds")
    parser.add_argument("--min_clients", type=int, default=2, help="Minimum number of clients")
    parser.add_argument("--save_dir", type=str, default="fl_models", help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs per round")
    parser.add_argument("--port", type=int, default=8080, help="Port for server")
    args = parser.parse_args()
    
    # Create strategy with proper client IDs
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
        fit_metrics_aggregation_fn=weighted_average  # Add this line
    )
    
    # Start server with specified port
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

if __name__ == "__main__": 
    main()