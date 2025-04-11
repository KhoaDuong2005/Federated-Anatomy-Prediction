import subprocess
import threading
import time
import os
import argparse
import sys
from pathlib import Path

def start_server(rounds, min_clients, save_dir, epochs):
    """Start the Flower server"""
    cmd = [
        sys.executable, 
        "fl_server.py", 
        "--rounds", str(rounds), 
        "--min_clients", str(min_clients),
        "--save_dir", save_dir,
        "--epochs", str(epochs)
    ]
    print(f"Starting server with command: {' '.join(cmd)}")
    subprocess.run(cmd)

def start_client(client_dir, server_address):
    """Start a Flower client"""
    cmd = [
        sys.executable, 
        "fl_client.py", 
        "--client_dir", client_dir, 
        "--server_address", server_address
    ]
    print(f"Starting client with command: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Run federated learning simulation")
    parser.add_argument("--data_dir", type=str, default="client_data", 
                        help="Base data directory containing client_X folders")
    parser.add_argument("--num_clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated learning rounds")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs per round")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080", help="Server address")
    parser.add_argument("--save_dir", type=str, default="fl_models", help="Directory to save models")
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Start server in a separate thread
    server_thread = threading.Thread(
        target=start_server,
        args=(args.rounds, args.num_clients, args.save_dir, args.epochs)
    )
    server_thread.daemon = True
    server_thread.start()
    
    # Wait for server to start
    print("Starting server...")
    time.sleep(5)
    
    # Start clients
    client_threads = []
    for i in range(args.num_clients):
        client_dir = os.path.join(args.data_dir, f"client_{i+1}")
        thread = threading.Thread(
            target=start_client,
            args=(client_dir, args.server_address)
        )
        thread.start()
        client_threads.append(thread)
    
    # Wait for clients to finish
    for thread in client_threads:
        thread.join()
    
    # Server will terminate automatically after all rounds

if __name__ == "__main__":
    main()