import os
import json

def validate_modality(modality: str):
    if modality == None:
        return None
    allowed = ["mri", "xray", "ct"]
    if modality not in allowed:
        raise ValueError(f"Invalid modality {modality}, currently supported: {', '.join(allowed)}")
    return modality

def validate_body_part(body_part: str):
    if body_part == None:
        return None
    allowed = ["chest", "brain", "leg", "abdomen"]
    if body_part not in allowed:
        raise ValueError(f"Invalid body_part {body_part}, currently supported: {', '.join(allowed)}")
    return body_part

def validate_client_data(client_dir) -> dict:
    config_path = os.path.join(client_dir, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found in directory: {client_dir}, you can create config file using --cfg flag")

    with open(config_path, 'r') as file:
        try:
            config = json.load(file)
        except Exception as e:
            raise ValueError(f"Error decoding JSON from config file: {e}")

    if "label_format" not in config:
        raise ValueError("Config file missing 'label_format' key")

    if config["label_format"] not in ["folder", "name", "csv"]:
        raise ValueError(f"Invalid label_format: {config['label_format']}, currently supported formats are 'folder', 'name', 'csv'")

    return config