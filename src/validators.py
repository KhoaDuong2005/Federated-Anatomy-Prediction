import os
import json

def validate_modality(modality: str):
    allowed = ["mri", "xray", "ct", "unknown"]
    if modality not in allowed:
        raise ValueError(f"Invalid modality {modality}, currently supported: {', '.join(allowed)}")
    return modality

def validate_body_part(body_part: str):
    allowed = ["chest", "brain", "leg", "abdomen", "unknown"]
    if body_part not in allowed:
        raise ValueError(f"Invalid body_part {body_part}, currently supported: {', '.join(allowed)}")
    return body_part

def validate_is_anatomy(is_anatomy: str):
    if is_anatomy == "True":
        return "True"
    elif is_anatomy == "False":
        return "False"
    elif is_anatomy == None:
        return None
    raise ValueError(f"Invalid input {is_anatomy}, currently support [True, False]")

def validate_client_data(client_dir) -> dict:
    config_path = os.path.join(client_dir, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found in directory: {client_dir}")

    with open(config_path, 'r') as file:
        try:
            config = json.load(file)
        except Exception as e:
            raise ValueError(f"Error decoding JSON from config file: {e}")

    if "label_format" not in config:
        raise ValueError("Config file missing 'label_format' key")

    if config["label_format"] not in ["label_folder", "label_name", "label_csv"]:
        raise ValueError(f"Invalid label_format: {config['label_format']}, currently supported formats are 'label_folder', 'label_name', 'label_csv'")

    return config