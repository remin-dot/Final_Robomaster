# src/config_loader.py
import os
import yaml

def load_config(config_path="config/settings.yaml"):
    """Read configuration values ​​from a YAML file and return them as a dictionary."""
    # Find the absolute path based on the current file's location.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, config_path)
    
    with open(full_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)