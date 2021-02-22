
import yaml

def read_yaml(path: str) -> dict:
    """Returns YAML file as dict"""
    with open(path, 'r') as file_in:
        config = yaml.safe_load(file_in)
    return config

