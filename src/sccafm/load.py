import yaml
import pandas as pd
from importlib import resources


def load_resources(filename: str) -> pd.DataFrame:
    """
    Load a CSV file from the packaged `resources/` folder.

    Args:
        filename (str): Name of the CSV file inside `resources/`

    Returns:
        pd.DataFrame: The contents of the CSV file
    
    Example:
        df = load_resources("token_dict.csv")
    """
    # Use importlib.resources to get the file inside the installed package
    try:
        with resources.as_file(
            resources.files("resources").joinpath(filename)
        ) as file_path:
            return pd.read_csv(file_path)

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Resource file '{filename}' not found!"
        )


def load_cfg(filename: str):
    """
    Load a config file from the packaged `configs/` folder

    Args:
        filename (str): Name of the YAML file inside `configs/`

    Returns:
        dict: The contents of the YAML file

    Example:
        cfg = load_cfg("model.yaml")        
    """
    try:
        with resources.as_file(
            resources.files("configs").joinpath(filename)
        ) as file_path:
            with open(file_path, "r") as f:
                return yaml.safe_load(f)

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Config file '{filename}' not found!"
        )    
