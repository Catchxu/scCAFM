import yaml
import pandas as pd
from importlib import resources
from typing import Optional
from pathlib import Path


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
    local_path = Path(filename).expanduser()
    if local_path.is_file():
        return pd.read_csv(local_path)

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
    local_path = Path(filename).expanduser()
    if local_path.is_file():
        with open(local_path, "r") as f:
            return yaml.safe_load(f)

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


def load_tf_list(tfs: Optional[pd.DataFrame] = None):
    if tfs is not None:
        try:
            tf_list = tfs["TF"].tolist()
        except:
            raise ValueError(
                "tfs doesn't have a column 'TF'!"
            )
    else:
        tf_list = None
    
    return tf_list
