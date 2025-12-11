import pandas as pd
from importlib import resources


def load_resources(filename: str) -> pd.DataFrame:
    """
    Load a CSV file from the packaged `resources/` folder.

    Args:
        filename (str): Name of the CSV file inside `resources/`.

    Returns:
        pd.DataFrame: The contents of the CSV file.
    
    Example:
        df = load_resources("token_dict.csv")
    """
    # Use importlib.resources to get the file inside the installed package
    with resources.as_file(resources.files("resources").joinpath(filename)) as file_path:
        return pd.read_csv(file_path)