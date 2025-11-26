import os
import yaml
from src.Wine_Quality_Prediction import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from box.exceptions import BoxValueError

"""_______The common functionality______"""

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Docstring for read_yaml
    
    Args:
        path_to_yaml (str) : path like input

    Raises:
        ValueError : if yaml file is emmpty
        e : empty file

    Returns:
        ConfigBox : ConfigBox type 
    """

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file : {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Docstring for create_directories
    
    Args:
        path_to_directories (list) : list of the path of directories
        ignore_log (bool, optional) : ignore if multiple dirs is to be created. Defaults 
    """

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at : {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Docstring for save_json
    
    Args:
        path (Path): path to json file
        data (dict): data to be saved in jason file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file is saved at : {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Docstring for load_json
    
    Args:
        path (Path) : path to json file

    returns:
        ConfigBox : data as class attributes instead of dict
    """
    with open(path) as f:
        content  = json.load(f)
    
    logger.info(f"json file loaded succesfully from : {path}")
    return ConfigBox(content)


@ensure_annotations
def save_model(data: Any, path: Path):
    """
    Docstring for save_model
    
    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """

    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")

@ensure_annotations
def load_model(path: Path) -> Any:
    """
    Docstring for load_model
    
    Args:
        path (Path) : path to binary file
    
    Returns:
        Any: object stored in the file 
    """