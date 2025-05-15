import json
from typing import Any, Dict, List
import numpy as np

def load_config(config_path: str) -> dict:
    """
    Load a JSON config file and return its contents as a dict.
    """
    if config_path is None:
        raise ValueError("config_path must be provided to load configuration.")
    with open(config_path, 'r') as f:
        return json.load(f)

def assign_env_config(obj: object, config_data: dict, schema: dict) -> None:
    """
    Assigns configuration values from config_data to attributes on obj,
    validating keys against schema and checking types.
    """
    for key, value in config_data.items():
        if key not in schema:
            raise AttributeError(f"{obj!r} has no config attribute '{key}'")
        expected_type = schema[key]
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Config '{key}' expects type {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
        setattr(obj, key, value)

def decode_action_util(action: np.ndarray, n: int) -> Dict[str, np.ndarray]:
    """
    Decode a flat action array into structured components.

    Parameters:
        action: combined action array of shape (2*n + 1,)
        n: number of compressors (first segment length)

    Returns:
        A dict with keys 'maintenance_action', 'production_rate', and 'external_purchase'.
    """
    maintenance_action = np.round(action[:n]).astype(int)
    production_rate = action[n:2*n]
    external_purchase = action[-1:].copy()

    return {
        "maintenance_action": maintenance_action,
        "production_rate": production_rate,
        "external_purchase": external_purchase
    }


def encode_observation_util(state: Dict[str, Any], keys: List[str]) -> np.ndarray:
    """
    Flattens a structured observation dict into a flat NumPy array.

    Parameters:
        state: dict mapping observation names to sequences (lists or arrays)
        keys: ordered list of keys to extract and concatenate

    Returns:
        1D NumPy array of concatenated observations (dtype float32).
    """
    arrays = []
    for key in keys:
        if key not in state:
            raise KeyError(f"Observation key '{key}' missing from state dict")
        arr = np.array(state[key], dtype=np.float32)
        arrays.append(arr)
    # concatenate all pieces into one flat array
    return np.concatenate(arrays).astype(np.float32)
