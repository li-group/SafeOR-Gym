import json
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

def decode_action(action):
    """
    Decode the action from the action space into a structured format.

    Parameters:
    - action (np.ndarray): The raw action array.

    Returns:
    - dict: Structured action.
    """
    return {'lambda': action}

def encode_observation(state):
    """
    Encode the observation into a flattened NumPy array.

    Parameters:
    - state (dict): The state dictionary containing 'electricity_prices', 'demand', and 'IV'.
    - lookahead_days (int): Number of lookahead days.
    - products (list): List of products.

    Returns:
    - np.ndarray: Flattened observation array.
    """
    electricity_prices = np.array(state['electricity_prices'], dtype=np.float32)
    demand = np.array(state['demand'], dtype=np.float32).flatten()
    IV = np.array(state['IV'], dtype=np.float32)

    flat_state = np.concatenate([electricity_prices, demand, IV])
    return flat_state
