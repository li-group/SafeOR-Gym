import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from gymnasium.utils import seeding
import json
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os, sys
from typing import Any, ClassVar, List, Tuple, Optional, Dict
import random

def convert_tuple_keys_to_string(d: dict) -> dict:
    """Convert all tuple keys to string representations."""
    new_dict = {}
    for key, value in d.items():
        if isinstance(key, tuple):
            key = str(key)  # Convert tuple to string
        if isinstance(value, dict):
            value = convert_tuple_keys_to_string(value)  # Recursively process nested dictionaries
        new_dict[key] = value
    return new_dict

def convert_string_keys_to_tuple(d: dict) -> dict:
    """Convert all string keys that represent tuples back to actual tuples."""
    new_dict = {}
    for key, value in d.items():
        if isinstance(key, str):
            try:
                # Attempt to convert the string representation of a tuple back into a tuple
                key = ast.literal_eval(key) if key.startswith("(") and key.endswith(")") else key
            except (ValueError, SyntaxError):
                pass  # If it's not a valid tuple string, leave it as is
        if isinstance(value, dict):
            value = convert_string_keys_to_tuple(value)  # Recursively process nested dictionaries
        new_dict[key] = value
    return new_dict
def get_bin(n):
    return f"{(n//12)*12+1}-{(n//12+1)*12}"
def cfg_to_omni(env_config,cfg_id: int, layout, device = "cuda:0", total_steps = 120000, parallel=1, algo = "CPO"):
    with open(f"./{cfg_id}.yaml", "r") as f:
        s = "".join(f.readlines())
    cfg = yaml.load(s, Loader=yaml.FullLoader)
    
    # /!\ To pass arguments to your environment it might be necessary to modify the reference config file
    # Open the file below (depending on your OS) and add the following to the end of your file: "  env_cfgs: {}"
    #if os.name == "posix":
        #cfg_ref = f"/opt/conda/lib/python3.10/site-packages/omnisafe/configs/on-policy/{algo}.yaml"
    #else:
    cfg_ref = f".local/lib/python3.9/site-packages/omnisafe/configs/on-policy/{algo}.yaml"
        
    with open(cfg_ref, "r") as f:
        s = "".join(f.readlines())
    ref: dict = yaml.load(s, Loader=yaml.FullLoader)["defaults"]
    
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': int(total_steps),
            'device': device,
            'parallel': parallel
        },
        
        'algo_cfgs': {
            "steps_per_epoch": 2000,
            "update_iters": 10,
            "batch_size": cfg["model"]["batch_size"],
            "entropy_coef": cfg["model"]["ent_coef"],
            "reward_normalize": cfg["reward_normalizer"],
            "cost_normalize": cfg["reward_normalizer"],
            "obs_normalize": cfg["obs_normalizer"],
        },
        
        "logger_cfgs":{"log_dir": f"./logs_os/{layout}/{get_bin(cfg_id)}/{str(cfg_id)}",
                       "save_model_freq": 25},
                
        "model_cfgs":{
            "actor":{
                "hidden_sizes": [cfg["model"]["arch_layersize"]] * cfg["model"]["arch_n"],
                "activation": cfg["model"]["act_fn"].lower(),
                "output_activation": cfg["model"]["act_fn"].lower(),
                #"activation": cfg["model"]["act_fn"].lower(),
                # "lr": cfg["model"]["lr"]
            },
            "critic":{
                "hidden_sizes": [cfg["model"]["arch_layersize"]] * cfg["model"]["arch_n"],
                "activation": cfg["model"]["act_fn"].lower(),
                # "lr": None
            }
        },
        
        'env_cfgs': env_config
    }
    print("TO see:",cfg["model"])
    for k in custom_cfgs.keys():
        if k not in ref.keys():
            ref[k] = custom_cfgs[k]
            continue
        
        for k2 in custom_cfgs[k].keys():
            if isinstance(custom_cfgs[k][k2], dict):
                for k3 in custom_cfgs[k][k2]:
                    ref[k][k2][k3] = custom_cfgs[k][k2][k3]
                    
            else:
                ref[k][k2] = custom_cfgs[k][k2]
    
    return ref

def assign_env_config(self, kwargs):
    print("Assigning configuration...")
    print(kwargs)
    for key, value in kwargs.items():
        #print(f"Trying to set {key} to {value}")
        if hasattr(self, key):
            #print(f"Setting {key} to {value}")
            setattr(self, key, value)
        else:
            #print(f"{self} has no attribute, {key}")
            raise AttributeError(f"{self} has no attribute, {key}")


def flatten_dict(dictionary, parent_key='', separator=';'):
    items = []
    for key, value in dictionary.items():
        # Convert tuple keys to a string format before concatenating
        if isinstance(key, tuple):
            key = '_'.join(map(str, key))  # Convert tuple to string format (e.g., ('a', 'b') -> 'a_b')

        new_key = parent_key + separator + str(key) if parent_key else key

        if isinstance(value, dict):
            # If the value is a dictionary (including empty ones), recurse
            if value:
                items.extend(flatten_dict(value, new_key, separator=separator).items())
            #else:
                # Add empty dictionaries as well
                #items.append((new_key, {}))
        else:
            items.append((new_key, value))
    return dict(items)

def convert_dict_to_tuple_keys(data):
    result = {}
    for outer_key, value in data.items():
        if isinstance(value, dict):
            # If the value is a dictionary, iterate over its items
            for inner_key, inner_value in value.items():
                result[(outer_key, inner_key)] = inner_value
        else:
            # If the value is not a dictionary, store it with the outer_key as a tuple
            result[(outer_key)] = value
    return result

def flatten_and_track_mappings(dictionary, separator=';'):
    # Flatten the dictionary using the updated flatten_dict function
    flattened_dict = flatten_dict(dictionary, separator=separator)
    
    # Track the mappings of index to the key split by separator
    mappings = []
    for index, (key, value) in enumerate(flattened_dict.items()):
        # Ensure the key is a string before splitting by separator
        mapped_key = key.split(separator)  # Split the string key by the separator
        mappings.append((index, mapped_key))
    
    # Convert the flattened values to a numpy array of float32, but skip non-numeric values
    flattened_values = []
    for value in flattened_dict.values():
        if isinstance(value, (int, float)):  # Only include numeric types
            flattened_values.append(value)
        elif isinstance(value, dict):  # Skip dictionaries
            flattened_values.append(0)  # Or set a default value for empty or nested dictionaries
    
    # Convert the valid numeric values to a numpy array
    flattened_array = np.array(flattened_values).astype("float32")
    
    return flattened_array, mappings



def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value
    return(dic)
def reconstruct_dict(flattened_array, mappings, separator=';'):
    reconstructed_dict = {}
    for index, keys in mappings:
        nested_set(reconstructed_dict, keys, flattened_array[index])
    return reconstructed_dict
def get_jsons(layout):
    import json
    with open(f"./configs/json/connections_{layout}.json" ,"r") as f:
        connections_s = f.readline()
    connections = json.loads(connections_s)

    with open(f"./configs/json/action_sample_{layout}.json" ,"r") as f:
        action_sample_s = f.readline()
    action_sample = json.loads(action_sample_s)
    return connections, action_sample

def clip(x,a,b):
    if x>b:
        return b
    elif x<a:
        return a
    return x
def get_sbp(connections):
    # Function to obtain the list of source, blending, demand tank names from the connections
    
    sources = list(connections["source_blend"].keys())
    
    b_list = list(connections["blend_blend"].keys())
    for b in connections["blend_blend"].keys():
        b_list += connections["blend_blend"][b]
    b_list += list(connections["blend_demand"].keys())
    blenders = list(set(b_list))
    
    p_list = []
    for p in connections["blend_demand"].keys():
        p_list += connections["blend_demand"][p]
    demands = list(set(p_list))
    
    return sources, blenders, demands