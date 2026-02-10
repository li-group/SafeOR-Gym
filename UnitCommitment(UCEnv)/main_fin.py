# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


############################################################################################################
# Benchmark running script: Ideally we need a table with each environment in the rows and each
# algorithm in the columns with the average reward/cost obtained
# See https://github.com/PKU-Alignment/omnisafe/tree/main/benchmarks/on-policy
############################################################################################################

import warnings
import torch
from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train

import yaml
import os
import numpy as np
import omnisafe
from omnisafe.envs.core import CMDP, env_register
from cmdp_env import UnitCommitmentMasterEnvSafe


import ast


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
    return f"{(n // 12) * 12 + 1}-{(n // 12 + 1) * 12}"


# def cfg_to_omni(env_config, cfg_id: int, layout, device="cuda:0", total_steps=120000, parallel=1, algo="CPO"):
#     with open(f"./{cfg_id}.yaml", "r") as f:
#         s = "".join(f.readlines())
#     cfg = yaml.load(s, Loader=yaml.FullLoader)
#
#     # /!\ To pass arguments to your environment it might be necessary to modify the reference config file
#     # Open the file below (depending on your OS) and add the following to the end of your file: "  env_cfgs: {}"
#     # if os.name == "posix":
#     # cfg_ref = f"/opt/conda/lib/python3.10/site-packages/omnisafe/configs/on-policy/{algo}.yaml"
#     # else:
#     cfg_ref = f".local/lib/python3.9/site-packages/omnisafe/configs/on-policy/{algo}.yaml"
#
#     with open(cfg_ref, "r") as f:
#         s = "".join(f.readlines())
#     ref: dict = yaml.load(s, Loader=yaml.FullLoader)["defaults"]
#
#     custom_cfgs = {
#         'train_cfgs': {
#             'total_steps': int(total_steps),
#             'device': device,
#             'parallel': parallel
#         },
#
#         'algo_cfgs': {
#             "steps_per_epoch": 2000,
#             "update_iters": 10,
#             "batch_size": cfg["model"]["batch_size"],
#             "entropy_coef": cfg["model"]["ent_coef"],
#             "reward_normalize": cfg["reward_normalizer"],
#             "cost_normalize": cfg["reward_normalizer"],
#             "obs_normalize": cfg["obs_normalizer"],
#         },
#
#         "logger_cfgs": {"log_dir": f"./logs_os/{layout}/{get_bin(cfg_id)}/{str(cfg_id)}",
#                         "save_model_freq": 25},
#
#         "model_cfgs": {
#             "actor": {
#                 "hidden_sizes": [cfg["model"]["arch_layersize"]] * cfg["model"]["arch_n"],
#                 "activation": cfg["model"]["act_fn"].lower(),
#                 "output_activation": cfg["model"]["act_fn"].lower(),
#                 # "activation": cfg["model"]["act_fn"].lower(),
#                 # "lr": cfg["model"]["lr"]
#             },
#             "critic": {
#                 "hidden_sizes": [cfg["model"]["arch_layersize"]] * cfg["model"]["arch_n"],
#                 "activation": cfg["model"]["act_fn"].lower(),
#                 # "lr": None
#             }
#         },
#
#         'env_cfgs': env_config
#     }
#     print("TO see:", cfg["model"])
#     for k in custom_cfgs.keys():
#         if k not in ref.keys():
#             ref[k] = custom_cfgs[k]
#             continue
#
#         for k2 in custom_cfgs[k].keys():
#             if isinstance(custom_cfgs[k][k2], dict):
#                 for k3 in custom_cfgs[k][k2]:
#                     ref[k][k2][k3] = custom_cfgs[k][k2][k3]
#
#             else:
#                 ref[k][k2] = custom_cfgs[k][k2]
#
#     return ref


# def assign_env_config(self, kwargs):
#     print("Assigning configuration...")
#     for key, value in kwargs.items():
#         # print(f"Trying to set {key} to {value}")
#         if hasattr(self, key):
#             # print(f"Setting {key} to {value}")
#             setattr(self, key, value)
#         else:
#             # print(f"{self} has no attribute, {key}")
#             raise AttributeError(f"{self} has no attribute, {key}")


# def flatten_dict(dictionary, parent_key='', separator=';'):
#     items = []
#     for key, value in dictionary.items():
#         # Convert tuple keys to a string format before concatenating
#         if isinstance(key, tuple):
#             key = '_'.join(map(str, key))  # Convert tuple to string format (e.g., ('a', 'b') -> 'a_b')
#
#         new_key = parent_key + separator + key if parent_key else key
#
#         if isinstance(value, dict):
#             # If the value is a dictionary (including empty ones), recurse
#             if value:
#                 items.extend(flatten_dict(value, new_key, separator=separator).items())
#             # else:
#             # Add empty dictionaries as well
#             # items.append((new_key, {}))
#         else:
#             items.append((new_key, value))
#     return dict(items)


# def convert_dict_to_tuple_keys(data):
#     result = {}
#     for outer_key, value in data.items():
#         if isinstance(value, dict):
#             # If the value is a dictionary, iterate over its items
#             for inner_key, inner_value in value.items():
#                 result[(outer_key, inner_key)] = inner_value
#         else:
#             # If the value is not a dictionary, store it with the outer_key as a tuple
#             result[(outer_key)] = value
#     return result


# def flatten_and_track_mappings(dictionary, separator=';'):
#     # Flatten the dictionary using the updated flatten_dict function
#     flattened_dict = flatten_dict(dictionary, separator=separator)
#
#     # Track the mappings of index to the key split by separator
#     mappings = []
#     for index, (key, value) in enumerate(flattened_dict.items()):
#         # Ensure the key is a string before splitting by separator
#         mapped_key = key.split(separator)  # Split the string key by the separator
#         mappings.append((index, mapped_key))
#
#     # Convert the flattened values to a numpy array of float32, but skip non-numeric values
#     flattened_values = []
#     for value in flattened_dict.values():
#         if isinstance(value, (int, float)):  # Only include numeric types
#             flattened_values.append(value)
#         elif isinstance(value, dict):  # Skip dictionaries
#             flattened_values.append(0)  # Or set a default value for empty or nested dictionaries
#
#     # Convert the valid numeric values to a numpy array
#     flattened_array = np.array(flattened_values).astype("float32")
#
#     return flattened_array, mappings


# def nested_set(dic, keys, value):
#     for key in keys[:-1]:
#         dic = dic.setdefault(key, {})
#     dic[keys[-1]] = value
#     return (dic)
#
#
# def reconstruct_dict(flattened_array, mappings, separator=';'):
#     reconstructed_dict = {}
#     for index, keys in mappings:
#         nested_set(reconstructed_dict, keys, flattened_array[index])
#     return reconstructed_dict


def recurse(eg, current, path=[], ):
    if isinstance(current, dict):
        for k, v in current.items():
            recurse(eg, v, path + [str(k)])
    else:
        key = 'env_cfgs:' + ':'.join(path)
        val = current if isinstance(current, list) else [current]
        eg.add(key, val)


if __name__ == '__main__':
    debug_use = True

    eg = ExperimentGrid(exp_name='Benchmark_supp_UC1')

    if debug_use == True:
        base_policy = ['PPO']
        naive_lagrange_policy = ['TRPOLag']
        first_order_policy = [] # ['P3O', 'OnCRPO']
        second_order_policy = [] # ['CPO']
        off_policy = [] #['DDPGLag']
        episodes_per_epoch = 2
        steps_per_epoch = [24*episodes_per_epoch]
        total_steps = [24*episodes_per_epoch*2]
        num_episodes = 2

    else:
        base_policy = ["SACLag", "SACPID", "FOCOPS"]
        naive_lagrange_policy = []# ['TRPOLag']
        first_order_policy = []# ['P3O', 'OnCRPO']
        second_order_policy = []# ['CPO']
        off_policy = []# ['DDPGLag']
        episodes_per_epoch = 100
        steps_per_epoch = [24 * episodes_per_epoch]
        total_steps = [24 * episodes_per_epoch * 500]
        num_episodes = 10

    window_lens = [episodes_per_epoch]
    algos = base_policy + naive_lagrange_policy + first_order_policy + second_order_policy + off_policy
    num_pool = len(algos)
    compare_num = len(algos)

    # Set the environments.
    mujoco_envs = [
        'UC-v0', 'UC-v1'
    ]
    eg.add('env_id', mujoco_envs)

    eg.add(
    'env_cfgs:env_init_config:config_path',
    ['unit_commitment_config.json'])

    # Set the device.
    available_gpus = list(range(torch.cuda.device_count()))
    if len(available_gpus) > 0:
        gpu_id = [0]
        device = ['cuda:0']
    else:
        warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
        # if you want to use CPU, please set gpu_id = None
        gpu_id = None
        device = ['cpu']

    eg.add('algo', algos)
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('logger_cfgs:use_tensorboard', [True])
    eg.add('logger_cfgs:window_lens', window_lens)
    eg.add('train_cfgs:vector_env_nums', [1])
    eg.add('train_cfgs:torch_threads', [1])

    eg.add('env_cfgs:env_init_config:scale_action', [True])
    eg.add('env_cfgs:env_init_config:penalty_factor_UT', [1.])
    eg.add('env_cfgs:env_init_config:penalty_factor_DT', [1.])
    eg.add('env_cfgs:env_init_config:penalty_factor_RampUp', [0.1])
    eg.add('env_cfgs:env_init_config:penalty_factor_RampDown', [0.1])

    eg.add('model_cfgs:actor:output_activation', ['tanh'])
    eg.add('algo_cfgs:steps_per_epoch', steps_per_epoch)
    eg.add('train_cfgs:total_steps', total_steps)
    eg.add('seed', [0])
    eg.add('train_cfgs:device', device)


    eg.run(train, num_pool=num_pool, gpu_id=gpu_id)

    eg.analyze(parameter='algo', values=None, compare_num=compare_num)
    eg.evaluate(num_episodes=num_episodes)

    print(dir(eg))
