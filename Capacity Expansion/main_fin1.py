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

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from gymnasium.utils import seeding
import json
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os, sys
#from utils import *
#from PIL import Image, ImageDraw, ImageFont
from omnisafe.envs.core import CMDP, env_register, env_unregister
from omnisafe.common.logger import Logger
import random

import torch as th
import yaml
import os
import numpy as np
import omnisafe

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

        new_key = parent_key + separator + key if parent_key else key

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

@env_register
class Cap_exp_env_Omni(CMDP):
    _support_envs = ['Capacity-Expansion']
    need_auto_reset_wrapper = True  # Whether `AutoReset` Wrapper is needed
    need_time_limit_wrapper = True  # Whether `TimeLimit` Wrapper is needed
    num_envs = 1
    def __init__(self, env_id, **kwargs):
        """
            Initialize the Cap_exp_env_Omni environment.

            Args:
                env_id (str): Identifier for the environment, which determines the layout. See _support_envs
                D: Multiplier for L2 based penalty on each violation
                P: Fixed penalty (L0 based) on each violation 
                eps: small bound to help round negligble power transfers to 0
                gencap: Dictionary corresponding to capacity of different generators ({generator:capacity})
                regions: List of regions
                demand: Dicitionary corresponding to demand in different regions ({region:demand})
                T: Number of time periods
                maxgen: Maximum number of generators of each type in each regions ({generator,region}:Maximum number)
                installcost: Dictionary of installation costs of generators  ({"generators" : {generator:cost}}) 
                generators: list of generators
                env_spec_log: Components to be noted

        """
        
        super().__init__(env_id) 
        print(kwargs)
        self._device = kwargs.get('device', 'cuda' if th.cuda.is_available() else 'cpu')
        self.device = kwargs.get('device', 'cuda' if th.cuda.is_available() else 'cpu')
        self.gencap = {"i1":0}
        self.demand = {"r1":{"1":0},"r2":{"1":0}}
        self.regions = ["r1","r2"]
        self.eps = 10**(-3)
        self.D = 10**3
        self.P = 10**3
        self.T = 1
        self.maxgen = {"i1":{ "r1": 10,"r2":10}}
        self.installcost = {"generators" : {"i1":0}}
        with open("./action_sample_base_1.json" ,"r") as f:
            action = f.read()
        self.action_sample = json.loads(action)
        #print(kwargs)
        #self._device = "cpu"
        
        self.render_mode= 'rgb_array',
        self.camera_name= None,
        self.camera_id = None,
        self.width = 256,
        self.height = 256
        try:
            del kwargs["dev"]
        except: 
            pass
        try:
            del kwargs["render_mode"]
        except:
            pass
        assign_env_config(self, kwargs)
        
        
        self.generators = list(self.gencap.keys())
        self.regions = list(self.demand.keys())
        self.env_spec_log = {'Total penalty: Generator lower bound violations':0,'Number of Generator lower bound violations':0,
                             'Total penalty: Generator upper bound violations':0,'Number of Generator upper bound violations':0,
                             'Number of Transmission power bound violations':0, 'Number of Demand violations': 0,'Total penalty: Demand violations': 0,
                             'Number of negative power violations': 0,'Total penalty: negative power violations': 0
                             }


        self.reset()
        #print(self.flatt_state.shape[0])
        self.flatt_state, self.mapping_obs = flatten_and_track_mappings(self.state)
        max_demand_value = max(max(region_demand.values()) for region_demand in self.demand.values())
        self.obs_high = np.concatenate([np.repeat(100,len(self.generators)*len(self.regions)),np.repeat(max_demand_value,len(self.regions)),np.repeat(self.T,1)])
        
        
        self._observation_space = Box(low=0, high=self.obs_high, shape=(self.flatt_state.shape[0],))
        
        self.flatt_act_sample, self.mapping_act = flatten_and_track_mappings(self.action_sample)
        #print(self.mapping_act)
        
        low_list = []
        high_list = []
        for i in self.generators:
            for r in self.regions:
                low_list.append(0)
                high_list.append(self.maxgen[i][r])
        self.action_low = np.array(low_list)
        self.action_high = np.array(high_list)
        self._action_space = Box(low=-1, high=1, shape=(len(self.flatt_act_sample),)) #Change the bounds
    
    def step(self, action_scaled: th.Tensor):
        """
        The state is kept track of in a human-readable dict "self.state"
        After updating it from the action, we flatten it and return it along with the reward, costs, truncated and terminated informations

        Args:
            action (torch.Tensor): model output
        """
        low_torch = torch.tensor(self.action_low, dtype=action_scaled.dtype, device=action_scaled.device)
        high_torch = torch.tensor(self.action_high, dtype=action_scaled.dtype, device=action_scaled.device)
        action = (action_scaled + 1) / 2 * (high_torch - low_torch) + low_torch

        self.t += 1
        self.state["t"] = self.t
        print("New step")
        print(self.state)
        print(self.flatt_state)
        self.pens_step = {k:0 for k in self.env_spec_log.keys()}
        reward = 0
        cost = 0
        if(not isinstance(action, list)):
            action = action.tolist()
        action = reconstruct_dict(action, self.mapping_act)
        for v in action.keys():
            action[v] = convert_dict_to_tuple_keys(action[v])
        action = self.sanitize_action(action)
        action,cost = self.check_bounds_cost(action,cost)
        print(action)
        reward = self.update_genreward(action,reward)
        cost = self.check_dem_cost(action,cost)
        #reward-=cost
        self.reward_ep+=reward
        self.cost_ep+=cost
        if self.t == self.T:
            self.terminated = True
        self.update_dem()
        self.flatt_state, _ = flatten_and_track_mappings(self.state)
        for k in self.env_spec_log.keys():
            self.env_spec_log[k] += self.pens_step[k]
        print("device",self._device)
       # return th.tensor(self.flatt_state).to(self._device), th.tensor(reward).to(self._device), th.tensor(cost).to(self._device), th.tensor(self.terminated).to(self._device), \
                #th.tensor(self.truncated).to(self._device), {"dict_state": self.state, "pen_tracker": self.pens_step, "terminated": self.terminated, "truncated": self.truncated}
        flatt_state_tensor = th.tensor(self.flatt_state, device=self._device)
        return flatt_state_tensor, th.tensor(reward, device=self._device), th.tensor(cost, device=self._device), th.tensor(self.terminated, device=self._device), th.tensor(self.truncated, device=self._device), {}
    def get_new_start_state(self):
        print(self.demand)
        self.state = {"num_gen" : {(i,r):0 for i in self.generators for r in self.regions},
                      "Dem":{r:self.demand[r][str(1)] for r in self.regions}
                      }
        self.state["t"] = self.t
    def reset(self, seed = 0, options = None):
        self.t = self.reward_ep = 0
        self.cost_ep = 0
        self.get_new_start_state()
        self.truncated  = self.terminated = False
        self.penalty = 0
        #print(self.state)

        self.flatt_state, _ = flatten_and_track_mappings(self.state)

        if self.t == self.T:
            self.terminated = True
        return th.tensor(self.flatt_state).to(self._device), {"dict_state": self.state, "terminated": self.terminated, "truncated": self.truncated}

    def check_bounds_cost(self,action,cost):
        for i in self.generators:
            for r in self.regions:
                #print(action["addgen"])
                if(action["addgen"][(i,r)]<0):
                    cost+=(action["addgen"][(i,r)])**2*self.D+self.P
                    self.pens_step['Total penalty: Generator lower bound violations']+=(action["addgen"][(i,r)])**2*self.D+self.P
                    self.pens_step['Number of Generator lower bound violations']+=1
                    action["addgen"][(i,r)] = 0
                if(action["addgen"][(i,r)]+self.state["num_gen"][i,r]>self.maxgen[i][r]):
                    #print(cost)
                    cost+=(-self.maxgen[i][r]+self.state["num_gen"][i,r]+action["addgen"][(i,r)])**2*self.D+self.P
                    #print(cost)
                    self.pens_step['Total penalty: Generator upper bound violations']+=(-self.maxgen[i][r]+self.state["num_gen"][i,r]+action["addgen"][(i,r)])**2*self.D+self.P
                    self.pens_step['Number of Generator upper bound violations']+=1
                    action["addgen"][(i,r)] = self.maxgen[i][r]-self.state["num_gen"][i,r]
        return action,cost
    def update_genreward(self,action,reward):
        for i in self.generators:
            for r in self.regions:
                    reward+=-action["addgen"][(i,r)]*self.installcost["generators"][i]
                    #print(reward)
                    self.state["num_gen"][i,r]+=action["addgen"][(i,r)]
        return reward
    def check_dem_cost(self,action,cost):
        for r in self.regions:
            total_avail = sum(self.state["num_gen"][(i,r)]*self.gencap[i] for i in self.generators)
            #print(total_avail)
            if(self.state["Dem"][r]> total_avail):
                cost+=self.P
                cost+=self.D*(self.state["Dem"][r]-total_avail)**2
                self.pens_step['Number of Demand violations']+=1
                self.pens_step['Total penalty: Demand violations']+=self.D*(self.state["Dem"][r]-total_avail)**2
                
            if(total_avail<0):
                cost+=self.D*(total_avail)**2
                self.pens_step['Number of negative power violations']+=1
                self.pens_step['Total penalty: negative power violations']+=self.D*(total_avail)**2
        return cost
    def sanitize_action(self,action):
        action["addgen"] = {r:round(p) for r,p in action["addgen"].items()}
        return action
    def update_dem(self):
        #print("t",self.t)
        for r in self.regions:
            if(self.terminated == False):
                self.state["Dem"][r] = self.demand[r][str(self.t+1)]
            else:
                self.state["Dem"][r] = 0
    def set_seed(self, seed: int):
        random.seed(seed)
    def spec_log(self, logger: Logger) -> None: 
        # Omnisafe method called at the end of each epoch. Averaged values are logged
        for key, value in self.env_spec_log.items():
            logger.store({key: value})
            self.env_spec_log[key] = 0.0
    def render(self, mode='human'):
        print("state:",f"{self.state}")
        print("cost:",f"{self.cost_ep}")
        print("Specification:",f"{self.env_spec_log}")
    def close(self):
        pass    
    @property
    def max_episode_steps(self) -> None:
        """The max steps per episode."""
        print("It goes in")
        return self.T
    
def recurse(eg,current, path=[],):
    if isinstance(current, dict):
        for k, v in current.items():
            recurse(eg,v, path + [str(k)])
    else:
        key = 'env_cfgs:' + ':'.join(path)
        val = current if isinstance(current, list) else [current]
        eg.add(key, val)

if __name__ == '__main__':
    eg = ExperimentGrid(exp_name='Benchmark_Safety_simple_2')

    base_policy = ['PolicyGradient', 'NaturalPG', 'TRPO', 'PPO']
    naive_lagrange_policy = ['PPOLag', 'TRPOLag', 'RCPO']
    first_order_policy = ['CUP', 'FOCOPS', 'P3O']
    second_order_policy = ['CPO', 'PCPO']
    #saute_policy = ['PPOSaute', 'TRPOSaute']
    

    # Set the environments.
    mujoco_envs = [
        'Capacity-Expansion'
    ]
    eg.add('env_id', mujoco_envs)

    # Set the device.
    avaliable_gpus = list(range(torch.cuda.device_count()))
    gpu_id = [0]
    # if you want to use CPU, please set gpu_id = None
    # gpu_id = None

    if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
        warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
        gpu_id = None
    env_config = {
    'T': 2,
    'gencap': {"i1": 10},
    'maxgen': {"i1":{ "r1": 1,"r2":1}},
    'installcost': {"generators": {"i1": 10}},
    'demand': {"r1":{"1": 10,"2":10},"r2":{"1": 10,"2":10}}}
    eg.add('algo', base_policy+naive_lagrange_policy+first_order_policy+second_order_policy)
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('logger_cfgs:use_tensorboard', [True])
    eg.add('train_cfgs:vector_env_nums', [1])
    eg.add('train_cfgs:torch_threads', [1])
   
    recurse(eg, env_config)
    eg.add('model_cfgs:actor:output_activation', ['tanh'])
    eg.add('algo_cfgs:steps_per_epoch', [20])
    eg.add('train_cfgs:total_steps', [100])
    eg.add('seed', [0])
    eg.add('train_cfgs:device', ['cuda:0'])
    # total experiment num must can be divided by num_pool
    # meanwhile, users should decide this value according to their machine
    eg.run(train, num_pool=12, gpu_id=gpu_id)

    
    eg.analyze(parameter='algo', values=None, compare_num=12)
    #eg.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    a = eg.evaluate(num_episodes=10)
    
    print(dir(eg))
    #print("Check",dir(eg._evaluator))
    #print("Check1",dir(eg.evaluate))
    #print("Check3",eg._evaluator._actor)
   # print("Check4",eg._evaluator._cfgs)
    