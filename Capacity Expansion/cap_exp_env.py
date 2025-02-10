import numpy as np
from pyomo.environ import *
#from pyomo import *
#import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.contrib.iis import *
import openpyxl
import random
import json

from stable_baselines3 import PPO
#from bayes_opt import acquisition

import numpy as np
import gymnasium as gym
# from gymnasium import spaces
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from gymnasium.utils import seeding
import json
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os, sys
from utils import *
from PIL import Image, ImageDraw, ImageFont


def assign_env_config(self, kwargs):
    print("Assigning configuration...")
    for key, value in kwargs.items():
        print(f"Trying to set {key} to {value}")
        if hasattr(self, key):
            print(f"Setting {key} to {value}")
            setattr(self, key, value)
        else:
            print(f"{self} has no attribute, {key}")
            raise AttributeError(f"{self} has no attribute, {key}")

'''def flatten_dict(dictionary, parent_key='', separator=';'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)'''
'''def flatten_and_track_mappings(dictionary, separator=';'):
    flattened_dict = flatten_dict(dictionary, separator=separator)
    mappings = [(index, key.split(separator)) for index, (key, value) in enumerate(flattened_dict.items())]
    flattened_array = np.array([value for key, value in flattened_dict.items()]).astype("float32")
    return flattened_array, mappings'''


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
class Cap_exp_env(gym.Env):
    def __init__(self,*args, **kwargs):
        super().__init__()
        self.gen_cap = {"i1":0}
        self.trans_cap = {1:0}
        self.demand = {("r1",1):0,("r2",1):0}
        self.regions = ["r1","r2"]
        self.eps = 10**(-3)
        self.D = 10**3
        self.P = 10**3
        self.T = 1
        self.trans_dict = {1:("r1","r2")}
        self.install_cost = {"generators" : {"i1":0},"transmission":{1:0}}
        self.removal_cost = {"generators" : {"i1":0},"transmission":{1:0}}
        with open("./action_sample_base.json" ,"r") as f:
            action = f.read()
        self.action_sample = json.loads(action)
        assign_env_config(self, kwargs)
        self.generators = list(self.gen_cap.keys())
        self.transmission_lines = list(self.trans_cap.keys())
        self.reset()
        #print(self.state)
        self.flatt_state, self.mapping_obs = flatten_and_track_mappings(self.state)
        self.obs_high = np.concatenate([np.repeat(100,len(self.generators)*len(self.regions)),np.repeat(1,len(self.transmission_lines)),np.repeat(max(self.demand.values()),len(self.regions)),np.repeat(self.T,1)])
        #print(self.state)
        #print(self.flatt_state)
        #print(self.mapping_obs)
        #print(self.trans_cap)
        self.observation_space = Box(low=0, high=self.obs_high, shape=(self.flatt_state.shape[0],),dtype=np.int32)
        
        self.flatt_act_sample, self.mapping_act = flatten_and_track_mappings(self.action_sample)
        #print(self.mapping_act)
        if self.trans_cap:
            max_trans_cap = max(self.trans_cap.values())
        else:
            max_trans_cap = 0
        self.action_space = Box(low=-max(10,max_trans_cap), high=max(10,max_trans_cap), shape=(len(self.flatt_act_sample),)) #Change the bounds

    
    def reset(self):
        self.t = self.reward = 0
        self.cost = 0
        self.get_new_start_state()
        self.terminated = False
        self.penalty = 0
        self.flatt_state, _ = flatten_and_track_mappings(self.state)
        return self.flatt_state, {"dict_state": self.state, "terminated": self.terminated}
    def get_new_start_state(self):
        self.state = {"num_gen" : {(i,r):0 for i in self.generators for r in self.regions},
                      "num_trans":{l:0 for l in self.transmission_lines},"Dem":{r:0 for r in self.regions}
                      }
        self.state["t"] = self.t
    
    def check_bounds_cost(self,action):
        for i in self.generators:
            for r in self.regions:
                if(action["add_gen"][(i,r)]<0):
                    self.cost+=(action["add_gen"](i,r))**2+self.P
                    action["add_gen"] = 0
                if(action["ret_gen"][(i,r)]<0):
                    self.cost+=(action["ret_gen"](i,r))**2+self.P
                    action["ret_gen"] = 0
                if(action["ret_gen"][(i,r)]>self.state["num_gen"][(i,r)]):
                    self.cost+=(action["ret_gen"](i,r)-self.state["num_gen"][(i,r)])**2+self.P
                    action["ret_gen"] = self.state["num_gen"][(i,r)]
        if(len(self.transmission_lines)>0):
             for l in self.transmission_lines:
                if(action["pow_flow"][l]<-self.trans_cap[l]):
                    self.cost+=self.P
                    action["pow_flow"][l] = -self.trans_cap[l]
                if(action["pow_flow"][l]>self.trans_cap[l]):
                    self.cost+=self.P
                    action["pow_flow"][l] = self.trans_cap[l]  
        return action
    def update_genreward(self,action):
        for i in self.generators:
            for r in self.regions:
                    self.reward+=-action["add_gen"][(i,r)]*self.install_cost["generators"][i]
                    self.reward+=-action["ret_gen"][(i,r)]*self.removal_cost["generators"][i]
                    #print(self.reward)
                    self.state["num_gen"][i,r]+=action["add_gen"][(i,r)]-action["ret_gen"][(i,r)]
    def update_tlreward(self,action):
        if(len(self.transmission_lines)>0):
            for l in self.transmission_lines:
                if(action["pow_flow"][l]!=0 and self.state["num_trans"][l]==0):
                    #print("yes")
                    self.state["num_trans"][l] = 1     
                    self.reward-=self.install_cost["transmission"][l]
    def check_dem_cost(self,action):
        '''for r1 in self.regions:
            for r2 in self.regions:
                if(r1!=r2 and (r1,r2) not in self.transmission_lines):
                    action["pow_flow"][(r1,r2)] = -action["pow_flow"][("r2,r1")]'''
        #print(self.transmission_lines)
        #print(self.trans_dict)
        if(len(self.transmission_lines)>0):
            action["pow_flow_tup"] = ({v: action["pow_flow"][l] for l,v in self.trans_dict.items()})
            action["pow_flow_tup"].update({v[::-1]: -action["pow_flow"][l] for  l,v in self.trans_dict.items()})
            #print(action)
            for r in self.regions:
                total_avail = sum(self.state["num_gen"][(i,r)]*self.gen_cap[i] for i in self.generators)+sum(action["pow_flow_tup"][l] for l in action["pow_flow_tup"].keys() if l[1]==r)
                #print(total_avail)
                #print(r)
                if(self.state["Dem"][r]> total_avail):
                    self.cost+=self.P
                    #print(self.cost)
                    self.cost+=self.D*(self.state["Dem"][r]-total_avail)**2
                    #print(self.cost)
                if(total_avail<0):
                    self.cost+=self.D*(total_avail)**2
                    #(self.cost)
        else:
            for r in self.regions:
                total_avail = sum(self.state["num_gen"][(i,r)]*self.gen_cap[i] for i in self.generators)
                #print(total_avail)
                if(self.state["Dem"][r]> total_avail):
                    self.cost+=self.P
                    self.cost+=self.D*(self.state["Dem"][r]-total_avail)**2
                    
                if(total_avail<0):
                    self.cost+=self.D*(total_avail)**2
    def sanitize_action(self,action):
        if(len(self.transmission_lines)>0):
            action["pow_flow"] = {l: (0 if -self.eps <= p <= self.eps else p) for l, p in action["pow_flow"].items()}
        #print(action)
        action["add_gen"] = {r:round(p) for r,p in action["add_gen"].items()}
        action["ret_gen"] = {r:round(p) for r,p in action["ret_gen"].items()}
        return action
    def update_dem(self):
        for r in self.regions:
            self.state["Dem"][r] = self.demand[r,self.t]

    def step(self, action: th.Tensor):
        self.t += 1
        self.reward = 0
        self.cost = 0
        if(not isinstance(action, list)):
            action = action.tolist()
        
        action = reconstruct_dict(action, self.mapping_act)
        for v in action.keys():
            action[v] = convert_dict_to_tuple_keys(action[v]) 
        #print(action)
        action = self.sanitize_action(action)
        action = self.check_bounds_cost(action)
        #print(action)
        self.update_genreward(action)
        self.update_tlreward(action)
        self.update_dem()
        self.check_dem_cost(action)
        self.reward-=self.cost
        self.state["t"] = self.t
        
        self.flatt_state, _ = flatten_and_track_mappings(self.state)
        
        if self.t == self.T:
            self.terminated = True
        return self.flatt_state, self.reward, self.terminated, {"dict_state": self.state, "terminated": self.terminated}

    def render(self, mode='human'):
        print("state:",f"{self.state}")
        print("cost:",f"{self.cost}")
    def close(self):
        pass
config = {'T' : 2,'regions' : ["r1","r2"],
          'gen_cap' : {"i1":10},
          'trans_cap' : {"1":10},
          'install_cost' : {"generators" : {"i1":10},"transmission":{"1":0.1}},
          'removal_cost' : {"generators" : {"i1":5}},
          'demand':{("r1",1):5,("r2",1):5,("r1",2):5,("r2",2):5},
          'trans_dict' : {"1":("r1","r2")}}
env = Cap_exp_env(**config)

state = env.reset()
done = False
rew_tot = 0

action_1 = [1,1,0,0,5]
state, reward, done, info = env.step(action_1)
env.render()
print(reward)

action_1 = [0,0,1,1,5]
state, reward, done, info = env.step(action_1)
env.render()
print(reward)
env.close()
