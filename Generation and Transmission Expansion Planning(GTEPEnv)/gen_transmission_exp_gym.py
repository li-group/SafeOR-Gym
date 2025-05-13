import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from gymnasium.utils import seeding
import json
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os, sys
from utils import assign_env_config,flatten_and_track_mappings,reconstruct_dict,convert_dict_to_tuple_keys,flatten_dict
from typing import Any, ClassVar, List, Tuple, Optional, Dict
import random
class Generator_transmission_expansion_env(gym.Env):
    '''
    The capacity expansion problem is a combinatorial optimization problem that involves determining the optimal capacities of system components—
    here, power generators as well as the transmission lines to install—across multiple time periods. The goal is to meet regional power demands 
    while minimizing the overall cost. This version of the environment focuses specifically on planning decisions for generator and transmission line installations.

    Problem Setup
        - We consider a set of regions, each with a time-dependent power demand.
        - A set of generator types can be installed in each region.
        - A set of probable transmission lines
        - Constraints include:
            - Maximum allowable generators of each type per region.
            - The absolute value of the power flow between 2 regions to be within the capacity of the transmission lines between these 2 regions
            - Satisfying power demand at every time step.
        - Violating these constraints results in penalties, and installation costs are incurred for added generators.
        - The agent's objective is to minimize total cost = installation cost + penalties.

    State space:
        The state space is a dictionary with the following keys:
            - num_gen: A dictionary with keys as tuples (i,r) where i is the generator type and r is the region. The value is the number of generators 
            of type i in region r.
            - num_tl: A dictionary with keys as transmission lines and the value being 1 if the transmission line is installed and 0 if it is not installed.
            - Dem:  A dictionary with keys as region identifiers and values representing the demand at the current time period (t). Although demand 
            is initially generated for time t+1, by the time the environment reaches time t, this becomes the active demand for planning. 
            - t: The current time period.
        All values are flattened into a single array for the observation space.
    Action space:
        The action space is a dictionary with the following keys:
            - addgen: A dictionary with keys as tuples (i,r) where i is the generator type and r is the region. The value is the number of generators 
            of type i to be added in region r.
            - pow_flow: A dictionary with keys as "r1_r2" where r1 and r2 are the regions between the transmission line can be built or is built
        In practice, the values in addgen are integer values between 0 and the maximum number of generators of each type in each region. Similarly the values in pow_flow
        can be between the negative of capacity of the transmission line and the positive of the capacity of the transmission line.  For the sake of
        training the agent, we use a flattened continuous action space with values between -1 and 1. The addgen part is scaled to the range between 0 and the maximum number 
        of generators of each type in each region and then rounded to the nearest integer. Similialy, the pow_flow part is scaled to the range between the the negative 
        of capacity of the transmission line and the positive of the capacity of the transmission line.
    Transition to next state:
        The num_gen is updated by adding the corresponding addgen value after adjusting for bound constraints. The num_tl for a transmission line is updated to 1 
        if there is power flow in the transmission line and the num_tl was 0 for the previous time period.
    Cost:
        The cost is the penalties incurred for violating the constraints:
            - Number of generators and power flow within the bounds
            - Demand satisfaction  
    Reward:
        The reward is the negative of the sum of installation cost of the generators added and the cost in the current time period. 
    Starting State:
        We start with no generators in any region and the demand for each region at time period 1 and with time t = 0. 
    Termination:
        The episode terminates when the time period t reaches T.
    '''
    def __init__(self, env_id: str,
                 **kwargs: Any) -> None:
        
        """
            Initialize the environment.

            Args:
                env_id (str): Identifier for the environment
                env_spec_log: Components to be noted
                D: Multiplier for L2 based penalty on each violation
                P: Fixed penalty (L0 based) on each violation 
                eps: small bound to help round negligble power transfers to 0
                gencap: Dictionary corresponding to capacity of different generators ({generator:capacity})
                tlcap: Dictionary corresponding to capacity of different transmission lines ({transmission line:capacity})
                demand: Dicitionary corresponding to demand in different regions ({region:demand})
                T: Number of time periods
                maxgen: Maximum number of generators of each type in each regions ({generator,region}:Maximum number)
                installcost: Dictionary of installation costs of generators and transmission lines ({"generators" : {generator:cost},"transmission":{transmission line:cost}}) 
                config_file: Json file with configuration of environment
                action_space_file: Json file with sample actions dictionary corresponding to config_file

        """

        super().__init__()
        self.env_id = env_id
        self.env_spec_log = {'Total penalty: Negative action':0,'Number of negative action violations':0,
                             'Total penalty: more generators than maximum possible':0,'Number of more generators than maximum possible':0,
                              'Number of Transmission power bound violations':0,'Number of Demand violations': 0,'Total penalty: Demand violations': 0,

                             }
        self._device = kwargs.get('device', 'cuda' if th.cuda.is_available() else 'cpu')
        
        #Default parameters
        self.D = 10
        self.P = 10
        self.eps = 10**(-3)
        self.gencap = {"i1":0}
        self.tlcap = {"r1_r2":0}
        self.demand = {"r1":{"1":0},"r2":{"1":0}}
        self.T = 1
        self.maxgen = {"i1":{ "r1": 10,"r2":10}}
        self.installcost = {"generators" : {"i1":0},"transmission":{"r1_r2":0}}
        self.window_len = 2
        self.action_sample_file = 'gen_trans_exp_default_action_sample.json'
        self.config_file = kwargs.get('config_file', '')
        with open(self.config_file ,"r") as f:
            env_config_read = f.read()
        env_config = json.loads(env_config_read)
        assign_env_config(self, env_config['env_init_cfgs'])
        self.generators = list(self.gencap.keys())
        self.transmission_lines = list(self.tlcap.keys())
        self.regions = list(self.demand.keys())
        with open(self.action_sample_file ,"r") as f:
            action = f.read()
        self.action_sample = json.loads(action)
        self.reset()
        self.flatt_state, self.mapping_obs = flatten_and_track_mappings(self.state)
        self.max_gen = max(v for inner_dict in self.maxgen.values() for v in inner_dict.values())
        self.max_tl = 1
        self.max_demand_value = max(max(region_demand.values()) for region_demand in self.demand.values())
        self.obs_high = np.concatenate([np.repeat(self.max_gen,len(self.generators)*len(self.regions)),np.repeat(self.max_tl,len(self.transmission_lines)),np.repeat(self.max_demand_value,len(self.regions)*self.window_len),np.repeat(self.T,1)])
        self.observation_space = Box(low=0, high=self.obs_high, shape=(self.flatt_state.shape[0],))
        self.flatt_act_sample, self.mapping_act = flatten_and_track_mappings(self.action_sample)
        low_list = [0] * len(self.mapping_act)
        high_list = [0] * len(self.mapping_act)
        for(k,val) in self.mapping_act:
            if(val[0]=='addgen'):
                low_list[k] = 0
                high_list[k] = self.maxgen[val[1]][val[2]]
            else:
                low_list[k] = -self.tlcap[val[1]]
                high_list[k] = self.tlcap[val[1]]
        self.action_low = np.array(low_list)
        self.action_high = np.array(high_list)
        self.action_space = Box(low=-1, high=1, shape=(len(self.flatt_act_sample),)) 
        
    def get_new_start_state(self):
        '''Function to get a new start state.'''
        self.state = {"num_gen" : {(i,r):0 for i in self.generators for r in self.regions}, "num_tl":{l:0 for l in self.transmission_lines},
                      "Dem":{(r,k):self.demand[r][str(k+1)] if k+1<self.T else 0 for r in self.regions for k in range(0,self.window_len)}
                      }
        self.state["t"] = self.t
    def reset(self, seed = 0, options = None):
        '''Reset the environment to the initial state'''
        self.t = self.reward_ep = 0
        self.cost_ep = 0
        self.cost = 0
        self.reward = 0
        self.get_new_start_state()
        self.truncated  = self.terminated = False
        self.flatt_state, _ = flatten_and_track_mappings(self.state)
        return th.tensor(self.flatt_state).to(self._device), {"dict_state": self.state, "terminated": self.terminated, "truncated": self.truncated}
    def compute_cost_bounds_santize_actions(self,action):
        '''Function to check if the action gives  a number of generators in a region less than or equal to the maximum number of generators in that region. 
        If this constraint is violated, we adjust the actions and calculte the penalty for the violation to the cost.
        The penalty is the sum of a scaled L0 violation and a scaled L2 violation''' 
        
        bounds_cost = 0
        for i in self.generators:
            for r in self.regions:
                if(action["addgen"][i][r]+self.state["num_gen"][i,r]>self.maxgen[i][r]):
                    bounds_cost+=(-self.maxgen[i][r]+self.state["num_gen"][i,r]+action["addgen"][i][r])**2*self.D+self.P
                    self.pens_step['Total penalty: more generators than maximum possible']+=(-self.maxgen[i][r]+self.state["num_gen"][i,r]+action["addgen"][i][r])**2*self.D+self.P
                    self.pens_step['Number of more generators than maximum possible']+=1
                    action["addgen"][i][r] = self.maxgen[i][r]-self.state["num_gen"][i,r]
        return action,bounds_cost

   
    def update_num_gen_and_num_tl(self,action):
        '''Function to update the number of generators in each region as well as the number of different transmision lines.'''
        for i in self.generators:
            for r in self.regions:
                self.state["num_gen"][i,r]+=action["addgen"][i][r]
        if(len(self.transmission_lines)>0):
            for l in self.transmission_lines:
                if(action["powflow"][l]!=0 and self.state["num_tl"][l]==0):
                    self.state["num_tl"][l] = 1     
    
    
    def compute_reward_gen(self,action):
        '''Function to calculate the reward for the installation of the generators. It is calulcated the negative of the installation cost of the generators added.'''
        gen_reward = sum(
        -action["addgen"][i][r] * self.installcost["generators"][i]
        for i in self.generators for r in self.regions
    )
        return gen_reward
    
    def compute_reward_tl(self,action):
        '''Function to calculate the reward for the installation of the transmission lines. It is calulcated the negative of the installation cost of the transmission lines added.'''
        tl_reward = 0
        if(len(self.transmission_lines)>0):
            for l in self.transmission_lines:
                if(action["powflow"][l]!=0 and self.state["num_tl"][l]==0):
                    tl_reward-=self.installcost["transmission"][l]
        return tl_reward

    def compute_cost_demand_satisfaction(self,action):
        '''Function to check the violations of the demand constraints. If the demand is not satisfied, we calculate a penalty for the violation. 
        The penalty is the sum of a scaled L0 violation and a scaled L2 violation when the demand is violated.'''
        dem_cost = 0
        for r in self.regions:
            total_avail = sum(self.state["num_gen"][(i,r)]*self.gencap[i] for i in self.generators)
            if(len(self.transmission_lines)>0):
                total_avail+=sum(action["powflow_tup"][l] for l in action["powflow_tup"].keys() if l[1]==r)
            if(self.state["Dem"][r,0]> total_avail):
                dem_cost+=self.D*(self.state["Dem"][r,0]-total_avail)**2+self.P
                self.pens_step['Number of Demand violations']+=1
                self.pens_step['Total penalty: Demand violations']+=self.D*(self.state["Dem"][r,0]-total_avail)**2+self.P
        return dem_cost

    def sanitize_action(self,action_scaled):
        '''Function to santize the action. In the case of generators, we scale the action to the space of [0, maxgen] and round the action to the nearest integer. 
        In the case of transmission lines we scale the action to [-transcap,transcap] round neglible power flows to 0. We also add the power flows for the reverse
        direction which is the negative of the forward direction'''
        action_scaled = th.as_tensor(action_scaled, device=self._device)
        low_torch = th.as_tensor(self.action_low, dtype=action_scaled.dtype, device=self._device)
        high_torch = th.as_tensor(self.action_high, dtype=action_scaled.dtype, device=self._device)
        action = (action_scaled + 1) / 2 * (high_torch - low_torch) + low_torch #Scale the action
        if(not isinstance(action, list)):
            action = action.tolist()
        action = reconstruct_dict(action, self.mapping_act)
        if(len(self.transmission_lines)>0):
            action["powflow"] = {l: (0 if -self.eps <= p <= self.eps else p) for l, p in action["powflow"].items()}
            action["powflow_tup"] = ({tuple(key.split('_')): action["powflow"][key] for key in self.transmission_lines})
            action["powflow_tup"].update({tuple(key.split('_'))[::-1]: -action["powflow"][key] for key in self.transmission_lines})
        action["addgen"] = {
            outer_k: {inner_k: round(inner_v) for inner_k, inner_v in inner_d.items()}
            for outer_k, inner_d in action["addgen"].items()
        }
        return action
    def update_demand(self):
        '''Function to update the demand for each region. The demand is updated to the next time period if exists else 0.'''
        for r in self.regions:
            for k in range(self.window_len):
                if(self.t+k+1<=self.T):
                    self.state["Dem"][r,k] = self.demand[r][str(self.t+k+1)]
                else:
                    self.state["Dem"][r,k] = 0
    def step(self, action_scaled: th.Tensor):
        self.t += 1
        self.state["t"] = self.t
        self.pens_step = {k:0 for k in self.env_spec_log.keys()}
        action_scaled = action_scaled.clip(-1,1)
        action = self.sanitize_action(action_scaled)
        action,bounds_cost = self.compute_cost_bounds_santize_actions(action)
        reward = self.compute_reward_gen(action)+self.compute_reward_tl(action)
        self.update_num_gen_and_num_tl(action)
        cost = bounds_cost+self.compute_cost_demand_satisfaction(action)
        self.reward_ep+=reward
        self.cost_ep+=cost
        self.reward = reward
        self.cost = cost
        if self.t == self.T:
            self.terminated = True
        self.update_demand()
        self.flatt_state, _ = flatten_and_track_mappings(self.state)
        for k in self.env_spec_log.keys():
            self.env_spec_log[k] += self.pens_step[k]
        flatt_state_tensor = th.tensor(self.flatt_state, device=self._device)
        return flatt_state_tensor, th.tensor(reward-cost, device=self._device), th.tensor(self.terminated, device=self._device), th.tensor(self.truncated, device=self._device), {}
    
    @property
    def max_episode_steps(self) -> int:
        return self.T
    
    def render(self, mode='human'):
        print("state:",f"{self.state}")
        print("Specification:",f"{self.env_spec_log}")

    def close(self) -> None:
        return None

    def set_seed(self, seed: int) -> None:
        random.seed(seed)

