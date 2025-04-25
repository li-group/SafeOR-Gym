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
class Generator_expansion_env(gym.Env):
    '''
    The capacity expansion problem is a combinatorial optimization problem that involves determining the optimal capacities of system components—
    here, power generators—across multiple time periods. The goal is to meet regional power demands while minimizing the overall cost.
    This version of the environment focuses specifically on planning decisions for generator installations.

    Problem Setup
        - We consider a set of regions, each with a time-dependent power demand.
        - A set of generator types can be installed in each region.
        - Constraints include:
            - Maximum allowable generators of each type per region.
            - Satisfying power demand at every time step.
        - Violating these constraints results in penalties, and installation costs are incurred for added generators.
        - The agent's objective is to minimize total cost = installation cost + penalties.

    State space:
        The state space is a dictionary with the following keys:
            - num_gen: A dictionary with keys as tuples (i,r) where i is the generator type and r is the region. The value is the number of generators 
            of type i in region r.
            - Dem:  A dictionary with keys as region identifiers and values representing the demand at the current time period (t). Although demand 
            is initially generated for time t+1, by the time the environment reaches time t, this becomes the active demand for planning. 
            - t: The current time period.
        All values are flattened into a single array for the observation space.
    Action space:
        The action space is a dictionary with the following keys:
            - addgen: A dictionary with keys as tuples (i,r) where i is the generator type and r is the region. The value is the number of generators 
            of type i to be added in region r.
        In practice, the action space is an integer value between 0 and the maximum number of generators of each type in each region. For the sake of
        training the agent, we use a flattened continuous action space with values between -1 and 1. The action is scaled to the range between 0 and the maximum number 
        of generators of each type in each region and then rounded to the nearest integer. 
    Transition to next state:
        The num_gen is updated by adding the corresponding addgen value after adjusting for bound constraints.
    Cost:
        The cost is the penalties incurred for violating the constraints.  
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
                P: Fixed penalty (L0 based) on each violation 0
                gencap: Dictionary corresponding to capacity of different generators ({generator:capacity})
                regions: List of regions
                demand: Dicitionary corresponding to demand in different regions ({region:demand})
                T: Number of time periods
                maxgen: Maximum number of generators of each type in each regions ({generator,region}:Maximum number)
                installcost: Dictionary of installation costs of generators  ({"generators" : {generator:cost}}) 
                generators: list of generators
                

        """

        super().__init__()
        self.env_id = env_id
        self.env_spec_log = {'Total penalty: Negative action':0,'Number of negative action violations':0,
                             'Total penalty: more generators than maximum possible':0,'Number of more generators than maximum possible':0,
                              'Number of Demand violations': 0,'Total penalty: Demand violations': 0,'num_steps':0,'num_episodes':0,'sum reward':0,'sum reward square':0,
                              'sum cost':0,'sum cost square':0
                             }
        self._device = kwargs.get('device', 'cuda' if th.cuda.is_available() else 'cpu')
        #Default parameters
        self.gencap = {"i1":0}
        self.demand = {"r1":{"1":0},"r2":{"1":0}}
        self.eps = 10**(-3)
        self.D = 10
        self.P = 10
        self.T = 1
        self.maxgen = {"i1":{ "r1": 10,"r2":10}}
        self.installcost = {"generators" : {"i1":0}}
        '''self.render_mode= 'rgb_array',
        self.camera_name= None,
        self.camera_id = None,
        self.width = 256,
        self.height = 256'''
        assign_env_config(self, kwargs)
        self.generators = list(self.gencap.keys())
        self.regions = list(self.demand.keys())
        with open("./action_sample_base_1.json" ,"r") as f:
            action = f.read()
        self.action_sample = json.loads(action)
        self.reset()
        self.flatt_state, self.mapping_obs = flatten_and_track_mappings(self.state)
        self.max_gen = 100
        self.max_demand_value = max(max(region_demand.values()) for region_demand in self.demand.values())
        self.obs_high = np.concatenate([np.repeat(self.max_gen,len(self.generators)*len(self.regions)),np.repeat(self.max_demand_value,len(self.regions)),np.repeat(self.T,1)])
        self.observation_space = Box(low=0, high=self.obs_high, shape=(self.flatt_state.shape[0],))
        self.flatt_act_sample, self.mapping_act = flatten_and_track_mappings(self.action_sample)
        low_list = []
        high_list = []
        for i in self.generators:
            for r in self.regions:
                low_list.append(0)
                high_list.append(self.maxgen[i][r])
        self.action_low = np.array(low_list)
        self.action_high = np.array(high_list)
        self.action_space = Box(low=-1, high=1, shape=(len(self.flatt_act_sample),)) 
    def get_new_start_state(self):
        #Function to get a new start state. 
        #print(self.demand)
        self.state = {"num_gen" : {(i,r):0 for i in self.generators for r in self.regions},
                      "Dem":{r:self.demand[r][str(1)] for r in self.regions}
                      }
        self.state["t"] = self.t
    def reset(self, seed = 0, options = None):
        ## Reset the environment to the initial state
        self.t = self.reward_ep = 0
        self.cost_ep = 0
        self.cost = 0
        self.reward = 0
        self.get_new_start_state()
        self.truncated  = self.terminated = False
        self.flatt_state, _ = flatten_and_track_mappings(self.state)
        return th.tensor(self.flatt_state).to(self._device), {"dict_state": self.state, "terminated": self.terminated, "truncated": self.truncated}
    def check_bounds_cost(self,action):
        #Check the bounds of the action. If the action is out of bounds, set it to the bounds and add the penalty for the violation to the cost.
        #The penalty is the sum of a scaled L0 violation and a scaled L2 violation.
        
        bounds_cost = 0
        for i in self.generators:
            for r in self.regions:
                if(action["addgen"][(i,r)]<0):
                    bounds_cost+=(action["addgen"][(i,r)])**2*self.D+self.P
                    self.pens_step['Total penalty: Negative action']+=(action["addgen"][(i,r)])**2*self.D+self.P
                    self.pens_step['Number of negative action violations']+=1
                    action["addgen"][(i,r)] = 0
                if(action["addgen"][(i,r)]+self.state["num_gen"][i,r]>self.maxgen[i][r]):
                    bounds_cost+=(-self.maxgen[i][r]+self.state["num_gen"][i,r]+action["addgen"][(i,r)])**2*self.D+self.P
                    self.pens_step['Total penalty: more generators than maximum possible']+=(-self.maxgen[i][r]+self.state["num_gen"][i,r]+action["addgen"][(i,r)])**2*self.D+self.P
                    self.pens_step['Number of more generators than maximum possible']+=1
                    action["addgen"][(i,r)] = self.maxgen[i][r]-self.state["num_gen"][i,r]
        return action,bounds_cost
    def update_num_gen(self,action):
        #Function to update the number of generators in each region.
        for i in self.generators:
            for r in self.regions:
                self.state["num_gen"][i,r]+=action["addgen"][(i,r)]
    def update_genreward(self,action):
        #Function to calculate the reward for the installation of the generators. 
        #It is calulcated the negative of the installation cost of the generators added.
        gen_reward = sum(
        -action["addgen"][(i, r)] * self.installcost["generators"][i]
        for i in self.generators for r in self.regions
    )
        return gen_reward
    def _compute_reward(self,action):
        #Function to calculate the reward excluding the violation of constraints. 
        reward = 0
        reward += self.update_genreward(action)
        return reward
    def check_dem_cost(self):
        #Function to check the violations of the demand constraints. If the demand is not satisfied, add the penalty to the cost.
        #The penalty is the sum of a scaled L0 violation and a scaled L2 violation whe the demand is violated.
        dem_cost = 0
        for r in self.regions:
            total_avail = sum(self.state["num_gen"][(i,r)]*self.gencap[i] for i in self.generators)
            #print(total_avail)
            if(self.state["Dem"][r]> total_avail):
                dem_cost+=self.P
                dem_cost+=self.D*(self.state["Dem"][r]-total_avail)**2
                self.pens_step['Number of Demand violations']+=1
                self.pens_step['Total penalty: Demand violations']+=self.D*(self.state["Dem"][r]-total_avail)+self.P
        return dem_cost
    def _compute_cost(self,bounds_cost):
        #Funtion to calculate the net cost.
        cost = 0
        cost += bounds_cost
        cost += self.check_dem_cost()
        
        return cost
    def sanitize_action(self,action):
        #Function to santize the action. In this case, we round the action to the nearest integer.
        action["addgen"] = {r:round(p) for r,p in action["addgen"].items()}
        return action
    def update_dem(self):
        #Function to update the demand for each region. The demand is updated to the next time period.
        for r in self.regions:
            if(self.terminated == False):
                self.state["Dem"][r] = self.demand[r][str(self.t+1)]
            else:
                self.state["Dem"][r] = 0
    def step(self, action_scaled: th.Tensor):
        
        action_scaled = th.as_tensor(action_scaled, device=self._device)
        low_torch = th.as_tensor(self.action_low, dtype=action_scaled.dtype, device=self._device)
        high_torch = th.as_tensor(self.action_high, dtype=action_scaled.dtype, device=self._device)
        action = (action_scaled + 1) / 2 * (high_torch - low_torch) + low_torch #Scale the actions

        self.t += 1
        self.state["t"] = self.t
        self.pens_step = {k:0 for k in self.env_spec_log.keys()}
        if(not isinstance(action, list)):
            action = action.tolist()
        action = reconstruct_dict(action, self.mapping_act)
        for v in action.keys():
            action[v] = convert_dict_to_tuple_keys(action[v])
        action = self.sanitize_action(action)
        action,bounds_cost = self.check_bounds_cost(action)
        print("action:",action)
        self.update_num_gen(action)
        reward = self.update_genreward(action)
        cost = bounds_cost+self.check_dem_cost()
        self.reward_ep+=reward
        self.cost_ep+=cost
        self.reward = reward
        self.cost = cost
        self.pens_step['num_steps']+=1
        self.pens_step['sum reward']+=self.reward
        self.pens_step['sum reward square']+=self.reward**2
        self.pens_step['sum cost']+=self.cost
        self.pens_step['sum cost square']+=self.cost**2
        if self.t == self.T:
            self.pens_step['num_episodes'] = 1
            self.terminated = True
        self.update_dem()
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