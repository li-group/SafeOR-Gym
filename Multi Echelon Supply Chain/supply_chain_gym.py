# Importing libraries
import numpy as np
from pyomo.environ import *
#from pyomo import *
#import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.contrib.iis import *
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
from PIL import Image, ImageDraw, ImageFont

def flatten_dict(dictionary, parent_key='', separator=';'):
    """
    Recursively flatten a nested dictionary or list into a dict of (string_key -> value).

    Changes from the original:
    - We also flatten lists, producing sub-keys like "parent;listKey;0", "parent;listKey;1", etc.
    - Non-numeric or empty placeholders are handled in a consistent way.
    """
    items = []
    for key, value in dictionary.items():
        # Convert the current key to string for safe concatenation
        str_key = str(key)
        new_key = parent_key + separator + str_key if parent_key else str_key

        if isinstance(value, dict):
            # Recursively flatten any nested dictionaries
            if value:
                items.extend(flatten_dict(value, new_key, separator=separator).items())
            else:
                # If it's an empty dict, store it as 0 or skip it
                items.append((new_key, 0.0))

        elif isinstance(value, list):
            # Flatten each item in the list
            for i, elem in enumerate(value):
                child_key = f"{new_key}{separator}{i}"
                if isinstance(elem, dict):
                    # If list element is another dict, recurse
                    if elem:
                        items.extend(flatten_dict(elem, child_key, separator=separator).items())
                    else:
                        items.append((child_key, 0.0))
                elif isinstance(elem, list):
                    # If list element is itself a list, wrap it in a dict for recursion
                    # or flatten inline. Here we wrap in a dict to reuse flatten_dict logic:
                    sub_list_dict = {i: elem}
                    items.extend(flatten_dict(sub_list_dict, child_key, separator=separator).items())
                else:
                    # Base case: numeric or string or something else
                    items.append((child_key, elem))
        
        else:
            # Base case: numeric or string or other
            items.append((new_key, value))

    return dict(items)

def flatten_and_track_mappings(dictionary, separator=';'):
    """
    1) Recursively flatten the input (dict + possibly nested lists)
    2) Build a mapping from array index -> path
    3) Convert any non-numeric to 0
    4) Return (flattened_array, index_mapping)
    """
    # 1) Flatten
    flattened_dict = flatten_dict(dictionary, separator=separator)

    # 2) Build index->keypath mappings and numeric array
    mappings = []
    flattened_values = []
    for index, (key, value) in enumerate(flattened_dict.items()):
        path_components = key.split(separator)
        mappings.append((index, path_components))

        # 3) Convert non-numeric to 0.0
        if isinstance(value, (int, float)):
            flattened_values.append(value)
        else:
            flattened_values.append(0.0)

    # 4) Convert to numpy array
    flattened_array = np.array(flattened_values, dtype=np.float32)
    return flattened_array, mappings

import gymnasium as gym
import numpy as np
# from your_cap_exp_helpers import (
#     flatten_and_track_mappings, reconstruct_dict,
#     nested_set, flatten_dict, convert_dict_to_tuple_keys
# )

class InvMgmtEnv(gym.Env):
    """
    Inventory Management Environment for multi-echelon supply chain,
    matching the function names/structure from the capacity-expansion environment.
    """

    def __init__(self, *args, **kwargs):
        """
        Functions like Cap_exp_env.__init__:
         1) assign_env_config
         2) set penalty parameters
         3) define placeholder for spaces
         4) reset
        """
        super().__init__()
        # 1) Assign environment config
        self.assign_env_config(kwargs)

        # 2) Penalty multipliers
        self.eps = 1e-3    # small action threshold
        self.D = 1e3       # cost factor if out-of-bounds
        self.P = 1e3       # cost factor if out-of-bounds

        # 3) Observation/action spaces are defined after we build the state in get_new_start_state
        self.observation_space = None
        self.action_space = None

        # 4) Reset environment
        self.reset()

        # print("Reset observation shape:", self.observation_space.shape)
        


    def assign_env_config(self, kwargs):
        """
        Mirrors cap_exp_env.assign_env_config, but uses the inventory defaults:
        - Uses set defaults
        - Overwrites from kwargs
        - Raises AttributeError if attribute isn't recognized
        """

        # Main environment horizon, node counts, etc. (unchanged)
        self.T = 360
        self.num_markets = 1
        self.num_retailers = 1
        self.num_distributors = 2
        self.num_producers = 3
        self.num_raw_distributors = 4
        self.num_total_nodes = (
            self.num_markets + self.num_retailers
            + self.num_distributors + self.num_producers
            + self.num_raw_distributors
        )

        # Example inventory defaults
        self.initial_inv = {1: 100, 2: 120, 3: 80, 4: 300, 5: 250, 6: 220}
        # Higher holding cost at the market/retailer vs. downstream to reflect
        # more expensive storage near the end of the chain
        self.inventory_holding_cost = {
            1: 0.04,  # market
            2: 0.03,  # retailer
            3: 0.02,
            4: 0.015,
            5: 0.018,
            6: 0.017
        }

        # Try to ensure the final route (1->0) is quite profitable (like a retail price),
        # while other routes are smaller markups.
        self.unit_price = {
            (1,0): 15.0,  # final sale to market is quite valuable
            (2,1): 2.5,
            (3,1): 2.7,
            (4,2): 2.0,
            (5,2): 1.5,
            (6,2): 1.6,
            (4,3): 1.8,
            (6,3): 1.8,
            (7,4): 0.5,
            (7,5): 0.4,
            (8,5): 0.35,
            (8,6): 0.4
        }

        # Slightly larger pipeline holding cost to reflect more expensive upstream shipments
        self.material_holding_cost = {
            (2,1): 0.02, (3,1): 0.03,
            (4,2): 0.02, (5,2): 0.015, (6,2): 0.015,
            (4,3): 0.02, (6,3): 0.02,
            (7,4): 0.0,  (7,5): 0.01,
            (8,5): 0.008, (8,6): 0.01
        }

        # Lead times remain the same if you like your structure:
        # (unmodified)
        self.lead_times = {
            (2,1):5, (3,1):3, (4,2):8, (5,2):9, (6,2):11,
            (4,3):10, (6,3):12, (7,4):0, (7,5):1, (8,5):2, (8,6):0
        }

        # Higher operating cost so that producing is not too cheap
        self.operating_cost = {
            4: 2.0,
            5: 2.5,
            6: 3.0
        }

        # Production yield remains 1 if each unit of raw is 1 unit final
        self.production_yield = {4: 1, 5: 1, 6: 1}

        # If letting backlog occur, we penalize heavily (avoid ignoring demand)
        self.unfulfilled_utility_penalty = {
            (1,0): 1e5
        }

        # Demand parameters: higher mean so there's a reason to produce
        self.demand_parameters = {
            'mean': 50,
            'std': 5,
            'p': 0.7,
            # We'll rely on environment seeding for the random seed
        }

        # Inventory capacities
        self.inv_capacity = {1: 300, 2: 300, 3: 300, 4: 500, 5: 500, 6: 400}

        # Reordering capacities for routes
        # self.reordering_route_capacity = {
        #     (2,1): 30, (3,1): 45, (4,2): 80, (5,2): 55, (6,2): 40,
        #     (4,3): 60, (6,3): 35, (7,4): 100, (7,5): 80, (8,5): 80, (8,6): 100
        # }
        self.reordering_route_capacity = {
            (2,1): 100, (3,1): 100, (4,2): 100, (5,2): 100, (6,2): 100,
            (4,3): 100, (6,3): 100, (7,4): 100, (7,5): 100, (8,5): 100, (8,6): 100
        }

        # j_in, j_out remain as is
        self.j_in = {1:[2,3], 2:[4,5,6], 3:[4,6], 4:[7], 5:[7,8], 6:[8]}
        self.j_out = {1:[0], 2:[1], 3:[1], 4:[2,3], 5:[2], 6:[2,3], 7:[4,5], 8:[5,6]}


        # Overwrite with user config
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{self} has no attribute, {key}")

        # Build route lists
        self.main_nodes = list(range(
            self.num_markets,
            self.num_markets + self.num_retailers
            + self.num_distributors + self.num_producers
        ))
        self.reordering_routes = list(self.material_holding_cost.keys())
        self.retailer_routes = [
            rt for rt in self.unit_price
            if rt[0] in range(self.num_markets, self.num_markets + self.num_retailers)
        ]

    def get_new_start_state(self):
        """
        Matches Cap_exp_env.get_new_start_state:
        - Setup arrays
        - Generate demand
        - Build self.state dict
        - Then define obs & action space shape
        """
        # Initialize environment arrays for horizon T
        self.I = np.zeros((self.T+1, len(self.main_nodes)))  # on-hand
        self.Tt = {rt: np.zeros(self.T+1) for rt in self.reordering_routes}  # pipeline
        self.R = {rt: np.zeros(self.T+1) for rt in self.reordering_routes}   # reorder
        self.Rp = {rt: np.zeros(self.T+1) for rt in self.reordering_routes}  # arrived
        self.Ss = {rt: np.zeros(self.T+2) for rt in self.retailer_routes}    # sales
        self.Bb = {rt: np.zeros(self.T+2) for rt in self.retailer_routes}    # backlog
        self.Dd = {rt: np.zeros(self.T+1) for rt in self.retailer_routes}    # demand

        # Generate random demand with self.np_random
        self.demand = {}
        for rt in self.retailer_routes:
            self.demand[rt] = self.np_random.normal(
                loc=self.demand_parameters['mean'],
                scale=self.demand_parameters['std'],
                size=self.T
            )

        # Initialize on-hand inventory
        for idx, node in enumerate(self.main_nodes):
            self.I[0, idx] = self.initial_inv.get(node, 0.0)

        # Build dictionary-based state for time t=0
        self.state = {
            "on_hand_inventory": {},
            "pipeline_inventory": {},
            "sales": {},
            "backlog": {},
            "demand_window": {},
            "t": self.t
        }
        for idx, node in enumerate(self.main_nodes):
            self.state["on_hand_inventory"][node] = float(self.I[self.t, idx])
        for rt in self.reordering_routes:
            self.state["pipeline_inventory"][rt] = [0.0]*self.lead_times[rt]
        for rt in self.retailer_routes:
            self.state["sales"][rt] = 0.0
            self.state["backlog"][rt] = 0.0

        # Define obs and action spaces after we see the final shape
        obs_size = (
            len(self.main_nodes)
            + sum(self.lead_times[rt] for rt in self.reordering_routes)
            + len(self.retailer_routes)  # sales
            + len(self.retailer_routes)  # backlog
            + len(self.retailer_routes)  # demand
            + 1 # For time period
        )
        obs_low = np.zeros(obs_size, dtype=np.float32)
        obs_high = np.full(obs_size, np.inf, dtype=np.float32)
        # limit on-hand by inv_capacity
        for idx, node in enumerate(self.main_nodes):
            obs_high[idx] = self.inv_capacity.get(node, np.inf)

        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)

        act_low = []
        act_high = []
        for rt in self.reordering_routes:
            act_low.append(0.0)
            cap = self.reordering_route_capacity[rt]
            act_high.append(cap if cap is not None else np.inf)
        self.action_space = gym.spaces.Box(
            low=np.array(act_low, dtype=np.float32),
            high=np.array(act_high, dtype=np.float32),
            dtype=np.float32
        )

        print("act_low", act_low)
        print("act_high", act_high)

    def reset(self, seed=None, options=None):
        """
        Matches Cap_exp_env.reset:
         - set t=0, cost=0, reward=0, terminated=False
         - call get_new_start_state
         - flatten
         - return flat obs, info
        """

         # Let Gym handle seeding => sets self.np_random
        super().reset(seed=seed)
        
        self.t = 0
        self.reward = 0.0
        self.cost = 0.0
        self.terminated = False

        self.get_new_start_state()

        flat_obs, self.mapping_obs = flatten_and_track_mappings(self.state)
        self.flatt_state = flat_obs

        return self.flatt_state, {"dict_state": self.state, "terminated": self.terminated}

    def sanitize_action(self, action_dict):
        """
        - set small values to zero
        - can round if desired
        """

        for rt in action_dict:
            if abs(action_dict[rt]) < self.eps:
                action_dict[rt] = 0.0
        return action_dict

    def update_dem(self):
        """
        - Realize new demand every step_size
        """
        # Ensure we are within horizon
        if 1 <= self.t < self.T:
            for rt in self.retailer_routes:
                # Simply copy the single demand value at time t
                self.Dd[rt][self.t] = self.demand[rt][self.t]


    def check_bounds_cost(self, action_dict, next_obs=None):
        """
        1) Clip actions if out of [0, route_capacity], apply penalty.
        2) If next_obs is provided, clip out-of-bounds (below 0 or above obs_high)
           and apply penalty.
        """
        # Action checks
        for rt in self.reordering_routes:
            val = action_dict[rt]
            if val < 0.0:
                diff = abs(val)
                self.cost += diff**2 + self.P
                action_dict[rt] = 0.0
            max_val = self.reordering_route_capacity[rt]
            if val > max_val:
                diff = val - max_val
                self.cost += diff**2 + self.P
                action_dict[rt] = max_val

        # Observation checks
        if next_obs is not None:
            low_b = self.observation_space.low
            high_b = self.observation_space.high
            for i in range(len(next_obs)):
                if next_obs[i] < low_b[i]:
                    diff = abs(next_obs[i] - low_b[i])
                    self.cost += diff**2 + self.P
                    next_obs[i] = low_b[i]
                elif next_obs[i] > high_b[i]:
                    diff = next_obs[i] - high_b[i]
                    self.cost += diff**2 + self.P
                    next_obs[i] = high_b[i]

        return action_dict, next_obs

    def calculate_reward(self):
        """
        The 'original' inventory environment reward function, separate from step.
        Summarizes the cost components, sets self.cost, then does reward = -cost.
        Returns final reward.
        """
        t = self.t
        # Summation of all cost/profit terms
        inv_cost = 0.0
        op_cost = 0.0
        pipeline_cost = 0.0
        backlog_penalty = 0.0
        total_sales = 0.0

        # 1) Inventory + operating
        for idx, node in enumerate(self.main_nodes):
            inv_level = self.I[t, idx]
            inv_cost += inv_level * self.inventory_holding_cost.get(node, 0.0)
            # Operating cost (production nodes)
            if node in self.operating_cost:
                outflow = sum(
                    self.R[(node,k)][t] 
                    for k in self.j_out.get(node, []) if (node,k) in self.R
                )
                # yield factor
                op_cost += (outflow / self.production_yield[node]) * self.operating_cost[node]

        # 2) Pipeline + reorder-based revenue
        for rt in self.reordering_routes:
            pipeline_val = self.Tt[rt][t]
            pipeline_cost += pipeline_val * self.material_holding_cost[rt]
            # partial "wholesale" sale:
            reorder_amount = self.R[rt][t]
            wholesale_price = self.unit_price.get(rt, 0.0)
            total_sales += reorder_amount * wholesale_price

        # 3) Retail sales + backlog penalty
        for rt in self.retailer_routes:
            b_now = self.Bb[rt][t]
            backlog_penalty += b_now * self.unfulfilled_utility_penalty.get(rt, 0.0)
            # Also final sale
            s_now = self.Ss[rt][t]
            retail_price = self.unit_price.get(rt, 0.0)
            total_sales += s_now * retail_price

        # 4) net cost => cost = everything but sales is positive
        self.total_cost = (inv_cost + op_cost + pipeline_cost + backlog_penalty) - total_sales

    def step(self, raw_action):
        """
        - Convert action
        - sanitize_action
        - check_bounds_cost (actions)
        - perform environment updates
        - update demand
        - compute reward via self.calculate_reward
        - flatten next state
        - check_bounds_cost (observations)
        - check if done
        - return
        """

        truncated = False
        self.reward = 0.0
        self.cost = 0.0
        self.total_cost = 0.0

        print("raw_action",raw_action)
        # 1) Convert action
        if isinstance(raw_action, dict):
            action_dict = raw_action
        elif isinstance(raw_action, np.ndarray):
            action_dict = {}
            for i, rt in enumerate(self.reordering_routes):
                action_dict[rt] = raw_action[i]
        else:
            raise ValueError("Action must be dict or np.ndarray")

        # 2) sanitize_action
        action_dict = self.sanitize_action(action_dict)

        # 3) check_bounds_cost for actions
        action_dict, _ = self.check_bounds_cost(action_dict)

        print("action_dict",action_dict)
        
        # 4) environment dynamics
        self.t += 1
        t = self.t

        # Post reorder
        for rt in self.reordering_routes:
            self.R[rt][t] = action_dict[rt]
            if t - self.lead_times[rt] >= 1:
                arrive_t = t - self.lead_times[rt]
                self.Rp[rt][t] = self.R[rt][arrive_t]

        # Production/distribution nodes
        for node in range(
            self.num_markets + self.num_retailers,
            self.num_total_nodes - self.num_raw_distributors
        ):
            inflow = sum(
                self.Rp.get((k,node), np.zeros(self.T+1))[t] 
                for k in self.j_in.get(node, [])
            )
            outflow = sum(
                self.R.get((node,k), np.zeros(self.T+1))[t]
                for k in self.j_out.get(node, [])
            )
            idx = node - self.num_markets
            self.I[t, idx] = self.I[t-1, idx] + inflow - outflow

        # Retailer nodes
        for node in range(self.num_markets, self.num_markets + self.num_retailers):
            inflow = sum(
                self.Rp.get((k,node), np.zeros(self.T+1))[t]
                for k in self.j_in.get(node, [])
            )
            sold = sum(
                self.Ss.get((node,k), np.zeros(self.T+2))[t]
                for k in self.j_out.get(node, [])
            )
            idx = node - self.num_markets
            self.I[t, idx] = self.I[t-1, idx] + inflow - sold

        # Pipeline
        for rt in self.reordering_routes:
            self.Tt[rt][t] = self.Tt[rt][t-1] - self.Rp[rt][t] + self.R[rt][t]

        # Demand update
        self.update_dem()

        # Retailer sales
        for node in range(self.num_markets, self.num_markets + self.num_retailers):
            avail = self.I[t, node - self.num_markets]
            for succ in self.j_out.get(node, []):
                needed = self.Dd[(node,succ)][t] + self.Bb[(node,succ)][t]
                made_sale = min(needed, avail)
                self.Ss[(node,succ)][t+1] = made_sale
                avail -= made_sale

        # Backlog
        for rt in self.retailer_routes:
            self.Bb[rt][t+1] = self.Bb[rt][t] + self.Dd[rt][t] - self.Ss[rt][t+1]

        # 5) calculate_reward in a separate function
        self.calculate_reward()  # sets self.cost & self.reward

        # 6) Build next state dictionary
        self.state["t"] = t

        # (A) On-hand inventory
        for idx, node in enumerate(self.main_nodes):
            self.state["on_hand_inventory"][node] = float(self.I[t, idx])

        # (B) Pipeline inventory for each route, always exactly lead_times[rt] items
        for rt in self.reordering_routes:
            lt = self.lead_times[rt]
            pipeline_vals = []
            for i in range(lt):
                idx_t = (t - lt) + i
                if idx_t >= 0:
                    pipeline_vals.append(self.R[rt][idx_t])
                else:
                    pipeline_vals.append(0.0)
            # Make sure this line is INSIDE the for-loop
            self.state["pipeline_inventory"][rt] = pipeline_vals

        # (C) Sales and backlog
        for rt in self.retailer_routes:
            self.state["sales"][rt] = float(self.Ss[rt][t])
            self.state["backlog"][rt] = float(self.Bb[rt][t])

        # (D) Demand: only the current time step (no rolling window)
        for rt in self.retailer_routes:
            if 1 <= t < self.T:
                # Wrap the single demand in a list if we want to keep the same structure
                self.state["demand_window"][rt] = [self.Dd[rt][t]]
            else:
                # If t is out of range, store 0.0
                self.state["demand_window"][rt] = [0.0]


        # print("state",self.state, t)
        # Flatten
        flat_obs, _ = flatten_and_track_mappings(self.state)

        # print("flat_obs", flat_obs)
        self.flatt_state = flat_obs

        # print("flat_obs", flat_obs)
        # Debug: Print the shape and compare it to observation_space.shape

        # 7) check_bounds_cost for observations
        _, clipped_obs = self.check_bounds_cost(action_dict, next_obs=self.flatt_state)
        self.flatt_state = clipped_obs

        # 8) check if done
        if self.t >= self.T:
            self.terminated = True
        
        self.reward =- (self.total_cost + self.cost)

        info = {"dict_state": self.state, "terminated": self.terminated}
        return self.flatt_state, self.reward, self.terminated, truncated, info

    def render(self, mode='human'):
        """
        Printing out Time step, Reward and costs
        """
        print(f"Time step: {self.t}, Step Reward: {self.reward}, Step Cost: {self.cost}")

    def close(self):
        pass
