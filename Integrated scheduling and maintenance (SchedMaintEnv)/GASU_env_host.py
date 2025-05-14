'''
ASU Joint Production and Maintenance Environment: Gasesous Demand (GAN and GOX): NO INVENTORY 
Akshdeep Singh Ahluwalia 
'''

import random
import json
from typing import Any, ClassVar, List, Tuple, Optional, Dict
import torch
import numpy as np
import scipy.stats as stats
# from or_gym.utils import assign_env_config
import gymnasium as gym
from gymnasium.spaces.utils import flatten_space
from gymnasium import spaces    
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, MultiBinary
from gymnasium.utils import seeding
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os, sys
from PIL import Image, ImageDraw, ImageFont
import math
from pathlib import Path
import matplotlib.pyplot as plt
             
# --- Plant Class ---
class GASU(gym.Env):

    """
        GASUEnv simulates the operation and maintenance of compressors in an air
        separation unit (ASU) focused on meeting gaseous product demand. Since in-
        ventorying gaseous products is impractical, the system allows for external prod-
        uct purchases when demand exceeds production capacity. The ASU consists of
        a set C of n compressors (n = 3), where each compressor c ∈ C has a maximum
        capacity denoted by Capc. The agent must decide on a daily basis whether
        each compressor should operate at a production level rc ∈ [0, 1] of its maximum
        capacity or undergo maintenance, based on its condition. Over an episode of
        length T , the agent also determines daily external purchase quantities, aiming
        to minimize the total operational cost, which includes both production and pur-
        chase costs.

        Industrial Scheduling and Maintenance Environment: Gasesous Demand (GAN and GOX): NO INVENTORY
        
        Observation:
            Type: Dict(5)
            Num     Observation
            0       Demand
            1       Electricity price 
            2       Time Left to Complete Maintenance (of each compressor) 
            3       Time Since Last Maintenance (of each compressor) 
            4       Can Do Maintenance (of each compressor) 

        Action:
            Type: Box(3 * n =(compressors))
            Num     Action
            0       Maintenance action (0 or 1)    : for each compressor
            1       Production rate (0 to 1)       : for each compressor
            2       External purchase (0 to 1)     : of some maximum capacity

        Reward:
            Type: float
            Cost = Production cost + External purchase cost
            Production cost = Production rate * Capacity * Specific energy * Electricity price
            External purchase cost = External purchase Quantity * External purchase price * External purchase capacity

        Cost:
            Type: float
            - Cost for maintenance duration: For agent to learn the maintenance duration
            - Cost for maintenance failure time: For agent to learn the maintenance MTTF
            - Cost for early maintenance: For agent to learn that maintenance is only possible to do after TSLM > mntr
            - Cost for ramping during maintenance: For agent to learn to stop ramping during maintenance
            - Cost for demand satisfaction: For agent to learn the demand satisfaction
        
        Termination:
            The episode ends after 31 days (1-31) of simulation.
        
        Compressor metadata:
            {
                "comp_id": "C1",          -> Unique identifier for the compressor
                "capacity": 500,          -> Capacity of the compressor (ton/day)
                "specific_energy": 450,   -> Specific energy consumption of the compressor (KWh/t)
                "mttf": 18,               -> mean time to failure (days)
                "mttr": 2                 -> mean time to repair (days)
                "mntr": 5                 -> minimum no repair time (days)
            }
            Note: mttf, mttr, mntr \in Z  (set of integers)
    """
    _CONFIG_SCHEMA = {
        "demand": list,
        "electricity_prices": list,
        "compressors": list, 
    }

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        
        state_horizon = 30
        action_horizon = 1
        T = 31
        """
            At the start of each day (1-31): 
                - 30 days of demand data
                - 30 days of price data

            at the start of 31st day: we have data of 31st day itself and 29 coming days (total 30 days, making simulation days = 60)
            i.e., episode length (T) = 31 days (1-31)
        """
        # Processing the input arguments
        self.env_id = env_id
        self._device = kwargs.get('device', 'cuda' if th.cuda.is_available() else 'cpu')
        
        self.config_path = kwargs.get("config_path", None)
        if self.config_path is None:
            raise ValueError("config_path must be provided for MyEnv.")
        self.config_data = self._load_config() 

        # Assign attributes like demand, electricity_prices
        self.assign_env_config(self.config_data)

        self.demand_array = np.array(self.config_data["demand"])
        self.price_array = np.array(self.config_data["electricity_prices"])
        self.compressors = self.config_data["compressors"]

        
        # First initialize the compressors
        self.compressors = {}
        self.initialize_compressors()
        
        # Initialize the environment
        self.action_horizon = action_horizon
        self.T = T    # episode length (days)
        self._max_episode_steps = self.T
        self.simulation_days = self.T + state_horizon - action_horizon
        self.state_horizon = state_horizon
        self.current_day = 0    # Start of Day: "current_day + 1"
        
        # self._initialize_simulation_data()
        self._initialize_observation_space()
        self._initialize_action_space()
        self._initialize_state()
        self.get_external_purchase_price()

        self.Penalty_maint_duration = 50
        self.Penalty_maint_failure_time = 100
        self.Penalty_early_maint = 75
        self.Penalty_ramp = 1
        self.Penalty_demand = 0.5
        
        self.terminated = False
        self.truncated = False
        self.reward_ep = 0
        self.cost_ep = 0

        self.env_spec_log = {'Number of Maintenance-Duration Violation': 0,
                             'Penalty of Maintenance-Duration Violation': 0,
                             'Number of Maintenance-Failure Violation': 0,
                             'Penalty of Maintenance-Failure Violation': 0,
                             'Number of Early-Maintenance Violation': 0,
                             'Penalty of Early-Maintenance Violation': 0,
                             'Number of Ramping-in-Maintenance Violation': 0,
                             'Penalty of Ramping-in-Maintenance Violation': 0,
                             'Number of Demand-Unsatisfaction Violation': 0,
                             'Penalty of Demand-Unsatisfaction Violation': 0,
                            }
        # self._device = kwargs.get('device', 'cuda' if th.cuda.is_available() else 'cpu')
    

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def assign_env_config(self, kwargs):
        # print("Assigning configuration...")
        # print(len(kwargs), "kwargs")
        for key, value in kwargs.items():
            # print(f"Trying to set {key} to {value!r}")
            # 1) ensure it's in the schema
            if key not in self._CONFIG_SCHEMA:
                raise AttributeError(f"{self!r} has no config attribute '{key}'")
            # 2) type‐check
            expected_type = self._CONFIG_SCHEMA[key]
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Config '{key}' expects type {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
            # 3) finally setattr
            # print(f"Setting {key} to {value!r}")
            setattr(self, key, value)
        
    def _initialize_simulation_data(self):
        demand = self.config_data["demand"]
        electricity_prices = self.config_data["electricity_prices"]

        self.demand_array = np.array(demand)
        self.price_array = np.array(electricity_prices)

        # return self.demand_array, self.price_array
             
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.current_day = 0
        
        self.reward_ep = 0
        self.cost_ep = 0

        self.cost = 0
        self.reward = 0

        self.terminated = False 
        self.truncated = False
        self.info = {}

        self._initialize_state()
        self.flatt_state = self.encode_observation(self.state)
        
        flatt_state_tensor = th.tensor(self.flatt_state, dtype=th.float32, device=self._device)
        # print("self.flatt_state_tensor from reset", flatt_state_tensor.shape)

        return flatt_state_tensor, {"dict_state": self.state, "terminated": self.terminated, "truncated": self.truncated}

    # def _initialize_observation_space(self):
        # self.max_mttr = max(comp.mttr for comp in self.compressors.values())
        # self.total_capacity = sum(comp.capacity for comp in self.compressors.values())
        # n = len(self.compressors)
        # S = self.state_horizon

        # self.observation_space = Dict({
        #     "demand": Box(
        #         low=0.0,
        #         high=self.total_capacity * 10,
        #         shape=(S,),
        #         dtype=np.float32
        #     ),
        #     "electricity_price": Box(
        #         low=0.0,
        #         high=10.0,
        #         shape=(S,),
        #         dtype=np.float32
        #     ),
        #     "TLCM": Box(  # Time Left to Complete Maintenance (days)
        #         low=0.0,
        #         high=self.max_mttr,
        #         shape=(n,),
        #         dtype=np.float32
        #     ),
        #     "TSLM": Box(  # Time Since Last Maintenance (days)
        #         low=0.0,
        #         high=100.0,
        #         shape=(n,),
        #         dtype=np.float32
        #     ),
        #     "CDM": MultiBinary(n)  # Can Do Maintenance: binary mask
        # })

    def _initialize_observation_space(self):
        self.max_mttr = max(comp.mttr for comp in self.compressors.values())
        self.total_capacity = sum(comp.capacity for comp in self.compressors.values())
        n = len(self.compressors)
        S = self.state_horizon

        # Compute the total observation dimension
        # obs_dim = S * 2 + n * 3  # demand, electricity_price, TLCM, TSLM, CDM
        self.reset()
        # obs_dim = self.flatt_state.shape[0]
        obs_dim = self.flatt_state.shape

        # Define lower and upper bounds for each component
        low = np.array(
            [0.0] * S +                        # demand
            [0.0] * S +                        # electricity_price
            [0.0] * n +                        # TLCM
            [0.0] * n +                        # TSLM
            [0.0] * n                          # CDM (binary)
        )
        high = np.array(
            [self.total_capacity * 10] * S +   # demand
            [10.0] * S +                       # electricity_price
            [self.max_mttr] * n +              # TLCM
            [100.0] * n +                      # TSLM
            [1.0] * n                          # CDM
        )

        # high_dim = high.shape[0]
        # low_dim = low.shape[0]

        # Set observation space as a flat Box
        # self.observation_space = Box(
        #     low=low,
        #     high=high, 
        #     shape=(obs_dim,),
        #     dtype=np.float32
        # )
        self.observation_space = Box(
            low=0,
            high=1000, 
            # shape=(obs_dim,),
            shape=(69,),
            dtype=np.float32
        )



    def _initialize_action_space(self):
        n = len(self.compressors)
        
        self.max_capacity = max(comp.capacity for comp in self.compressors.values())
        self.max_purchase_quantity = self.max_capacity 
        # Create lower and upper bounds
        low = np.array([0] * n + [0.0] * n + [0.0], dtype=np.float32)          # maintenance (0), production (0), purchase (0)
        # high = np.array([1] * n + [1.0] * n + [10000.0], dtype=np.float32)     # maintenance (1), production (1), purchase (10000)
        high = np.array([1] * n + [1.0] * n + [1.0], dtype=np.float32)            # maintenance (1), production (1), purchase (1)

        high_dim = high.shape[0]

        self.action_space = Box(low=low,
        high=high, 
        shape=(high_dim,),
        dtype=np.float32)

        '''
        REMARKS:
        1. Maintenance and Ramp cannot be coupled, since ramp can be zero if demand is zero, however, that doesn't suggest that the compressor is under maintenance.  
        2. Maybe also use maximum external purchase capacity, and use a coefficent in (0-1) in the action space.
        '''
   
    def update_information_state(self):

        start_idx = self.current_day     # Start of Day: "current_day + 1"
        end_idx = self.current_day + self.state_horizon
        self.state_demand = self.demand_array[start_idx:end_idx]   # Start from (idx: current_day) till (idx: self.current_day + self.state_horizon - 1)
        self.state_price = self.price_array[start_idx:end_idx]

        # Finally update the state of the compressor in self.state
        self.state["demand"] = self.state_demand
        self.state["electricity_price"] = self.state_price

    def update_compressor_physical_condition_state(self, action=None):
            
        """
        At current_day == 0, set physical condition state values directly
        from the compressor metadata dict (self.compressors).
        """
        comp_ids = list(self.compressors.keys())
        n_comp = len(comp_ids)
        if self.current_day == 0:
            tlcm = np.array([self.compressors[cid].TLCM for cid in comp_ids], dtype=np.int32)
            tslm = np.array([self.compressors[cid].TSLM for cid in comp_ids], dtype=np.float32)
            cdm  = np.array([self.compressors[cid].CDM  for cid in comp_ids], dtype=np.int32)

            # Assign to state
            self.state["TLCM"] = tlcm         # Time Left to Complete Maintenance, if started (days)
            self.state["TSLM"] = tslm         # Time Since Last Maintenance (days)
            self.state["CDM"] = cdm           # Can Do Maintenance (derived from 'TSLM and mntr')
        
        else:
            maint_action = action["maintenance_action"]
            tlcm = self.state["TLCM"]
            tslm = self.state["TSLM"]
            cdm = self.state["CDM"]

            for i, cid in enumerate(comp_ids):
                comp = self.compressors[cid]
                mttr = comp.mttr
                mntr = comp.mntr

                # --- TLCM update: if under maintenance, set countdown
                if maint_action[i] == 1 and cdm[i] > 0:
                    tlcm[i] = mttr - 1
                elif maint_action[i] == 1 and cdm[i] == 0:
                    tlcm[i] -= 1
                # else:
                #     tlcm[i] = 0  # tlcm becomes 0 only when maintenance is completed

                # --- TSLM update
                tslm[i] = 0 if maint_action[i] == 1 else tslm[i] + 1

                # --- CDM update
                cdm[i] = 1 if tslm[i] >= mntr else 0

            # Write back updated values to state
            self.state["TLCM"] = tlcm
            self.state["TSLM"] = tslm
            self.state["CDM"] = cdm

    def _initialize_state(self):

        n_comp = len(self.compressors)
        horizon = self.state_horizon

        self.state = {
            "demand": np.zeros(horizon, dtype=np.float32),
            "electricity_price": np.zeros(horizon, dtype=np.float32),
            "TLCM": np.zeros(n_comp, dtype=np.float32),
            "TSLM": np.zeros(n_comp, dtype=np.float32),
            "CDM": np.zeros(n_comp, dtype=np.int32),
        }

        self.update_information_state()
        self.update_compressor_physical_condition_state()

        # UPDATE STATE TO MATCH: Gym-compliant obs tensor for learning !!
        # return self.state

    def production_and_external_purchase_cost(self, action):
        cost = 0
        price_today = self.price_array[self.current_day]
        comp_ids = list(self.compressors.keys())

        for i, cid in enumerate(comp_ids):
            comp = self.compressors[cid]
            ramp_cost = (
                action["production_rate"][i]
                * comp.capacity
                * comp.specific_energy
                * price_today
            )
            cost += ramp_cost

        # Add external purchase cost
        cost += action["external_purchase"][0] * self.external_purchase_price * self.max_purchase_quantity
        return cost
    
    def maintenance_duration_penalty(self, action):
        penalty = 0
        comp_ids = list(self.compressors.keys())
        tlcm = self.state["TLCM"]
        tslm = self.state["TSLM"]
        maint_action = action["maintenance_action"]

        for i, cid in enumerate(comp_ids):
            # Case 1: Interrupting maintenance before it's done
            if maint_action[i] != 1 and tlcm[i] > 0:
                penalty += self.Penalty_maint_duration * math.exp(tlcm[i])

            # Case 2: Negative TLCM (overrun) — theoretically shouldn't happen if logic is right, but safe to guard.
            elif maint_action[i] == 1 and tlcm[i] == 0 and (tslm[i] == 0):
                penalty += self.Penalty_maint_duration * math.exp(tlcm[i])            
            elif tlcm[i] < 0 and maint_action[i] == 1:
                penalty += self.Penalty_maint_duration * math.exp(-tlcm[i])

        if penalty > 0:
            self.env_spec_log['Number of Maintenance-Duration Violation'] += 1
            self.env_spec_log['Penalty of Maintenance-Duration Violation'] += penalty

        return penalty

    def maintenance_failure_time_penalty(self, action):
        penalty = 0
        comp_ids = list(self.compressors.keys())
        tslm = self.state["TSLM"]

        for i, cid in enumerate(comp_ids):
            comp = self.compressors[cid]
            if tslm[i] == comp.mttf and action["maintenance_action"][i] == 0:
                penalty += self.Penalty_maint_failure_time 
            elif tslm[i] > comp.mttf and action["maintenance_action"][i] == 0:
                penalty += self.Penalty_maint_failure_time * (tslm[i] - comp.mttf)
        
        if penalty > 0:
            self.env_spec_log['Number of Maintenance-Failure Violation'] += 1
            self.env_spec_log['Penalty of Maintenance-Failure Violation'] += penalty

        return penalty

    def early_maintenance_penalty(self, action):
        penalty = 0
        comp_ids = list(self.compressors.keys())
        tslm = self.state["TSLM"]
        tlcm = self.state["TLCM"]
        cdm = self.state["CDM"]
        maintenance_actions = action["maintenance_action"]

        for i, cid in enumerate(comp_ids):
            comp = self.compressors[cid]
            if maintenance_actions[i] == 1 and cdm[i] == 0 and tlcm[i] == 0:
                penalty += self.Penalty_early_maint * tslm[i]

        if penalty > 0:
            self.env_spec_log['Number of Early-Maintenance Violation'] += 1
            self.env_spec_log['Penalty of Early-Maintenance Violation'] += penalty
        return penalty

    def ramp_penalty(self, action):
        penalty = 0
        comp_ids = list(self.compressors.keys())
        production_rates = action["production_rate"]
        maintenance_actions = action["maintenance_action"]

        for i, cid in enumerate(comp_ids):
            comp = self.compressors[cid]
            production_quantity = production_rates[i] * comp.capacity

            # Penalize if compressor is under maintenance AND producing
            if maintenance_actions[i] == 1 and production_quantity > 0:
                penalty += self.Penalty_ramp * production_quantity/10

        if penalty > 0:
            self.env_spec_log['Number of Ramping-in-Maintenance Violation'] += 1
            self.env_spec_log['Penalty of Ramping-in-Maintenance Violation'] += penalty
        return penalty

    def demand_penalty(self, action):
        penalty = 0
        comp_ids = list(self.compressors.keys())
        production = 0

        for i, cid in enumerate(comp_ids):
            comp = self.compressors[cid]
            production += action["production_rate"][i] * comp.capacity

        # External purchase
        ext_purchase = action["external_purchase"][0]* self.max_purchase_quantity

        # Total supply
        total_supplied = production + ext_purchase

        # Demand on current day
        demand_today = self.demand_array[self.current_day]

        # Apply penalty if demand not met exactly
         
        if abs(demand_today - total_supplied) > 10:    # OLD--> Absolute error 
        # if abs(demand_today - total_supplied) > 100:    # OLD--> Absolute error 
        # # if abs(demand_today - total_supplied)/demand_today > 0.1:    # NEW--> Relative error: 10% of demand
        #     penalty = self.Penalty_demand * abs(demand_today - total_supplied)/10
        # elif abs(demand_today - total_supplied) > 10 and abs(demand_today - total_supplied) <= 100:
        #     penalty = self.Penalty_demand * abs(demand_today - total_supplied)
        # elif abs(demand_today - total_supplied) > 1 and abs(demand_today - total_supplied) <= 10:
            penalty = self.Penalty_demand * abs(demand_today - total_supplied)


        if penalty > 0:
            self.env_spec_log['Number of Demand-Unsatisfaction Violation'] += 1
            self.env_spec_log['Penalty of Demand-Unsatisfaction Violation'] += penalty

        return penalty
      
    def sanitize_action(self, action):

        """
        Sanitize the action: To resolve state bound constraints
        """
        comp_ids = list(self.compressors.keys())
        n_comp = len(comp_ids)
        maintenance_action = action["maintenance_action"]
        production_rate = action["production_rate"]
        external_purchase = action["external_purchase"][0] * self.max_purchase_quantity

        for i, cid in enumerate(comp_ids):
            comp = self.compressors[cid]

            # Do maintenance if tslm >= mttf
            if self.state["TSLM"][i] >= comp.mttf and maintenance_action[i] != 1:
                maintenance_action[i] = 1
                production_rate[i] = 0
            
            # Do not ramp if under maintenance
            if maintenance_action[i] == 1 and production_rate[i] > 0:
                production_rate[i] = 0
            
            # Do not maintain if cdm is 0
            if self.state["CDM"][i] == 0 and maintenance_action[i] == 1:
                maintenance_action[i] = 0

            # Keep maintaining if tlcm > 0
            if self.state["TLCM"][i] > 0 and maintenance_action[i] != 1:
                maintenance_action[i] = 1
                production_rate[i] = 0

        # update action dict
        action["maintenance_action"] = maintenance_action
        action["production_rate"] = production_rate

    def decode_action(self, action):
        n = len(self.compressors)
        maintenance_action = np.round(action[:n]).astype(int)
        production_rate = action[n:2*n]
        external_purchase = action[-1:]  # keeps it as shape (1,)

        return {
            "maintenance_action": maintenance_action,
            "production_rate": production_rate,
            "external_purchase": external_purchase
        }
    
    def encode_observation(self, state):
        """
        Converts a structured observation dictionary into a flat NumPy array
        compatible with the Box observation space.
        """
        demand = np.array(state["demand"], dtype=np.float32)                        # shape (S,)
        electricity_price = np.array(state["electricity_price"], dtype=np.float32)  # shape (S,)
        tlcm = np.array(state["TLCM"], dtype=np.float32)                            # shape (n,)
        tslm = np.array(state["TSLM"], dtype=np.float32)                            # shape (n,)
        cdm = np.array(state["CDM"], dtype=np.float32)                              # shape (n,), encoded as float
        flatt_state = np.concatenate([demand, electricity_price, tlcm, tslm, cdm]).astype("float32")

        # print("flatt_state shape ", flatt_state.shape)

        # print("flatt_state shap [0]", flatt_state.shape[0])
        
        return flatt_state

    def step(self, raw_action):
        truncated = False

        if isinstance(raw_action, torch.Tensor):
            raw_action = raw_action.to(self._device)
            action = raw_action.cpu().numpy()
        else:
            action = raw_action

        action = raw_action.numpy() if torch.is_tensor(raw_action) else raw_action
        
        # make sure the action is within the valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action_dict = self.decode_action(action)
        
        ## COST INCURRED AS PER THE PLANT CONFIGURATION and EXTERNAL PURCHASE
        real_cost = self.production_and_external_purchase_cost(action_dict)

        ## PENALTY 1: TO LEARN THE MAINTENANCE DURATION
        penalty_LMD = self.maintenance_duration_penalty(action_dict)

        ## PENALTY 2: TO LEARN THE MAINTENANCE MTTF 
        penalty_MTTF = self.maintenance_failure_time_penalty(action_dict)

        ## PENALTY 3: TO LEARN that MAINTAINENCE is ONLY POSSIBLE TO DO AFTER TSLM > mntr
        penalty_maint = self.early_maintenance_penalty(action_dict)

        ## PENALTY 4: TO LEARN TO STOP RAMPING DURING MAINTENANCE
        penalty_ramp = self.ramp_penalty(action_dict)

        ## PENALTY 5: TO LEARN THE DEMAND SATISFACTION
        penalty_demand = self.demand_penalty(action_dict)

        # Positive cost (penalties) incurred.
        self.cost_ep += penalty_LMD + penalty_MTTF + penalty_maint + penalty_ramp + penalty_demand
        cost = penalty_LMD + penalty_MTTF + penalty_maint + penalty_ramp + penalty_demand
        self.cost = cost
        
        # Reward is negative of operational expense
        self.reward_ep += -real_cost
        reward = - real_cost
        self.reward = reward
        
        self.current_day += self.action_horizon      # 1 day
        self.sanitize_action(action_dict)            # To prevent state bound violations     
        self.update_information_state()
        self.update_compressor_physical_condition_state(action_dict)

        if self.current_day == self.T:       # one month (31 days)
            self.terminated = True                 # end of episode


        self.flatt_state = self.encode_observation(self.state)
        flatt_state_tensor = th.tensor(self.flatt_state, dtype = th.float32, device=self._device)
        return flatt_state_tensor, th.tensor(reward-cost, dtype = th.float32, device=self._device), th.tensor(self.terminated, dtype = th.bool, device=self._device), th.tensor(self.truncated, dtype = th.bool, device=self._device), {}

    def _get_state(self, mode):
        if mode == 'dict':
            return self.state
        elif mode == 'flatt':
            return self.flatt_state
        elif mode == 'tensor':
            return th.tensor(self.flatt_state, device=self._device)
        else:
            raise ValueError("Invalid mode. Choose from 'dict', 'flatt', or 'tensor'.")
    
    @property
    def max_episode_steps(self) -> int:
        return self.T

    def render(self, mode='human'):
        print("state:", f"{self._get_state(mode='dict')}")
        print("reward:", f"{self.reward_ep}")
        print("cost", f"{self.cost_ep}")
        print("specification:", f"{self.env_spec_log}")

    def close(self) -> None:
        return None

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def render_information_state(self):
        return self.state_demand, self.state_price
    
    def render_initial_compressor_state(self):
        return self.compressors
    
    def sample_action(self):
        return self.action_space.sample()

    def sample_state(self):
        return self.observation_space.sample()

    def get_external_purchase_price(self):
        self.average_price = np.mean(self.price_array)
        average_compressor_price = sum(
            comp.specific_energy for comp in self.compressors.values()
        ) / len(self.compressors) * self.average_price             # (KWh/t) * ($/kWh) = $/t
        alpha = 2                                                  # multiplier for external purchase price
        external_purchase_price = alpha*average_compressor_price
        self.external_purchase_price = external_purchase_price     # $/ton
        return self.external_purchase_price
    
    def get_external_purchase_capacity(self):
        self.max_capacity = max(comp.capacity for comp in self.compressors.values())
        return self.max_capacity
    
    def initialize_compressors(self):
        try:
            compressor_configs = self.config_data.get("compressors", [])
            if not compressor_configs:
                raise ValueError("No compressors found in config data.")
            
            self.compressors = {
                cfg["comp_id"]: Compressor(**cfg) for cfg in compressor_configs
            }
        except Exception as e:
            print(f"Error initializing compressors: {e}")

    def compressor_info(self):
        comp_info_dict = {
            comp_id: comp.get_dict()
            for comp_id, comp in self.compressors.items()
        }
        return comp_info_dict
    
    def plot_complete_simulation_data(self):
        # Simulate and plot both in a single figure
        # import matplotlib.pyplot as plt
        # Create a figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot the simulated electricity price
        axes[0].plot(self.price_array, label="Simulated Electricity Price ($/kWh)")
        self.average_price = np.mean(self.price_array)
        axes[0].axhline(y=self.average_price, color='r', linestyle='--', label='Average Price')
        axes[0].set_title("Simulated Industrial Electricity Price Over Time")
        axes[0].set_xlabel("Day")
        axes[0].set_ylabel("Price ($/kWh)")
        axes[0].legend()
        axes[0].grid(True)

        # Plot the simulated demand
        axes[1].plot(self.demand_array, label="Simulated Demand")
        average_demand = np.mean(self.demand_array)
        axes[1].axhline(y=average_demand, color='r', linestyle='--', label='Average Demand')
        axes[1].axhline(y=self.total_capacity, color='g', linestyle='--', label='Maximum Capacity of the Plant')
        axes[1].set_title("Simulated Demand with Seasonal, Step, and Noise Effects")
        axes[1].set_xlabel("Day")
        axes[1].set_ylabel("Demand (ton/day)")
        axes[1].legend()
        axes[1].grid(True)
        # Adjust layout and show the combined plot
        plt.tight_layout()
        plt.show()

# --- Compressor Class ---
class Compressor:
    def __init__(self, comp_id, capacity, specific_energy, mttf, mttr, mntr, TLCM, TSLM, CDM):
        self.scale = 750
        self.comp_id = comp_id
        self.capacity = capacity
        self.specific_energy = specific_energy/self.scale
        self.mttf = mttf
        self.mttr = mttr
        self.mntr = mntr
        self.TLCM = TLCM
        self.TSLM = TSLM
        self.CDM = CDM

    def get_dict(self):
        comp_dict = {
            "comp_id": self.comp_id,
            "capacity": self.capacity,
            "specific_energy": self.specific_energy,
            "mttf": self.mttf,
            "mttr": self.mttr,
            "mntr": self.mntr,
            "TLCM": self.TLCM,
            "TSLM": self.TSLM,
            "CDM": self.CDM,
            
        }
        return comp_dict

    def info(self):
        return (
            f"Compressor {self.comp_id} → "
            f"Capacity: {self.capacity} ton/day, "
            f"Specific energy: {self.specific_energy} KWh/t, "
            f"MTTF: {self.mttf} days, "
            f"MTTR: {self.mttr} days, "
            f"MNTR: {self.mntr} days "
            f"TLCM: {self.TLCM} "
            f"TSLM: {self.TSLM} "
            f"CDM: {self.CDM} "
        )

def main():
    base_dir   = Path(__file__).resolve().parent
    config_fp  = base_dir / "gasu_config.json"
    if not config_fp.is_file():
        raise FileNotFoundError(f"Couldn’t find config.json at {config_fp}")
    
    env = GASU(env_id='GASU-v0', config_path= config_fp)
    # Reset returns (obs_tensor, info_dict)
    obs, info = env.reset()
    print("Manual rollout start...")
    i = 1
    done = False
    while not done:
        # Sample action (still a NumPy array)
        action = env.action_space.sample()
        # Step returns
        #    (state_tensor, reward_tensor, done_tensor, truncated_tensor, info_dict)
        obs, reward, done, truncated, info = env.step(action)

        # You can still do:
        print(f"Step {i}, obs shape={obs.shape}, reward={reward}, done={done}")

        # Checking `done' on a scalar BoolTensor works:
        if done:
            obs, info = env.reset()
            print("Episode reset")
        i += 1
if __name__ == "__main__":
    main()