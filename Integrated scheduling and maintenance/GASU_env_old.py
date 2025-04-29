'''
ASU Joint Production and Maintenance Environment: Gasesous Demand (GAN and GOX): NO INVENTORY 
Akshdeep Singh Ahluwalia 
'''

import random
from typing import Any, ClassVar, List, Tuple, Optional, Dict
import torch
import numpy as np
import scipy.stats as stats
# from or_gym.utils import assign_env_config
import gymnasium as gym
from gymnasium.spaces.utils import flatten_space
from gymnasium import spaces        # https://gymnasium.farama.org/
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, MultiBinary
from gymnasium.utils import seeding
import json
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os, sys
from PIL import Image, ImageDraw, ImageFont
import math

# --- Compressor Class ---
class Compressor:
    def __init__(self, comp_id, capacity, specific_energy, mttf, mttr, mntr, TLCM, TSLM, CDM):
        self.comp_id = comp_id
        self.capacity = capacity
        self.specific_energy = specific_energy
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
    
class Demand_simulator:
    def __init__(self, duration, compressors, gamma=1):
        self.seed = 42
        np.random.seed(self.seed)
        self.duration = duration
        self.compressors = compressors
        self.total_capacity = sum(comp.capacity for comp in compressors.values())
        self.gamma = gamma
        self.max_demand = self.gamma * self.total_capacity

    def get_demand_array(self):
        days = np.arange(self.duration)

        # Seasonal effect: strong sine wave with 1 full cycle per 30 days
        seasonal_effect = 0.6 + 0.4 * np.sin(2 * np.pi * days / 35)

        # Step changes: every 30 days, simulate a change in base demand level
        base_levels = np.random.uniform(0.6, 0.8, size=(self.duration // 30 + 1))
        step_effect = np.repeat(base_levels, 30)[:self.duration]

        # Random noise: small normally distributed fluctuations
        noise = np.random.normal(0, 0.03, self.duration)

        # Combine effects and scale to maximum allowed demand
        demand = 1.2*(seasonal_effect * step_effect + noise) * self.max_demand

        # Clip to [0, max_demand]
        demand = np.clip(demand, 0, self.max_demand)
        demand = demand + 300
        self.demand_array = demand
        return self.demand_array
    
class Electric_price_simulator:
    def __init__(self, duration):
        self.duration = duration
        self.price_array = np.zeros(duration)  # $/kWh
        self.seed = 42
        np.random.seed(self.seed)

    def get_price_array(self):
        base_price = 0.0832           # average industrial price in the US in $/kWh
        days = np.arange(self.duration)

        # Weekly cycle: higher prices Mon-Fri, lower Sat-Sun
        weekly_cycle = 1 + 0.015 * np.sin(2 * np.pi * days / 7)

        # Seasonal trend: gradual increase
        seasonal_trend = 1 + 0.0002 * days  # mild upward drift

        # Random noise: ±0.005 around the trend
        random_fluctuation = np.random.normal(0, 0.003, size=self.duration)

        # Final price calculation
        price = base_price * weekly_cycle * seasonal_trend + random_fluctuation

        # Ensure non-negative prices
        price = np.clip(price, 0, None)

        self.price_array = price
        return self.price_array  # $/kWh
           
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
            2       Time Left to Complete Maintenance 
            3       Time Since Last Maintenance 
            4       Can Do Maintenance 

        Action:
            Type: Box(3)
            Num     Action
            0       Maintenance action (0 or 1)
            1       Production rate (0 to 1)
            2       External purchase (0 to 10000)

        Reward:
            Type: float
            Cost = Production cost + External purchase cost
            Production cost = Production rate * Capacity * Specific energy * Electricity price
            External purchase cost = External purchase Quantity * External purchase price

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
        self.env_id = env_id
        
        # First initialize the compressors
        self.compressors = {}
        self.initialize_compressors("compressors_config.json")

        # Initialize the environment
        self.action_horizon = action_horizon
        self.T = T    # episode length (days)
        self._max_episode_steps = self.T
        self.simulation_days = self.T + state_horizon - action_horizon
        self.state_horizon = state_horizon
        self.current_day = 0    # Start of Day: "current_day + 1"
        
        self._initialize_simulation_data()
        self._initialize_observation_space()
        self._initialize_action_space()
        self._initialize_state()

        self.Penalty_maint_duration = 1 * 1e2
        self.Penalty_maint_failure_time = 1 * 1e5
        self.Penalty_early_maint = 1 * 1e5
        self.Penalty_ramp = 1 * 1e5
        self.Penalty_demand = 1 * 1e5
        
        self.terminated = False
        self.truncated = False
        self.reward = 0
        self.cost = 0

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

    def _initialize_simulation_data(self):
        demand_simulator = Demand_simulator(self.simulation_days, self.compressors)
        demand_array = demand_simulator.get_demand_array()
        self.demand_array = demand_array

        price_simulator = Electric_price_simulator(self.simulation_days)
        price_array = price_simulator.get_price_array()
        self.price_array = price_array
        return self.demand_array, self.price_array
             
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.current_day = 0
        # self._initialize_simulation_data()
        self._initialize_state()
        self.done = False
        self.terminated = False
        self.reward = 0
        self.info = {} 
        self.flatt_state = self.encode_observation(self.state)
        return self.flatt_state, {"dict_state": self.state, "terminated": self.terminated}

    def _initialize_observation_space(self):
        self.max_mttr = max(comp.mttr for comp in self.compressors.values())
        self.total_capacity = sum(comp.capacity for comp in self.compressors.values())
        n = len(self.compressors)
        S = self.state_horizon

        self.observation_space = Dict({
            "demand": Box(
                low=0.0,
                high=self.total_capacity * 10,
                shape=(S,),
                dtype=np.float32
            ),
            "electricity_price": Box(
                low=0.0,
                high=10.0,
                shape=(S,),
                dtype=np.float32
            ),
            "TLCM": Box(  # Time Left to Complete Maintenance (days)
                low=0.0,
                high=self.max_mttr,
                shape=(n,),
                dtype=np.float32
            ),
            "TSLM": Box(  # Time Since Last Maintenance (days)
                low=0.0,
                high=100.0,
                shape=(n,),
                dtype=np.float32
            ),
            "CDM": MultiBinary(n)  # Can Do Maintenance: binary mask
        })

    def _initialize_action_space(self):
        n = len(self.compressors)
        
        # Create lower and upper bounds
        low = np.array([0] * n + [0.0] * n + [0.0], dtype=np.float32)          # maintenance (0), production (0), purchase (0)
        high = np.array([1] * n + [1.0] * n + [10000.0], dtype=np.float32)     # maintenance (1), production (1), purchase (10000)

        self.action_space = Box(low=low, high=high, dtype=np.float32)

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

        return self.state

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
        cost += action["external_purchase"][0] * self.external_purchase_price
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
                penalty += -self.Penalty_maint_duration * math.exp(tlcm[i])

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
                penalty += self.Penalty_ramp * production_quantity

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
        ext_purchase = action["external_purchase"][0]

        # Total supply
        total_supplied = production + ext_purchase

        # Demand on current day
        demand_today = self.demand_array[self.current_day]

        # Apply penalty if demand not met exactly
        if abs(demand_today - total_supplied) > 10:
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
        external_purchase = action["external_purchase"][0]

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
        flatt_state = np.concatenate([demand, electricity_price, tlcm, tslm, cdm])
        
        return flatt_state

    def step(self, action):
        truncated = False

        # make sure the action is within the valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action_dict = self.decode_action(action)
        
        ## COST INCURRED AS PER THE PLANT CONFIGURATION and EXTERNAL PURCHASE
        cost = self.production_and_external_purchase_cost(action_dict)

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

        self.info[self.current_day+1] = {}
        self.info[self.current_day+1]["cost"] = cost
        # Log penalty presence as 1 (incurred) or 0 (not incurred)
        self.info[self.current_day+1]["penalty_LMD"] = int(penalty_LMD > 0)
        self.info[self.current_day+1]["penalty_MTTF"] = int(penalty_MTTF > 0)
        self.info[self.current_day+1]["penalty_maint"] = int(penalty_maint > 0)
        self.info[self.current_day+1]["penalty_ramp"] = int(penalty_ramp > 0)
        self.info[self.current_day+1]["penalty_demand"] = int(penalty_demand > 0)
        # USE self.logger for above.

        # Positive cost (penalties) incurred.
        self.cost += penalty_LMD + penalty_MTTF + penalty_maint + penalty_ramp + penalty_demand
        self.reward += -cost
        
        self.current_day += self.action_horizon      # 1 day
        self.sanitize_action(action_dict)            # To prevent state bound violations     
        self.update_information_state()
        self.update_compressor_physical_condition_state(action_dict)

        if self.current_day == self.T:       # one month (31 days)
            self.done = True                 # end of episode
        
        self.flatt_state = self.encode_observation(self.state)

        return self.flatt_state, self.reward - self.cost, self.done, truncated, self.info
    
    @property
    def max_episode_steps(self) -> int:
        return self.T

    def render(self, mode='human'):
        print("state:", f"{self._get_state(mode='dict')}")
        print("reward:", f"{self.reward}")
        print("cost:", f"{self.cost}")
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

    def initialize_compressors(self, json_path):
        try:
            with open(json_path, "r") as f:
                config_data = json.load(f)
            self.compressors = {
                cfg["comp_id"]: Compressor(**cfg) for cfg in config_data
            }
        except Exception as e:
            print(f"Error loading compressors: {e}")

    def compressor_info(self):
        comp_info_dict = {
            comp_id: comp.get_dict()
            for comp_id, comp in self.compressors.items()
        }
        return comp_info_dict
    
    def plot_complete_simulation_data(self):
        # Simulate and plot both in a single figure
        import matplotlib.pyplot as plt
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
        axes[1].set_ylabel("Demand (Nm³/h)")
        axes[1].legend()
        axes[1].grid(True)
        # Adjust layout and show the combined plot
        plt.tight_layout()
        plt.show()

    # def plot_state_information(self):

        
    
    

