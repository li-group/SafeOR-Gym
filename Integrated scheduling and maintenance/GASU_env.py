'''
ASU Joint Prodcution and Maintenance Environment: Gasesous Demand (GAN and GOX): NO INVENTORY 
Akshdeep Singh Ahluwalia 

SAFE RL: TO ENSURE Gaseous Demand Satisfaction and SAFE Maintainence of the compressors, 
Salient features:
• To ensure gaseous demand satisfaction (either through production or external purchase (expensive))
• Maintainence decision to happen only after certain days of operation 
    (to make it available for production to happen, i.e., failure happens right the next day after the mean time to failure.)
• To start maintaince on or before the MTTR (to stop production if mean time to failuer has passed and then start maintainence.)
• To keep duration of maintenence (or the compressor not be used) for at least for MTTR days

Production_rate -> between 0 and 1
Maintaining the compressor -> 0 or 1
Production_rate <= Capacity * (1 - maintenance_decision) 

Simulation Horizon:  
31 days (1-31)
'''

import numpy as np
import gymnasium as gym
from gymnasium import spaces    # https://gymnasium.farama.org/

from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, MultiBinary
from gymnasium.utils import seeding
import json
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os, sys
from PIL import Image, ImageDraw, ImageFont
import random
from scipy.spatial import ConvexHull
import json

"""
at the start of day (1-31): 
    - 30 days of demand data
    - 30 days of price data

at the start of 31st day: we have data of 31st day itself and 29 coming days (total 30 days, making simulation days = 60)

i.e., episode length = 31 days (1-31)
"""

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
    
class Demand_simulator():

    def __init__(self, duration, compressors):
        # Placeholder implementation
        self.duration = duration
        self.demand_array = np.zeros(duration)  # Example: Initialize with zeros
        self.total_capacity = sum(comp.capacity for comp in compressors.values())

    def get_demand_array(self):
        # Placeholder implementation
        return self.demand_array 
    
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
        seasonal_effect = 0.6 + 0.4 * np.sin(2 * np.pi * days / 10)

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
        base_price = 0.0832  # average industrial price in the US in $/kWh
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

    def __init__(self, T, state_horizon, action_horizon):
        
        ## ALSO PUT IN ACTION HORIZON, i.e., 1 day (here)
        ## however, make it user defined, how long the action has to be predicted and also to be re-optimized. 
        # here action horizon = 1 day
        # Action_horizon would also be needed to pass to the optimizer to fetch optimal action of appropriate length (action_horizon)
        self.action_horizon = action_horizon

        ## UPDATE STATE IN THE STEP FUNCTION WOULD DIRECTLY DEPEND ON HOW LONG THE ACTION HORIZON IS 
        
        # First initialize the compressors
        self.compressors = {}
        self.initialize_compressors("compressors_config.json")
        """ compressor exemplar data format
        {
            "comp_id": "C1",          -> id
            "capacity": 500,          -> ton/day  (150)
            "specific_energy": 450,   -> KWh/t    (400)
            "mttf": 18,               -> mean time to failure (days)
            "mttr": 2                 -> mean time to repair (days)
            "mntr": 5                 -> minimum no repair time (days)
        }
        Rest we have "C2" and "C3" with different values

        Note: mttf, mttr, mntr \in Z  (set of integers)
        """

        # Initialize the environment
        self.T = T    # episode length (days)
        self.simulation_days = self.T + state_horizon - action_horizon
        self.state_horizon = state_horizon
        self.current_day = 0  # Start of Day: "current_day + 1"
        
        self._initialize_simulation_data()
        self._initialize_observation_space()
        self._initialize_action_space()
        self._initialize_state()

        self.Penalty_maint_duration = 1 * 1e4
        self.Penalty_maint_failure_time = 1 * 1e4
        self.Penalty_early_maint = 1 * 1e5
        self.Penalty_ramp = 1 * 1e5
        self.Penalty_demand = 1 * 1e5
        
        self.done = False
        self.info = {}  

    def _initialize_simulation_data(self):
        demand_simulator = Demand_simulator(self.simulation_days, self.compressors)
        demand_array = demand_simulator.get_demand_array()
        self.demand_array = demand_array

        price_simulator = Electric_price_simulator(self.simulation_days)
        price_array = price_simulator.get_price_array()
        self.price_array = price_array     

    def reset(self):
        self.current_day = 0
        # self._initialize_simulation_data()
        self._initialize_state()
        self.done = False
        self.reward = 0
        self.info = {} 

    def _initialize_observation_space(self):

        self.max_mttr = max(comp.mttr for comp in self.compressors.values())
        self.total_capacity = sum(comp.capacity for comp in self.compressors.values())
        self.observation_space = Dict({
            "demand": Box(
                low=0,
                high=self.total_capacity * 10,
                shape=(self.state_horizon,),
                dtype=np.float32
            ),
            "electricity_price": Box(
                low=0,
                high=10,
                shape=(self.state_horizon,),
                dtype=np.float32
            ),
            "TLCM": Box(         # Time Left to Complete Maintenance, if started (days)
                low=0,
                high= self.max_mttr,
                shape= (len(self.compressors),),
                dtype= np.float32
            ), 
            "TSLM": Box(        # Time Since Last Maintenance (days)
                low=0,
                high=100,
                shape=(len(self.compressors),),
                dtype=np.float32
            ),                                                  
            "CDM": MultiBinary(len(self.compressors)),          # Can Do Maintenance (derived from 'TSLM and mntr')                                             
        })

    def _initialize_action_space(self):
        
        # Define the Action Space
        self.action_space = Dict({
            "maintenance_action": MultiDiscrete([2] * len(self.compressors)),     # 0 or 1 for each compressor: indicating maintenance or not
            "production_rate": Box(low=0, high=1, shape=(len(self.compressors),), dtype=np.float32),
            "external_purchase": Box(low=0, high=10000, shape=(1,), dtype=np.float32)
        })

        '''
        _NOTE_: Maintenance and Ramp cannot be coupled, since ramp can be zero if demand is zero, 
        however, that doesn't suggest that the compressor is under maintenance.  '''
        # should I use (0-1) for external purchase or (0-10000) for external purchase?
        # I think (0-10000) is better, as it is more interpretable
    
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
        maint_action = action["maintenance_action"]

        for i, cid in enumerate(comp_ids):
            # Case 1: Interrupting maintenance before it's done
            if maint_action[i] != 1 and tlcm[i] > 0:
                penalty += self.Penalty_maint_duration * tlcm[i]

            # Case 2: Negative TLCM (overrun) — theoretically shouldn't happen if logic is right, but safe to guard.
            elif tlcm[i] < 0:
                penalty += -self.Penalty_maint_duration * tlcm[i]
        return penalty

    def maintenance_failure_time_penalty(self, action):
        penalty = 0
        comp_ids = list(self.compressors.keys())
        tslm = self.state["TSLM"]

        for i, cid in enumerate(comp_ids):
            comp = self.compressors[cid]
            if tslm[i] > comp.mttf:
                penalty += self.Penalty_maint_failure_time * (tslm[i] - comp.mttf)
        return penalty

    def early_maintenance_penalty(self, action):
        penalty = 0
        comp_ids = list(self.compressors.keys())
        tslm = self.state["TSLM"]
        cdm = self.state["CDM"]
        maintenance_actions = action["maintenance_action"]

        for i, cid in enumerate(comp_ids):
            comp = self.compressors[cid]
            if maintenance_actions[i] == 1 and cdm[i] == 0:
                penalty += self.Penalty_early_maint * tslm[i]
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
        if demand_today != total_supplied:
            penalty = self.Penalty_demand * abs(demand_today - total_supplied)
        return penalty
      
    def step(self, action):
        truncated = False

        ## COST INCURRED AS PER THE PLANT CONFIGURATION and EXTERNAL PURCHASE
        cost = self.production_and_external_purchase_cost(action)

        ## PENALTY 1: TO LEARN THE MAINTENANCE DURATION
        penalty_LMD = self.maintenance_duration_penalty(action)

        ## PENALTY 2: TO LEARN THE MAINTENANCE MTTF 
        penalty_MTTF = self.maintenance_failure_time_penalty(action)

        ## PENALTY 3: TO LEARN that MAINTAINENCE is ONLY POSSIBLE TO DO AFTER TSLM > mntr
        penalty_maint = self.early_maintenance_penalty(action)

        ## PENALTY 4: TO LEARN TO STOP RAMPING DURING MAINTENANCE
        penalty_ramp = self.ramp_penalty(action)

        ## PENALTY 5: TO LEARN THE DEMAND SATISFACTION
        penalty_demand = self.demand_penalty(action)

        self.info[self.current_day+1] = {}
        self.info[self.current_day+1]["cost"] = cost
        # Log penalty presence as 1 (incurred) or 0 (not incurred)
        self.info[self.current_day+1]["penalty_LMD"] = int(penalty_LMD > 0)
        self.info[self.current_day+1]["penalty_MTTF"] = int(penalty_MTTF > 0)
        self.info[self.current_day+1]["penalty_maint"] = int(penalty_maint > 0)
        self.info[self.current_day+1]["penalty_ramp"] = int(penalty_ramp > 0)
        self.info[self.current_day+1]["penalty_demand"] = int(penalty_demand > 0)

        # TOTAL cost = cost + penalties
        total_cost = cost + penalty_LMD + penalty_MTTF + penalty_maint + penalty_ramp + penalty_demand
        # self.reward = -total_cost
        self.reward += -total_cost

        self.current_day += self.action_horizon      # 1 day
        
        self.update_information_state()
        self.update_compressor_physical_condition_state(action)

        if self.current_day == self.T:       # one month (31 days)
            self.done = True                 # end of episode
        return self.state, self.reward, self.done, truncated, self.info

    def render(self):
        return self.state

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

        
    
    

