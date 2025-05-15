'''
ASU Production Scheduling: Liquid Products (LIN, LOX, LAR)
Akshdeep Singh Ahluwalia 
'''

import numpy as np
import gymnasium as gym
from gymnasium import spaces      
from typing import Any, ClassVar, List, Tuple, Optional, Dict
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from gymnasium.utils import seeding
import json
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os, sys
from PIL import Image, ImageDraw, ImageFont
import random
from scipy.spatial import ConvexHull
from typing import Any
import torch
import matplotlib.pyplot as plt
# from utils import load_config, assign_env_config,
from utils import decode_action, encode_observation
from utils import load_config, assign_env_config
from pathlib import Path

class ASUEnv(gym.Env):

    """
        ASU Production Scehduling Environment for SafeRL

        This environment simulates the operation of an Air Separation Unit (ASU) for liquid products.

        Attributes:
        Observation Space:
            - electricity_prices: Electricity prices for the next 24 hours.
            - demand: Demand forecast for each product for the next 24 hours.
            - IV: Current inventory levels for each product.
        
        Action Space:
            - lambda: Production quantities for each product, constrained by the convex hull of production data.
        
        Reward:
            - Negative of the total cost, which includes production cost

        Penalties: 
            - Inventory exceeding maximum capacity.
            - Demand shortfall at the end of the day.

    """
    _CONFIG_SCHEMA = {
        "demand": dict,         # Outer: str(int), Inner: dict of products with float values
        "price": dict,          # Outer: str(int), Inner: dict of hours with float values
        "products": list,       # List of product strings: ['LIN', 'LOX', 'LAR']
        "IV_u": dict,           # Dict of product: float (max inventory)
        "fixed_cost": list,     # List with one float (e.g., [36])
        "dynamic_cost": list,   # List with one float (e.g., [0.24192])
        "IV_i": dict,           # Dict of product: float (initial inventory)
        "liq_prod_data": dict   # Dict: {product -> {str(int) -> float}}
    }

    def __init__(self, env_id: str, **kwargs: Any) -> None:

        self.name = env_id
        lookahead = 4
        self.lookahead_days = lookahead
        days_to_simulate = 7
        T = days_to_simulate
        self.T = T*24
        self._max_episode_steps = self.T

        self.env_id = env_id
        self._device = kwargs.get('device', 'cuda' if th.cuda.is_available() else 'cpu')
    
        config_path = kwargs.get('config_path')
        if config_path is None:
            raise ValueError("config_path must be provided for ASUEnv.")
        # Load and assign configuration, Load and stash raw config, then assign each field
        config_data = load_config(config_path)
        self.config_data = config_data              # ← add this line
        assign_env_config(self, config_data, self._CONFIG_SCHEMA)

        # Simulation time counters
        self.current_hour = 0  # hour in current day (0-23)
        self.current_day = 0   # start of the first day (0-indexed) 

        # Initialize parameters, state, observation and action spaces
        self._initialize_params()
        self._initialize_simulation_data()
        self._initialize_observation_space()
        self._initialize_action_space()


        self.terminated = False
        self.truncated = False
        self.reward_ep = 0
        self.cost_ep = 0

        self.Penalty_inventory = 1
        self.Penalty_demand = 20
        self.env_spec_log = {'Number of Inventory Violation': 0,
                        'Cost of Inventory Violation': 0,
                        'Number of Demand Violation': 0,
                        'Cost of Demand Violation': 0,
                        }

    def _initialize_simulation_data(self):
        # Convert string keys to integers
        self.dict_demand = {int(k): v for k, v in self.config_data["demand"].items()}
        self.dict_prices  = {
            int(day): {int(hour): price for hour, price in day_dict.items()}
            for day, day_dict in self.config_data["price"].items()
        }
        # return self.dict_prices, self.dict_demand

    def _initialize_params(self):
        """Load configuration and operational parameters."""
        self.products = self.config_data["products"]        # e.g., ['LIN', 'LOX', 'LAR']
        self.IV_u = self.config_data["IV_u"]                # Maximum inventory capacities
        self.fixed_cost = self.config_data["fixed_cost"][0]
        self.unit_prod_cost = self.config_data["dynamic_cost"][0]
        self.IV_i = self.config_data["IV_i"]                # Initial inventory levels
        # return self.products, self.IV_u, self.fixed_cost, self.unit_prod_cost, self.IV_i

    def _initialize_observation_space(self):
        self.reset()
        """Define the observation space as a single Box."""

        # Dimensions
        num_products = len(self.products)
        price_dim = 24 * (1 + self.lookahead_days)
        demand_dim = num_products * price_dim
        iv_dim = num_products

        # High bounds
        price_high = np.full((price_dim,), np.inf, dtype=np.float32)
        demand_high = np.full((demand_dim,), np.inf, dtype=np.float32)
        iv_high = np.array([self.IV_u[prod] for prod in self.products], dtype=np.float32)
        self.iv_high = np.array([self.IV_u[prod] for prod in self.products], dtype=np.float32)

        # Concatenate all high bounds into a single high vector
        high = np.concatenate([price_high, demand_high, iv_high])
        low = np.zeros_like(high, dtype=np.float32)

        # Define the single Box space
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _initialize_action_space(self):
        """  Define the action space.  """
        max_quantities = np.array([self.IV_u[prod] for prod in self.products], dtype=np.float32)

        ### Data preprocessing to yield Convex Hull ###
        # liq_prod_data = self.loaded_data['liq_prod_data']
        liq_prod_data = self.config_data["liq_prod_data"]
            # Convert liq_prod_data to a list of tuples
        points = [(liq_prod_data['LIN'][i], liq_prod_data['LOX'][i], liq_prod_data['LAR'][i]) for i in liq_prod_data['LIN']]
        # Compute the convex hull
        points_np = np.array(points)                         # Convert list of tuples to a NumPy array
        hull = ConvexHull(points_np)
        self.extreme_points_liqp = points_np[hull.vertices]       # Extract the vertices (extreme points)
        row_liqprod, colliq_prod = self.extreme_points_liqp.shape
        self.row_liqprod = row_liqprod

        self.action_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(self.row_liqprod,),  # Number of convex hull vertices
        dtype=np.float32)

    def _initialize_state(self):
        """Initialize the state with default values."""
        
        self.electricity_prices = np.zeros(24 * (1+self.lookahead_days), dtype=np.float32)
        self.demand = np.zeros((len(self.products), 24 * (1+self.lookahead_days)), dtype=np.float32)
        
        # Set initial inventory levels from IV_i
        self.IV = np.zeros(len(self.products), dtype=np.float32)

        # Combine into the state dictionary
        self.state = {
            'electricity_prices': self.electricity_prices,
            'demand': self.demand,
            'IV': self.IV,
        }

        self.update_demand_and_electricty_state()
        self.update_physical_state()

    def get_demand_and_el_states_for_the_day(self):

        start_day = self.current_day + 1
        selected_days = range(start_day, start_day + self.lookahead_days + 1)
        # print(f"start_day = {start_day}")
        # print(f"Available days = {list(self.dict_demand.keys())}")

        product_demand = {}
        product_demand = {
            product: {day: self.dict_demand[day][product] for day in selected_days}
            for product in self.dict_demand[start_day]
        }
        electricity_prices = {}
        index = 1
        for day in selected_days:
            for hour in range(1, 25):  # Hours 1 to 24
                electricity_prices[index] = self.dict_prices[day][hour]
                index += 1

        return product_demand, electricity_prices
    
    def update_physical_state(self):

        if self.current_hour == 0 and self.current_day == 0:
            # Initialize the inventory state at the start of the first day
            self.IV = np.array([self.IV_i[prod] for prod in self.products], dtype=np.float32)

        self.state['IV'] = self.IV 

    def update_demand_and_electricty_state(self):
        """
        Update the full-week observation arrays using new data.
        This method is intended to be called at the start of a new day (e.g. day 1, day 2, ...).

        It updates the full-week electricity prices and demand forecasts.
        For demand, only the value at the end of each day (the 24th hour) is updated from product_demand.
        
        Args:
            current_day (int): The current day index (e.g., 1, 2, ...).
            product_demand (dict): Demand data keyed by product and day.
                                For example, product_demand['LIN'][1] gives the demand for LIN on day 1.
            electricity_prices (dict): Electricity prices keyed by hour (starting at 1) over the full forecast horizon.
        """

        product_demand, electricity_prices = self.get_demand_and_el_states_for_the_day()
        
        forecast_days = 1 + self.lookahead_days    # e.g., if lookahead_days is 6, then forecast 7 days in total.
        total_hours = 24 * forecast_days
        new_full_demand = np.zeros((len(self.products), total_hours), dtype=np.float32)
        
        # For each product and each forecast day, update only the end-of-day (24th hour).
        for idx, prod in enumerate(self.products):
            for d in range(self.current_day+1, self.current_day+1 + forecast_days):
                # Calculate the index for the end-of-day (24th hour) for day d.
                # For the first day (d == current_day), the 24th hour is at index 23 (0-indexed).
                # For subsequent days, it is (d - current_day + 1) * 24 - 1.
                # hour_index = (d - self.current_day + 1) * 24 - 1   # OLD
                # hour_index = (d - self.current_day) * 24 - 1
                hour_index = (d - self.current_day) * 24 - 1
                # Retrieve the demand value for product 'prod' on day d. Default to 0 if missing.
                demand_value = product_demand.get(prod, {}).get(d, 0)
                new_full_demand[idx, hour_index] = demand_value

        # Build the full forecast for electricity prices.
        # Assumes electricity_prices has keys 1, 2, ... total_hours.
        new_full_prices = np.array(
            [electricity_prices.get(i, 0) for i in range(1, total_hours + 1)],
            dtype=np.float32)

        # Store these full-week forecasts for later shifting during simulation.
        self.full_demand = new_full_demand
        self.full_prices = new_full_prices

        # Also update the state immediately (with no shifting yet).
        self.state['demand'] = new_full_demand.copy()
        self.state['electricity_prices'] = new_full_prices.copy()

        # if self.current_hour == 0:
        self.demand_today = self.state['demand'][:, 23]  # Demand forecast for the current day (for each product) # Check demand for the present day: the 24th hour in the shifted demand array (index 23)
        
    def shift_observation(self):
        """
        Shift the current observation based on the number of hours elapsed in the current day.
        For electricity prices, shift left by the number of hours elapsed and pad the end with 1000.
        For demand, shift left and pad the end with zeros.
        This method should be called after each hour of simulation.
        """
        t = self.current_hour  # Hours elapsed in the current day (0 <= t < 24)
        if t > 0:
            # Shift demand:
            remaining_demand = self.full_demand[:, t:]
            padding_demand = np.zeros((len(self.products), t), dtype=np.float32)
            updated_demand = np.concatenate((remaining_demand, padding_demand), axis=1)
            self.state['demand'] = updated_demand

            # Shift electricity prices:
            remaining_prices = self.full_prices[t:]
            # Compute the average of the remaining prices (if available)
            if remaining_prices.size > 0:
                avg_price = np.mean(remaining_prices)
            else:
                avg_price = 0.0
            # Pad the end with the average price
            padding_prices = np.full((t,), avg_price, dtype=np.float32)
            updated_prices = np.concatenate((remaining_prices, padding_prices))
            self.state['electricity_prices'] = updated_prices

    def production_quantity_and_cost(self, lambda_action):
        """
        Calculate production cost based on lambda_action.
        Returns the production vector (for each product) and production cost.
        Production vector is computed as the weighted sum of the convex hull extreme points.
        The dynamic cost is the total production * self.unit_prod_cost
        If production is active (sum(lambda) > 0), the fixed cost is added.
        """
        # Weighted production vector: dot product of lambda_action (shape: [n_extreme_points])
        # with extreme_points_liqp (shape: [n_extreme_points, 3]) => result shape: (3,)
        production_vector = np.dot(lambda_action, self.extreme_points_liqp)
        total_production = np.sum(production_vector)
        dynamic_cost_total = total_production *  self.unit_prod_cost     #  self.dynamic_cost
        fixed_cost_total = self.fixed_cost if np.sum(lambda_action) > 0 else 0.0
        prod_cost = (dynamic_cost_total + fixed_cost_total) * self.state['electricity_prices'][0]  # Use the first hour's price for cost calculation
        return production_vector, prod_cost
    
    def demand_penalty(self, ship_quantity):
        # --- Penalty 1: Demand Shortfall at End of Day ---
        # Calculate demand shortfall
        demand_penalty = 0
        demand_shortfall = np.maximum(self.demand_today - ship_quantity, 0)
        demand_penalty = self.Penalty_demand * np.sum(demand_shortfall)

        if demand_penalty > 0:
            self.env_spec_log['Number of Demand Violation'] += 1
            self.env_spec_log['Cost of Demand Violation'] += demand_penalty
        return demand_penalty

    def inventory_penalty(self):
        # --- Penalty 2: Inventory Exceeding Maximum --- 
        # For each product, if new_IV > iv_high, impose a penalty proportional to the excess
        # inventory_excess = np.maximum(new_IV - self.iv_high, 0)

        inventory_penalty = 0
        inventory_excess = np.maximum(self.IV - self.iv_high, 0)
        if np.sum(inventory_excess) > 10000:
            inventory_excess = inventory_excess/5

        inventory_penalty = self.Penalty_inventory * np.sum(inventory_excess)

        if inventory_penalty > 0:
            self.env_spec_log['Number of Inventory Violation'] += 1
            self.env_spec_log['Cost of Inventory Violation'] += inventory_penalty
        return inventory_penalty
    
    def sanitize_action(self, inventory_penalty):
        """
        Sanitize the action to prevent tank overflow.
        If the action leads to an inventory level exceeding IV_u, adjust the action.
        """
        # update self.IV to respect the upper bound
        if inventory_penalty > 0:
            self.IV = np.minimum(self.IV, self.iv_high)

    def update_information_state(self):
        """
        Update the information state based on the current hour and day.
        This method is called at the end of each hour to update the state with new information.
        """
        # If one day (24 hours) is complete, you might call update_observation externally to refresh the full forecasts.
        if self.current_hour == 0:
            self.update_demand_and_electricty_state()
        else:
            self.shift_observation()

    def step(self, raw_action):

        trucnated = False

        if isinstance(raw_action, torch.Tensor):
            raw_action = raw_action.to(self._device)
            action = raw_action.cpu().numpy()
        else:
            action = raw_action

        action = raw_action.numpy() if torch.is_tensor(raw_action) else raw_action

        # make sure the action is within the valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # action_dict = self.decode_action(action) 
        action_dict = decode_action(action)

        lambda_action = action_dict['lambda']

        # Validate action dimensions
        assert lambda_action.shape == (self.row_liqprod,), \
            f"Lambda action must match the number of extreme points ({self.row_liqprod})."
        
        total = np.sum(lambda_action)
        if total > 0:
            lambda_action = lambda_action / total

        # --- Production Cost and Production Vector ---
        production_vector, prod_cost = self.production_quantity_and_cost(lambda_action)

        # PENALTY 1: DEMAND SHORTFALL
        # demand_today = self.state['demand'][:, 23]  # Demand forecast for the current day (for each product) # Check demand for the present day: the 24th hour in the shifted demand array (index 23)
        if self.current_hour == 23:
            # At hour 23, we need to ship out the demand for the day
            ship_quantity = np.minimum(self.demand_today, self.IV + production_vector)  # since, we can ship only what we have in inventory
            # Update inventory after shipping
            # new_IV = self.IV + production_vector - ship_quantity
            self.IV += production_vector - ship_quantity
            demand_penalty = self.demand_penalty(ship_quantity)
        else:
            # For other hours, just update inventory with production
            # new_IV = self.IV + production_vector
            self.IV += production_vector
            demand_penalty = 0
        
        # PENALTY 2: INVENTORY 
        inventory_penalty = self.inventory_penalty()

        # Positive cost (penalties) incurred
        self.cost_ep += inventory_penalty + demand_penalty
        cost = inventory_penalty + demand_penalty
        self.cost = cost

        # Reward is negative of operational expense
        self.reward_ep += - prod_cost
        reward = - prod_cost
        self.reward = reward

        # Increment simulation time and day
        self.current_hour += 1
        if self.current_hour == 24:
            self.current_hour = 0   # END of certain day = Start of next day
            self.current_day += 1   

        # Now santize the action to land the correct physical state
        action = self.sanitize_action(inventory_penalty)

        self.update_physical_state()
        self.update_information_state()

        if self.current_day*24 == self.T:
            self.terminated = True

        # self.flatt_state = self.encode_observation(self.state)
        self.flatt_state = encode_observation(self.state)
        flatt_state_tensor = th.tensor(self.flatt_state, dtype = th.float32, device=self._device)

        # print(self.env_spec_log)

        # PRINT flattened OBS SPACE dimension throughout the episode to check.
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

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment to its initial state."""
        self.current_hour = 0
        self.current_day = 0

        self.reward_ep = 0
        self.cost_ep = 0
        self.cost = 0
        self.reward = 0
        self.terminated = False 
        self.truncated = False
        self.info = {}

        self._initialize_state()
        # self.flatt_state = self.encode_observation(self.state)
        self.flatt_state = encode_observation(self.state)
        flatt_state_tensor = th.tensor(self.flatt_state, dtype=th.float32, device=self._device)

        # print("self.flatt_state_tensor from reset", flatt_state_tensor.shape)
        return flatt_state_tensor, {"dict_state": self.state, "terminated": self.terminated, "truncated": self.truncated}
            
    def sample_action(self):
        return self.action_space.sample()
    
    def sample_state(self):
        return self.observation_space.sample()

    def _plot_demand_state(self):
        """Plot the demand state for visualization."""
        hours = np.arange(24 * (1+self.lookahead_days))
        plt.figure(figsize=(12, 6))
        for idx, prod in enumerate(self.products):
            plt.plot(hours, self.state['demand'][idx], label=f'Demand {prod}')
        plt.xlabel('Hour')
        plt.ylabel('Demand')
        plt.title('Demand Forecast Over a Week')
        plt.legend()
        plt.show()

    def _plot_electricity_prices_state(self):
        """Plot the electricity prices for visualization."""
        hours = np.arange(24 * (1+self.lookahead_days))
        plt.figure(figsize=(12, 6))
        plt.plot(hours, self.state['electricity_prices'], label='Electricity Prices')
        plt.xlabel('Hour')
        plt.ylabel('Price')
        plt.title('Electricity Prices Over a Week')
        plt.legend()
        plt.show()

def main():
    base_dir   = Path(__file__).resolve().parent
    config_fp  = base_dir / "asuenv_config.json"
    if not config_fp.is_file():
        raise FileNotFoundError(f"Couldn’t find config.json at {config_fp}")
  
    env_id = 'ASU1'
    env = ASUEnv(env_id, config_path = config_fp)
    # 3) reset returns (obs_tensor, info_dict)
    obs, info = env.reset()
    print("Manual rollout start...")
    
    i = 1
    num_action = 0

    terminated = False
    while not terminated:

        # step 1: Get the observation
        print(f"start of day{i}: observation shape", obs.shape[0]) 

        # step 2: take action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        terminated = terminated.item()
        num_action += 1
        
        print(f"Step {i}, obs shape={obs.shape}, reward={reward}, done={terminated}")

        if terminated:
            print("Shape of last observation:", obs.shape[0])
            print("Episode reset")
            print(f"Total actions taken: {num_action}")
            obs, info = env.reset()
        i += 1

if __name__ == "__main__":
    main()

    ## THE ERROR WAS DUE TO: 
    """ we have below:
    self.current_day = 0       # Start of Day: "current_day + 1"

    took action-
    self.current_day += 1      # 1 day
    when self.self.current_day = self.T, i.e. start of day 32 (since T=31)
    WE STILL UPDATE the state: for start of day 32 and return False for done

    Therefore, need to have data to support that.
    """
