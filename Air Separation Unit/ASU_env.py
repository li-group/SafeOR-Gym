'''
ASU Environment: Liquid Products
Akshdeep Singh Ahluwalia 

SAFE RL: TO ENSURE INVENTORY BOUNDS and Demand Satisfaction
'''

"""  
Physical state: Initial inventory
Information state: El prices and demand forecast
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces    # https://gymnasium.farama.org/

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
from simulate_state import Get_electricity_dataframe, Get_demand_dataframe

# Constants for penalties (tune these as necessary)
INVENTORY_PENALTY_FACTOR = 20.0      # Penalty per unit of inventory exceeding maximum
DEMAND_PENALTY_FACTOR = 20.0         # Penalty per unit demand shortfall at end of day

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'asu_config.json')
# Open the configuration file using the computed path
with open(config_path, 'r') as config_file:
    ASU_DATA_FILES = json.load(config_file)

class ASUEnv(gym.Env):
    def __init__(self, asu_name, lookahead, days_to_simulate):
        super().__init__()
        self.name = asu_name

        # Simulation time counters
        self.current_hour = 0  # hour in current day (0-23)
        self.current_day = 1
        self.days_to_simulate = days_to_simulate

        self.lookahead_days = lookahead
        self.full_week_prices = np.zeros(24 * (1+self.lookahead_days), dtype=np.float32)
        self.full_week_demand = np.zeros((3, 24 * (1+self.lookahead_days)), dtype=np.float32)

        # Initialize parameters, state, observation and action spaces
        self._initialize_params()
        self._initialize_simulation_data()
        self._initialize_observation_space()
        self._initialize_action_space()
        self._initialize_state()

        self.reward = 0
        self.done = False
        self.info = {}

    def _initialize_simulation_data(self):
        # Electricity simulation
        data_simulation_days = 31
        electricity_simulator = Get_electricity_dataframe(data_simulation_days + np.maximum(0, self.days_to_simulate - data_simulation_days), self.name)
        price_df = electricity_simulator.simulate_electricity_prices()
        # electricity_simulator.get_electricity_plot()
        # Create new columns for the day and the hour (adding 1 to the hour so it goes from 1 to 24)
        price_df['day'] = price_df['Timestamp'].dt.day
        price_df['hour'] = price_df['Timestamp'].dt.hour + 1
        # Group by day and create a nested dictionary for each day's prices
        price_dict = {
            day: dict(zip(group['hour'], group['Electricity_Price']))
            for day, group in price_df.groupby('day')
        }
        self.dict_prices = price_dict

        # Demand simulation
        demand_simulator =  Get_demand_dataframe(data_simulation_days + np.maximum(0, self.days_to_simulate - data_simulation_days), self.name)
        demand_df = demand_simulator.simulate_daily_target_demand() 
        # demand_simulator.get_demand_plot()
        demand_dict = {i + 1: row.to_dict() for i, (date, row) in enumerate(demand_df.iterrows())}
        self.dict_demand = demand_dict

    def _initialize_params(self):
        """Load configuration and operational parameters."""
        
        data_file = ASU_DATA_FILES.get(self.name)
        if data_file is None:
            raise ValueError(f"Unknown ASU identifier: {self.name}")
        with open(data_file, 'r') as file:
            self.loaded_data = json.load(file)

        # Set parameters from the loaded data file
        self.products = self.loaded_data['products']  # e.g., ['LIN', 'LOX', 'LAR']
        self.IV_u = self.loaded_data['IV_u']           # Maximum inventory capacities
        self.IV_l = self.loaded_data['IV_l']
        self.IV_f = self.loaded_data['IV_f']
        self.fixed_cost = self.loaded_data['fixed_cost'][0]
        self.dynamic_cost = self.loaded_data['dynamic_cost'][0]
        self.IV_i = self.loaded_data['IV_i']           # Initial inventory levels
        self.total_hours = self.loaded_data['total_hours'][0]

    def _initialize_observation_space(self):
        """Define the observation space."""
        # Build the high bound for inventory from IV_u
        self.iv_high = np.array([self.IV_u[prod] for prod in self.products], dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            'electricity_prices': spaces.Box(low=0, high=np.inf, shape=(24 * (1+self.lookahead_days),), dtype=np.float32),
            'demand': spaces.Box(low=0, high=np.inf, shape=(len(self.products), 24 * (1+self.lookahead_days)), dtype=np.float32),
            'IV': spaces.Box(low=0, high=self.iv_high, shape=(len(self.products),), dtype=np.float32)
        })

    def _initialize_action_space(self):
        """  Define the action space.  """
        max_quantities = np.array([self.IV_u[prod] for prod in self.products], dtype=np.float32)

        ### Data preprocessing to yield Convex Hull ###
        liq_prod_data = self.loaded_data['liq_prod_data']
            # Convert liq_prod_data to a list of tuples
        points = [(liq_prod_data['LIN'][i], liq_prod_data['LOX'][i], liq_prod_data['LAR'][i]) for i in liq_prod_data['LIN']]
        # Compute the convex hull
        points_np = np.array(points)                         # Convert list of tuples to a NumPy array
        hull = ConvexHull(points_np)
        self.extreme_points_liqp = points_np[hull.vertices]       # Extract the vertices (extreme points)
        row_liqprod, colliq_prod = self.extreme_points_liqp.shape
        self.row_liqprod = row_liqprod

        self.action_space = spaces.Dict({
            'lambda': spaces.Box(low=0, high=1, shape=(self.row_liqprod,), dtype=np.float32),
        })

    def _initialize_state(self):
        """Initialize the state with default values."""
        
        self.electricity_prices = np.zeros(24 * (1+self.lookahead_days), dtype=np.float32)
        self.demand = np.zeros((len(self.products), 24 * (1+self.lookahead_days)), dtype=np.float32)
        
        # Set initial inventory levels from IV_i
        self.IV = np.array([self.IV_i[prod] for prod in self.products], dtype=np.float32)
        # Combine into the state dictionary
        self.state = {
            'electricity_prices': self.electricity_prices,
            'demand': self.demand,
            'IV': self.IV,
        }

        self.update_demand_and_electricty_state()

    def get_demand_and_el_states_for_the_day(self):
        start_day = self.current_day
        selected_days = range(start_day, start_day + self.lookahead_days + 1)
        
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
            for d in range(self.current_day, self.current_day + forecast_days):
                # Calculate the index for the end-of-day (24th hour) for day d.
                # For the first day (d == current_day), the 24th hour is at index 23 (0-indexed).
                # For subsequent days, it is (d - current_day + 1) * 24 - 1.
                hour_index = (d - self.current_day + 1) * 24 - 1
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
        The dynamic cost is the total production * self.dynamic_cost.
        If production is active (sum(lambda) > 0), the fixed cost is added.
        """
        # Weighted production vector: dot product of lambda_action (shape: [n_extreme_points])
        # with extreme_points_liqp (shape: [n_extreme_points, 3]) => result shape: (3,)
        production_vector = np.dot(lambda_action, self.extreme_points_liqp)
        total_production = np.sum(production_vector)
        dynamic_cost_total = total_production * self.dynamic_cost
        fixed_cost_total = self.fixed_cost if np.sum(lambda_action) > 0 else 0.0
        prod_cost = (dynamic_cost_total + fixed_cost_total) * self.state['electricity_prices'][0]  # Use the first hour's price for cost calculation
        return production_vector, prod_cost
    
    def demand_penalty(self, ship_quantity):
        # --- Penalty 1: Demand Shortfall at End of Day ---
        # Calculate demand shortfall
        demand_shortfall = np.maximum(self.demand_today - ship_quantity, 0)
        demand_penalty = DEMAND_PENALTY_FACTOR * np.sum(demand_shortfall)
        return demand_penalty

    def inventory_penalty(self, new_IV):
        # --- Penalty 2: Inventory Exceeding Maximum --- 
        # For each product, if new_IV > iv_high, impose a penalty proportional to the excess
        inventory_excess = np.maximum(new_IV - self.iv_high, 0)
        inventory_penalty = INVENTORY_PENALTY_FACTOR * np.sum(inventory_excess)
        return inventory_penalty

    def step(self, action):

        if self.current_hour == 0:
            self.demand_today = self.state['demand'][:, 23]  # Demand forecast for the current day (for each product) # Check demand for the present day: the 24th hour in the shifted demand array (index 23)
            
        # Extract action components
        lambda_action = action['lambda']

        # Validate action dimensions
        assert lambda_action.shape == (self.row_liqprod,), \
            f"Lambda action must match the number of extreme points ({self.row_liqprod})."
        
        total = np.sum(lambda_action)
        if total > 0:
            lambda_action = lambda_action / total

        # --- Production Cost and Production Vector ---
        production_vector, prod_cost = self.production_quantity_and_cost(lambda_action)

        # --- Update Inventory and Penalty 2: Demand Shortfall (only at the end of the day) ---
        # demand_today = self.state['demand'][:, 23]  # Demand forecast for the current day (for each product) # Check demand for the present day: the 24th hour in the shifted demand array (index 23)
        if self.current_hour == 23:
            # At hour 23, we need to ship out the demand for the day
            ship_quantity = np.minimum(self.demand_today, self.IV + production_vector)  # since, we can ship only what we have in inventory
            # Update inventory after shipping
            new_IV = self.IV + production_vector - ship_quantity
            demand_penalty = self.demand_penalty(ship_quantity)
        else:
            # For other hours, just update inventory with production
            new_IV = self.IV + production_vector
            demand_penalty = 0
            
        inventory_penalty = self.inventory_penalty(new_IV)

        total_penalty = inventory_penalty + demand_penalty
        
        # Increment simulation time
        self.current_hour += 1

        # If one day (24 hours) is complete, you might call update_observation externally to refresh the full forecasts.
        if self.current_hour >= 24:
            self.current_hour = 0
            self.current_day += 1   
            self.update_demand_and_electricty_state()

        # --- Update Inventory State ---
        self.IV = new_IV.copy()
        # Shift the observation based on elapsed hours
        self.shift_observation()  
        self.state['IV'] = self.IV      

        # Total cost: production cost plus penalties
        total_cost = prod_cost + total_penalty
        reward = -total_cost

        if self.current_day > self.days_to_simulate:
            self.done = True

        return self.state, reward, self.done, self.info

    def reset(self):
        """Reset the environment to its initial state."""
        self.current_hour = 0
        self.current_day = 1
        self._initialize_state()
        self.done = False
        self.reward = 0
        # return self.state, self.reward, self.done, self.info
    
    def render(self):
        """Return the current state of the environment."""
        return self.state
    
    def sample_action(self):
        """Sample a random action from the action space."""
        action = {}
        action['lambda'] = self.action_space['lambda'].sample()
        return action

    def _plot_demand_state(self):
        """Plot the demand state for visualization."""
        import matplotlib.pyplot as plt
        hours = np.arange(24 * (1+self.lookahead_days))
        plt.figure(figsize=(12, 6))
        for idx, prod in enumerate(self.products):
            plt.plot(hours, self.state['demand'][idx], label=f'Demand {prod}')
        plt.xlabel('Hour')
        plt.ylabel('Demand')
        plt.title('Demand Forecast Over a Week')
        plt.legend()
        plt.show()

    def _plot_electricity_prices(self):
        """Plot the electricity prices for visualization."""
        import matplotlib.pyplot as plt
        hours = np.arange(24 * (1+self.lookahead_days))
        plt.figure(figsize=(12, 6))
        plt.plot(hours, self.state['electricity_prices'], label='Electricity Prices')
        plt.xlabel('Hour')
        plt.ylabel('Price')
        plt.title('Electricity Prices Over a Week')
        plt.legend()
        plt.show()