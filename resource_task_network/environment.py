import yaml
import json
import pickle
import logging
import random
from typing import Any, ClassVar, List, Tuple, Optional, Dict
from types import SimpleNamespace

import numpy as np

import torch
import pyomo.environ as po
from pyomo.opt import SolverFactory

import gymnasium as gym
from gymnasium.spaces.utils import flatten_space

from omnisafe.typing import OmnisafeSpace
from omnisafe.envs.core import CMDP, env_register
from omnisafe.common.logger import Logger


#@env_register
class RTNEnv(gym.Env):
    """
    Resource Task Network
    Env Registration : rtn-v0

    The Resource Task Network (RTN) is a modeling framework for plant scheduling which
    requires scheduling processes (tasks) in a plant to meet product demands while trying to
    minimize cost of running processes. The environment accepts an action and checks for
    feasibility, sanitising the action if infeasible. The sanitized action is used to 
    compute the transition function which updates the inventory (state). If certain actions
    lead to unfixable inventory violations, the inventory is clamped with added costs.

    Observations (state s_t) : 
        - Inventory levels (ndarray) shape = (resources, ) : The current inventory of each resource
        - Pending outputs (ndarray) shape = (max_tau * (products + intermediates + equipments), ) : The 
        products that will be produced/delivered in upcoming time steps
        - future demand (ndarray) shape = (T * products, ) : The future demand from current time step.
        It is padded with 0s to keep a consistent size.

    Actions (a_t) :
        - task batch size (ndarray) shape = (tasks, ) : The batch size of each task to be processed.

    Reward (r_t) :
        - Revenue (scalar) : The revenue earned from meeting the product demand.
        - Prices (scalar) : The utility cost for running the task.
        - Penalty (scalar) : The penalty incurred due to not meeting the demand. (1.5 * price of product)
        - Total reward = revenue - prices - penalties

    Cost (c_t) : 
        - Inventory lower bound (scalar) : Total no of violations due to inventory falling below the lower bound.
        - Inventory upper bound (scalar) : Total no of violations due to inventory going above the upper bound.
        - Equipment lower bound (scalar) : Total no of violations due to using unavailable equipments.
        - Total cost = inv lower bound + inv upper bound + equip lower bound

    Starting state (s_0) : 
        - The initial inventory level as specified in the environment config file

    Episode termination :
        - When the horizon ends. It is not truncated even under constraint violations.

    The following functions are included:
        - init : to initialize all the configuration parameters:
        - reset : resets the timestep to 0 and all the inventory to the initial inventory
        - step : changes the state based on the action given by the agent
        - get_state : gets the current state based on the inventory + incoming products + future demand
        - get_padded_future_demand : post pads the demand with 0 to keep a consistent state shape 
        - sanitize_action : sanitizes the action to ensure inventory and equipment lower bounds are not violated (adds a cost to the RL agent in that case)
        - check_inventory_bounds : checks for any inventory constraints violations
        - fix_inventory : fixes the inventory incase it overflows due to actions (tasks) that have been already executed. Assigns
        - cost to the agent in that case
        - compute_resource_change : computes the change (comsumption) of the reactants to the task
        - compute_cost : computes the constraint violations
        - compute_utility_cost : computes the cost of the utilities
        - compute_reward : computes the reward for that action
        - compute_sanitization_cost :  computes the cost of sanitizing the action (a larger degree of sanitization implies a higher cost)
        - compute_product_change : computes the delivery of products at that time step due to previous action (task) execution

            
    """
        

    _support_envs: ClassVar[List[str]] = ['rtn-v0']
    _action_space: OmnisafeSpace
    _observation_space: OmnisafeSpace
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        #super(RTNEnv, self).__init__(env_id, **kwargs)
        """
        Function to initialise all the environment parameters:
            - T : horizon
            - reactants : materials used at the start
                - X0 : initial amount of inventory present
                - Xmin : minimum inventory
                - Xmax : maximum inventory
                - cost : cost of ordering more reactants
            - intermediates : materials that might be produced or consumed in a task (but no supply or demand exist)
                - X0 : initial amount of inventory present
                - Xmin : minimum inventory
                - Xmax : maximum inventory
                - cost : set to 0 (as cannot be ordered or sold)
            = products : materials that are always produced and never consumed (demand exists for each product at a given time step)
                - X0 : initial amount of inventory present
                - Xmin : minimum inventory
                - Xmax : maximum inventory
                - cost : cost of selling products (cannot sell more than the demand)
            - tasks : processes that are run at a given time step
                - tau : processing time of the task
                - vmin : minimum batch size that can be processed in that task
                - vmax : maximum batch size that can be processed in that task
                = stoich : the stoichiometry of the reactants and the products that go through the process
                - equipments : the set of equipments that are used in that task
            - equipments : used in a task to run the process (a given no of units exist for this equipment)
                - X0 : initial amount of inventory present
                - Xmin : minimum inventory
                - Xmax : maximum inventory
            - utility : utility used for running tasks
                - cost : cost of the particular utility at that time step
            
            - demand : demand of each product across the horizon


        Inputs:
            - env_id : str

        Outputs:
            - None : initializes the environment state
        
        """

        # Set up debug flag and logger.
        self.config_file = kwargs.get('config_file')        
        self.debug = kwargs.get('debug', False)
        self.sanitization_cost_weight = kwargs.get('sanitization_cost_weight')
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.env_spec_log = {"Penalties/inv_lb" : 0, "Penalties/inv_ub" : 0, "Penalties/equip_lb" : 0}
        
        if self.debug:
            logging.basicConfig(level = logging.DEBUG)
            self.logger.setLevel(logging.DEBUG)
            self.logger.debug('---- Enabling debugging ----')
        else:
            self.logger.setLevel(logging.WARNING)

        self.logger.debug('---- Building environment ---- ')
        self.load_data_from_file()

        if self.demand is not None:
            self.num_periods = len(self.demand[list(self.demand.keys())[0]])
            self.T = self.num_periods
            self.logger.debug(f'---- Setting the horizon to {self.T} ----')

        self.logger.debug('---- Loaded all files ----')

        self.t = 0
        self.cost = 0.0
        self.reward = 0.0
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = kwargs.get('verbose', False)

        # Resource names by category
        self.reactants = list(self.raws_dict.keys())
        self.products = list(self.prods_dict.keys())
        self.intermediates = list(self.ints_dict.keys())
        self.equipments = list(self.equipment_dict.keys())

        # Create combined lists
        self.materials = self.reactants + self.intermediates + self.products
        self.resources = self.reactants + self.intermediates + self.products + self.equipments
        self.pending_resources = self.intermediates + self.products + self.equipments
        self.pending_resource_indices = {r: i for i, r in enumerate(self.pending_resources)}


        # For reactants, intermediates, products, initial inventory comes from the pickle.
        self.initial_inventory = {r: (self.raws_dict.get(r) or self.prods_dict.get(r) or self.ints_dict.get(r) or self.equipment_dict.get(r))['X0']  for r in self.resources}

        # Resource bounds: for reactants/intermediates/products, from pickle;
        # for equipments, lower bound = 0, upper bound = available units.
        self.resource_bounds = {r: ((self.raws_dict.get(r) or self.prods_dict.get(r) or self.ints_dict.get(r) or self.equipment_dict.get(r))['Xmin'], (self.raws_dict.get(r) or self.prods_dict.get(r) or self.ints_dict.get(r) or self.equipment_dict.get(r))['Xmax']) for r in self.resources}

        self.logger.debug(f'---- No of reactants : {len(self.reactants)}, intermediates : {len(self.intermediates)}, products : {len(self.prods_dict)}, equipments : {len(self.equipments)}')

        # Build the inventory vector from initial_inventory.
        self.inventory = np.array([self.initial_inventory[r] for r in self.resources], dtype=np.float32)
        self.lower_bounds = np.array([self.resource_bounds[r][0] for r in self.resources], dtype=np.float32)
        self.upper_bounds = np.array([self.resource_bounds[r][1] for r in self.resources], dtype=np.float32)

        # Task parsing (unchanged)
        self.task_names = list(self.tasks_dict.keys())
        self.num_tasks = len(self.task_names)
        self.tasks = []
        self.min_batch = []
        self.max_batch = []
        self.task_utilities = []
        self.task_stoichs = []  # Each stoich: dict mapping resource->coefficient.
        self.task_equipments = []

        # Processing times and delayed production queue.
        self.task_taus = [self.tasks_dict[name].get('tau') for name in self.task_names]
        self.max_tau = max(self.task_taus)
        self.logger.debug(f'---- Maximum tau set to {self.max_tau} ----')
        self.delayed_production_queue = {t: [] for t in range(self.t + 1, self.t + self.max_tau + 1)}

        for task_name in self.task_names:
            task_equipments = []
            task_data = self.tasks_dict[task_name]
            vmin = task_data['Vmin']
            vmax = task_data['Vmax']
            self.min_batch.append(vmin)
            self.max_batch.append(vmax)
                        
            # Build stoichiometric vector.
            stoich = {r: 0.0 for r in self.resources}
            
            # Consumption of raws
            for _, r in task_data['raws'].items():
                stoich[r] -= task_data['raw_dist'][r]
            
            # Consumption of intermediate reactants
            for _, r in task_data['int_react'].items():
                stoich[r] -= task_data['int_react_dist'][r]
            
            # Production of intermediate products
            for _, r in task_data['int_prod'].items():
                stoich[r] += task_data['int_prod_dist'][r]
            
            # Production of final products
            for _, r in task_data['prods'].items():
                stoich[r] += task_data['prod_dist'][r]
            
            # NEW: Subtract 1 unit for each equipment used (immediate consumption).
            for _, eq in task_data['equipments'].items():
                task_equipments.append(eq)                
                stoich[eq] = -1.0
            
            self.task_stoichs.append(stoich)
            self.task_equipments.append(task_equipments)
            self.task_utilities.append(task_data.get('utilities', {}))  # Use provided utilities.

        self.min_batch = np.array(self.min_batch, dtype=np.float32)
        self.max_batch = np.array(self.max_batch, dtype=np.float32)

        #self.logger.debug(f'---- Tasks : {self.task_equipments},{self.task_stoichs},{self.task_utilities} ----')

        # Define action space (unchanged).
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(self.task_names),), dtype=np.float32)

        # Define observation space.
        # Note that inventory now includes equipments.
        self.raw_observation_space = gym.spaces.Dict({
            "inventory": gym.spaces.Box(low=self.lower_bounds, high=self.upper_bounds, shape=(len(self.resources),), dtype=np.float32),
            "pending_outputs": gym.spaces.Box(low=0.0, high=np.inf, shape=(self.max_tau * len(self.pending_resources),), dtype=np.float32),
            "demand": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.T * len(self.products),), dtype=np.float32)
        })
        self.observation_space = gym.spaces.utils.flatten_space(self.raw_observation_space)
        self.logger.debug(f'---- Shape of observation space : inv ({len(self.resources)}, ), pending outputs : ({self.max_tau * (len(self.pending_resources))}, ), demand : ({len(self.products) * self.T}, )')


    def load_data_from_file(self):
        """
        Loads the data from a given environment configuration file.

        Inputs : 
            - None (load the data file specified through self.config_file)

        Outputs : 
            - Loads the important data through reactants, intermediates, products, equipments, tasks, demand and utility_cost
        
        """
        data_file = json.load(open(self.config_file, 'r'))

        self.raws_dict = data_file['reactants']
        self.ints_dict = data_file['intermediates']
        self.prods_dict = data_file['products']
        self.equipment_dict = data_file['equipments']
        self.tasks_dict = data_file['tasks']

        self.demand = {}
        for k, v in data_file['demand'].items():
            self.demand[k] = [v1 for _, v1 in v.items()]

        self.utility_costs = {}
        for k, v in data_file['utility_costs'].items():
            self.utility_costs[k] = v



    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[torch.Tensor, dict]:
        """
        Resets the environment to the initial state. Does not reset the env_spec_log

        Inputs:
            - seed

        Outputs:
            - state (ndarray)
        
        
        """
        self.t = 0
        self.terminated = False
        self.truncated = False

        if seed is not None:
            self.set_seed(seed)

        # Reset inventory (including equipments) to initial values.
        self.inventory = np.array([self.initial_inventory[r] for r in self.resources], dtype=np.float32)
        self.delayed_production_queue = {t: [] for t in range(self.t + 1, self.t + self.max_tau + 1)}
        state = self._get_state()

        return torch.from_numpy(state).float().to(self.device), {}

    def _get_current_demand(self) -> np.ndarray:
        demand_vec = np.array([self.demand.get(prod, [0.0] * self.T)[self.t] if self.t < len(self.demand.get(prod, [])) else 0.0 for prod in self.products], dtype=np.float32)
        
        return demand_vec

    def _get_padded_future_demand(self) -> np.ndarray:
        """
        Obtain the demand from time step t till the end of the horizon.
        Then pad it with 0s to keep a consistent shape across all states

        Inputs:
            - Nothing (taken from self)
        Outputs:
            - future demand + padding (ndarray)
        
        """
        
        future_demand = []

        for t_future in range(self.t, self.T):
            for prod in self.products:
                # Access demand using integer index
                demand_list = self.demand.get(prod, [])
                demand_val = demand_list[t_future] if t_future < len(demand_list) else 0.0
                future_demand.append(demand_val)

        # Pad to full length
        total_length = self.T * len(self.products)
        future_demand += [0.0] * (total_length - len(future_demand))

        return np.array(future_demand, dtype=np.float32)

    def _get_state(self) -> np.ndarray:
        """
        Construct the observation vector for this current timestep.
            - Inventory for all resources (including equipments)
            - Pending outputs for resources that will be delivered at t+1.
            - Full demand trajectory (for products) + padding

        Inputs:
            - None

        Outputs:
            - state (ndarray)

        """
        self.logger.debug(f'---- Obtaining state at time step {self.t+1} ----')
        # Current state has the inventory
        self.logger.debug(f'---- Inventory : {self.inventory} ----')
        state_parts = [self.inventory]

        # Pending outputs for t+1
        pending = np.zeros((self.max_tau, len(self.pending_resources)), dtype=np.float32)

        for dt in range(1, self.max_tau + 1):
            delivery_time = self.t + dt
            if delivery_time in self.delayed_production_queue:
                for res, amount in self.delayed_production_queue[delivery_time]:
                    if res in self.pending_resource_indices:
                        res_idx = self.pending_resource_indices[res]
                        pending[dt - 1, res_idx] += amount
            
        # State has the pending outputs then
        self.logger.debug(f'---- Pending outputs : {pending.flatten()} ----')
        state_parts.append(pending.flatten())

        # State has the future demand (+ padding) finally
        self.logger.debug(f'---- Padded future demand : {self._get_padded_future_demand()} ----')
        state_parts.append(self._get_padded_future_demand())

        return np.concatenate(state_parts)


    def _check_inventory_bounds(self, new_inventory: np.ndarray) -> bool:
        """
        Checks if any of the inventory bounds have been violated.

        Inputs
            - inventory (ndarray)

        Outputs
            - violated_or_not (bool)
        """
        
        if np.any(new_inventory < self.lower_bounds) or np.any(new_inventory > self.upper_bounds):
            violated = [(self.resources[i], new_inventory[i], self.lower_bounds[i], self.upper_bounds[i]) for i in range(len(self.resources)) if new_inventory[i] < self.lower_bounds[i] or new_inventory[i] > self.upper_bounds[i]]
            
            self.logger.debug(f'---- Inventory bounds violated at ----')
            for res, val, lb, ub in violated:
                self.logger.debug(f"---- {res}: {val:.3f} (bounds: {lb:.3f}–{ub:.3f}) ----")
            
            self.env_spec_log["Penalties/inv_ub"] += 1
            return False
        
        return True
    

    def _fix_inventory(self, new_inventory: np.ndarray) -> np.ndarray:
        """
        Fixes the inventory in case of any overflow due to delivery of products based on prior executed task

        Inputs
            - inventory (ndarray)

        Outputs
            - updated inventory (ndarray)
        """
        if np.any(new_inventory < self.lower_bounds) or np.any(new_inventory > self.upper_bounds):
            violated = [(i, self.resources[i], new_inventory[i], self.lower_bounds[i], self.upper_bounds[i]) for i in range(len(self.resources)) if new_inventory[i] < self.lower_bounds[i] or new_inventory[i] > self.upper_bounds[i]]
            
            self.logger.debug(f'---- Fixing violated inventory bounds ----')
            for idx, res, val, lb, ub in violated:
                if val < lb:
                    new_inventory[idx] = lb
                elif val > ub:
                    new_inventory[idx] = ub
            
            return new_inventory
        
        raise NotImplementedError
        

    def _compute_resource_change(self, action: np.ndarray) -> np.ndarray:
        """
        Compute net change from tasks stoichiometry for reactants 
        Equipment consumption is applied separately.
        Products are entered into the buffer.

        Inputs
            - action (batch size of each task)

        Outputs
            - total_change (ndarray)
        """
        
        total_change = np.zeros_like(self.inventory, dtype=np.float32)
        
        for i, stoich in enumerate(self.task_stoichs):
            material_change = np.array([stoich.get(r, 0.0) for r in self.materials], dtype=np.float32)
            equipment_change = np.zeros(len(self.equipments))

            # Make sure no products are changed immediately.
            material_change = np.clip(material_change, a_min = None, a_max = 0.0)

            change = np.append(material_change, equipment_change)
            total_change += action[i] * change
        
        return total_change

    # def _compute_equipment_change(self, action: np.ndarray) -> np.ndarray:
    #     """
    #     Compute units of each equipment (stoichiometry for equipments)
        
    #     Inputs 
    #         - action (batch size of each task)
        
    #     Outputs
    #         - None (updates inventory directly)
        
    #     """
    #     for i in range(self.num_tasks):
    #         if action[i]:
    #             equip_ids = self.task_equipments[i]  # Equipment used by this task
                
    #             for eq in equip_ids:
    #                 eq_idx = self.resources.index(eq)
    #                 self.inventory[eq_idx] -= 1  # Use one unit of equipment


    def _compute_cost(self, inventory: np.ndarray) -> float:
        """
        Cost is defined as the number of constraint violations:
        - Inventory lower bound violation
        - Inventory upper bound violation

        Inputs:
            - inventory (np.ndarray): updated inventory after applying transition

        Returns:
            - cost (float): number of violated constraints
        """
        cost = 0.0

        # Inventory bound violations
        below_lb = inventory < self.lower_bounds
        above_ub = inventory > self.upper_bounds
        cost += np.sum(below_lb) + np.sum(above_ub)

        return cost #float(cost)

    def _compute_utility_cost(self, action: np.ndarray) -> float:
        """
        Compute the utility-based operating cost of executing tasks (based on batch size and utility usage).
            - Extract the utility cost at timestep t
            - Iterate over all utilities and add them multiplied by Uf
            - 
        Inputs:
            - action : ndarray
        
        Outputs:
            - utility_cost : float
        """
        total_cost = 0.0
        utility_prices = self.utility_costs

        for i in range(self.num_tasks):
            batch = action[i]
            utility_usage = self.task_utilities[i]

            for util, uf in utility_usage.items():
                price = utility_prices[util].get(str(int(self.t) + 1))
                total_cost += uf * price * batch
        
        self.logger.debug(f'---- Utility cost computed : {total_cost}')

        return total_cost

    def _compute_reward(self, new_inventory: np.ndarray, action: np.ndarray) -> float:
        """
        Compute reward based on fulfilled demand (revenue) and utility cost.
        Apply penalty for unmet demand, ensuring product inventory does not go below min bound.

        Inputs:
            - new_inventory : ndarray
            - action : ndarray

        Outputs:
            - reward : float

        """

        # Initialize revenue from 0
        revenue = 0.0
        num_reactants = len(self.reactants)
        num_intermediates = len(self.intermediates)
        product_indices = range(
            num_reactants + num_intermediates, 
            num_reactants + num_intermediates + len(self.products)
        )
        product_indices_list = list(product_indices)

        current_demand = self._get_current_demand()
        product_price = {prod: self.prods_dict[prod]['cost'] for prod in self.products}

        # Extract product inventories
        prod_inventory = new_inventory[product_indices_list]

        # Minimum inventory bounds for products
        min_inventory = np.array([self.lower_bounds[idx] for idx in product_indices_list])

        # Available inventory to fulfill demand (without dropping below min bounds)
        available_to_fulfill = np.clip(prod_inventory - min_inventory, a_min=0.0, a_max=None)

        # Calculate revenue from fulfilled demand
        for idx, prod in enumerate(self.products):
            units_sold = min(available_to_fulfill[idx], -current_demand[idx])
            revenue += units_sold * product_price.get(prod)

        # Utility cost of executing the action
        utility_cost = self._compute_utility_cost(action)

        # Unmet demand = total demand - fulfilled amount
        unmet_demand = current_demand - np.minimum(current_demand, available_to_fulfill)
        unmet_penalty = sum(1.5 * unmet_demand[i] * product_price.get(prod) for i, prod in enumerate(self.products))

        reward = revenue - utility_cost - unmet_penalty  # 1.5 is a tunable penalty coefficient

        return reward

    def _compute_sanitization_cost(self, raw: np.ndarray, final: np.ndarray) -> float:
        """
        If action is getting sanitized then there should be a small cost associated with it
        The raw action is scaled as it a part of the model rather than env.

        Inputs:
            - raw : ndarray (raw action)
            - final : ndarray (sanitized action)

        Outputs:
            - cost : scalar (how much sanitized action is different from raw action) Higher values mean more sanitization was necessary.

        """
        if not isinstance(final, torch.Tensor):
            final = torch.from_numpy(final)
        if not isinstance(raw, torch.Tensor):
            raw = torch.from_numpy(raw)

        return (torch.abs(raw - final) > 1e-4).float().sum().item()
    
    def _deliver_products(self):
        """
        Compute the change of state due to incoming product deliveries due to completion of several tasks

        Inputs:
            - None
        Outputs : 
            - None (updates self.inventory and removes first element from delayed_production_queue)
        
        """
        # Handle delayed product deliveries (including any returns that are scheduled for t).
        if self.t in self.delayed_production_queue:
            # There are some products at the next time step coming in
            for res, amount in self.delayed_production_queue[self.t]:
                # Add the incoming amount for every product id
                res_idx = self.resources.index(res)
                self.inventory[res_idx] += amount
            
            # Remove the product delivery at time t to ensure multiple products aren't added
            del self.delayed_production_queue[self.t]

    def _schedule_products(self, new_inventory, action):
        """
        Compute the inventory change of products and intermediates due to completion of ongoing tasks.

        Inputs:
            - new_inventory (ndarray) : Current inventory
            - action (ndarray) : sanitized action

        Outputs : 
            - new_inventory : updated inventory after computing product deliveries
        
        """
        for i, batch in enumerate(action):
            if batch == 0:
                continue
            tau = self.task_taus[i]
            for res, coeff in self.task_stoichs[i].items():
                if coeff > 0:
                    delivery_time = self.t + tau
                    # ensure list exists
                    self.delayed_production_queue.setdefault(delivery_time, [])
                    self.delayed_production_queue[delivery_time].append((res, coeff * batch))

        
        # Increase equipment units if task is getting finished at this time step
        
        # if 'equipment_returns' in self.delayed_production_queue:
        #     if self.t in self.delayed_production_queue['equipment_returns']:
        #         # If there are some units going to be free at the next time step
        #         for eq, amount in self.delayed_production_queue['equipment_returns'][self.t]:
        #             eq_idx = self.resources.index(eq)
        #             new_inventory[eq_idx] += amount
        #         del self.delayed_production_queue['equipment_returns'][self.t]       

        return new_inventory
    
    def sanitize_action(self, raw_action: np.ndarray) -> np.ndarray:
        """
        Sanitize and scale raw actions from the RL policy output.
            - raw_action is assumed to be in [-1, 1]^num_tasks.
            - Scaled to [min_batch, max_batch]^num_tasks.
            - If |raw_action| <= 1e-4, the task is assumed to be not initiated.
            - Sets batch size to 0 or min batch based on the current resource inv and specified lower limits
            - Block tasks if there is insufficient equipment units for that task.

        Inputs:
            - raw action (torch.tensor)
        Outputs:
            - sanitized action (torch.tensor)
    
        """
        self.logger.debug('---- Sanitising action ----')
        duplicate_inventory = self.inventory.copy()

        scaled_action = 0.5 * (raw_action + 1.0) * (self.max_batch - self.min_batch) + self.min_batch
        self.logger.debug(f'---- Scaled action: {scaled_action} ----')
        
        candi_action = np.zeros_like(scaled_action)
        final_action = np.zeros_like(scaled_action)

        # Second pass: inventory bound feasibility (including effects of this action)
        for i in range(self.num_tasks):
            scaled_batch = scaled_action[i]
            raw_batch = raw_action[i]

            if np.abs(raw_batch) <= 1e-3:
                continue

            stoich = self.task_stoichs[i]
            max_feasible_batch = np.inf  # Initialize with no upper limit (start with the whole batch).

            # Check consumption side for each resource
            for j, r in enumerate(self.materials):
                coeff = stoich.get(r, 0.0)
                if coeff < 0:  # Only consider consumption terms
                    # How much we can afford to consume without violating the lower bound
                    available = self.inventory[j] - self.lower_bounds[j]
                    feasible_batch_r = available / abs(coeff)  # Maximum batch that does not violate the lower bound
                    max_feasible_batch = min(max_feasible_batch, feasible_batch_r)  # Get the limiting factor

            # If the action would deplete the resource below its lower bound, clamp it to the feasible batch size
            if max_feasible_batch < scaled_batch:
                # Try clamping to min_batch (if min_batch itself violates the lower bound, reject action)
                if max_feasible_batch < self.min_batch[i]:
                    candi_action[i] = 0.0  # Infeasible action, set to 0
                else:
                    candi_action[i] = max_feasible_batch  # Clamp batch to feasible amount

                self.logger.debug(f"---- Task {i+1} clamped to {candi_action[i]:.3f} ----")
                self.env_spec_log["Penalties/inv_lb"] += 1
            else:
                # Action is feasible, keep the scaled batch as predicted by the agent
                candi_action[i] = scaled_batch

        self.logger.debug(f'---- Action after inv based sanitization : {candi_action} ----')

        # First pass: equipment feasibility using current inventory
        for i in range(self.num_tasks):            
            equip_ids = self.task_equipments[i]
            if any(duplicate_inventory[self.resources.index(e)] <= 0 for e in equip_ids):
                self.env_spec_log["Penalties/equip_lb"] += 1
                continue
            else:
                # Reduce equipments to track combinationally infeasible tasks
                final_action[i] = candi_action[i]
                for e in equip_ids:
                    duplicate_inventory[self.resources.index(e)] -= 1 

        self.logger.debug(f'---- Action after equipment-based sanitization : {final_action} ----')

        return final_action

    
    def step(self, action) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Step function that updates the state based on the action 
            - Sanitization the action based on bounds
            - Computes inventory change for all resources (materials : _compute_resource_change)
            - Computes equipment unit change for all equipments ()

        Inputs:
            - action (ndarray)
        Outputs:
            - new state, reward - cost (feasibility checks), truncated (bool), terminated (bool) (ndarray)
    
        """
        self._deliver_products()
        self.logger.debug(f'---- Raw action: {action} ----')
        sanitized_action = self.sanitize_action(action)
        self.logger.debug(f'---- Final sanitized action: {sanitized_action} ----')
        sanit_cost = self._compute_sanitization_cost(action, sanitized_action)

        #_, reward, cost, terminated, truncated = self._transition(sanitized_action)

        
        # Start with the current inventory
        new_inventory = self.inventory.copy()

        new_inventory = self._schedule_products(new_inventory, sanitized_action)

        # Calculate change of each resource
        resource_change = self._compute_resource_change(sanitized_action)
        # Update inventory
        new_inventory += resource_change

        # Feasibility checks.
        # Important as production from buffer might cause inv surplus
        inv_ok = self._check_inventory_bounds(new_inventory)
        feasible = inv_ok 

        if not feasible:
            self.cost = self._compute_cost(self.inventory)
            self.reward = self._compute_reward(self.inventory, sanitized_action)  # Heavy penalty.
            truncated = False
            new_inventory = self._fix_inventory(new_inventory) # Bound the inventories between lower and upper bounds.
        else:
            # For each executed task, update equipment consumption.
           
                # For every equipment required, consume one unit immediately.
            for i, batch in enumerate(sanitized_action):
                if batch == 0:
                    continue
                
                tau = self.task_taus[i]
                
                for eq in self.task_equipments[i]:
                    eq_idx = self.resources.index(eq)
                    new_inventory[eq_idx] -= 1
                    
                    ret_time = self.t + tau
                    
                    self.delayed_production_queue.setdefault(ret_time, [])
                    self.delayed_production_queue[ret_time].append((eq, 1.0))

            # for i in range(self.num_tasks):
            #     tau = self.task_taus[i]
                
            #     if not sanitized_action[i]:
            #         continue
            #     for eq in self.task_equipments[i]:
            #         eq_idx = self.resources.index(eq)
            #         new_inventory[eq_idx] -= 1

                    # Schedule the return at t + tau.
                    # if 'equipment_returns' not in self.delayed_production_queue:
                    #     self.delayed_production_queue['equipment_returns'] = {}
                    
                    # if self.t + tau not in self.delayed_production_queue['equipment_returns']:
                    #     self.delayed_production_queue['equipment_returns'][self.t + tau] = []
                    # self.delayed_production_queue['equipment_returns'][self.t + tau].append((eq, 1.0))
            
            # State most likely to be feasible
            self.cost = 0.0
            # Calculate the reward for this action
            self.reward = self._compute_reward(new_inventory, sanitized_action)
            truncated = False

        # Update inventory for next time step
        self.inventory = new_inventory
        state = self._get_state()
        self.t += 1
        terminated = self.t >= self.T
        
        self.cost += self.sanitization_cost_weight * sanit_cost

        return (
            torch.from_numpy(state).float(),
            torch.tensor(self.reward - self.cost, dtype=torch.float32),
            torch.tensor(terminated, dtype=torch.bool),
            torch.tensor(truncated, dtype=torch.bool),
            {}
        )

    def render(self) -> Any:
        # For debugging: display the current time and inventory.
        self.logger.debug(f"Time: {self.t}, Inventory: {self.inventory}")
        return self.inventory

    def close(self) -> None:
        return None

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def spec_log(self, logger : Logger) -> None:
        for key, val in self.env_spec_log.items():
            logger.store({key : val})
            self.env_spec_log[key] = 0.0

    @property
    def max_episode_steps(self) -> int:
        return self.T  # or self.num_periods


@env_register
class SafeRTN(CMDP):
    _support_envs: ClassVar[List[str]] = ['rtn-v0']
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(env_id)
        
        self._device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        # Instantiate the environment object
        self._env = RTNEnv(env_id = env_id, **kwargs.get('env_init_config', {}))
        # Specify the action space for initialization by the algorithm layer
        self._action_space = self._env.action_space
        # Specify the observation space for initialization by the algorithm layer
        self._observation_space = self._env.observation_space

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[torch.Tensor, dict]:
        # Reset the environment
        obs, info = self._env.reset(seed=seed, options=options)
        
        # Convert the reset observations to a torch tensor.
        return (
            torch.as_tensor(obs, dtype=torch.float32, device=self._device),
            info,
        )

    def step(self, action : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # Read the dynamic information after interacting with the environment
        obs, neg_reward_minus_pos_cost, terminated, truncated, info = self._env.step(action.detach().cpu().numpy(),)

        cost = self._env.cost
        reward = neg_reward_minus_pos_cost + cost

        # Convert dynamic information into torch tensor.
        obs, reward, cost, terminated, truncated = (torch.as_tensor(x, dtype=torch.float32, device=self._device) for x in (obs, reward, cost, terminated, truncated))

        return obs, reward, cost, terminated, truncated, {}

    @property
    def max_episode_steps(self) -> int:
        # Return the maximum number of interaction steps per episode in the environment
        return self._env.T

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        # Release the environment instance after training ends
        self._env.close()

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def spec_log(self, logger: Logger) -> None:
        # Omnisafe method called at the end of each epoch. Averaged values are logged
        for key, value in self.env_spec_log.items():
            logger.store({key: float(value)})
            self.env_spec_log[key] = 0.0

    @property
    def env_spec_log(self):
        return self._env.env_spec_log