import json
import pickle
import logging
import random
import yaml
from typing import Any, ClassVar, List, Tuple, Optional, Dict
from types import SimpleNamespace

import numpy as np

import torch
import pyomo.environ as po
from pyomo.opt import SolverFactory

import gymnasium as gym
from gymnasium.spaces.utils import flatten_space

import omnisafe
from omnisafe.typing import OmnisafeSpace
from omnisafe.envs.core import CMDP, env_register
from omnisafe.common.logger import Logger

from utils import init_model

@env_register
class STNEnv(CMDP):
    _support_envs: ClassVar[List[str]] = ['stn-v0']
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(env_id, **kwargs)

        # … [load graph, stn_data, demand, utilities exactly as before] …

        # build self.min_batch, self.max_batch, self.task_taus, self.task_stoichs …  

        # now action space is *only* batch sizes:
        self._action_space = gym.spaces.Box(
            low=self.min_batch, high=self.max_batch, dtype=np.float32
        )

        # same obs space as before:
        self._observation_space = flatten_space(self.raw_observation_space)

        # sanitization weight
        self.sanitization_cost_weight = 10.0

    def sanitize_action(self, raw_batch: np.ndarray) -> np.ndarray:
        # zero out any tiny batch <1e-4
        batch = raw_batch.copy()
        batch[np.abs(batch) < 1e-4] = 0.0
        # clamp to [min_batch, max_batch]
        return np.clip(batch, self.min_batch, self.max_batch)

    def _compute_sanitization_cost(self, raw: np.ndarray, final: np.ndarray) -> float:
        # count how many entries were changed
        return float((np.abs(raw - final) > 1e-4).sum())



@env_register
class STNEnv(CMDP):
    _support_envs: ClassVar[List[str]] = ['stn-v0']
    _action_space: OmnisafeSpace
    _observation_space: OmnisafeSpace
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super(STNEnv, self).__init__(env_id, **kwargs)
        """
        Function to initialise all the environment parameters

        Inputs:
            - env_id : str

        Outputs:
            - None : initializes the environment state
        
        """

        self.config_file = "structured_environment_data.json"
        self.debug = kwargs.get('debug', False)
        self.sanitization_cost_weight = 1e1
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.env_spec_log = {"Penalties/inv_lb" : 0, "Penalties/inv_ub" : 0, "Penalties/equip_lb" : 0}

        if self.debug:
            logging.basicConfig(level = logging.DEBUG)
            self.logger.setLevel(logging.DEBUG)
            self.logger.debug('---- Enabling debugging ----')
        else:
            self.logger.setLevel(logging.WARNING)
        
        self.logger.debug('---- Building environment ----')
        self.load_data_from_file()

        if self.demand is not None:
            self.num_periods = len(self.demand[list(self.demand.keys())[0]])
            self.T = self.num_periods
            self.logger.debug(f'---- Setting the horizon to {self.T} ----')

        self.logger.debug('---- Loaded all files ----')

        self.t = 0

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

        # Initial storage
        self.initial_storage = {
            r: (self.raws_dict.get(r) or self.prods_dict.get(r) or self.ints_dict.get(r))['X0'] for r in self.resources
        }
        self.resource_bounds = {
            r: ((self.raws_dict.get(r) or self.prods_dict.get(r) or self.ints_dict.get(r))['Xmin'],
                (self.raws_dict.get(r) or self.prods_dict.get(r) or self.ints_dict.get(r))['Xmax'])
            for r in self.resources
        }
        self.state_storage = np.array([self.initial_storage[r] for r in self.resources], dtype=np.float32)
        self.lower_bounds = np.array([self.resource_bounds[r][0] for r in self.resources], dtype=np.float32)
        self.upper_bounds = np.array([self.resource_bounds[r][1] for r in self.resources], dtype=np.float32)

        self.equipment_limits = {e: self.equipment_dict[e]['Xmax'] for e in self.equipment_dict}

        # Task setup
        self.task_names = list(self.tasks_dict.keys())
        self.num_tasks = len(self.task_names)
        self.min_batch = []
        self.max_batch = []
        self.task_utilities = []
        self.task_stoichs = []
        self.task_taus = []
        self.max_tau = 0

        for task_name in self.task_names:
            task_data = self.tasks_dict[task_name]
            self.min_batch.append(task_data['Vmin'])
            self.max_batch.append(task_data['Vmax'])
            self.task_utilities.append(task_data.get('utilities', {}))
            self.task_taus.append(task_data.get('tau', 1))
            stoich = {r: 0.0 for r in self.resources}
            for r in task_data['raws']:
                stoich[r] -= task_data['raw_dist'][r]
            for r in task_data['int_react']:
                stoich[r] -= task_data['int_react_dist'][r]
            for r in task_data['int_prod']:
                stoich[r] += task_data['int_prod_dist'][r]
            for r in task_data['prods']:
                stoich[r] += task_data['prod_dist'][r]
            self.task_stoichs.append(stoich)

        self.max_tau = max(self.task_taus)
        self.delayed_queue = {t: [] for t in range(self.t + 1, self.t + self.max_tau + 1)}
        self.min_batch = np.array(self.min_batch, dtype=np.float32)
        self.max_batch = np.array(self.max_batch, dtype=np.float32)

        # Action & Observation spaces
        self._action_space = gym.spaces.Box(low=self.min_batch, high=self.max_batch, dtype=np.float32)

        self.raw_observation_space = gym.spaces.Dict({
            "inventory": gym.spaces.Box(low=self.lower_bounds, high=self.upper_bounds, shape=(len(self.resources),), dtype=np.float32),
            "pending_outputs": gym.spaces.Box(low=0.0, high=np.inf, shape=(self.max_tau * (len(self.resources) - len(self.reactants)),), dtype=np.float32),
            "demand": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.T * len(self.products),), dtype=np.float32)
        })
        self._observation_space = flatten_space(self.raw_observation_space)

    def load_data_from_file(self):
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

        self.state_storage = np.array([self.initial_storage[r] for r in self.resources], dtype=np.float32)
        self.delayed_queue = {t: [] for t in range(self.t + 1, self.t + self.max_tau + 1)}
        state = self._get_state(self._get_current_demand())
        
        return torch.from_numpy(state).float(), {}

    def _get_current_demand(self) -> np.ndarray:
        return np.array([self.demand.get(self.t, {}).get(prod, 0.0) for prod in self.products], dtype=np.float32)


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
        # Current state has the inventory
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
        state_parts.append(pending.flatten())

        # State has the future demand (+ padding) finally
        state_parts.append(self._get_padded_future_demand())

        return np.concatenate(state_parts)

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

    def step(self, action) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # 1) sanitize
        raw = action.astype(np.float32)
        batch = self.sanitize_action(raw)
        sanit_cost = self._compute_sanitization_cost(raw, batch)

        # 2) simulate same as before, but now only batch vector
        #    build state_change, schedule delayed outputs, etc…
        new_storage = self.state_storage.copy()
        # … [your existing loop that applies negative stoich immediate, positive stoich into delayed_queue] …

        # 3) apply deliveries
        # … [your existing code] …

        # 4) feasibility check on new_storage
        inv_ok = np.all(new_storage >= self.lower_bounds) and np.all(new_storage <= self.upper_bounds)

        if not inv_ok:
            # **RTN‐style**: do *not* truncate, revert storage, compute cost/reward at old state
            cost = self._compute_cost(self.state_storage)
            reward = self._compute_reward(self.state_storage, batch)
            truncated = False
            # leave new_storage unchanged
        else:
            # feasible branch
            cost = self._compute_cost(new_storage)
            reward = self._compute_reward(new_storage, batch)
            truncated = False
            # fulfill demand and advance storage
            # … [subtract demand from new_storage] …

        # 5) update for next step
        self.state_storage = new_storage
        self.t += 1
        terminated = self.t >= self.horizon

        # 6) add sanitization cost into total cost
        cost += self.sanitization_cost_weight * sanit_cost

        # 7) build next obs
        obs = self._get_state(self._get_current_demand())
        return (
            torch.from_numpy(obs).float(),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(cost, dtype=torch.float32),
            torch.tensor(terminated, dtype=torch.bool),
            torch.tensor(truncated, dtype=torch.bool),
            {}
        )

    def _check_inventory_bounds(self, new_inventory: np.ndarray) -> bool:
        if np.any(new_inventory < self.lower_bounds) or np.any(new_inventory > self.upper_bounds):
            violated = [(self.resources[i], new_inventory[i], self.lower_bounds[i], self.upper_bounds[i]) for i in range(len(self.resources)) if new_inventory[i] < self.lower_bounds[i] or new_inventory[i] > self.upper_bounds[i]]
            
            self.logger.debug(f'---- Inventory bounds violated at ----')
            for res, val, lb, ub in violated:
                self.logger.debug(f"---- {res}: {val:.3f} (bounds: {lb:.3f}–{ub:.3f}) ----")
            
            self.env_spec_log["Penalties/inv_ub"] += 1
            return False
        
        return True

    def _fix_inventory(self, new_inventory: np.ndarray) -> np.ndarray:
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

    def _transition(self, action) -> Tuple[np.ndarray, float, float, bool, bool]:
        """
        Transition function that updates the state based on the action 
            - Computes inventory change for all resources (materials : _compute_resource_change)
            - Computes equipment unit change for all equipments ()

        Inputs:
            - (sanitized) action (ndarray)
        Outputs:
            - new state, reward, cost (feasibility checks), truncated (bool), terminated (bool) (ndarray)
    
        """
        # Start with the current inventory
        new_inventory = self.inventory

        # Handle delayed product deliveries (including any returns that are scheduled for t).
        if self.t in self.delayed_production_queue:
            # There are some products at the next time step coming in
            for res, amount in self.delayed_production_queue[self.t]:
                # Add the incoming amount for every product id
                res_idx = self.resources.index(res)
                new_inventory[res_idx] += amount
            
            # Remove the product delivery at time t to ensure multiple products aren't added
            del self.delayed_production_queue[self.t]
        
        # Increase equipment units if task is getting finished at this time step
        if 'equipment_returns' in self.delayed_production_queue:
            if self.t in self.delayed_production_queue['equipment_returns']:
                # If there are some units going to be free at the next time step
                for eq, amount in self.delayed_production_queue['equipment_returns'][self.t]:
                    eq_idx = self.resources.index(eq)
                    new_inventory[eq_idx] += amount
                del self.delayed_production_queue['equipment_returns'][self.t]

        # Schedule production of products in a buffer
        for i in range(self.num_tasks):
            tau = self.task_taus[i]
            batch = action[i]
            task_stoich = self.task_stoichs[i]
            
            for res, coeff in task_stoich.items():
                # Products get added to production queue
                if coeff > 0:  
                    delivery_time = self.t + tau
                    # Check if other tasks ending at that time
                    if delivery_time not in self.delayed_production_queue:
                        self.delayed_production_queue[delivery_time] = []
                    
                    self.delayed_production_queue[delivery_time].append((res, coeff * batch))

        # Calculate change of each resource
        resource_change = self._compute_resource_change(action)
        # Update inventory
        new_inventory += resource_change

        # Feasibility checks.
        #equip_ok = self._check_equipment_constraints(action)
        # Important as production from buffer might cause inv surplus
        inv_ok = self._check_inventory_bounds(new_inventory)
        #demand_ok = self._check_demand_satisfaction(new_inventory)
        feasible = inv_ok #and demand_ok

        if not feasible:
            cost = self._compute_cost(self.inventory)
            reward = self._compute_reward(self.inventory, action)  # Heavy penalty.
            truncated = False
            new_inventory = self._fix_inventory(new_inventory) # Bound the inventories between lower and upper bounds.
        else:
            # For each executed task, update equipment consumption.
            for i in range(self.num_tasks):
                tau = self.task_taus[i]
                
                if not action[i]:
                    continue
                # For every equipment required, consume one unit immediately.
                for eq in self.task_equipments[i]:
                    eq_idx = self.resources.index(eq)
                    new_inventory[eq_idx] -= 1

                    # Schedule the return at t + tau.
                    if 'equipment_returns' not in self.delayed_production_queue:
                        self.delayed_production_queue['equipment_returns'] = {}
                    
                    if self.t + tau not in self.delayed_production_queue['equipment_returns']:
                        self.delayed_production_queue['equipment_returns'][self.t + tau] = []
                    self.delayed_production_queue['equipment_returns'][self.t + tau].append((eq, 1.0))
            
            # State most likely to be feasible
            cost = 0.0
            # Calculate the reward for this action
            reward = self._compute_reward(new_inventory, action)
            truncated = False

        # Update inventory for next time step
        self.inventory = new_inventory
        self.t += 1
        terminated = self.t >= self.T

        return new_inventory, reward, cost, terminated, truncated


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
        self.logger.debug(f'')
        action_dict = unflatten(self.raw_action_space, action)
        run, batch = np.array(action_dict["run"], dtype=np.int32), np.array(action_dict["batch"], dtype=np.float32)
        batch = run * batch

        state_change = np.zeros_like(self.state_storage)
        for i in range(self.num_tasks):
            if run[i]:
                for res, coeff in self.task_stoichs[i].items():
                    idx = self.resources.index(res)
                    if coeff < 0:
                        state_change[idx] += coeff * batch[i]
                    else:
                        delivery_t = self.t + self.task_taus[i]
                        if delivery_t not in self.delayed_queue:
                            self.delayed_queue[delivery_t] = []
                        self.delayed_queue[delivery_t].append((res, coeff * batch[i]))

        # Apply incoming deliveries
        if self.t in self.delayed_queue:
            for res, amt in self.delayed_queue[self.t]:
                state_change[self.resources.index(res)] += amt
            del self.delayed_queue[self.t]

        new_storage = self.state_storage + state_change
        feasible = (
            np.all(new_storage >= self.lower_bounds) and
            np.all(new_storage <= self.upper_bounds)
        )

        if not feasible:
            reward, cost = -1e6, 1e6
            new_storage = self.state_storage.copy()
            truncated = True
        else:
            cost = self._compute_cost(run, batch)
            reward = self._compute_reward(new_storage, cost)
            truncated = False
            # Fulfill demand
            prod_start = len(self.reactants) + len(self.intermediates)
            demand = self._get_current_demand()
            new_storage[prod_start:] -= demand

        self.state_storage = new_storage
        self.t += 1
        terminated = self.t >= self.horizon
        state = self._get_state(self._get_current_demand())

        return (
            torch.from_numpy(state).float(),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(cost, dtype=torch.float32),
            torch.tensor(terminated, dtype=torch.bool),
            torch.tensor(truncated, dtype=torch.bool),
            {}
        )

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

    @property
    def max_episode_steps(self) -> int:
        return self.horizon

    def render(self) -> Any:
        print(f"[t={self.t}] Storage: {self.state_storage}")
        return self.state_storage

    def close(self) -> None:
        return

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @property
    def max_episode_steps(self) -> int:
        return self.horizon  # or self.num_periods


    

