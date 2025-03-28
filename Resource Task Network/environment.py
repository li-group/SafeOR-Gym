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
class RTNEnv(CMDP):
    _support_envs: ClassVar[List[str]] = ['rtn-v0']
    _action_space: OmnisafeSpace
    _observation_space: OmnisafeSpace
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super(RTNEnv, self).__init__(env_id, **kwargs)

        print(f'---- Building environment ---- ')
       
        # Load environment configuration (e.g., horizon)
        #env_config_file = kwargs['env_cfgs']['env_config']
        env_config_file = "env_config.yaml"
        
        with open(env_config_file, 'r') as f:
            env_config_data = yaml.safe_load(f)

        # Load pickle files for the RTN graph, resources/tasks, and demand.
        rtn_graph_file = env_config_data.get('graph_file')
        with open(rtn_graph_file, 'rb') as f:
            self.rtn_graph = pickle.load(f)

        rtn_res_tasks_file = env_config_data.get('res_n_task_file')
        with open(rtn_res_tasks_file, 'rb') as f:
            self.rtn_res_tasks = pickle.load(f)

        demand_file = env_config_data.get('demand_file')
        with open(demand_file, 'rb') as f:
            self.demand = pickle.load(f)
        
        elec_cost_file = env_config_data.get('elec_cost_file')
        with open(elec_cost_file, 'rb') as f:
            self.elec_cost = pickle.load(f)

        utility_cost_file = env_config_data.get('utility_cost_file')
        with open(utility_cost_file, 'rb') as f:
            self.utility_costs = pickle.load(f)

        if self.demand is not None:
            self.num_periods = len(self.demand)
            self.horizon = self.num_periods

        print('---- Loaded all files ----')
        
        self.t = 0
        self.verbose = kwargs.get('verbose', False)
        
        # Unpack the tuple structure from RTNResourcesTasks.pickle
        self.raws_dict, self.prods_dict, self.ints_dict, self.tasks_dict, self.equipment_dict, self.utility_dict = self.rtn_res_tasks

        # Resource names by category
        self.reactants = list(self.raws_dict.keys())
        self.products = list(self.prods_dict.keys())
        self.intermediates = list(self.ints_dict.keys())
        self.resources = self.reactants + self.intermediates + self.products

        # Initial inventory and bounds
        self.initial_inventory = {
            r: (self.raws_dict.get(r) or self.prods_dict.get(r) or self.ints_dict.get(r))['X0'] for r in self.resources
        }
        self.resource_bounds = {
            r : ((self.raws_dict.get(r) or self.prods_dict.get(r) or self.ints_dict.get(r))['Xmin'],
            (self.raws_dict.get(r) or self.prods_dict.get(r) or self.ints_dict.get(r))['Xmax'])
            for r in self.resources
        }
        self.inventory = np.array([self.initial_inventory[r] for r in self.resources], dtype=np.float32)
        self.lower_bounds = np.array([self.resource_bounds[r][0] for r in self.resources], dtype=np.float32)
        self.upper_bounds = np.array([self.resource_bounds[r][1] for r in self.resources], dtype=np.float32)

        # Equipment limits
        self.equipment_limits = {
            e: self.equipment_dict[e]['Xmax'] for e in self.equipment_dict
        }

        # Task parsing
        self.task_names = list(self.tasks_dict.keys())
        self.num_tasks = len(self.task_names)
        self.tasks = []
        self.min_batch = []
        self.max_batch = []
        self.task_utilities = []
        self.task_stoichs = []  # List of dicts: {resource_name: stoich_coeff}
        self.task_equipments = []

        # Processing time of each task
        self.task_taus = [self.tasks_dict[name].get('tau', 1) for name in self.task_names]
        self.max_tau = max(self.task_taus)
        self.delayed_production_queue = {t: [] for t in range(self.t + 1, self.t + self.max_tau + 1)}

        for task_name in self.task_names:
            task_data = self.tasks_dict[task_name]

            vmin = task_data['Vmin']
            vmax = task_data['Vmax']
            self.min_batch.append(vmin)
            self.max_batch.append(vmax)

            # Equipment used
            self.task_equipments.append(task_data['equipments'])

            # Build stoichiometric vector for this task
            stoich = {r: 0.0 for r in self.resources}

            # Raw consumption
            for r in task_data['raws']:
                stoich[r] -= task_data['raw_dist'][r]

            # Intermediate reactants
            for r in task_data['int_react']:
                stoich[r] -= task_data['int_react_dist'][r]

            # Intermediate products
            for r in task_data['int_prod']:
                stoich[r] += task_data['int_prod_dist'][r]

            # Final products
            for r in task_data['prods']:
                stoich[r] += task_data['prod_dist'][r]

            self.task_stoichs.append(stoich)

            # Cost (if not provided, assume proportional to batch size)
            self.task_utilities.append(task_data.get('utilities', {}))  # placeholder, or infer based on energy, etc.
            
        self.min_batch = np.array(self.min_batch, dtype=np.float32)
        self.max_batch = np.array(self.max_batch, dtype=np.float32)

        # Define the action space
        self.raw_action_space = gym.spaces.Dict({
            "run": gym.spaces.MultiBinary(self.num_tasks),
            "batch": gym.spaces.Box(low=self.min_batch, high=self.max_batch, dtype=np.float32)
        })
        self._action_space = gym.spaces.utils.flatten_space(self.raw_action_space)


        # Define the observation space
        self.raw_observation_space = gym.spaces.Dict({
        "inventory": gym.spaces.Box(low=self.lower_bounds, high=self.upper_bounds, dtype=np.float32),
        "demand": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.products),), dtype=np.float32),
        "t": gym.spaces.Box(low=0, high=self.horizon, shape=(), dtype=np.int32),
        "pending_outputs": gym.spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(self.max_tau * len(self.resources),),
            dtype=np.float32
        )})
        self._observation_space = gym.spaces.utils.flatten_space(self.raw_observation_space)
        

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[torch.Tensor, dict]:
        self.t = 0
        self.terminated = False
        self.truncated = False
        
        if seed is not None:
            self.set_seed(seed)
        
        # Reset inventory to initial levels.
        self.inventory = np.array([self.initial_inventory[r] for r in self.resources], dtype=np.float32)
        current_demand = self._get_current_demand()
        state = self._get_state(current_demand)
        
        return torch.from_numpy(state).float(), {}

    def _get_current_demand(self) -> np.ndarray:
        # The demand pickle is assumed to be a dict with time step keys mapping to a dict of product demands.
        demand_t = self.demand.get(self.t, {r: 0.0 for r in self.products})
        demand_vec = np.array([demand_t.get(prod, 0.0) for prod in self.products], dtype=np.float32)
        
        return demand_vec

    def _get_state(self, current_demand: np.ndarray) -> np.ndarray:
        # Inventory
        state_parts = [self.inventory]

        # Demand
        state_parts.append(current_demand)

        # Time
        state_parts.append(np.array([self.t], dtype=np.float32))

        # Pending outputs — shape: (max_tau, num_resources)
        pending = np.zeros((self.max_tau, len(self.resources)), dtype=np.float32)
        for dt in range(1, self.max_tau + 1):
            delivery_time = self.t + dt
            if delivery_time in self.delayed_production_queue:
                for res, amount in self.delayed_production_queue[delivery_time]:
                    res_idx = self.resources.index(res)
                    pending[dt - 1, res_idx] += amount

        # Flatten and add to state
        state_parts.append(pending.flatten())

        # Final state vector
        return np.concatenate(state_parts)


    def _check_equipment_constraints(self, action_run: np.ndarray, action_batch: np.ndarray) -> bool:
        # Check that equipment usage does not exceed available capacity.
        equipment_usage = {}
        for i, task in enumerate(self.tasks):
            if action_run[i]:
                eq = task.get("equipment", None)
                if eq is not None:
                    equipment_usage[eq] = equipment_usage.get(eq, 0.0) + action_batch[i]
        for eq, usage in equipment_usage.items():
            limit = self.equipment_limits.get(eq, np.inf)
            if usage > limit:
                if self.verbose:
                    logging.error(f"Equipment {eq} usage {usage} exceeds limit {limit}.")
                return False
        return True

    def _check_inventory_bounds(self, new_inventory: np.ndarray) -> bool:
        if np.any(new_inventory < self.lower_bounds) or np.any(new_inventory > self.upper_bounds):
            if self.verbose:
                logging.error(f"Inventory bounds violated. Inventory: {new_inventory}, "
                              f"Lower: {self.lower_bounds}, Upper: {self.upper_bounds}")
            return False
        return True

    def _check_demand_satisfaction(self, new_inventory: np.ndarray) -> bool:
        # For products, verify that available inventory meets current demand.
        num_reactants = len(self.reactants)
        num_intermediates = len(self.intermediates)
        product_indices = range(num_reactants + num_intermediates, len(self.resources))
        current_demand = self._get_current_demand()
        prod_inventory = new_inventory[list(product_indices)]
        if np.any(prod_inventory < current_demand):
            if self.verbose:
                logging.error(f"Demand not satisfied. Product inventory: {prod_inventory}, Demand: {current_demand}")
            return False
        return True

    def _compute_resource_change(self, action_run: np.ndarray, action_batch: np.ndarray) -> np.ndarray:
        # Compute the total change in resources based on task stoichiometry.
        total_change = np.zeros_like(self.inventory, dtype=np.float32)
        
        for i, task in enumerate(self.task_stoichs):
            if action_run[i]:
                stoich_dict = task.get("stoich", {})
                stoich_vec = np.array([stoich_dict.get(r, 0.0) for r in self.resources], dtype=np.float32)
                total_change += action_batch[i] * stoich_vec
        
        return total_change

    def _compute_cost(self, action_run: np.ndarray, action_batch: np.ndarray) -> float:
        utility_prices = self.utility_costs.get(self.t, {})  # e.g. {'utility_1': 0.8, 'utility_2': 1.5}

        # Sum of operating cost for the tasks that are run.
        total_cost = 0.0
        for i, task in enumerate(self.tasks):
            if action_run[i] == 0:
                continue
            batch = action_batch[i]
            utility_usage = self.task_utilities[i]
            
            for util, uf in utility_usage.items():
                price = utility_prices.get(util, 0.0)
                total_cost += uf * price * batch
        
        return total_cost

    def _compute_reward(self, new_inventory: np.ndarray, cost: float) -> float:
        # Reward: revenue from meeting demand minus operating cost.
        revenue = 0.0
        num_reactants = len(self.reactants)
        num_intermediates = len(self.intermediates)
        
        product_indices = range(num_reactants + num_intermediates, len(self.resources))
        
        current_demand = self._get_current_demand()
        product_revenue = {prod : self.prods_dict[prod]['cost'] for prod in self.products}
        #product_revenue = self.rtn_res_tasks.get("", {prod: 1.0 for prod in self.products})
        prod_inventory = new_inventory[list(product_indices)]
        
        for idx, prod in enumerate(self.products):
            units_sold = min(prod_inventory[idx], current_demand[idx])
            revenue += units_sold * product_revenue.get(prod, 1.0)
        
        reward = revenue - cost
        
        return reward

    def _transition(self, action) -> Tuple[np.ndarray, float, float, bool, bool]:
        # Extract action details.
        action_run = np.array(action.get("run"), dtype=np.int32)
        action_batch = np.array(action.get("batch"), dtype=np.float32)
        
        # Force batch size to zero if task is not run.
        action_batch = action_run * action_batch

        resource_change = self._compute_resource_change(action_run, action_batch)
        new_inventory = self.inventory + resource_change

        # Feasibility checks.
        equip_ok = self._check_equipment_constraints(action_run, action_batch)
        inv_ok = self._check_inventory_bounds(new_inventory)
        demand_ok = self._check_demand_satisfaction(new_inventory)

        feasible = equip_ok and inv_ok and demand_ok

        if not feasible:
            cost = self._compute_cost(action_run, action_batch)
            reward = -1e6 - cost  # heavy penalty for infeasibility
            truncated = True
            new_inventory = self.inventory.copy()  # inventory remains unchanged on infeasible actions
        else:
            cost = self._compute_cost(action_run, action_batch)
            reward = self._compute_reward(new_inventory, cost)
            truncated = False
            
            # For products, subtract the demand to simulate order fulfillment.
            num_reactants = len(self.reactants)
            num_intermediates = len(self.intermediates)
            
            product_indices = range(num_reactants + num_intermediates, len(self.resources))
            
            current_demand = self._get_current_demand()
            new_inventory[list(product_indices)] -= current_demand

        # Handle delayed product deliveries due this timestep
        if self.t in self.delayed_production_queue:
            for res, amount in self.delayed_production_queue[self.t]:
                res_idx = self.resources.index(res)
                new_inventory[res_idx] += amount
            del self.delayed_production_queue[self.t]

        # Schedule new task outputs for future
        for i in range(self.num_tasks):
            if action_run[i] == 0:
                continue
            tau = self.task_taus[i]
            batch = action_batch[i]
            task_stoich = self.task_stoichs[i]
            for res, coeff in task_stoich.items():
                if coeff > 0:  # Only schedule products
                    delivery_time = self.t + tau
                    if delivery_time not in self.delayed_production_queue:
                        self.delayed_production_queue[delivery_time] = []
                    self.delayed_production_queue[delivery_time].append((res, coeff * batch))

        self.inventory = new_inventory
        self.t += 1
        terminated = self.t >= self.horizon

        return new_inventory, reward, cost, terminated, truncated

    def step(self, action) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        action_unflat = gym.spaces.utils.unflatten(self.raw_action_space, action)
        new_inventory, reward, cost, terminated, truncated = self._transition(action_unflat)
        
        current_demand = self._get_current_demand()
        state = self._get_state(current_demand)
        
        return (torch.from_numpy(state).float(),
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(cost, dtype=torch.float32),
                torch.tensor(terminated, dtype=torch.bool),
                torch.tensor(truncated, dtype=torch.bool),
                {})

    def render(self) -> Any:
        # Simple render: print time step and current inventory.
        print(f"Time: {self.t}, Inventory: {self.inventory}")
        return self.inventory

    def close(self) -> None:
        return None

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @property
    def max_episode_steps(self) -> int:
        return self.horizon  # or self.num_periods
    
    def _solve_rtn_milp(self):
        """
        Solve the full RTN MILP model using the data available in RTNEnv.
        Returns a dictionary with optimal actions and the total profit.
        """
        print("🔧 Solving RTN MILP via Pyomo + Gurobi...")

        # --- 1. Build a minimal args namespace (as expected by MILP builder) ---
        args = SimpleNamespace()
        args.horizon = self.horizon

        # --- 2. Unpack the required RTN data ---
        net = self.rtn_graph
        resources_and_task = self.rtn_res_tasks
        demand_dict = self.demand

        # Build supply_dict for reactants
        reactants = list(resources_and_task[0].keys())
        supply_dict = {
            r: {t: 0.0 for t in range(1, self.horizon + 1)}
            for r in reactants
        }

        # Convert utility cost time series into expected format
        utility_cost_dict = {
            u: {"cost": [self.utility_costs[t].get(u, 0.0) for t in range(self.horizon)]}
            for u in resources_and_task[5].keys()
        }

        # --- 3. Build and solve the Pyomo model ---
        model = init_model(args, net, resources_and_task, demand_dict, supply_dict, utility_cost_dict)
        solver = SolverFactory("gurobi")
        results = solver.solve(model, tee=False)

        # --- 4. Extract results if optimal ---
        if results.solver.termination_condition != po.TerminationCondition.optimal:
            print("❌ MILP Solver failed to find an optimal solution.")
            return None

        print("MILP Solver found an optimal solution.")

        # Extract optimal task schedule and batch sizes
        optimal_actions = {
            "run": {(i, t): po.value(model.N[i, t]) for i in model.I for t in model.T},
            "batch": {(i, t): po.value(model.E[i, t]) for i in model.I for t in model.T},
            "objective": po.value(model.obj)
        }

        return optimal_actions




'''

class RTNEnvironment(CMDP):
    _support_envs = ClassVar[list[str]] = ['rtn-v0']
    _action_space: OmnisafeSpace
    _observation_space : OmnisafeSpace
    metadata: ClassVar[dict[str, int]] = {}
    env_spec_log: dict[str, Any]
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1


    def __init__(self, env_id : str, **kwargs: Any, ) -> None:
        super().__init__(env_id)
        
        self.env_spec_log = {}
        self.verbose = False
        self.pyomo_verbose = False
        self.model_type = 'normal'

        if not self.pyomo_verbose:
            logging.getLogger('pyomo.core').setLevel(logging.ERROR)

        env_config_file = kwargs['args_config']
        with open(env_config_file, 'r') as f:
            env_config_data = yaml.safe_load(f)

        self.num_periods = env_config_data['env']['num_periods']
        self._max_episode_steps = self.num_periods
        self.horizon = self.num_periods

        # No of material nodes
        self.num_reactants = 3 #kwargs['num_reactants']
        self.num_products = 2 #kwargs['num_products']
        self.num_intermediates = 2 #kwargs['num_intermediates']

        self.num_task = 3 #kwargs['num_tasks']

        # Parameters of the MILP
        self.I_max = np.array([5.0, 5.0, 5.0])
        self.I_cost = np.array([2.0, 1.0, 4.0])
        self.M_cap = np.array([10., 10., 10., 15., 15., 8. , 8.])

        #assign_env_config(self, kwargs)

        self.min_demand = np.min(self.demand)
        self.max_demand = np.max(self.demand)

        # Initial Actions
        self.discrete_action0 = np.array([0., 0., 0.])
        self.continuous_action0 = np.array([0., 0., 0.])

        self.discrete_action_prev = np.array([0., 0., 0.])
        self.continuous_action_prev = np.array([0., 0., 0.])

        self.init_ED()
        self.reset()

        self.action_space = gym.spaces.Dict({
            'n' : gym.spaces.MultiBinary(self.num_tasks),
            'e' : gym.spaces.Box(low = np.min(self.task_min_batch_size), high = np.max(self.task_max_batch_size))
            })
         
        self._action_space = self.action_space

        self.observation_space = gym.spaces.Dict({
            "x" : gym.spaces.Box(low = np.array([0]), high = np.max(self.M_cap), shape = (), dtype = float),
            "demand" : gym.spaces.Box(low = self.min_demand, high = self.max_demand, shape = (), dtype = float),
            "t" : gym.spaces.Box(low = 0, high = self.horizon, shape = (), dtype = int)

        })
        self.observation_space = flatten_space(self.observation_space)
        self._observation_space = self.observation_space

    def define_parameters(self):
        self.num_reactants = 3 #kwargs['num_reactants']
        self.num_products = 2 #kwargs['num_products']
        self.num_intermediates = 2 #kwargs['num_intermediates']

        self.num_task = 3 #kwargs['num_tasks']

        # Parameters of the MILP
        self.I_max = np.array([5.0, 5.0, 5.0])
        self.I_cost = np.array([2.0, 1.0, 4.0])
        self.M_cap = np.array([10., 10., 10., 15., 15., 8. , 8.])

        #assign_env_config(self, kwargs)

        self.min_demand = np.min(self.demand)
        self.max_demand = np.max(self.demand)



    def init_ED(self):

        # Fill up with pyomo model declaration with required constraints etc.
        # Copy straight from MILP folder from RTNML
        self.num_reactants = 3 #kwargs['num_reactants']
        self.num_products = 2 #kwargs['num_products']
        self.num_intermediates = 2 #kwargs['num_intermediates']

        self.num_task = 3 #kwargs['num_tasks']

        # Parameters of the MILP
        self.I_max = np.array([5.0, 5.0, 5.0])
        self.I_cost = np.array([2.0, 1.0, 4.0])
        self.M_cap = np.array([10., 10., 10., 15., 15., 8. , 8.])

        #assign_env_config(self, kwargs)

        self.min_demand = np.min(self.demand)
        self.max_demand = np.max(self.demand)


    def reset(self, seed : int | None = None, options : dict[str, Any] | None = None) -> tuple[torch.Tensor, dict]:
        self.t = 0
        self.terminated = False
        self.truncated = False

        # Bus assumed to be 1 only, drop network constraints
        # physical states are always reset to the given initial values
        self.u_prev = self.u0_prev
        self.u = self.u0
        self.p_prev = self.p0_prev
        self.v = np.maximum(0, self.u - self.u_prev)
        self.w = - np.minimum(0, self.u - self.u_prev)
        self.v_UT = {i: np.zeros(self.v_UT_len[i]) for i in
                     range(self.num_gen)}  # assume no change in the past UT-1 periods, didn't accept user input
        self.w_DT = {i: np.zeros(self.w_DT_len[i]) for i in
                     range(self.num_gen)}  # assume no change in the past DT-1 periods, didn't accept user input
        # update the first element of v_UT and w_DT, as v and w at time t (t=0)
        self.v_UT, self.w_DT = self._roll_records(self.v, self.w)
        self.v_UT_arr, self.v_UT_index_mapping = self._vectorize_records(self.v_UT)
        self.w_DT_arr, self.w_DT_index_mapping = self._vectorize_records(self.w_DT)

        # Make sure the initial state is feasible (sometimes demands are sampled too high or too low)
        MAX_ITER = 20
        curr_iter = 0
        self.feasible = False

        while not self.feasible and curr_iter < MAX_ITER:
            # belief states are reset with some randomness
            self.D_forecast = self._forecast_net_load()
            # Update the ED model for t=0
            self._update_ED(self.D_forecast, self.u, self.u_prev, self.v, self.w, self.p_prev)
            # Solve the updated ED model for t=0
            opt_results = self._solve_ED()
        if self.feasible:
            # Update p and p_bar for t=0
            self.p = np.array([self.model.p[i].value for i in self.model.G_set])
            self.p_bar = np.array([self.model.p_bar[i].value for i in self.model.G_set])
            self.reward = self.cost = 0
            new_init_state = self._get_state_arr()
            return new_init_state, {}
        else:
            raise ValueError("Cannot find a feasible solution for the initial state")

    def _roll_records(self, v, w):
        v_UT = self.v_UT
        w_DT = self.w_DT
        for i in range(self.num_gen):
            if v_UT[i].size > 0:
                v_UT[i] = np.roll(v_UT[i], 1)
                v_UT[i][0] = v[i]
            if w_DT[i].size > 0:
                w_DT[i] = np.roll(w_DT[i], 1)
                w_DT[i][0] = w[i]
        return v_UT, w_DT

    @staticmethod
    def _vectorize_records(records):
        sorted_keys = sorted(records.keys())
        records_arr = np.concatenate([records[key] for key in sorted_keys])
        index_mapping = {}
        current_index = 0
        for key in sorted_keys:
            value = records[key]
            index_mapping[key] = (current_index, current_index + len(value))
            current_index += len(value)
        return records_arr, index_mapping

    @staticmethod
    def _retrieve_sub_records(key, index_mapping, records_arr):
        start, end = index_mapping[key]
        return records_arr[start:end]

    def _decide_must_on_off(self):
        """
        Determine which generators must be kept on or off for the next time period.
        """
        must_on = np.zeros(self.num_gen)
        must_off = np.zeros(self.num_gen)
        sum_v = np.array([np.sum(self.v_UT[i]) for i in range(self.num_gen)])
        sum_w = np.array([np.sum(self.w_DT[i]) for i in range(self.num_gen)])
        must_on[sum_v == self.u] = 1  # True if w_{t+1} must be 0 (sum_v == self.u means w_{t+1} must be 0)
        must_off[sum_w == (1 - self.u)] = 1  # True if v_{t+1} must be 0 (sum_w == (1 - self.u) means v_{t+1} must be 0)
        if self.NoMinUTDT.size > 0:
            must_on[self.NoMinUTDT] = 0  # UT=0 or DT=0 means no constraint
            must_off[self.NoMinUTDT] = 0  # UT=0 or DT=0 means no constraint
        return must_on, must_off

    def _enforce_MinUTDT(self, action):
        """
        Enforce an action to satisfy Minimum Up-Time and Down-Time Constraints
        """
        temp_v_new, temp_w_new = self._reckless_move(action)
        temp_v_new[self.must_off] = 0
        temp_w_new[self.must_on] = 0
        # backward derive u_new, which is the feasible action
        u_new = self.u + temp_v_new - temp_w_new
        feasible_action = u_new
        return feasible_action

    @staticmethod
    def _reckless_move(u_new, u_curr):
        """
        get v_{t+1} and w_{t+1} from u_{t+1} and u_t
        """
        v_new = np.maximum(0, u_new - u_curr)
        w_new = - np.minimum(0, u_new - u_curr)
        return v_new, w_new

    def _is_MinUTDT(self, action):
        """
        Check if an action satisfies Minimum Up-Time and Down-Time Constraints
        """
        # remember v_prev and w_prev, v and w, and u haven't been updated to t+1
        # u_new = action
        v_new, w_new = self._reckless_move(action, self.u)
        # in total, UT or DT terms
        # t+1 (new, given by reckless move), t+0 (current), t-1, ..., t-UT+2, so we need [:-1]
        # records has UT or DT terms, t+0, t-1, ..., t-UT+1
        sum_v = np.array([np.sum(self.v_UT[i][:-1]) for i in range(self.num_gen)]) + v_new
        sum_w = np.array([np.sum(self.w_DT[i][:-1]) for i in range(self.num_gen)]) + w_new
        UT_violation = np.any(sum_v > action)
        DT_violation = np.any(sum_w > (1 - action))
        if any([UT_violation, DT_violation]):
            self.feasible = False
            self.cost += 1e8
        else:
            self.feasible = True
            self.cost += 0

    def _update_ED(self, D_forecast, u, u_prev, v, w, p_prev):
        self.model.D_forecast.set_value(D_forecast)
        self.model.u.store_values({i + 1: u[i] for i in range(len(u))})
        self.model.v.store_values({i + 1: v[i] for i in range(len(v))})
        self.model.w.store_values({i + 1: w[i] for i in range(len(w))})
        self.model.u_prev.store_values({i + 1: u_prev[i] for i in range(len(u_prev))})
        self.model.p_prev.store_values({i + 1: p_prev[i] for i in range(len(p_prev))})

    def _solve_ED(self):
        results = self.solver.solve(self.model, tee=self.pyomo_verbose)
        # if infeasible, set the feasible flag to False
        if results.solver.termination_condition is not pe.TerminationCondition.optimal:
            self.feasible = False
        else:
            self.feasible = True
        return results

    def _get_state_dict(self):
        self.state = {'u': self.u,
                      'v_UT_arr': self.v_UT_arr,
                      'w_DT_arr': self.w_DT_arr,
                      'D_forecast': self.D_forecast,
                      'p': self.p,
                      't': self.t}
        
        return self.state

    def _get_state_arr(self):
        u = self.u
        v_UT, v_index_mapping = self._vectorize_records(self.v_UT)
        w_DT, w_index_mapping = self._vectorize_records(self.w_DT)
        D_forecast = np.array([self.D_forecast])
        p = self.p
        t = np.array([self.t])
        flattened_state = np.concatenate([u, v_UT, w_DT, D_forecast, p, t])
        return torch.from_numpy(flattened_state)

    def _transition(self, action):
        self._is_MinUTDT(action)
        # Update the time to t+1
        self.t += 1

        if self.feasible:
            # D_forecast is at t+1, no update so far
            D_forecast = self.D_forecast
            # Update u for t+1
            u = action
            # Update u_prev for t+1
            u_prev = self.u
            # Update v and w for t+1
            v, w = self._reckless_move(u, u_prev)
            # Update p_prev for t+1
            p_prev = self.p
            # Update the ED model for t+1
            self._update_ED(D_forecast, u, u_prev, v, w, p_prev)
            # Solve the updated ED model for t+1
            opt_results = self._solve_ED()
            
            if self.feasible:
                # Update p and p_bar for t+1
                self.p = np.array([self.model.p[i].value for i in self.model.G_set])
                self.p_bar = np.array([self.model.p_bar[i].value for i in self.model.G_set])
                # Update D_forecast to t+2
                if self.t < self.num_periods:
                    self.D_forecast = self._forecast_net_load()
                # Accept the update u for t+1
                self.u = u
                # Accept the update u_prev for t+1
                self.u_prev = u_prev
                # Accept the update v and w for t+1
                self.v = v
                self.w = w
                # Update v_UT and w_DT for t+1, vectorize them
                self.v_UT, self.w_DT = self._roll_records(v, w)
                self.v_UT_arr, self.v_UT_index_mapping = self._vectorize_records(self.v_UT)
                self.w_DT_arr, self.w_DT_index_mapping = self._vectorize_records(self.w_DT)
                # Accept the update p_prev for t+1
                self.p_prev = p_prev

    def _get_reward(self):
        if self.feasible:
            total_cost = self.model.c_total_imp.value - self.model.crelax_imp.value
            self.reward = - total_cost
        else:
            self.reward = - 1e9

        return self.reward

    def _get_cost(self):
        self.cost = self.model.crelax_imp.value
        return self.cost

    def _get_terminated(self):
        if self.t >= self.num_periods:
            self.terminated = True
        else:
            self.terminated = False

        return self.terminated

    def _get_truncated(self):
        if not self.feasible:
            self.truncated = True
        else:
            self.truncated = False

        return self.truncated

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self._transition(action)
        
        new_state = self._get_state_arr()
        new_reward = self._get_reward()
        new_cost = self._get_cost()
        
        is_terminated = self._get_terminated()
        is_truncated = self._get_truncated()
        
        return new_state, torch.as_tensor(self.reward), torch.as_tensor(self.cost), torch.as_tensor(self.terminated), torch.as_tensor(self.truncated), {}

    def logg(self, *args):
        if self.verbose:
            print(*args)

        return

    @property
    def max_episode_steps(self) -> int:
        return 24

    def spec_log(self, logger: Logger) -> None:
        return None

    def set_seed(self, seed: int) -> None:
        random.seed(seed)

    def render(self) -> Any:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def close(self) -> None:
        return None

        
'''
