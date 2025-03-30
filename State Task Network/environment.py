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
    _action_space: OmnisafeSpace
    _observation_space: OmnisafeSpace
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super(STNEnv, self).__init__(env_id, **kwargs)

        env_config_file = "env_config.yaml"
        with open(env_config_file, 'r') as f:
            env_config_data = yaml.safe_load(f)

        # Load core environment components
        with open(env_config_data['graph_file'], 'rb') as f:
            self.stn_graph = pickle.load(f)
        with open(env_config_data['res_n_task_file'], 'rb') as f:
            self.stn_data = pickle.load(f)
        with open(env_config_data['demand_file'], 'rb') as f:
            self.demand = pickle.load(f)
        with open(env_config_data['utility_cost_file'], 'rb') as f:
            self.utility_costs = pickle.load(f)

        self.num_periods = len(self.demand)
        self.horizon = self.num_periods
        self.t = 0

        # Unpack STN structure
        self.raws_dict, self.prods_dict, self.ints_dict, self.tasks_dict, self.equipment_dict, self.utility_dict = self.stn_data

        self.reactants = list(self.raws_dict.keys())
        self.products = list(self.prods_dict.keys())
        self.intermediates = list(self.ints_dict.keys())
        self.resources = self.reactants + self.intermediates + self.products

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
        self.raw_action_space = gym.spaces.Dict({
            "run": gym.spaces.MultiBinary(self.num_tasks),
            "batch": gym.spaces.Box(low=self.min_batch, high=self.max_batch, dtype=np.float32)
        })
        self._action_space = flatten_space(self.raw_action_space)

        self.raw_observation_space = gym.spaces.Dict({
            "state_storage": gym.spaces.Box(low=self.lower_bounds, high=self.upper_bounds, dtype=np.float32),
            "demand": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.products),), dtype=np.float32),
            "t": gym.spaces.Box(low=0, high=self.horizon, shape=(), dtype=np.int32),
            "pending_outputs": gym.spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.max_tau * len(self.resources),),
                dtype=np.float32)
        })
        self._observation_space = flatten_space(self.raw_observation_space)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[torch.Tensor, dict]:
        if seed is not None:
            self.set_seed(seed)
        self.t = 0
        self.state_storage = np.array([self.initial_storage[r] for r in self.resources], dtype=np.float32)
        self.delayed_queue = {t: [] for t in range(self.t + 1, self.t + self.max_tau + 1)}
        return torch.from_numpy(self._get_state(self._get_current_demand())).float(), {}

    def _get_current_demand(self) -> np.ndarray:
        return np.array([self.demand.get(self.t, {}).get(prod, 0.0) for prod in self.products], dtype=np.float32)

    def _get_state(self, demand: np.ndarray) -> np.ndarray:
        pending = np.zeros((self.max_tau, len(self.resources)), dtype=np.float32)
        for dt in range(1, self.max_tau + 1):
            delivery_t = self.t + dt
            if delivery_t in self.delayed_queue:
                for res, amount in self.delayed_queue[delivery_t]:
                    pending[dt - 1, self.resources.index(res)] += amount
        obs_dict = {
            "state_storage": self.state_storage,
            "demand": demand,
            "t": np.array([self.t], dtype=np.int32),
            "pending_outputs": pending.flatten()
        }
        return flatten(self.raw_observation_space, obs_dict)

    def step(self, action) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
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

    def _compute_cost(self, run: np.ndarray, batch: np.ndarray) -> float:
        prices = self.utility_costs.get(self.t, {})
        return sum(
            batch[i] * sum(prices.get(util, 0.0) * uf for util, uf in self.task_utilities[i].items())
            for i in range(self.num_tasks) if run[i]
        )

    def _compute_reward(self, storage: np.ndarray, cost: float) -> float:
        revenue = 0.0
        prod_start = len(self.reactants) + len(self.intermediates)
        demand = self._get_current_demand()
        prod_inv = storage[prod_start:]
        for i, prod in enumerate(self.products):
            units = min(prod_inv[i], demand[i])
            revenue += units * self.prods_dict[prod]['cost']
        return revenue - cost

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

