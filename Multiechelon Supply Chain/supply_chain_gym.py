'''
Sai Madhukiran Kompalli

Multi-Echelon Inventory Management Environment

'''

import numpy as np
import gymnasium as gym
from gymnasium.spaces.utils import flatten_space
from utils import assign_env_config, flatten_and_track_mappings
import torch
import random


class InvMgmtEnv(gym.Env):
    """
    ------------------------------------------------------------------
    Multi-Echelon Inventory-Management Environment
    ------------------------------------------------------------------

    Problem Description
       This environment represents a five-layer supply chain containing
       markets, retailers, distributors, producers, and raw-material
       distributors.  During every period the decision-maker chooses
       continuous reorder quantities on each transportation route.  
       Orders travel through fixed lead times, replenish on-hand stock,
       satisfy stochastic customer demand, and may create backlogs.
       After receiving the orders, the simulator
         • updates all on-hand and pipeline inventories,  
         • realises market demand and fulfils it if inventory is available,  
         • carries forward any unfulfilled demand as backlog, and  
         • computes the net profit (revenues minus costs and penalties)  
       which is returned as the step reward.
       [ add citation here ]

    Observation Space
       The flattened observation vector contains, in this order:
         • On-hand inventory for every inventory-holding node.  
         • Pipeline inventory for each reordering route, one element
           per period of that route’s lead time (first in transit, second
           in transit, …).  
         • For each retailer-to-market link: sales made this period,
           backlog carried to next period, realised demand.  
         • Scaled time index (current period divided by total horizon).
       The length of the vector equals:
         number of main nodes
         + total pipeline slots across all routes
         + three times the number of retailer routes
         + one time-index element.

    Action Space
       A flat Box whose length equals the number of reordering routes.
       The agent supplies a value in the range −1 to 1 for every route;
       the environment rescales that to a physical order quantity between
       zero and the pre-defined capacity of that route.  Values below a
       small threshold are treated as zero.

    Key Environment Parameters (all can be overridden via **kwargs)
         • T – planning horizon in periods  
         • inv_capacity – storage capacity at each node  
         • initial_inv – starting on-hand inventory  
         • inventory_holding_cost – per-unit per-period cost  
         • operating_cost and production_yield for producer nodes  
         • material_holding_cost – cost for inventory in transit  
         • unit_price – wholesale or retail selling price per route  
         • lead_times – integer shipping lags per route  
         • reordering_route_capacity – maximum order size per route  
         • demand_parameters – mean, standard deviation, seed for demand  
         • unfulfilled_utility_penalty – cost per unit of backlog  
         • P – large constant added to every quadratic penalty term  
         • eps – numerical threshold below which orders are treated as zero
    """


    def __init__(self, env_id: str = 'InvMgmt-v0', **kwargs):
        """
        Initialize the environment by setting defaults, applying overrides,
        building spaces, and resetting to the initial state.

        Parameters
        ----------
        env_id : str
            Identifier for the environment variant.
        **kwargs
            Keyword arguments to override default configuration attributes.

        """
        super().__init__()
        self.env_id = env_id

        # Default configuration parameters
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
        self.initial_inv = {1: 100, 2: 120, 3: 80, 4: 300, 5: 250, 6: 220}
        self.inventory_holding_cost = {
            1: 0.04, 2: 0.03, 3: 0.02, 4: 0.015, 5: 0.018, 6: 0.017
        }
        self.unit_price = {
            (1,0): 15.0, (2,1): 2.5, (3,1): 2.7,
            (4,2): 2.0, (5,2): 1.5, (6,2): 1.6,
            (4,3): 1.8, (6,3): 1.8,
            (7,4): 0.5, (7,5): 0.4,
            (8,5): 0.35, (8,6): 0.4
        }
        self.material_holding_cost = {
            (2,1): 0.02, (3,1): 0.03,
            (4,2): 0.02, (5,2): 0.015, (6,2): 0.015,
            (4,3): 0.02, (6,3): 0.02,
            (7,4): 0.0,  (7,5): 0.01,
            (8,5): 0.008, (8,6): 0.01
        }
        self.lead_times = {
            (2,1): 1, (3,1): 1, (4,2): 1, (5,2): 1, (6,2): 1,
            (4,3): 1, (6,3): 1, (7,4): 1, (7,5): 1, (8,5): 1, (8,6): 1
        }
        self.operating_cost = {4: 2.0, 5: 2.5, 6: 3.0}
        self.production_yield = {4: 1, 5: 1, 6: 1}
        self.unfulfilled_utility_penalty = {(1,0): 1e5}
        self.demand_parameters = {'mean': 30, 'std': 2, 'p': 0.7, 'seed': 42}
        self.inv_capacity = {1: 300, 2: 300, 3: 300, 4: 500, 5: 500, 6: 400}
        self.reordering_route_capacity = {
            (2,1): 100, (3,1): 100, (4,2): 100, (5,2): 100, (6,2): 100,
            (4,3): 100, (6,3): 100, (7,4): 100, (7,5): 100, (8,5): 100, (8,6): 100
        }
        self.j_in = {1: [2,3], 2: [4,5,6], 3: [4,6], 4: [7], 5: [7,8], 6: [8]}
        self.j_out = {1: [0], 2: [1], 3: [1], 4: [2,3], 5: [2], 6: [2,3], 7: [4,5], 8: [5,6]}
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

        # Apply external overrides
        assign_env_config(self, kwargs)

        # Small threshold and penalty factors
        self.eps = 1e-3
        self.D = 1e3
        self.P = 1e3

        # Tracking of bound violations
        self.env_spec_log = {
            'Number of Action Bound Violations': 0,
            'Penalty of Action Bound Violations': 0,
            'Number of Observation Bound Violations': 0,
            'Penalty of Observation Bound Violations': 0
        }

        # Define observation and action spaces
        obs_size = (
            len(self.main_nodes)
            + sum(self.lead_times[rt] for rt in self.reordering_routes)
            + 3 * len(self.retailer_routes)  # sales, backlog, demand
            + 1  # time index
        )
        low_obs = np.zeros(obs_size, dtype=np.float32)
        high_obs = np.full(obs_size, np.inf, dtype=np.float32)
        for idx, node in enumerate(self.main_nodes):
            high_obs[idx] = self.inv_capacity.get(node, np.inf)
        self.raw_observation_space = gym.spaces.Box(low_obs, high_obs, dtype=np.float32)
        self.observation_space = flatten_space(self.raw_observation_space)

        # act_low = np.zeros(len(self.reordering_routes), dtype=np.float32)
        # act_high = np.array([
        #     self.reordering_route_capacity[rt] for rt in self.reordering_routes
        # ], dtype=np.float32)
        # self.raw_action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)
        # self.action_space = flatten_space(self.raw_action_space)

        # actions in [-1,1]
        self.raw_action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(len(self.reordering_routes),),
            dtype=np.float32
        )
        self.action_space = flatten_space(self.raw_action_space)

        # Initialize state
        self.reset()

    def _get_state(self, mode='arr'):
        """
        Return the current state either as a dict or a flattened array.

        Parameters
        ----------
        mode : {'arr', 'dict'}
            If 'dict', returns nested state dict; otherwise returns flattened numpy array.

        Returns
        -------
        np.ndarray or dict
        """
        if mode == 'dict':
            return self.state
        flat_obs, mapping = flatten_and_track_mappings(self.state)
        self.mapping_obs = mapping
        return flat_obs

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial conditions.

        Parameters
        ----------
        seed : int or None
            Random seed for reproducibility.
        options : dict, optional
            Additional options (ignored).

        Returns
        -------
        obs : np.ndarray
            Flattened initial observation.
        info : dict
            Contains 'dict_state' (nested state) and 'terminated' flag.
        """
        if seed is not None:
            self.set_seed(seed)
        self.t = 0
        self.reward = 0.0
        self.cost = 0.0
        self.terminated = False

        self._initialize_state_arrays()
        self._build_initial_state_dict()

        obs = self._get_state()
        return obs, {'dict_state': self.state, 'terminated': self.terminated}

    def _initialize_state_arrays(self):
        """
        Set up time-series arrays for inventory levels, orders, shipments, and demand.
        """
        self.I = np.zeros((self.T+1, len(self.main_nodes)))
        self.Tt = {rt: np.zeros(self.T+1) for rt in self.reordering_routes}
        self.R = {rt: np.zeros(self.T+1) for rt in self.reordering_routes}
        self.Rp = {rt: np.zeros(self.T+1) for rt in self.reordering_routes}
        self.Ss = {rt: np.zeros(self.T+1) for rt in self.retailer_routes}
        self.Bb = {rt: np.zeros(self.T+2) for rt in self.retailer_routes}
        self.Dd = {rt: np.zeros(self.T+1) for rt in self.retailer_routes}
        # Use demand_parameters['seed'] for deterministic demand
        seed = self.demand_parameters.get('seed', None)
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = self.np_random
        self.demand = {
            rt: rng.normal(
                loc=self.demand_parameters['mean'],
                scale=self.demand_parameters['std'],
                size=self.T
            ) for rt in self.retailer_routes
        }
        for idx, node in enumerate(self.main_nodes):
            self.I[0, idx] = self.initial_inv.get(node, 0.0)

    def _build_initial_state_dict(self):
        """
        Construct the nested state dictionary for time t=0.
        """
        self.state = {
            'on_hand_inventory': {},
            'pipeline_inventory': {},
            'sales': {},
            'backlog': {},
            'demand_window': {},
            't': self.t
        }
        for idx, node in enumerate(self.main_nodes):
            self.state['on_hand_inventory'][node] = float(self.I[self.t, idx])
        for rt in self.reordering_routes:
            self.state['pipeline_inventory'][rt] = [0.0] * self.lead_times[rt]

    def sanitize_action(self, action_dict):
        """
        Zero out orders below a small threshold.

        Parameters
        ----------
        action_dict : dict
            Proposed reorder quantities per route.

        Returns
        -------
        dict
            Sanitized action dictionary.
        """
        for rt, val in action_dict.items():
            if abs(val) < self.eps:
                action_dict[rt] = 0.0
        return action_dict

    def update_dem(self):
        """
        Realize new demand for retailers at current time step.
        """
        if 1 <= self.t < self.T:
            for rt in self.retailer_routes:
                self.Dd[rt][self.t] = self.demand[rt][self.t - 1]

    def check_action_bounds_cost(self, action_dict):
        """
        Clip out-of-bound actions and accumulate penalties.
        """
        for rt in self.reordering_routes:
            val = action_dict[rt]
            if val < 0.0:
                diff = -val
                self.cost += diff**2 + self.P
                self.env_spec_log['Number of Action Bound Violations'] += 1
                self.env_spec_log['Penalty of Action Bound Violations'] += diff**2 + self.P
                action_dict[rt] = 0.0
            max_val = self.reordering_route_capacity[rt]
            if val > max_val:
                diff = val - max_val
                self.cost += diff**2 + self.P
                self.env_spec_log['Number of Action Bound Violations'] += 1
                self.env_spec_log['Penalty of Action Bound Violations'] += diff**2 + self.P
                action_dict[rt] = max_val
        return action_dict

    def check_obs_bounds_cost(self, next_obs):
        """
        Clip out-of-bound observations and accumulate penalties.

        Parameters
        ----------
        next_obs : np.ndarray
            Flattened observation to be checked.

        Returns
        -------
        np.ndarray
            Clipped observation.
        """
        low_b = self.observation_space.low
        high_b = self.observation_space.high
        for i, x in enumerate(next_obs):
            if x < low_b[i]:
                diff = low_b[i] - x
                self.cost += diff**2 + self.P
                self.env_spec_log['Number of Observation Bound Violations'] += 1
                self.env_spec_log['Penalty of Observation Bound Violations'] += diff**2 + self.P
                next_obs[i] = low_b[i]
            elif x > high_b[i]:
                diff = x - high_b[i]
                self.cost += diff**2 + self.P
                self.env_spec_log['Number of Observation Bound Violations'] += 1
                self.env_spec_log['Penalty of Observation Bound Violations'] += diff**2 + self.P
                next_obs[i] = high_b[i]
        return next_obs

    def calculate_reward(self):
        """
        Compute the net cost (holding + operating + pipeline + backlog - sales) and
        set self.total_cost accordingly.
        """
        t = self.t
        inv_cost = op_cost = pipeline_cost = backlog_penalty = total_sales = 0.0
        for idx, node in enumerate(self.main_nodes):
            inv_cost += self.I[t, idx] * self.inventory_holding_cost[node]
            if node in self.operating_cost:
                outflow = sum(self.R[(node,k)][t] for k in self.j_out.get(node, []))
                op_cost += (outflow / self.production_yield[node]) * self.operating_cost[node]
        for rt in self.reordering_routes:
            pipeline_cost += self.Tt[rt][t] * self.material_holding_cost[rt]
            total_sales += self.R[rt][t] * self.unit_price.get(rt, 0.0)
        for rt in self.retailer_routes:
            backlog_penalty += self.Bb[rt][t+1] * self.unfulfilled_utility_penalty.get(rt, 0.0)
            total_sales += self.Ss[rt][t] * self.unit_price.get(rt, 0.0)
        self.total_cost = (inv_cost + op_cost + pipeline_cost + backlog_penalty) - total_sales
    
    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @property
    def max_episode_steps(self) -> int:
        return self.T

    def step(self, raw_action):
        """
        Apply an action to the environment and advance one time step.

        Parameters
        ----------
        raw_action : dict or np.ndarray
            Reorder quantities per route.

        Returns
        -------
        obs : np.ndarray
            Next flattened observation.
        reward : float
            Negative of total cost for this step.
        terminated : bool
            True if end of horizon reached.
        truncated : bool
            Always False (no truncation logic).
        info : dict
            Contains 'dict_state' and 'terminated'.
        """
        truncated = False
        self.reward = self.cost = 0.0
        self.total_cost = 0.0

        action = raw_action.numpy() if torch.is_tensor(raw_action) else raw_action

        # If the agent gave us a flat array in [-1,1], scale it to [0,capacity].
        # If they already gave us a dict of reorders, just use it directly.
        if isinstance(action, np.ndarray):
            action_dict = {}
            for i, rt in enumerate(self.reordering_routes):
                cap = self.reordering_route_capacity[rt]
                action_dict[rt] = (action[i] + 1.0) * 0.5 * cap
        elif isinstance(action, dict):
            action_dict = action
        else:
            raise ValueError("Action must be np.ndarray or dict")
        
        action_dict = self.sanitize_action(action_dict)
        action_dict = self.check_action_bounds_cost(action_dict)


        # Advance time and update flows
        self.t += 1
        t = self.t
        for rt in self.reordering_routes:
            self.R[rt][self.t] = action_dict[rt]
            arrive = self.t - self.lead_times[rt]
            if arrive >= 1:
                self.Rp[rt][self.t] = self.R[rt][arrive]

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
                self.Ss.get((node,k), np.zeros(self.T+1))[t - 1]
                for k in self.j_out.get(node, [])
            )
            idx = node - self.num_markets
            self.I[t, idx] = self.I[t-1, idx] + inflow - sold

        # Pipeline
        for rt in self.reordering_routes:
            self.Tt[rt][t] = self.Tt[rt][t-1] - self.Rp[rt][t] + self.R[rt][t]

        # Demand update
        self.update_dem()
        print(self.Dd)

        # Retailer sales
        for node in range(self.num_markets, self.num_markets + self.num_retailers):
            avail = self.I[t, node - self.num_markets]
            for succ in self.j_out.get(node, []):
                needed = self.Dd[(node,succ)][t] + self.Bb[(node,succ)][t]
                made_sale = min(needed, avail)
                self.Ss[(node,succ)][t] = made_sale
                avail -= made_sale

        # Backlog
        for rt in self.retailer_routes:
            self.Bb[rt][t+1] = self.Bb[rt][t] + self.Dd[rt][t] - self.Ss[rt][t]

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
                idx_t = (t - lt) + i + 1
                if idx_t >= 0:
                    pipeline_vals.append(self.Tt[rt][idx_t])
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

        # 7) check_bounds_cost for observations
        clipped_obs = self.check_obs_bounds_cost(next_obs=self.flatt_state)
        self.flatt_state = clipped_obs

        # 8) check if done
        if self.t >= self.T:
            self.terminated = True
        
        self.reward =- (self.total_cost + self.cost)

        info = {"dict_state": self.state, "terminated": self.terminated}
        return self.flatt_state, self.reward, self.terminated, truncated, info
    
    def render(self, mode='human'):
        """
        Display the current time step and performance metrics.
        """
        print(f"Time step: {self.t}, Reward: {self.reward}, Cost: {self.cost}")

    def close(self):
        """
        Clean up resources (none required).
        """
        pass

# # Manual Run
# env = InvMgmtEnv()
# obs, info = env.reset()
# print("Manual rollout start...")
# for i in range(1):
#     action = env.action_space.sample()
#     obs, reward, done, truncated, info = env.step(action)
#     print(f"Step {i + 1}, obs shape={obs.shape}, reward={reward}, done={done}")
#     print(info)
#     if done:
#         obs, info = env.reset()
#         print("Episode reset")