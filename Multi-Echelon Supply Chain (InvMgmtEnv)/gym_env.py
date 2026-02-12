'''
Sai Madhukiran Kompalli

Multi-Echelon Inventory Management Environment

'''
# export PATH="$CONDA_PREFIX/bin:$PATH"

# Import Libraries
import numpy as np
import gymnasium as gym
from gymnasium.spaces.utils import flatten_space
from utils import assign_env_config, flatten_and_track_mappings
import torch as th
import random
import json
import math
import json, os
from pathlib import Path


class InvMgmtEnv(gym.Env):
    '''
    Multi-Echelon Inventory Management Environment

    Problem Description
        This environment represents a five‐tier supply chain—markets, retailers,
        distributors, producers, and raw‐material suppliers—where each period the
        agent chooses continuous reorder quantities on every transportation route.
        Orders travel through fixed lead times, replenish on‐hand stock, satisfy
        stochastic customer demand, and may create backlogs. After order arrivals:
          • On‐hand and pipeline inventories are updated.
          • Market demand is realized and sales occur if inventory permits.
          • Unfulfilled demand rolls into backlogs.
          • Net profit (sales minus procurement, operating, holding, penalty costs)
            is computed as the step reward.

    State space:
        A dictionary with keys:
          - on_hand_inventory: node → current inventory level
          - pipeline_inventory: (i,j) → list of in‐transit quantities (one per lead time slot)
          - sales: (retailer, market) → units sold this period
          - backlog: (retailer, market) → unfulfilled demand carried forward
          - demand_window: (retailer, market) → lookahead demand for next periods
          - t: current time period
        Flattened into a single array for the observation space.

    Action space:
        A dict with one key:
          - reorder: continuous Box in [–1,1]^R (R = number of reordering routes)
        Each entry is linearly scaled to [0, capacity] for that route; values
        below eps are zeroed.

    Transition to next state:
        Pipeline flows advance by one period, on‐hand inventories and backlogs
        are updated based on arrivals, sales, and unmet demand, and new demand
        is sampled for the lookahead window.

    Cost:
        Quadratic penalties on actions outside [0, capacity] and on observation
        violations (inventory, pipeline, sales, backlog, demand).

    Reward:
        Net profit for the period = sales revenue
           – procurement cost
           – operating cost
           – holding cost (on‐hand + pipeline)
           – backlog penalty
        Minus any constraint penalties yields the step reward.

    Starting State:
        All on‐hand and pipeline inventories zero except initial_inv;
        backlogs and sales zero; demand window zero; t = 0.

    Termination:
        Episode ends when t reaches T.
    '''

    _CONFIG_SCHEMA = {
        "T": int,
        "num_markets": int,
        "num_retailers": int,
        "num_distributors": int,
        "num_producers": int,
        "num_raw_distributors": int,
        "initial_inv": dict,
        "inventory_holding_cost": dict,
        "unit_price": dict,
        "material_holding_cost": dict,
        "lead_times": dict,
        "operating_cost": dict,
        "production_yield": dict,
        "unfulfilled_utility_penalty": dict,
        "demand_parameters": dict,
        "inv_capacity": dict,
        "reordering_route_capacity": dict,
        "j_in": dict,
        "j_out": dict
    }

    def __init__(self, env_id: str = 'InvMgmt-v0', **kwargs):
        """
        Initialize the environment by setting defaults, applying environment configuration, resetting to the initial state, and
        building action and observation spaces.

        Parameters
        ----------
        env_id : str
            Identifier for the environment variant.
        **kwargs
            config_path : str
            (Path to the JSON configuration file)
        """
        super().__init__()
        self._device = 'cuda' if th.cuda.is_available() else 'cpu'

        # Assign environment ID
        self.env_id = env_id
        
        # Assign environment configuration parameters
        config_path = kwargs.pop('config_file', None)
        if config_path is None:
            raise ValueError("You must pass config_path in kwargs")

        raw_cfg = self.load_config(config_path)
        assign_env_config(self, raw_cfg)
        
        # Set up derived environment parameters
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
        self.num_total_nodes =  self.num_markets + self.num_retailers + self.num_distributors + self.num_producers + self.num_raw_distributors

        self.lookahead_horizon = int(math.ceil(self.T / 10))
        
        self.raw_distributors = list(range(
        self.num_markets + self.num_retailers + self.num_distributors +
        self.num_producers,
        self.num_markets + self.num_retailers + self.num_distributors +
        self.num_producers + self.num_raw_distributors
        ))



        # Threshold and penalty factors
        self.eps = 1e-3

        # Penalty *factors* (one per category)
        # These will multiply diff**2
        self.penalty_factors = {
            'action':             1e2,
            'on_hand_inventory':  1e1,
            'pipeline_inventory': 1e1,
            'sales':              1e1,
            'backlog':            1e1,
            'demand_window':      1e1
        }


        # Initializing Environment Spec Log
        self.env_spec_log = {
            # action‐space violations
            'Total penalty: Action below zero':        0,
            'Num action below zero':                   0,
            'Total penalty: Action above capacity':    0,
            'Num action above capacity':               0,

            # on‐hand inventory violations
            'Total penalty: on_hand_inventory':        0,
            'Num on_hand_inventory violations':        0,

            # pipeline inventory violations
            'Total penalty: pipeline_inventory':       0,
            'Num pipeline_inventory violations':       0,

            # sales violations
            'Total penalty: sales':                    0,
            'Num sales violations':                    0,

            # backlog violations
            'Total penalty: backlog':                  0,
            'Num backlog violations':                  0,

            # demand‐window violations
            'Total penalty: demand_window':            0,
            'Num demand_window violations':            0,

            # summary stats
            'num_steps':       0,
            'num_episodes':    0,
            'sum reward':      0,
            'sum reward square': 0,
            'sum cost':        0,
            'sum cost square': 0,
        }

        # Initialize state
        self.reset()

        # Define observation and action spaces
        obs_sample = self._get_state()
        obs_dim = len(obs_sample)

        low_obs = np.zeros(obs_dim, dtype=np.float32)
        high_obs = np.full(obs_dim, np.inf, dtype=np.float32)
        for idx, node in enumerate(self.main_nodes):
            high_obs[idx] = self.inv_capacity.get(node, np.inf)

        self.observation_space = gym.spaces.Box(
            low=low_obs,
            high=high_obs,
            shape=(obs_dim,),
            dtype=np.float32
            )

        # actions in [-1,1]
        act_dim = len(self.reordering_routes)

        # Continuous actions in [–1,1]^act_dim
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32
        )

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
        self.reward_ep = 0.0
        self.cost_ep = 0.0
        self.terminated = self.truncated =  False

        self._initialize_state_arrays()
        self._build_initial_state_dict()

        obs = self._get_state()
        obs_tensor = th.tensor(obs, dtype=th.float32, device=self._device)

        return obs_tensor, {'dict_state': self.state, 'terminated': self.terminated, 'truncated': self.truncated}

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

        # On-hand inventory
        for idx, node in enumerate(self.main_nodes):
            self.state['on_hand_inventory'][node] = float(self.I[self.t, idx])

        # Pipeline inventory: zeros for each lead-time slot
        for rt in self.reordering_routes:
            self.state['pipeline_inventory'][rt] = [0.0] * self.lead_times[rt]

        # Sales and backlog start at zero
        for rt in self.retailer_routes:
            self.state['sales'][rt]   = 0.0
            self.state['backlog'][rt] = 0.0

        # Demand window: initialize lookahead zeros
        for rt in self.retailer_routes:
            self.state['demand_window'][rt] = [0.0] * self.lookahead_horizon

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
        Populate self.Dd[rt][k] for k in [t, t+1, …, t+lookahead_horizon], clipped to [1, T].
        """
        for rt in self.retailer_routes:
            # Compute the range of periods we want to fill
            start = max(1, self.t)
            end   = min(self.T, self.t + self.lookahead_horizon)
            
            self.Dd[rt][start:end+1] = self.demand[rt][start-1:end]


    def check_action_bounds_cost(self, action_dict):
        """
        Check if actions are within bounds and apply penalties if not.
        """

        action_penalty = 0.0
        for rt, val in action_dict.items():
            if val < 0.0:
                diff    = -val
                penalty_low = diff * self.penalty_factors['action']
                action_penalty += penalty_low
                self.pens_step['Total penalty: Action below zero'] += penalty_low
                self.pens_step['Num action below zero']            += 1
                action_dict[rt] = 0.0

            cap = self.reordering_route_capacity[rt]
            if val > cap:
                diff    = val - cap
                penalty_high = diff * self.penalty_factors['action']
                action_penalty += penalty_high
                self.pens_step['Total penalty: Action above capacity'] += penalty_high
                self.pens_step['Num action above capacity']            += 1
                action_dict[rt] = cap

        return action_dict, action_penalty
    
    def check_obs_bounds_cost(self, next_obs):
        """
        Check if observations are within bounds and apply penalties if not.
        """
        low, high = self.observation_space.low, self.observation_space.high
        total_obs_penalty = 0.0
        clipped = next_obs.copy()

        for i, x in enumerate(next_obs):
            cat, _ = self.mapping_obs[i]
            # only penalize known categories
            if cat not in self.penalty_factors:
                continue
            pf = self.penalty_factors[cat]

            if x < low[i]:
                diff = low[i] - x
                penalty = diff * pf
                self.pens_step[f'Total penalty: {cat}'] += penalty
                self.pens_step[f'Num {cat} violations'] += 1
                clipped[i] = low[i]
                total_obs_penalty += penalty

            elif x > high[i]:
                diff = x - high[i]
                penalty = diff * pf
                self.pens_step[f'Total penalty: {cat}'] += penalty
                self.pens_step[f'Num {cat} violations'] += 1
                clipped[i] = high[i]
                total_obs_penalty += penalty

        return clipped, total_obs_penalty

    def calculate_reward(self):
        """
        Compute per-period profit:
          • revenue (shipments + sales)
          • minus procurement cost
          • minus operating cost
          • minus holding cost (on-hand + pipeline)
          • minus unfulfilled-demand penalty (only for t>=2)
        Returns the net profit for this period.
        """
        t = self.t

        # 1) Revenue: shipments on reordering_routes + sales on retailer_routes
        revenue = sum(
            self.R[rt][t] * self.unit_price.get(rt, 0.0)
            for rt in self.reordering_routes
            if rt[0] not in self.raw_distributors
        ) + sum(
            self.Ss[rt][t] * self.unit_price.get(rt, 0.0)
            for rt in self.retailer_routes
        )

        # 2) Procurement cost: what you pay upstream this period
        procurement = sum(
            self.R[rt][t] * self.unit_price.get(rt, 0.0)
            for rt in self.reordering_routes
        )

        # 3) Operating cost (producers only)
        operating = sum(
            (sum(self.R[(node, k)][t] for k in self.j_out.get(node, []))
             / self.production_yield[node])
            * self.operating_cost[node]
            for node in self.operating_cost
        )

        # 4) Holding cost: on-hand + pipeline
        onhand_hc = sum(
            self.I[t, idx] * self.inventory_holding_cost[node]
            for idx, node in enumerate(self.main_nodes)
        )
        pipeline_hc = sum(
            self.Tt[rt][t] * self.material_holding_cost[rt]
            for rt in self.reordering_routes
        )
        holding = onhand_hc + pipeline_hc

        # 5) Unfulfilled‐demand penalty (use backlog carried into this period)
        if t >= 2:
            backlog_pen = sum(
                self.Bb[rt][t] * self.unfulfilled_utility_penalty.get(rt, 0.0)
                for rt in self.retailer_routes
            )
        else:
            backlog_pen = 0.0

        # 6) Net profit
        return revenue - procurement - operating - holding, backlog_pen

    def set_seed(self, seed: int):
        """
        Set the random seed for reproducibility.
        Parameters
        ----------
        seed : int
            The random seed to set.
        """

        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)

    @property
    def max_episode_steps(self) -> int:
        """
        Maximum number of steps in an episode.
        Returns
        -------
        int
            Maximum number of steps in an episode.
        """
        
        return self.T

    def step(self, raw_action):
        """
        Execute one time step within the environment.
        Parameters
        ----------
        raw_action : np.ndarray or dict
            The action to take, either as a NumPy array or a dictionary.
        Returns
        -------
        obs_tensor : th.Tensor
            The observation tensor after taking the action.
        reward_tensor : th.Tensor
            The reward tensor after taking the action.
        done_tensor : th.Tensor
            A tensor indicating whether the episode has terminated.
        trunc_tensor : th.Tensor
            A tensor indicating whether the episode has been truncated.
        info : dict
            Additional information about the environment state.
        """

        self.truncated = False

        # per‐step log
        self.pens_step = {k:0 for k in self.env_spec_log.keys()}

        # 1) decode action
        action = raw_action.numpy() if th.is_tensor(raw_action) else raw_action

        if isinstance(action, np.ndarray):
            action_dict = {
                rt: (action[i] + 1.0) * 0.5 * self.reordering_route_capacity[rt]
                for i, rt in enumerate(self.reordering_routes)
            }
        elif isinstance(action, dict):
            action_dict = action
        else:
            raise ValueError("Action must be np.ndarray or dict")

        # 2) sanitize (zero‐threshold) – no logging
        action_dict = self.sanitize_action(action_dict)

        # 3) check action bounds, log into pens_step
        action_dict, action_penalty = self.check_action_bounds_cost(action_dict)

        # 4) advance time and flows
        self.t += 1
        t = self.t

        for rt in self.reordering_routes:
            self.R[rt][t] = action_dict[rt]
            arrive = t - self.lead_times[rt]
            if arrive >= 1:
                self.Rp[rt][t] = self.R[rt][arrive]

        # production/distribution
        for node in range(
            self.num_markets + self.num_retailers,
            self.num_total_nodes - self.num_raw_distributors
        ):
            inflow = sum(
                self.Rp.get((k, node), np.zeros(self.T+1))[t]
                for k in self.j_in.get(node, [])
            )
            outflow = sum(
                self.R.get((node, k), np.zeros(self.T+1))[t]
                for k in self.j_out.get(node, [])
            )
            idx = node - self.num_markets
            self.I[t, idx] = self.I[t-1, idx] + inflow - outflow

        # retailer nodes
        for node in range(self.num_markets, self.num_markets + self.num_retailers):
            inflow = sum(
                self.Rp.get((k, node), np.zeros(self.T+1))[t]
                for k in self.j_in.get(node, [])
            )
            sold = sum(
                self.Ss.get((node, k), np.zeros(self.T+1))[t-1]
                for k in self.j_out.get(node, [])
            )
            idx = node - self.num_markets
            self.I[t, idx] = self.I[t-1, idx] + inflow - sold

        # pipeline update
        for rt in self.reordering_routes:
            self.Tt[rt][t] = self.Tt[rt][t-1] - self.Rp[rt][t] + self.R[rt][t]

        # demand, sales, backlog updates
        self.update_dem()
        for node in range(self.num_markets, self.num_markets + self.num_retailers):
            avail = self.I[t, node - self.num_markets]
            for succ in self.j_out.get(node, []):
                needed = self.Dd[(node, succ)][t] + self.Bb[(node, succ)][t]
                made = min(needed, avail)
                self.Ss[(node, succ)][t] = made
                avail -= made
        for rt in self.retailer_routes:
            self.Bb[rt][t+1] = self.Bb[rt][t] + self.Dd[rt][t] - self.Ss[rt][t]

        # 5) Compute reward
        reward, backlog_pen = self.calculate_reward()   # sets self.total_cost

        # 6) rebuild state dict
        self.state["t"] = t
        for idx, node in enumerate(self.main_nodes):
            self.state["on_hand_inventory"][node] = float(self.I[t, idx])
        for rt in self.reordering_routes:
            lt = self.lead_times[rt]
            pipeline_vals = [
                self.Tt[rt][(t - lt) + i + 1] if (t - lt) + i + 1 >= 0 else 0.0
                for i in range(lt)
            ]
            self.state["pipeline_inventory"][rt] = pipeline_vals

        for rt in self.retailer_routes:
            self.state["sales"][rt]   = float(self.Ss[rt][t])
            self.state["backlog"][rt] = float(self.Bb[rt][t])

            # ————— Demand lookahead —————
            window = []
            for offset in range(self.lookahead_horizon):
                future_t = t + offset
                # Dd[rt][future_t] is only valid if 1 <= future_t <= T
                if 1 <= future_t <= self.T:
                    window.append(float(self.Dd[rt][future_t]))
                else:
                    window.append(0.0)
            self.state["demand_window"][rt] = window
            # ————————————————————————————————————————

        # 7) flatten & log obs-bound violations
        flat_obs, self.mapping_obs = flatten_and_track_mappings(self.state)
        clipped_obs, obs_penalty = self.check_obs_bounds_cost(flat_obs)

        self.flatt_state = clipped_obs

        # 9) compute cost
        cost = action_penalty + obs_penalty

        self.reward = reward - backlog_pen
        self.reward_ep += self.reward
        self.cost = cost
        self.cost += self.cost

        # 10) summary stats
        self.pens_step['num_steps']         += 1
        self.pens_step['sum reward']        += self.reward
        self.pens_step['sum reward square'] += self.reward**2
        self.pens_step['sum cost']          += self.cost
        self.pens_step['sum cost square']   += self.cost**2

        if self.t >= self.T:
            self.pens_step['num_episodes']  += 1
            self.terminated = True

        # 11) aggregate into master log
        for k in self.env_spec_log.keys():
            self.env_spec_log[k] += self.pens_step[k]

        flat_obs = self.flatt_state  # numpy 1-D array
        obs_tensor   = th.tensor(flat_obs,   dtype=th.float32, device=self._device)
        reward_tensor= th.tensor(reward - cost , dtype=th.float32, device=self._device)
        done_tensor  = th.tensor(self.terminated, dtype=th.bool,   device=self._device)
        trunc_tensor = th.tensor(self.truncated,  dtype=th.bool,   device=self._device)

        return obs_tensor, reward_tensor, done_tensor, trunc_tensor, {}
    
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

    def load_config(self, path):
        """
        Load and preprocess the JSON configuration file.
        Parameters
        ----------
        path : str
            Path to the JSON configuration file.
        Returns
        -------
        dict
            Preprocessed configuration dictionary.
        """
        
        cfg_full = json.load(open(path))

        cfg = cfg_full["env_init_cfgs"]

        # 1) integer-keyed dicts
        for name in ["initial_inv","inventory_holding_cost", "inv_capacity", "j_in", "j_out", "operating_cost", "production_yield"]:
            raw = cfg[name]
            cfg[name] = { int(k): v for k, v in raw.items() }

        # 2) tuple-keyed dicts
        def parse_tuple_dict(raw_d):
            out = {}
            for k, v in raw_d.items():
                # strip “(” and “)”, split on comma, map to int, build tuple
                nums = k.strip("()").split(",")
                tup  = tuple(int(x) for x in nums)
                out[tup] = v
            return out

        for name in [
            "unit_price",
            "material_holding_cost",
            "lead_times",
            "reordering_route_capacity",
            "unfulfilled_utility_penalty"
        ]:
            cfg[name] = parse_tuple_dict(cfg[name])

        return cfg
