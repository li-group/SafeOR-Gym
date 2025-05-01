'''
Sai Madhukiran Kompalli

Multi-Echelon Inventory Management Environment

'''
# export PATH="$CONDA_PREFIX/bin:$PATH"


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
        self._device = 'cuda' if th.cuda.is_available() else 'cpu'

        # Assign environment ID
        self.env_id = env_id
        
        # Assign environment configuration parameters
        config_path = kwargs.pop('config_path', None)
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


        # Threshold and penalty factors
        self.eps = 1e-3

        # Penalty *factors* (one per category)
        # These will multiply diff**2
        self.penalty_factors = {
            'action':             1e4,
            'on_hand_inventory':  1e2,
            'pipeline_inventory': 1e2,
            'sales':              1e2,
            'backlog':            1e2,
            'demand_window':      1e2
        }


        # Initializing Environment Spec Log
        self.env_spec_log = {
            # action‐space violations
            'Total penalty: Action below zero':        0,
            'Num action below zero':                   0,
            'Total penalty: Action above capacity':    0,
            'Num action above capacity':               0,

            # on‐hand inventory violations
            'Total penalty: On-hand Inventory':        0,
            'Num On-hand Inventory violations':        0,

            # pipeline inventory violations
            'Total penalty: Pipeline Inventory':       0,
            'Num Pipeline Inventory violations':       0,

            # sales violations
            'Total penalty: Sales':                    0,
            'Num Sales violations':                    0,

            # backlog violations
            'Total penalty: Backlog':                  0,
            'Num Backlog violations':                  0,

            # demand‐window violations
            'Total penalty: Demand Window':            0,
            'Num Demand Window violations':            0,

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
        # obs_size = (
        #     len(self.main_nodes)
        #     + sum(self.lead_times[rt] for rt in self.reordering_routes)
        #     + 2 * len(self.retailer_routes)     # sales + backlog
        #     + self.lookahead_horizon * len(self.retailer_routes)
        #     + 1  # time-index
        #             )

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

        # act_low = np.zeros(len(self.reordering_routes), dtype=np.float32)
        # act_high = np.array([
        #     self.reordering_route_capacity[rt] for rt in self.reordering_routes
        # ], dtype=np.float32)
        # self.raw_action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)
        # self.action_space = flatten_space(self.raw_action_space)

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
        # (requires self.lookahead_horizon defined in __init__)
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

    # def update_dem(self):
    #     """
    #     Realize new demand for retailers at current time step.
    #     """
    #     if 1 <= self.t < self.T:
    #         for rt in self.retailer_routes:
    #             self.Dd[rt][self.t] = self.demand[rt][self.t - 1]

    def update_dem(self):
        """
        Populate self.Dd[rt][k] for k in [t, t+1, …, t+lookahead_horizon], clipped to [1, T].
        """
        for rt in self.retailer_routes:
            # Compute the range of periods we want to fill
            start = max(1, self.t)
            end   = min(self.T, self.t + self.lookahead_horizon)
            
            # demand[rt] is length T, indexed 0…T-1; self.Dd[rt] is length T+1, indexed 0…T
            # We want to write demand[rt][start-1 : end] into Dd[rt][start : end+1]
            self.Dd[rt][start:end+1] = self.demand[rt][start-1:end]


    def check_action_bounds_cost(self, action_dict, pens_step):
        for rt, val in action_dict.items():
            if val < 0.0:
                diff    = -val
                penalty = diff**2 * self.penalty_factors['action']
                pens_step['Total penalty: Action below zero'] += penalty
                pens_step['Num action below zero']            += 1
                action_dict[rt] = 0.0

            cap = self.reordering_route_capacity[rt]
            if val > cap:
                diff    = val - cap
                penalty = diff**2 * self.penalty_factors['action']
                pens_step['Total penalty: Action above capacity'] += penalty
                pens_step['Num action above capacity']            += 1
                action_dict[rt] = cap

        return action_dict, pens_step['Total penalty: Action below zero'] \
                           + pens_step['Total penalty: Action above capacity']
    
    def check_obs_bounds_cost(self, next_obs, pens_step):
        low, high = self.observation_space.low, self.observation_space.high
        cat_map = {
            'on_hand_inventory':  ('On-hand Inventory',  'on_hand_inventory'),
            'pipeline_inventory': ('Pipeline Inventory', 'pipeline_inventory'),
            'sales':              ('Sales',              'sales'),
            'backlog':            ('Backlog',            'backlog'),
            'demand_window':      ('Demand Window',      'demand_window'),
        }

        total_obs_penalty = 0.0
        clipped = next_obs.copy()

        for i, x in enumerate(next_obs):
            cat, key = self.mapping_obs[i]
            if cat not in cat_map:
                continue
            human, attr = cat_map[cat]
            pf          = self.penalty_factors[attr]

            if x < low[i]:
                diff    = low[i] - x
                penalty = diff**2 * pf
                pens_step[f'Total penalty: {human}']  += penalty
                pens_step[f'Num {human} violations']  += 1
                clipped[i] = low[i]
                total_obs_penalty += penalty

            elif x > high[i]:
                diff    = x - high[i]
                penalty = diff**2 * pf
                pens_step[f'Total penalty: {human}']  += penalty
                pens_step[f'Num {human} violations']  += 1
                clipped[i] = high[i]
                total_obs_penalty += penalty

        return clipped, total_obs_penalty

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
        th.manual_seed(seed)

    @property
    def max_episode_steps(self) -> int:
        return self.T

    def step(self, raw_action):
        """
        Apply an action to the environment and advance one time step,
        with per-step logging and aggregation into self.env_spec_log.
        """
        self.truncated = False

        # reset costs for this step
        self.total_cost = 0.0
        # per‐step log
        pens_step = {k: 0 for k in self.env_spec_log}

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
        action_dict, action_penalty = self.check_action_bounds_cost(action_dict, pens_step)

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

        # 5) compute pure cost
        self.calculate_reward()   # sets self.total_cost
        pure_cost = self.total_cost

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

            # ————— Updated demand_window lookahead —————
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
        clipped_obs, obs_penalty = self.check_obs_bounds_cost(flat_obs, pens_step)
        self.flatt_state = clipped_obs

        # 8) termination
        if t >= self.T:
            self.terminated = True

        # 9) compute total reward & cost for this step
        step_cost   = action_penalty + obs_penalty
        step_reward = - (pure_cost + step_cost)

        self.reward = - pure_cost
        self.reward_ep += - pure_cost
        self.cost = step_cost
        self.cost += step_cost

        # 10) summary stats
        pens_step['num_steps']         += 1
        pens_step['sum reward']        += step_reward
        pens_step['sum reward square'] += step_reward**2
        pens_step['sum cost']          += step_cost
        pens_step['sum cost square']   += step_cost**2
        if self.terminated:
            pens_step['num_episodes']  += 1

        # 11) aggregate into master log
        for k, v in pens_step.items():
            self.env_spec_log[k] += v

        flat_obs = self.flatt_state  # numpy 1-D array
        obs_tensor   = th.tensor(flat_obs,   dtype=th.float32, device=self._device)
        reward_tensor= th.tensor(- (pure_cost + step_cost), dtype=th.float32, device=self._device)
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
        cfg = json.load(open(path))

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

def main():

    base_dir   = Path(__file__).resolve().parent
    config_fp  = base_dir / "config.json"
    if not config_fp.is_file():
        raise FileNotFoundError(f"Couldn’t find config.json at {config_fp}")

    # 1) load & preprocess your JSON config
    # config = load_config("/Users/skompall/Project Files/OR Gym/config.json")   

    # 2) instantiate
    env = InvMgmtEnv(env_id='InvMgmt-v0', config_path= config_fp)

    # 3) reset returns (obs_tensor, info_dict)
    obs, info = env.reset()
    print("Manual rollout start...")
    
    for i in range(3):
        # 4) sample action (still a NumPy array)
        action = env.action_space.sample()
        
        # 5) step returns 
        #    (state_tensor, reward_tensor, done_tensor, truncated_tensor, info_dict)
        obs, reward, done, truncated, info = env.step(action)
        
        # 6) you can still do:
        print(f"Step {i+1}, obs shape={obs.shape}, reward={reward}, done={done}")

        # 7) checking `done` on a scalar BoolTensor works:
        if done:
            obs, info = env.reset()
            print("Episode reset")

if __name__ == "__main__":
    main()


