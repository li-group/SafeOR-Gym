"""
Sai Madhukiran Kompalli

Battery Operation Environment
"""

import numpy as np
import gymnasium as gym
from gymnasium.spaces.utils import flatten_space
from gymnasium.spaces import Box
from utils import assign_env_config, flatten_and_track_mappings
import random
import torch

class BatteryOperationEnv(gym.Env):
    """
    Environment for strategic battery dispatch in a transmission grid.

    Problem Description
    -------
    We model a network of buses, transmission lines, generators, and energy storage
    units (batteries) over a finite time horizon (e.g. 24 hours).  At each time step,
    the environment accepts continuous control inputs for generator outputs, battery
    charge/discharge rates, slack generation, and load‐shedding.  It then applies a
    DC power‐flow calculation and linear battery SOC update to enforce network and
    storage constraints, and computes operating costs and penalties.

    Observation Space
    -----------------
    A flat Box of length (2·N + 2·L + 1), where N = number of buses and L = number of lines:
      1. demand[n]        ∈ [0, max_demand]       (length N)
      2. normalized SOC   ∈ [0, 1]                (length N)
      3. wildfire risk    ∈ [0, 1] (normalized)   (length L)
      4. flow ratio ℓ     ∈ [–1, 1]               (length L)
      5. time index       ∈ [0, 1]                (scalar)

    Action Space
    ------------
    A flat Box of length (G + 4·N), where G = number of generators:
      1. generator outputs g[g]      ∈ [g_min[g], g_max[g]]    (length G)
      2. battery charge rates p_c[n] ∈ [0, p_c_up[n]]           (length N)
      3. battery discharge p_d[n]    ∈ [0, p_d_up[n]]           (length N)
      4. slack generation g_slack[n] ∈ [0, gslack_max]         (length N)
      5. load‐shedding p_ls[n]       ∈ [0, max_demand]         (length N)

    Parameters
    ----------
    - time_periods: planning horizon T (e.g. 24 hours)
    - Buses, TransmissionLines: network topology
    - BusGeneratorLink: mapping each generator → bus
    - PowerDemandAtBus[(n,t)]: exogenous load time‐series
    - WildfireRisk[(ℓ,t)]: exogenous risk time‐series
    - LineSusceptance, LinePowerFlowLimit, VoltageAngle bounds
    - GeneratorLower/UpperLimit
    - BatteryUpLimit[n]: max energy capacity per bus
    - BatteryInitialCharge[n]: initial SOC
    - UpperBatteryChargeRate/DischargeRate[n]
    - ChargeEfficiency η, CarryOverRate h
    - GeneratorCost[(g,j)]: polynomial cost coefficients
    - Kls, Kslack: penalties for load‐shedding and slack
    - gslack_max: hard cap on slack generation
    - P, D: large constants for penalizing bound violations

    State Transitions (step method)
    --------------------------------
    1. Clip and unpack actions into (g, p_c, p_d, g_slack, p_ls).
    2. SOC update:  
       E_t = h·E_{t−1} + η·p_c – (1/η)·p_d  
       (matches the SOC_Balance constraint).
    3. Nodal injection:  
       P_n = ∑_{g→n} g_g + g_slack_n – demand_n + p_ls_n – p_c_n + p_d_n  
       (matches the power‐balance constraint).
    4. DC power‐flow solve:  
       B_red · θ_red = P_red  (θ_ref = 0).
    5. Line flows:  
       f = –B_lines · θ, then clipped to ±flow limits.
    6. Cost calculation:  
       generator cost + Kls·∑p_ls + Kslack·∑g_slack  
       plus any bound‐violation penalties.  
       reward = –total_cost.
    7. Advance time, rebuild observation.

    Termination
    -----------
    Episode terminates when the internal time index t exceeds the horizon T.
    """

    def __init__(self, env_id: str = 'Battery-v0', **kwargs):
        super().__init__()
        self.env_id = env_id

        # 1) Default parameters
        self.time_periods = 24
        self.Buses = [1, 2, 3]
        self.Generators = [1, 2]
        self.TransmissionLines = [(1,2), (2,3)]
        self.BusGeneratorLink = {1:1, 2:2}
        self.PowerDemandAtBus = {
            (n,t): 50.0
            for n in self.Buses
            for t in range(1, self.time_periods+1)
        }
        self.WildfireRisk = {
            (ℓ,t): 0.0
            for ℓ in self.TransmissionLines
            for t in range(1, self.time_periods+1)
        }
        self.LineSusceptance       = {(1,2):10.0, (2,3):10.0}
        self.LinePowerFlowLimit    = {(1,2):100.0, (2,3):100.0}
        self.LineUpperVoltageAngle = {(1,2):0.2, (2,3):0.2}
        self.LineLowerVoltageAngle = {(1,2):-0.2,(2,3):-0.2}
        self.GeneratorUpperLimit = {1:100.0, 2:100.0}
        self.GeneratorLowerLimit = {1:0.0,   2:0.0}
        self.BatteryUpLimit    = {n: 1.0 for n in self.Buses}  
        self.BatteryInitialCharge    = {n:0.5 for n in self.Buses}
        self.UpperBatteryChargeRate  = {n:1.0 for n in self.Buses}
        self.UpperBatteryDischargeRate ={n:1.0 for n in self.Buses}
        self.ChargeEfficiency  = 0.95
        self.CarryOverRate     = 0.999958
        self.PolynomialDegree  = 2
        self.GeneratorCost     = {(1,0):0.0,(1,1):20.0,(2,0):0.0,(2,1):25.0}
        self.Kls    = 20_000
        self.Kslack = 50 * self.Kls
        self.gslack_max = 1e4     

        # penalties
        self.P = 1e3
        self.D = 1e3

        # 2) Apply overrides
        assign_env_config(self, kwargs)

        # 3) Derived sizes & indices
        self.T       = int(self.time_periods)
        self.N       = len(self.Buses)
        self.G       = len(self.Generators)
        self.L       = len(self.TransmissionLines)
        self.bus_idx = {n:i for i,n in enumerate(self.Buses)}
        self.gen_idx = {g:i for i,g in enumerate(self.Generators)}
        self.line_idx= {ℓ:i for i,ℓ in enumerate(self.TransmissionLines)}

        # 4) Precompute DC-PF matrices
        # 4a) generator→bus incidence
        self.gen_to_bus = np.zeros((self.N, self.G), dtype=np.float32)
        for g in self.Generators:
            b = self.BusGeneratorLink[g]
            self.gen_to_bus[self.bus_idx[b], self.gen_idx[g]] = 1.0

        # 4b) susceptance Laplacian (for θ solve)
        B = np.zeros((self.N, self.N), dtype=np.float32)
        for ℓ, b_sus in self.LineSusceptance.items():
            i,j = self.bus_idx[ℓ[0]], self.bus_idx[ℓ[1]]
            B[i,i] += b_sus; B[j,j] += b_sus
            B[i,j] -= b_sus; B[j,i] -= b_sus
        self.B_reduced = B[1:,1:]   # reference bus = index 0

        # 4c) line-angle incidence for flows
        C = np.zeros((self.L, self.N), dtype=np.float32)
        for ℓ in self.TransmissionLines:
            i,j = self.bus_idx[ℓ[0]], self.bus_idx[ℓ[1]]
            idx = self.line_idx[ℓ]
            C[idx,i], C[idx,j] =  1.0, -1.0
        b_vec = np.array([self.LineSusceptance[ℓ] for ℓ in self.TransmissionLines],
                         dtype=np.float32)
        self.B_lines = np.diag(b_vec) @ C

        # 5) Battery capacity vector
        self.E_cap_max = np.array([self.BatteryUpLimit[n] for n in self.Buses],
                                  dtype=np.float32)

        # 6) Build action space: [g, p_c, p_d, g_slack, p_ls]
        g_low = np.array([self.GeneratorLowerLimit[g] for g in self.Generators],
                         dtype=np.float32)
        g_up  = np.array([self.GeneratorUpperLimit[g] for g in self.Generators],
                         dtype=np.float32)
        pc_max = np.array([self.UpperBatteryChargeRate[n] for n in self.Buses],
                          dtype=np.float32)
        pd_max = np.array([self.UpperBatteryDischargeRate[n] for n in self.Buses],
                          dtype=np.float32)
        gs_max = np.full(self.N, self.gslack_max, dtype=np.float32)
        max_d  = max(self.PowerDemandAtBus.values())
        ls_max = np.full(self.N, max_d, dtype=np.float32)

        low  = np.concatenate([g_low,
                               np.zeros(self.N),
                               np.zeros(self.N),
                               np.zeros(self.N),
                               np.zeros(self.N)])
        high= np.concatenate([g_up, pc_max, pd_max, gs_max, ls_max])
        self.action_space = Box(low, high, dtype=np.float32)

        # 7) Build observation space: [d, norm_SOC, r, flow_ratio, t]
        obs_low  = np.concatenate([
            np.zeros(self.N),        # demand ≥0
            np.zeros(self.N),        # SOC norm ≥0
            np.zeros(self.L),        # risk ≥0
            -np.ones(self.L),        # flow_ratio ∈[-1,1]
            np.zeros(1)              # time ∈[0,1]
        ])
        obs_high = np.concatenate([
            np.full(self.N, max_d),
            np.ones(self.N),
            np.ones(self.L),
            np.ones(self.L),
            np.ones(1)
        ])
        self.observation_space = Box(obs_low, obs_high, dtype=np.float32)

        # 8) Violation logging
        self.env_spec_log = {
            'Number of Action Bound Violations': 0,
            'Penalty of Action Bound Violations': 0,
            'Number of Observation Bound Violations': 0,
            'Penalty of Observation Bound Violations': 0
        }

        # 9) Seed & initial reset
        self.set_seed(None)
        self.reset()

    def _get_state(self, mode: str = 'arr'):
        """
        Build the observation from self.state.

        Returns
        -------
        np.ndarray or dict
        """
        if mode == 'dict':
            return self.state
        flat_obs, mapping = flatten_and_track_mappings(self.state)
        self.mapping_obs = mapping
        return flat_obs

    def _initialize_state(self):
        """
        Set up arrays for SOC, last_flow, and demand lookup.
        """
        # initial SOC
        self.E = np.array([self.BatteryInitialCharge[n] for n in self.Buses],
                          dtype=np.float32)
        # last line flows
        self.last_flow = np.zeros(self.L, dtype=np.float32)
        # demand lookup
        self.demand = {
            (n, t): self.PowerDemandAtBus[(n, t)]
            for n in self.Buses
            for t in range(1, self.T+1)
        }
        self.max_risk = max(self.WildfireRisk.values())
        self.p_lim_vec = np.array([self.LinePowerFlowLimit[ℓ]
                                   for ℓ in self.TransmissionLines],
                                  dtype=np.float32)

    def _build_initial_state(self):
        """
        Construct nested self.state dict.
        """
        self.state = {
            'demand':       {},
            'soc':          {},
            'risk':         {},
            'flow_ratio':   {},
            't':            self.t
        }
        # period 1 values
        for i, n in enumerate(self.Buses):
            self.state['demand'][n]     = float(self.demand[(n, self.t)])
            self.state['soc'][n]        = float(self.E[i] / self.E_cap_max[i])
        for i, ℓ in enumerate(self.TransmissionLines):
            self.state['risk'][ℓ]       = float(self.WildfireRisk[(ℓ, self.t)] / (self.max_risk or 1))
            self.state['flow_ratio'][ℓ] = float(self.last_flow[i] / self.p_lim_vec[i])
        # time index
        # (stored as scalar in dict for flattening)
        self.state['t'] = float(self.t / self.T)

    def reset(self, seed=None, options=None):
        """
        Reset to t=1, initialize SOC & flows, build initial state.

        Returns
        -------
        obs : np.ndarray
        info : dict
            {'dict_state': self.state, 'terminated': False}
        """
        if seed is not None:
            self.set_seed(seed)
        self.t = 1
        self.reward = 0.0
        self.cost   = 0.0
        self.terminated = False

        self._initialize_state()
        self._build_initial_state()
        obs = self._get_state()
        return obs, {'dict_state': self.state, 'terminated': self.terminated}

    def check_obs_bounds_cost(self, obs: np.ndarray) -> np.ndarray:
        """
        Clip & penalize observation violations.
        """
        clipped = obs.copy()
        low, high = self.observation_space.low, self.observation_space.high
        for i in range(len(clipped)):
            if clipped[i] < low[i]:
                diff = low[i] - clipped[i]
                self.env_spec_log['Number of Observation Bound Violations'] += 1
                self.env_spec_log['Penalty of Observation Bound Violations'] += diff**2 + self.P
                clipped[i] = low[i]
            elif clipped[i] > high[i]:
                diff = clipped[i] - high[i]
                self.env_spec_log['Number of Observation Bound Violations'] += 1
                self.env_spec_log['Penalty of Observation Bound Violations'] += diff**2 + self.P
                clipped[i] = high[i]
        return clipped

    def sanitize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Detect any out-of-bounds action components (using the raw input),
        log & penalize those violations, then clip to [low, high].

        Parameters
        ----------
        action : np.ndarray
            Raw continuous action vector of length G+4N.

        Returns
        -------
        np.ndarray
            Clipped action vector.
        """
        low, high = self.action_space.low, self.action_space.high
        clipped = action.copy()
        # check raw action before clipping
        for i in range(len(action)):
            if action[i] < low[i]:
                diff = low[i] - action[i]
                self.env_spec_log['Number of Action Bound Violations']    += 1
                self.env_spec_log['Penalty of Action Bound Violations']   += diff**2 + self.P
                clipped[i] = low[i]
            elif action[i] > high[i]:
                diff = action[i] - high[i]
                self.env_spec_log['Number of Action Bound Violations']    += 1
                self.env_spec_log['Penalty of Action Bound Violations']   += diff**2 + self.P
                clipped[i] = high[i]
        return clipped

    def step(self, raw_action):
        """
        Advance one hour with the continuous control vector
        [g, p_c, p_d, g_slack, p_ls].

        Returns
        -------
        obs        : np.ndarray
        reward     : float   ( = – total operating cost )
        terminated : bool    (True once the horizon is exhausted)
        truncated  : bool    (always False)
        info       : dict    {'dict_state': nested-state, 'terminated': terminated}
        """
        truncated = False
        self.cost = 0.0  # reset per-step penalty pot

        # 1) parse, log & clip action in one go
        act = raw_action.numpy() if torch.is_tensor(raw_action) else raw_action
        act = self.sanitize_action(act)

        # 2) unpack slices
        G, N = self.G, self.N
        g   = act[       : G]
        pc  = act[   G   : G+N]
        pd  = act[ G+N   : G+2*N]
        gs  = act[ G+2*N : G+3*N]
        ls  = act[ G+3*N : G+4*N]

        # 3a) SOC update
        self.E = (self.CarryOverRate * self.E
                  + self.ChargeEfficiency * pc
                  - (1.0 / self.ChargeEfficiency) * pd)

        # 3b) demand & enforce p_ls ≤ demand
        dem = np.array([self.demand[(n, self.t)] for n in self.Buses],
                       dtype=np.float32)
        ls  = np.minimum(ls, dem)

        # 3c) net nodal injection
        P = (self.gen_to_bus @ g) + gs - dem + ls - pc + pd

        # 3d) DC power-flow solve (θ[0]=0)
        theta = np.zeros(self.N, dtype=np.float32)
        theta[1:] = np.linalg.solve(self.B_reduced, P[1:])

        # 3e) line flows & penalize overloads
        f = -self.B_lines @ theta
        self.last_flow = f
        over = np.abs(f) - self.p_lim_vec
        viol = over[over > 0.0]
        if viol.size > 0:
            self.cost += np.sum(viol**2) + self.P * len(viol)

        # 4) cost & reward
        gen_cost = 0.0
        for j in range(self.PolynomialDegree):
            coeffs = np.array([self.GeneratorCost.get((g_id, j), 0.0)
                               for g_id in self.Generators],
                              dtype=np.float32)
            gen_cost += np.sum(coeffs * (g ** j))
        slack_cost = self.Kslack * np.sum(gs)
        shed_cost  = self.Kls    * np.sum(ls)
        total_cost = gen_cost + slack_cost + shed_cost + self.cost
        self.reward = -total_cost

        # 5) build obs before advancing time
        self._build_initial_state()  # refresh self.state for period t
        obs = self._get_state()
        obs = self.check_obs_bounds_cost(obs)

        # 6) advance time & termination
        self.t += 1
        terminated = (self.t > self.T)

        info = {'dict_state': self.state, 'terminated': terminated}
        return obs, self.reward, terminated, truncated, info

    def render(self, mode='human'):
        """
        Print time step, SOC, and performance.
        """
        soc = np.round(self.E / self.E_cap_max, 3)
        print(f"[t={self.t-1}/{self.T}] SOC={soc}, Reward={self.reward:.2f}")

    def close(self):
        """
        No resources to clean up.
        """
        pass

    def set_seed(self, seed: int):
        """
        Set random seeds for reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)
        try:
            torch.manual_seed(seed)
        except ImportError:
            pass
