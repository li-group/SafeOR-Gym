'''
Sai Madhukiran Kompalli

Grid Integrated Energy Storage Environment

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

class BatteryOperationEnv(gym.Env):
    _CONFIG_SCHEMA = {
        "time_periods": int,                             # T: total time steps
        "Buses": list,                                   # N buses
        "Generators": list,                              # G generators
        "TransmissionLines": list,                       # L lines
        "DeEnergizedLines": list,                  # static set of lines always de-energized
        "BusGeneratorLink": dict,                        # mapping g -> bus n
        "LineSusceptance": dict,                         # Bij for each line
        "LinePowerFlowLimit": dict,                      # f_max for each line
        "LineUpperVoltageAngle": dict,                   # θ_upper for each line
        "LineLowerVoltageAngle": dict,                   # θ_lower for each line
        "PowerDemandAtBus": dict,                        # mapping n -> {t: demand}
        "BatteryLowLimit": dict,                         # Emin_n
        "BatteryUpLimit": dict,                          # Emax_n
        "BatteryInitialCharge": dict,                    # E_n,0
        "LowerBatteryChargeRate": dict,                  # pc_min_n
        "UpperBatteryChargeRate": dict,                  # pc_max_n
        "LowerBatteryDischargeRate": dict,               # pd_min_n
        "UpperBatteryDischargeRate": dict,               # pd_max_n
        "ChargeEfficiency": float,                       # η
        "CarryOverRate": float,                          # γ
        "PolynomialDegree": int,                         # number of cost coefficients
        "GeneratorCost": dict,                           # mapping g -> [C0, C1, ..., C_{d-1}]
        "Kslack": float,                                 # penalty per unit slack
        "Kls": float,                                    # penalty per unit load-shedding
        "ThetaMax": float,                               # absolute angle bound
        "PhiBalance": float,                             # ϕ_bal
        "PhiThetaAction": float,                         # ϕ_θ,act (action clipping)
        "PhiPower": float,                               # ϕ_power
        "PhiCharge": float,                              # ϕ_charge
        "PhiDischarge": float,                           # ϕ_discharge
        "PhiSlack": float,                               # ϕ_slack (obs bound)
        "PhiShed": float,                                # ϕ_shed (obs bound)
        "PhiSOC": float,                                 # ϕ_soc
        "PhiTheta": float,                               # ϕ_θ (obs bound)
        "PhiFlowRatio": float,                           # ϕ_flow_ratio
        "smax": float,                                   # max slack per bus
        "dmax_global": float,                            # global max shed
        "DemandForecastHorizon": int,                     # k-hour forecast window
        "GeneratorLowerLimit": dict,    # pᵍ_min for each generator  
        "GeneratorUpperLimit": dict    # pᵍ_max for each generator

    }

    def __init__(self, env_id: str = 'Battery-v0', **kwargs):
        super().__init__()
        self.env_id = env_id
        self._device = 'cuda' if th.cuda.is_available() else 'cpu'

        # --- 1) Load and validate configuration ---
        config_path = kwargs.pop('config_path', None)
        if config_path is None:
            raise ValueError("`config_path` must be provided in kwargs")
        raw_cfg = self.load_config(config_path)
        assign_env_config(self, raw_cfg)

        # --- Forecast horizon k ---
        self.k = int(self.DemandForecastHorizon)

        # --- 2) Dimensions and indices ---
        self.Buses = list(self.Buses)
        self.Generators = list(self.Generators)
        self.TransmissionLines = list(self.TransmissionLines)
        self.T = int(self.time_periods)
        self.N = len(self.Buses)
        self.G = len(self.Generators)
        self.L = len(self.TransmissionLines)
        self.bus_idx = {b: i for i, b in enumerate(self.Buses)}
        self.gen_idx = {g: i for i, g in enumerate(self.Generators)}
        self.line_idx = {l: i for i, l in enumerate(self.TransmissionLines)}

        # Precompute generator-to-bus incidence
        self.gen_to_bus = np.zeros((self.N, self.G), dtype=np.float32)
        for g in self.Generators:
            b = self.BusGeneratorLink[g]
            self.gen_to_bus[self.bus_idx[b], self.gen_idx[g]] = 1.0

        self.GeneratorsAtBus = {
            n: [g for g, b in self.BusGeneratorLink.items() if b == n]
            for n in self.Buses
        }

        self.LinesToN = {
            n: [l for l in self.TransmissionLines if l[1] == n]
            for n in self.Buses
        }
        self.LinesFromN = {
            n: [l for l in self.TransmissionLines if l[0] == n]
            for n in self.Buses
        }


        # --- 3) Precompute static matrices ---
        self._prepare_power_flow_matrices()
        self.E_cap_min = np.array([self.BatteryLowLimit[n] for n in self.Buses], dtype=np.float32)
        self.E_cap_max = np.array([self.BatteryUpLimit[n] for n in self.Buses], dtype=np.float32)
        self.p_lim_vec = np.array([self.LinePowerFlowLimit[l] for l in self.TransmissionLines], dtype=np.float32)

        # --- 4) Penalty factors from config ---
        self.phi = {
            'balance':     self.PhiBalance,
            'theta_act':   self.PhiThetaAction,
            'power':       self.PhiPower,
            'charge':      self.PhiCharge,
            'discharge':   self.PhiDischarge,
            'slack':   self.PhiSlack,
            'shed':    self.PhiShed,
            'soc':         self.PhiSOC,
            'theta_obs':   self.PhiTheta,
            'flow_ratio':  self.PhiFlowRatio,
        }

        # --- 5) Action and observation spaces ---
        self.reset()
        self._build_action_space()
        self._build_observation_space()

        # --- 6) Loggers ---
        self.env_spec_log = {
        # action‐clip penalties
        'Total penalty: power below lower':   0.0,
        'Num power below lower':              0,
        'Total penalty: power above upper':   0.0,
        'Num power above upper':              0,
        'Total penalty: charge below lower':  0.0,
        'Num charge below lower':             0,
        'Total penalty: charge above upper':  0.0,
        'Num charge above upper':             0,
        'Total penalty: discharge below lower': 0.0,
        'Num discharge below lower':         0,
        'Total penalty: discharge above upper': 0.0,
        'Num discharge above upper':         0,
        'Total penalty: shed below lower':    0.0,
        'Num shed below lower':              0,
        'Total penalty: shed above upper':    0.0,
        'Num shed above upper':              0,
        'Total penalty: theta below lower':   0.0,
        'Num theta below lower':             0,
        'Total penalty: theta above upper':   0.0,
        'Num theta above upper':             0,
        # obs‐bounds penalties
        'Total penalty: SOC below lower':     0.0,
        'Num SOC below lower':               0,
        'Total penalty: SOC above upper':     0.0,
        'Num SOC above upper':               0,
        'Total penalty: theta below lower':   0.0,
        'Num theta below lower':             0,
        'Total penalty: theta above upper':   0.0,
        'Num theta above upper':             0,
        'Total penalty: flow_ratio below lower': 0.0,
        'Num flow_ratio below lower':       0,
        'Total penalty: flow_ratio above upper': 0.0,
        'Num flow_ratio above upper':       0,
        'Total penalty: slack below lower':  0.0,
        'Num slack below lower':            0,
        'Total penalty: slack above upper':  0.0,
        'Num slack above upper':            0,
        # summary stats
        'num_steps':         0,
        'num_episodes':      0,
        'sum reward':        0.0,
        'sum reward square': 0.0,
        'sum cost':          0.0,
        'sum cost square':   0.0,
        }

        # --- 7) Seed and reset ---
        seed = kwargs.pop('seed', None)
        if seed is not None:
            self.set_seed(seed)

    def _get_state(self, mode='arr'):
        """
        Return the current state as a flattened array or nested dict.

        Parameters
        ----------
        mode : {'arr', 'dict'}
            If 'dict', return nested state dict; else return flattened array.
        """
        if mode == 'dict':
            return self.state
        flat_obs, mapping = flatten_and_track_mappings(self.state)
        self.mapping_obs = mapping
        return flat_obs

    def _initialize_state(self):
        """
        Initialize time-series data for SOC, line flows, bus angles, slacks, and demand history.
        """
        # 1) Battery SOC at t=1
        self.E = np.array(
            [self.BatteryInitialCharge[n] for n in self.Buses],
            dtype=np.float32
        )
        # 2) No prior flows or angles
        self.last_flow = np.zeros(self.L, dtype=np.float32)
        # Store most recent bus angles (θn,t), initialize to zero
        self.theta = np.zeros(self.N, dtype=np.float32)
        # 3) Slack generation per bus
        self.slack = np.zeros(self.N, dtype=np.float32)
        # 4) Demand lookup {(n, t) → dn,t}
        self.demand = dict(self.PowerDemandAtBus)        
        self.max_demand = max(self.demand.values())


    def _build_state_dict(self):
        """
        Construct the nested state dict at the current timestep self.t,
        matching st = (SOC, Θ, Φ, f, s, D… , τ) as per §1.7.3.
        """
        t = self.t

        # — demand look-ahead window Dn,t:t+k−1 (zero-pad beyond T)
        demand_window = {}
        for i, n in enumerate(self.Buses):
            window = []
            for δ in range(self.DemandForecastHorizon):
                τ = t + δ
                window.append(self.demand.get((n, τ), 0.0))
            demand_window[n] = window

        # — normalized SOCn,t = En,t / Emax_n
        soc = {
            n: float(self.E[i] / self.E_cap_max[i])
            for i, n in enumerate(self.Buses)
        }

        # — normalized voltage-angle differences Θℓ,t
        theta_norm = {}
        for ℓ in self.TransmissionLines:
            i, j = ℓ
            raw = self.theta[self.bus_idx[i]] - self.theta[self.bus_idx[j]]
            θ_plus  = self.LineUpperVoltageAngle[ℓ]
            θ_minus = self.LineLowerVoltageAngle[ℓ]
            theta_norm[ℓ] = float(
                (2 * raw - (θ_plus + θ_minus)) / (θ_plus - θ_minus)
            )

        # — line-loading ratios Φℓ,t = fℓ,t / f_max, and raw flows fℓ,t
        flow_ratio = {}
        flow       = {}
        for idx, ℓ in enumerate(self.TransmissionLines):
            fℓ = self.p_lim_vec[idx]
            fl = self.last_flow[idx]
            flow_ratio[ℓ] = float(fl / (fℓ or 1.0))
            flow[ℓ]       = float(fl)

        # — slack sn,t per bus
        slack = {
            n: float(self.slack[self.bus_idx[n]])
            for n in self.Buses
        }

        # — normalized time τt = (t−1)/(T−1)
        tau = float((t - 1) / (self.T - 1))

        self.state = {
            'soc':           soc,
            'theta':         theta_norm,
            'flow_ratio':    flow_ratio,
            'flow':          flow,
            'slack':         slack,
            'demand_window': demand_window,
            'tau':           tau
        }

    def _prepare_power_flow_matrices(self):
        """
        Precompute matrices for DC power-flow per §1.7.2:
          1) Reduced susceptance Laplacian (for solving θ).
          2) Line susceptance-incidence matrix (for f = B_lines · θ).
        """
        # full N×N susceptance Laplacian B
        B = np.zeros((self.N, self.N), dtype=np.float32)
        for (i, j), Bij in self.LineSusceptance.items():
            ii, jj = self.bus_idx[i], self.bus_idx[j]
            B[ii, ii] += Bij
            B[jj, jj] += Bij
            B[ii, jj] -= Bij
            B[jj, ii] -= Bij
        # reduced by removing reference bus (index 0)
        self.B_reduced = B[1:, 1:]  # :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}

        # build L×N incidence C and form B_lines = diag(Bij) · C
        C = np.zeros((self.L, self.N), dtype=np.float32)
        for idx, (frm, to) in enumerate(self.TransmissionLines):
            i, j = self.bus_idx[frm], self.bus_idx[to]
            C[idx, i] =  1.0
            C[idx, j] = -1.0
        b_vec = np.array([self.LineSusceptance[line] for line in self.TransmissionLines],
                         dtype=np.float32)
        self.B_lines = np.diag(b_vec) @ C

    def _build_action_space(self):
        """
        Normalized input a_norm ∈ [−1,1]^(G+3N+(N−1)) is affinely mapped to:
          amin = (pmin, pc_min, pd_min, 0_N, -Θmax_{N-1})
          amax = (pmax, pc_max, pd_max, dmax_global·1_N, Θmax·1_{N−1})
        then clipped to bounds (§1.7.4) :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}.
        """
        # physical bounds
        pmin = np.array([self.GeneratorLowerLimit[g] for g in self.Generators], dtype=np.float32)
        pmax = np.array([self.GeneratorUpperLimit[g] for g in self.Generators], dtype=np.float32)
        pc_min = np.array([self.LowerBatteryChargeRate[n] for n in self.Buses], dtype=np.float32)
        pc_max = np.array([self.UpperBatteryChargeRate[n] for n in self.Buses], dtype=np.float32)
        pd_min = np.array([self.LowerBatteryDischargeRate[n] for n in self.Buses], dtype=np.float32)
        pd_max = np.array([self.UpperBatteryDischargeRate[n] for n in self.Buses], dtype=np.float32)
        zeros_N = np.zeros(self.N, dtype=np.float32)
        theta_max = np.full(self.N - 1, self.ThetaMax, dtype=np.float32)
        shed_max  = np.full(self.N, self.dmax_global, dtype=np.float32)

        self.amin = np.concatenate([pmin, pc_min, pd_min, zeros_N, -theta_max])
        self.amax = np.concatenate([pmax, pc_max, pd_max, shed_max,  theta_max])

        # policy sees normalized cube
        dim = self.G + 3*self.N + (self.N - 1)
        self.action_space = gym.spaces.Box(
            low  = -np.ones(dim, dtype=np.float32),
            high =  np.ones(dim, dtype=np.float32),
            dtype = np.float32
        )

    def _build_observation_space(self):
        """
        Observation vector st = [SOC, Θ, Φ, f, s, D, τ] flattened
        with bounds per §1.7.3: SOC∈[0,1]^N; Θ∈[−1,1]^L; Φ∈[−1,1]^L;
        f∈[−f_lim,f_lim]^L; s∈[0,smax]^N; D∈[0,dmax]^N×k; τ∈[0,1] :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}.
        """
        # SOC
        low_soc  = np.zeros(self.N, dtype=np.float32)
        high_soc = np.ones(self.N,  dtype=np.float32)
        # Θ and Φ
        low_theta   = -np.ones(self.L, dtype=np.float32)
        high_theta  =  np.ones(self.L,  dtype=np.float32)
        low_phi     = -np.ones(self.L, dtype=np.float32)
        high_phi    =  np.ones(self.L,  dtype=np.float32)
        # f flows
        low_flow  = -self.p_lim_vec
        high_flow =  self.p_lim_vec
        # slack s
        low_s     = np.zeros(self.N, dtype=np.float32)
        high_s    = np.full(self.N, self.smax, dtype=np.float32)
        # demand window D
        dk = self.N * self.DemandForecastHorizon
        low_D     = np.zeros(dk, dtype=np.float32)
        high_D    = np.full(dk, self.max_demand, dtype=np.float32)
        # time τ
        low_tau   = np.zeros(1, dtype=np.float32)
        high_tau  = np.ones(1,  dtype=np.float32)

        lows  = np.concatenate([low_soc, low_theta, low_phi, low_flow, low_s, low_D, low_tau])
        highs = np.concatenate([high_soc, high_theta, high_phi, high_flow, high_s, high_D, high_tau])
        self.observation_space = gym.spaces.Box(low=lows, high=highs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Reset the environment at t=1, reinitialize SOC, flows, angles, slacks, and state.

        Returns:
            obs (torch.Tensor): flattened initial observation.
            info (dict): {'dict_state', 'terminated', 'truncated'}.
        """
        if seed is not None:
            self.set_seed(seed)
        self.t = 1
        self.terminated = False
        self.truncated  = False
        self.reward = 0.0
        self.cost = 0.0
        self.reward_ep = 0.0
        self.cost_ep = 0.0

        # reinitialize dynamic variables
        self._initialize_state()
        self._build_state_dict()

        # get flattened obs and wrap as tensor
        obs = self._get_state()
        obs_tensor = th.tensor(obs, dtype=th.float32, device=self._device)

        return obs_tensor, {
            'dict_state': self.state,
            'terminated': self.terminated,
            'truncated': self.truncated
        }

    def sanitize_action(self, a_norm):
        """
        Decode normalized action a_norm ∈ [−1,1]^d to physical,
        clip to [amin,amax], and log action‐clip penalties into self.pens_step.

        Returns:
            a_phys (np.ndarray): clipped physical action.
            total_penalty (float): sum of clipping penalties this step.
        """
        # affine map back to physical range
        a_pre = 0.5 * (a_norm + 1.0) * (self.amax - self.amin) + self.amin
        a_phys = a_pre.copy()
        total_penalty = 0.0

        # define segments and their φ penalties from config
        segments = [
            ('power',     0,            self.G,                 self.phi['power']),
            ('charge',    self.G,       self.G + self.N,         self.phi['charge']),
            ('discharge', self.G + self.N, self.G + 2*self.N,    self.phi['discharge']),
            ('shed',      self.G + 2*self.N, self.G + 3*self.N,  self.phi['shed']),
            ('theta',     self.G + 3*self.N, self.G + 3*self.N + (self.N-1), self.phi['theta_act']),
        ]

        for name, start, end, phi in segments:
            low_key  = f"Total penalty: {name} below lower"
            low_cnt  = f"Num {name} below lower"
            high_key = f"Total penalty: {name} above upper"
            high_cnt = f"Num {name} above upper"

            for i in range(start, end):
                if a_pre[i] < self.amin[i]:
                    diff = self.amin[i] - a_pre[i]
                    p = diff * phi
                    total_penalty += p
                    a_phys[i] = self.amin[i]
                    self.pens_step[low_key] += p
                    self.pens_step[low_cnt] += 1

                elif a_pre[i] > self.amax[i]:
                    diff = a_pre[i] - self.amax[i]
                    p = diff * phi
                    total_penalty += p
                    a_phys[i] = self.amax[i]
                    self.pens_step[high_key] += p
                    self.pens_step[high_cnt] += 1

        return a_phys, total_penalty

    def check_obs_bounds_cost(self, obs):
        """
        Clip observations to their valid ranges and log obs‐clip penalties into self.pens_step.

        Returns:
            clipped (np.ndarray): clipped observation.
            total_penalty (float): sum of obs‐clip penalties this step.
        """
        low, high = self.observation_space.low, self.observation_space.high
        clipped = obs.copy()
        total_penalty = 0.0

        # only these state categories incur φ‐penalties
        penalized = {
            'soc':      self.phi['soc'],
            'theta':    self.phi['theta_obs'],
            'flow_ratio': self.phi['flow_ratio'],
            'slack':    self.phi['slack']
        }

        for idx, x in enumerate(obs):
            cat, _ = self.mapping_obs[idx]
            if cat not in penalized:
                continue

            lo, hi = low[idx], high[idx]
            phi = penalized[cat]

            low_key  = f"Total penalty: {cat} below lower"
            low_cnt  = f"Num {cat} below lower"
            high_key = f"Total penalty: {cat} above upper"
            high_cnt = f"Num {cat} above upper"

            if x < lo:
                diff = lo - x
                p = diff * phi
                total_penalty += p
                clipped[idx] = lo
                self.pens_step[low_key] += p
                self.pens_step[low_cnt] += 1

            elif x > hi:
                diff = x - hi
                p = diff * phi
                total_penalty += p
                clipped[idx] = hi
                self.pens_step[high_key] += p
                self.pens_step[high_cnt] += 1

        return clipped, total_penalty
    
    def step(self, a_norm):
        """
        Receive normalized action a_norm ∈ [−1,1]^(G+3N+(N−1)), decode, apply
        transition (seven steps), compute reward Πₜ and penalties Cₜ per §1.7.5–1.7.8.
        """

        # per‐step log
        self.pens_step = {k:0 for k in self.env_spec_log.keys()}

        # 1. Action decoding, clipping, penalty logging
        a_phys, C_action = self.sanitize_action(a_norm)  

        G, N = self.G, self.N
        # unpack: pg ∈ ℝᵍ, cn, pdn, ℓn, θ_{2..N} from a_phys
        pg    = a_phys[               :G]
        cn    = a_phys[         G     :G+N]
        pdn   = a_phys[     G+N         :G+2*N]
        ln    = a_phys[ G+2*N           :G+3*N]
        θ_vars = a_phys[G+3*N           :G+3*N+(N-1)]
        # reconstruct full bus‐angle vector θ with θ₁=0
        θ = np.zeros(N, dtype=np.float32)
        θ[1:] = θ_vars
        self.theta     = θ.copy()


        # 2. Battery SOC update: Eₙ,t₊₁ = γ Eₙ,t + η cₙ,t − (1/η) pdₙ,t :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
        self.E = (self.CarryOverRate * self.E
                  + self.ChargeEfficiency * cn
                  - (1.0 / self.ChargeEfficiency) * pdn)

        # 3. Enforce ℓₙ,t ≤ demand (and ≤ dmax_global via sanitize_action)
        dem = np.array([self.demand[(n, self.t)] for n in self.Buses], dtype=np.float32)
        ln  = np.minimum(ln, dem)

        # 4. Power-flow: fₗ = Bij (θᵢ−θⱼ) or 0 if ℓ∈Dₜ :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
        f = np.zeros(self.L, dtype=np.float32)
        for idx, (i, j) in enumerate(self.TransmissionLines):
            if (i, j) in self.DeEnergizedLines:
                f[idx] = 0.0
            else:
                Bij = self.LineSusceptance[(i, j)]
                f[idx] = Bij * (θ[self.bus_idx[i]] - θ[self.bus_idx[j]])
        
        self.last_flow = f.copy()

        # 5. Slack generation: sₙ,t = max{0, dₙ,t − ℓₙ,t − ∑_{g→n} p_g + cₙ − pdₙ
        #                                + inflows − outflows} :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}
        net_flow = np.zeros(N, dtype=np.float32)
        for idx, (i, j) in enumerate(self.TransmissionLines):
            net_flow[self.bus_idx[i]] -= f[idx]
            net_flow[self.bus_idx[j]] += f[idx]
        gen_inj = self.gen_to_bus @ pg
        sn = np.maximum(
            0.0,
            dem - ln - gen_inj + cn - pdn + net_flow
        )
        self.slack = sn.copy()

        # 6. Net injection & power‐balance penalty:
        #    Pₙ,t = ∑_{g→n} p_g + sₙ − dₙ + ℓₙ − cₙ + pdₙ
        P = gen_inj + sn - dem + ln - cn + pdn
        Δ = P - net_flow
        C_balance = self.PhiBalance * np.sum(np.abs(Δ))  # :contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}

        # 7. Advance time and build next state (including demand‐forecast window)
        self._build_state_dict()

        # Observation‐bound clipping & penalty C_obs
        obs_arr = self._get_state()
        obs, C_obs = self.check_obs_bounds_cost(obs_arr)

        # Operating‐cost components (for reward Πₜ)
        gen_cost   = 0.0
        for j in range(self.PolynomialDegree):
            coeffs = np.array([self.GeneratorCost.get((g, j), 0.0)
                               for g in self.Generators], dtype=np.float32)
            gen_cost += np.sum(coeffs * pg**j)
        slack_cost = self.Kslack * np.sum(sn)
        shed_cost  = self.Kls    * np.sum(ln)

        # Reward Πₜ = −(gen_cost + slack_cost + shed_cost) :contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}
        reward = - (gen_cost + slack_cost + shed_cost)

        # Update env‐level stats
        self.cost   = C_action + C_obs + C_balance
        self.reward = reward
        self.cost_ep   += self.cost
        self.reward_ep += self.reward


        self.pens_step['num_steps']         += 1
        self.pens_step['sum reward']        += self.reward
        self.pens_step['sum reward square'] += self.reward**2
        self.pens_step['sum cost']          += self.cost
        self.pens_step['sum cost square']   += self.cost**2
        if self.t >= self.T:
            self.pens_step['num_episodes']  += 1
            self.terminated = True
        else:
            self.t += 1

        # 11) aggregate into master log
        for k in self.env_spec_log.keys():
            self.env_spec_log[k] += self.pens_step[k]

        obs_tensor    = th.tensor(obs,    dtype=th.float32, device=self._device)
        reward_tensor = th.tensor(self.reward - self.cost, dtype=th.float32, device=self._device)
        done_tensor   = th.tensor(self.terminated,   dtype=th.bool,   device=self._device)
        trunc_tensor  = th.tensor(False,  dtype=th.bool,   device=self._device)

        return obs_tensor, reward_tensor, done_tensor, trunc_tensor, {}

    def render(self, mode='human'):
        """
        Print current step, normalized SOC, last flows, slack, and cumulative reward.
        """
        soc_pct = np.round(self.E / self.E_cap_max, 3)
        flows   = np.round(self.last_flow / self.p_lim_vec, 3)
        print(f"[t={self.t-1}/{self.T}] SOC={soc_pct.tolist()}, "
              f"flow_ratio={flows.tolist()}, reward={self.reward:.2f}") 

    def close(self):
        """
        No resources to clean up.
        """
        pass

    def set_seed(self, seed: int):
        """
        Set random seeds for reproducibility.
        """
        if seed is None:
            return
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


    def load_config(self, path):
        cfg = json.load(open(path))

        # parse lines as tuples
        cfg["TransmissionLines"] = [tuple(x) for x in cfg.get("TransmissionLines", [])]
        cfg["DeEnergizedLines"]  = [tuple(x) for x in cfg.get("DeEnergizedLines", [])]

        # 2) integer-keyed simple dicts
        for name in [
            "BusGeneratorLink",
            "GeneratorLowerLimit", "GeneratorUpperLimit", 
            "BatteryLowLimit", "BatteryUpLimit",
            "BatteryInitialCharge",
            "LowerBatteryChargeRate", "UpperBatteryChargeRate",
            "LowerBatteryDischargeRate", "UpperBatteryDischargeRate"
        ]:
            raw = cfg.get(name, {})
            cfg[name] = {int(k): v for k, v in raw.items()}

        # 3) tuple-keyed dicts
        def parse_tuple_dict(raw_d):
            out = {}
            for k, v in raw_d.items():
                i, j = k.strip("()").split(",")
                out[(int(i), int(j))] = v
            return out

        for name in [
            "LineSusceptance", "LinePowerFlowLimit",
            "LineUpperVoltageAngle", "LineLowerVoltageAngle",
            "GeneratorCost"
        ]:
            cfg[name] = parse_tuple_dict(cfg.get(name, {}))

        # 4) flatten PowerDemandAtBus
        raw_pd = cfg.get("PowerDemandAtBus", {})
        demand = {}
        for kb, inner in raw_pd.items():
            b = int(kb)
            for kt, val in inner.items():
                demand[(b, int(kt))] = float(val)
        cfg["PowerDemandAtBus"] = demand

        return cfg

def main():
    # 1) locate config.json
    base_dir  = Path(__file__).resolve().parent
    config_fp = base_dir / "config.json"
    if not config_fp.is_file():
        raise FileNotFoundError(f"Couldn’t find config.json at {config_fp}")

    # 2) instantiate environment
    env = BatteryOperationEnv(env_id='Battery-v0', config_path=config_fp, seed=42)

    # 3) reset returns (obs, info)
    obs, info = env.reset()
    print("Manual rollout start...")

    # 4) run for a few steps
    for i in range(24):
        # sample an action (NumPy array)
        raw_action = env.action_space.sample()
        # step returns obs, reward, done, truncated, info
        obs, reward, done, truncated, info = env.step(raw_action)

        print(
            f"Step {i+1} | obs.shape={obs.shape} | "
            f"reward={reward:.3f} | done={done} | truncated={truncated}"
        )

        if done:
            obs, info = env.reset()
            print("Episode reset")

if __name__ == "__main__":
    main()

