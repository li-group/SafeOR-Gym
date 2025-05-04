'''
Unit Commitment
Hao Chen
'''
import random
from typing import Any, ClassVar, List, Tuple, Optional, Dict

import torch
import numpy as np

import scipy.stats as stats
# from or_gym.envs.power_system.forecast import get_random_25hr_forecast

import gymnasium as gym
from gymnasium.spaces.utils import flatten_space


class UnitCommitmentMasterEnv(gym.Env):
    """
    Unit Commitment Environment
    UC-v0: single-bus system without network constraints
    UC-v1: multiple-buses system with network constraints

    The Unit Commitment Problem is a combinatorial optimization problem in which the objective is to
    determine the optimal schedule for turning power units on or off, generating and dispatching power over a given time horizon.
    The goal is to minimize production costs while meeting demand and satisfying various operational constraints such as
    generation limits, ramping rates, and minimum uptime/downtime of units, reserve requirements, and network constraints.
    This version is typically solved in an online fashion, where decisions are made in real-time based on
    the current state of the system and the forecast of demand for the next time period.

    Observation:
        'u_seq': dictionary of binary vectors indicating the on/off status of ith generator from t-max(UT, DT)-1 to t
        'D_forecast': forecasted demand of bus n at time t+1
        'p': power output of ith generator at time t

    Actions:
        'on_off': binary vector indicating the on/off status of ith generator at time t+1
        'power': power output of ith generator at time t+1
        'angle': angle of bus n at time t+1 (only for UC-v1)

    Reward:
        The reward is the negative of the following costs:
        - production cost: the cost of generating power from the generators
        - start-up cost: the cost of starting up a generator from off to on
        - shut-down cost: the cost of shutting down a generator from on to off
        - load shedding cost: the cost of not meeting or exceeding the demand
        - reserve cost: the cost of not meeting the reserve requirement

    Cost:
        The cost is the positive of the following costs related to safety:
        - minimum up-time/down-time cost: the cost of violating the minimum up-time/down-time constraint
        - ramping up/down cost: the cost of violating the ramping up/down constraint
        - irreparable cost: the cost of violating ramping down and minimum down-time and being irreparable

    Starting State:
        if no_change_before_0 = True,
        then the on-off status of the generators does not change before time 0
        if no_change_before_0 = False,
        then the on-off status of the generators from -max(UT, DT)-1 to 0 is initialized with given u0_seq

    Episode Termination:
        The episode terminates when the time step reaches T (24 hours).
        Invalid action is repaired inside the environment and does not terminate the episode, instead a cost is applied

    P_max: maximum power output of ith generator
    P_min: minimum power output of ith generator
    UT: minimum up time of ith generator
    DT: minimum down time of ith generator
    RU: maximum limit of ramp up rate of ith generator
    RD: maximum limit of ramp down rate of ith generator
    SU: maximum limit of start up rate of ith generator
    SD: maximum limit of shut down rate of ith generator
    a: power generation cost coefficient, quadratic term
    b: power generation cost coefficient, linear term
    c: power generation cost coefficient, constant term
    hot_cost: hot start cost of ith generator
    cold_cost: cold start cost of ith generator
    cold_hrs: cold start hour of ith generator
    C_SD: shut down cost of ith generator
    C_LS: load shedding cost
    C_RP: reserve penalty cost
    R: reserve requirement

    D: demand of bus n at time t
    u: on/off status of ith generator at time t
    v: start-up status of ith generator at time t
    w: shut-down status of ith generator at time t
    p: power output of ith generator at time t
    r: reserve of ith generator at time t
    pi: angle of bus n at time t
    """

    def __init__(self, env_id: str,
                 **kwargs: Any) -> None:
        super().__init__()

        self._device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.env_id = env_id

        self.env_spec_log = {'Number of Minimum Up-time Violation': 0,
                             'Penalty of Minimum Up-time Violation': 0,
                             'Number of Minimum Down-time Violation': 0,
                             'Penalty of Minimum Down-time Violation': 0,
                             'Number of Ramping Up Violation': 0,
                             'Penalty of Ramping Up Violation': 0,
                             'Number of Ramping Down Violation': 0,
                             'Penalty of Ramping Down Violation': 0,
                             # 'Number of Irreparable Violation': 0,
                             # 'Penalty of Irreparable Violation': 0,
                             }

        self.penalty_factor_UT = 100
        self.penalty_factor_DT = 100
        self.penalty_factor_RampUp = 100
        self.penalty_factor_RampDown = 100
        # self.penalty_factor_irreparable = 1000000

        self.T = 24
        self._max_episode_steps = self.T
        self.num_gen = 5
        self.generators = range(self.num_gen)
        if self.env_id == 'UC-v0':
            self.num_bus = 1
            self.gen_bus = {i: 0 for i in self.generators}
            self.bus_gen = {0: [0, 1, 2, 3, 4]}
            self.num_line = 0
            self.line_bus = {}
            self.B = np.array([20])
            self.F_max = np.array([0])
            self.F_min = np.array([0])
            self.Pi_max = np.array([0])
            self.Pi_min = np.array([0])
            self.pi = self.pi0 = np.array([0])
            self.loc = [300.0]
            self.scale = [70.0]
            self.deterministic_demand = np.array([[362.], [191.], [303.], [263.], [416.], [302.],
                                                  [328.], [234.], [357.], [266.], [333.], [325.],
                                                  [343.], [285.], [290.], [329.], [245.], [305.],
                                                  [311.], [254.], [385.], [214.], [197.], [227.]])
        elif self.env_id == 'UC-v1':
            self.num_bus = 4
            self.gen_bus = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}
            self.bus_gen = {0: [0], 1: [1], 2: [2], 3: [3, 4]}
            self.num_line = 5
            self.line_bus = {0: (0, 1), 1: (0, 2), 2: (1, 2), 3: (1, 3), 4: (2, 3)}
            self.B = np.array([20, 20, 20, 20, 20])
            self.F_max = np.array([100, 100, 100, 100, 100])
            self.F_min = np.array([-100, -100, -100, -100, -100])
            self.Pi_max = np.array([0, 0.2, 0.2, 0.2])  # angle_1 = 0
            self.Pi_min = np.array([0, -0.2, -0.2, -0.2])  # angle_1 = 0
            self.pi = self.pi0 = np.array([0, 0, 0, 0])
            self.loc = [253.1,  60.3,  60.3,  70.1]
            self.scale = [22.6,  6.0,  6.0,  8.4]
            self.deterministic_demand = np.array([[226.,  57.,  52.,  59.], [233.,  57.,  65.,  77.],
                                                  [258.,  59.,  48.,  78.], [272.,  67.,  55.,  67.],
                                                  [247.,  55.,  65.,  65.], [232.,  65.,  69.,  63.],
                                                  [213.,  57.,  59.,  69.], [245.,  71.,  60.,  74.],
                                                  [243.,  59.,  72.,  61.], [263.,  63.,  55.,  56.],
                                                  [291.,  60.,  55.,  72.], [235.,  58.,  59.,  73.],
                                                  [234.,  59.,  67.,  65.], [253.,  47.,  54.,  63.],
                                                  [267.,  52.,  47.,  55.], [223.,  58.,  57.,  72.],
                                                  [239.,  67.,  62.,  67.], [260.,  60.,  56.,  63.],
                                                  [234.,  61.,  62.,  76.], [241.,  59.,  54.,  84.],
                                                  [298.,  63.,  63.,  76.], [235.,  55.,  52.,  80.],
                                                  [273.,  63.,  75.,  80.], [276.,  54.,  73.,  70.]])

        else:
            raise ValueError(f"Unknown env_id: {env_id}")
        self.buses = range(self.num_bus)
        self.lines = range(self.num_line)
        self.from_bus = []
        self.to_bus = []
        for line in self.lines:
            self.from_bus.append(self.line_bus[line][0])
            self.to_bus.append(self.line_bus[line][1])
        self.from_bus_lines = {i: [] for i in self.buses}
        self.to_bus_lines = {i: [] for i in self.buses}
        for line, (from_bus, to_bus) in self.line_bus.items():
            self.from_bus_lines[from_bus].append(line)
            self.to_bus_lines[to_bus].append(line)

        self.t = 0
        self.terminated = False
        self.truncated = False
        self.reward = 0
        self.cost = 0

        # default forecast
        self.model_type = 'deterministic'
        self.nyiso_path = "./or_gym/envs/power_system/" + 'data/'

        # default information
        self.P_max = np.array([455, 130, 130, 80, 55])
        self.P_min = np.array([150,  20,  20, 20, 55])
        self.a = np.array([0.00048, 0.00200, 0.00211, 0.00712, 0.00413])
        self.b = np.array([16.19, 16.60, 16.50, 22.26, 25.92])
        self.c = np.array([1000, 700, 680, 370, 660])
        self.UT = np.array([8, 5, 5, 3, 1])
        self.DT = np.array([8, 5, 5, 3, 1])
        self.RU = np.array([300, 85, 85, 55, 55])
        self.RD = np.array([300, 85, 85, 55, 55])
        self.SU = np.array([300, 85, 85, 55, 55])
        self.SD = np.array([300, 85, 85, 55, 55])
        self.hot_cost = np.array([4500, 550, 560, 170, 30])
        self.cold_cost = np.array([9000, 1100, 1120, 340, 60])
        self.cold_hrs = np.array([5, 4, 4, 2, 0])
        self.C_SD = np.array([0, 0, 0, 0, 0])
        self.C_LS = 100
        self.C_RP = 100
        self.R = 10

        # default initial values for physical state variables
        self.scale_action = False
        self.no_change_before_0 = True
        self.u_seq = self.u0_seq = {0: np.ones(8 + 1),  # assume only 1st generator is on
                                    1: np.zeros(5 + 1),
                                    2: np.zeros(5 + 1),
                                    3: np.zeros(3 + 1),
                                    4: np.zeros(1 + 1)}  # assume no change happened from - [max(UT, DT)+1] to 0
        # assume only 1st generator is on
        self.u_prev = self.u0_prev = np.array([1, 0, 0, 0, 0])
        self.u = self.u0 = np.array([1, 0, 0, 0, 0])
        self.v, self.w = self._reckless_move(self.u, self.u_prev)
        self.v_seq, self.w_seq = self._u2vw_seq(self.u_seq)

        self.p_prev = self.p0_prev = np.array([300, 0, 0, 0, 0])
        self.p = self.p0 = np.array([300, 0, 0, 0, 0])

        assign_env_config(self, kwargs)

        # initialize belief model/data
        self.D_forecast = np.zeros(self.num_bus)
        self.forecast_model = ForecastModel(self.model_type, self.loc, self.scale, self.num_bus,
                                            self.deterministic_demand, self.nyiso_path)

        self.reset()

        if env_id == 'UC-v0':
            if self.scale_action:
                self.raw_action_space = gym.spaces.Dict({
                    "on_off": gym.spaces.Box(low=-1, high=1, shape=(self.num_gen,), dtype=np.float32),
                    "power": gym.spaces.Box(low=-1, high=1, shape=(self.num_gen,), dtype=np.float32)
                })
            else:
                self.raw_action_space = gym.spaces.Dict({
                    "on_off": gym.spaces.MultiBinary(self.num_gen),
                    "power": gym.spaces.Box(low=self.P_min, high=self.P_max, dtype=np.float32)
                })
            self.action_space = flatten_space(self.raw_action_space)

            self.raw_observation_space = gym.spaces.Dict(
                {
                    "u_seq": gym.spaces.MultiBinary((np.maximum(self.UT, self.DT) + 1).sum()),
                    "D_forecast": gym.spaces.Box(low=0, high=np.ones(self.num_bus) * self.P_max.sum(), dtype=np.float32),
                    "p": gym.spaces.Box(low=0, high=self.P_max, dtype=np.float32)
                })
            self.observation_space = flatten_space(self.raw_observation_space)

        elif env_id == 'UC-v1':
            if self.scale_action:
                self.raw_action_space = gym.spaces.Dict({
                    "on_off": gym.spaces.Box(low=-1, high=1, shape=(self.num_gen,), dtype=np.float32),
                    "power": gym.spaces.Box(low=-1, high=1, shape=(self.num_gen,), dtype=np.float32),
                    "angle": gym.spaces.Box(low=-1, high=1, shape=(self.num_bus-1,), dtype=np.float32)
                })
            else:
                self.raw_action_space = gym.spaces.Dict({
                    "on_off": gym.spaces.MultiBinary(self.num_gen),
                    "power": gym.spaces.Box(low=self.P_min, high=self.P_max, dtype=np.float32),
                    "angle": gym.spaces.Box(low=self.Pi_min[1:], high=self.Pi_max[1:], dtype=np.float32)
                })
            self.action_space = flatten_space(self.raw_action_space)

            self.raw_observation_space = gym.spaces.Dict(
                {
                    "u_seq": gym.spaces.MultiBinary((np.maximum(self.UT, self.DT) + 1).sum()),
                    "D_forecast": gym.spaces.Box(low=0, high=np.ones(self.num_bus) * self.P_max.sum(), dtype=np.float32),
                    "p": gym.spaces.Box(low=0, high=self.P_max, dtype=np.float32),
                    "pi": gym.spaces.Box(low=self.Pi_min, high=self.Pi_max, dtype=np.float32)
                })
            self.observation_space = flatten_space(self.raw_observation_space)
        else:
            raise ValueError(f"Unknown env_id: {env_id}")

        action_low = {"on_off": [],
                      "power": [],
                      "angle": []}
        action_high = {"on_off": [],
                       "power": [],
                       "angle": []}
        for i in self.generators:
            action_low["on_off"].append(-1.)
            action_high["on_off"].append(1.)
            action_low["power"].append(self.P_min[i])
            action_high["power"].append(self.P_max[i])
        for n in self.buses:
            action_low["angle"].append(self.Pi_min[n])
            action_high["angle"].append(self.Pi_max[n])

        self.action_low = action_low
        self.action_high = action_high

    def _get_state(self, mode="arr") -> np.ndarray:
        if self.env_id == 'UC-v0':
            obs_dict = {
                "u_seq": self.u_seq,
                "D_forecast": self.D_forecast,
                "p": self.p,
            }
            obs_arr = np.concatenate([self._vectorize_seq(self.u_seq),
                                      self.D_forecast,
                                      self.p])
        elif self.env_id == 'UC-v1':
            obs_dict = {
                "u_seq": self.u_seq,
                "D_forecast": self.D_forecast,
                "p": self.p,
                "pi": self.pi
            }
            obs_arr = np.concatenate([self._vectorize_seq(self.u_seq),
                                      self.D_forecast,
                                      self.p,
                                      self.pi])
        else:
            raise NotImplementedError

        if mode == "dict":
            return obs_dict
        elif mode == "arr":
            return obs_arr
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.set_seed(seed)
        self.t = 0
        self.terminated = False
        self.truncated = False
        self.reward = 0
        self.cost = 0

        self.pi = self.pi0
        self.p_prev = np.array([self.p0_prev[i] for i in self.generators], dtype=float)
        self.p = np.array([self.p0[i] for i in self.generators], dtype=float)
        if self.no_change_before_0:
            self.u_prev = self.u0_prev
            self.u = self.u0
            self.u_seq = {}
            for i in self.generators:
                if self.u[i]:
                    self.u_seq.update({i: np.ones(max(self.UT[i], self.DT[i]) + 1)})
                else:
                    self.u_seq.update({i: np.zeros(max(self.UT[i], self.DT[i]) + 1)})
        else:
            self.u_seq = self.u0_seq
            self.u_prev = np.concatenate([self.u_seq[i][1] for i in self.generators])
            self.u = np.concatenate([self.u_seq[i][0] for i in self.generators])
        self.v, self.w = self._reckless_move(self.u, self.u_prev)
        self.v_seq, self.w_seq = self._u2vw_seq(self.u_seq)

        # forecast the demand
        self.forecast_model.reset()
        self.D_forecast = self._forecast_demand()

        return self._get_state(), {}

    @staticmethod
    def _vectorize_seq(seq: Dict[int, np.ndarray]) -> np.ndarray:
        sorted_keys = sorted(seq.keys())
        seq_vector = np.concatenate([seq[key] for key in sorted_keys])
        return seq_vector

    @staticmethod
    def _reckless_move(u_new: np.ndarray, u_curr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        get v_{t+1} and w_{t+1} from u_{t+1} and u_t
        """
        v_new = np.maximum(0, u_new - u_curr)
        w_new = - np.minimum(0, u_new - u_curr)
        return v_new, w_new

    def _roll_seq(self):
        """
        roll the sequence of u_seq, v_seq and w_seq
        """
        for i in self.generators:
            self.u_seq[i] = np.roll(self.u_seq[i], 1)
            self.u_seq[i][0] = self.u[i]
            self.v_seq[i] = np.roll(self.v_seq[i], 1)
            self.v_seq[i][0] = self.v[i]
            self.w_seq[i] = np.roll(self.w_seq[i], 1)
            self.w_seq[i][0] = self.w[i]

    def _u2vw_seq(self, u_seq: Dict[Any, np.ndarray]) -> Tuple[Dict[Any, np.ndarray], Dict[Any, np.ndarray]]:
        """
        convert u_seq to v_seq and w_seq
        """
        v_seq = {}
        w_seq = {}
        for i in self.generators:
            u_diff_seq = u_seq[i][:-1] - u_seq[i][1:]
            v_seq.update({i: np.maximum(0, u_diff_seq[:self.UT[i]])})
            w_seq.update({i: - np.minimum(0, u_diff_seq[:self.DT[i]])})
        return v_seq, w_seq

    def _evaluate_UTDT(self, u_new, v_new, w_new):
        # seq has UT or DT terms, t+0, t-1, ..., t-UT+1
        # we need UT or DT terms but, t+1 (new, given by reckless move), t+0 (current), t-1, ..., t-UT+2
        sum_v = v_new + np.array([np.sum(self.v_seq[i][:-1]) for i in self.generators])
        sum_w = w_new + np.array([np.sum(self.w_seq[i][:-1]) for i in self.generators])
        UT_violation = sum_v > u_new
        DT_violation = sum_w > (1 - u_new)
        UT_cost = np.sum(np.maximum(sum_v - u_new, 0)) * self.penalty_factor_UT
        DT_cost = np.sum(np.maximum(sum_w - (1 - u_new), 0)) * self.penalty_factor_DT
        return UT_violation, DT_violation, UT_cost, DT_cost

    def _evaluate_Ramp(self, u_new, u_curr, p_new, p_curr, v_new, w_new):
        RampUp_violation = p_new - p_curr > self.RU * u_curr + self.SU * v_new
        RampDown_violation = p_curr - p_new > self.RD * u_new + self.SD * w_new
        RampUp_cost = np.sum(np.maximum(
            p_new - p_curr - (self.RU * u_curr + self.SU * v_new), 0)) * self.penalty_factor_RampUp
        RampDown_cost = np.sum(np.maximum(
            p_curr - p_new - (self.RD * u_new + self.SD * w_new), 0)) * self.penalty_factor_RampDown
        return RampUp_violation, RampDown_violation, RampUp_cost, RampDown_cost

    def _repair_action(self, on_off: np.ndarray, power: np.ndarray, angle: np.ndarray):
        """
        Repair the action that violates
        Minimum Up-Time, Minimum Down-Time, Ramping Up and Ramping Down Constraints

        Minimum Up-Time implies the generators that must be on
        Minimum Down-Time implies the generators that must be off
        Ramping up implies a tighter bound on the power of the generators that are on
        Ramping down implies
        (1) the generators that must be on (if turning off, the decrease will exceed the max shut down rate)
        (2) a tighter bound on the power of the generators that are good to be off

        the must_on and must off obtained from Minimum Up-Time and Minimum Down-Time won't contradict with each other
        but the must on and must off obtained from Ramping down and Minimum Down-Time may contradict with each other
        if not contradicted, the action can be repaired by fixing all the must on&off and applying tighter bounds
        if contradicted, the action cannot be repaired and leads to truncated or a very large cost
        """
        u_new = on_off
        u_curr = self.u
        v_new, w_new = self._reckless_move(u_new, u_curr)
        p_new = power
        p_curr = self.p
        pi_new = angle

        # # check contradiction
        # # we must keep it off, as turning on will result in a too early turn-on
        # must_off = np.array([np.sum(self.w_seq[i][:-1]) for i in self.generators]) > 0
        # # we must keep it on, as turning off will result in a too large decrease
        # must_on = p_curr > self.SD
        # contradiction = must_on & must_off

        # repair part of the action
        repaired_pi_new = np.minimum(np.maximum(pi_new, self.Pi_min), self.Pi_max)
        # the decision, on, implies turn-on and violates DT, so must keep it off
        # the decision, off, implies turn-off and violates UT, so must keep it on
        UT_violation, DT_violation, UT_cost, DT_cost = self._evaluate_UTDT(u_new, v_new, w_new)
        repaired_u_new = np.where(DT_violation, 0, np.where(UT_violation, 1, u_new))
        repaired_v_new, repaired_w_new = self._reckless_move(repaired_u_new, u_curr)

        # if np.any(contradiction):
        #     # self.truncated = True # DON'T TRUNCATE EVEN THOUGH WE CANNOT REPAIR THE ACTION
        #     repaired_p_new = u_new * np.minimum(np.maximum(p_new, self.P_min), self.P_max)
        #     # ADD UP A VERY LARGE COST INSTEAD
        #     irreparable_violation = contradiction
        #     irreparable_cost = np.sum(contradiction) * self.penalty_factor_irreparable
        #     self.cost += irreparable_cost
        #
        #     self.env_spec_log['Number of Irreparable Violation'] += np.sum(irreparable_violation)
        #     self.env_spec_log['Penalty of Irreparable Violation'] += irreparable_cost
        #
        # else:
        #     # repair the rest of the action
        #     ub = p_curr + self.RU * u_curr + self.SU * repaired_v_new
        #     lb = p_curr - self.RD * repaired_u_new - self.SD * repaired_w_new
        #     repaired_p_new = repaired_u_new * np.minimum(np.maximum(p_new, lb), ub)
        # return repaired_u_new, repaired_v_new, repaired_w_new, repaired_p_new, repaired_pi_new
        # return u_new, v_new, w_new, p_new, pi_new

        ub = p_curr + self.RU * u_curr + self.SU * repaired_v_new
        lb = p_curr - self.RD * repaired_u_new - self.SD * repaired_w_new
        repaired_p_new = repaired_u_new * np.minimum(np.maximum(p_new, lb), ub)
        return repaired_u_new, repaired_v_new, repaired_w_new, repaired_p_new, repaired_pi_new

    def _compute_reserve(self, u_new: np.ndarray,
                         v_new: np.ndarray, w_new: np.ndarray, p_new: np.ndarray) -> np.ndarray:
        """
        the reserve, r, must satisfy the constraints:
        p_{t+1} + r_{t+1} <= P_max * u_{t+1}
        p_{t+1} + r_{t+1} - p_{t} <= RU * u_{t} + SU * v_{t+1}
        """
        u_curr = self.u
        p_curr = self.p

        r = np.maximum(
                np.minimum(self.P_max * u_new - p_new,
                           self.RU * u_curr + self.SU * v_new + p_curr - p_new),
                0)
        return r

    def _compute_power_flow(self, pi_new: np.ndarray) -> np.ndarray:
        flow = self.B * (pi_new[self.from_bus] - pi_new[self.to_bus])
        repaired_flow = np.minimum(np.maximum(flow, self.F_min), self.F_max)
        return repaired_flow

    def _compute_power_slack(self, p_new: np.ndarray, flow: np.ndarray, demand: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        flow_in = np.array([np.sum(flow[self.to_bus_lines[b]]) for b in self.buses])
        flow_out = np.array([np.sum(flow[self.from_bus_lines[b]]) for b in self.buses])
        flow_diff = flow_in - flow_out
        s = demand - np.array([np.sum(p_new[self.bus_gen[b]]) for b in self.buses]) - flow_diff
        overflow = np.maximum(0, s)
        underflow = - np.minimum(0, s)
        return overflow, underflow

    def _compute_production_reward(self, p_new):
        return - np.sum(self.a * (p_new ** 2) + self.b * p_new + self.c)

    def _compute_startup_reward(self, v_new):
        return - np.sum(self.hot_cost * v_new)

    def _compute_shutdown_reward(self, w_new):
        return - np.sum(self.C_SD * w_new)

    def _compute_fulfillment_reward(self, overflow, underflow):
        return - self.C_LS * np.sum(overflow + underflow)

    def _compute_reservation_reward(self, reserve):
        return - self.C_RP * np.maximum(self.R - np.sum(reserve), 0)

    def _forecast_demand(self) -> np.ndarray:
        demand = self.forecast_model.forecast()
        return demand

    def _compute_reward(self, u_new, v_new, w_new, p_new, pi_new, demand) -> np.float32:
        reward = 0

        # compute the reserve
        reserve = self._compute_reserve(u_new, v_new, w_new, p_new)
        # compute the flow
        flow = self._compute_power_flow(pi_new)
        # compute the power slack
        overflow, underflow = self._compute_power_slack(p_new, flow, demand)

        # compute the reward
        reward += self._compute_production_reward(p_new)
        reward += self._compute_startup_reward(v_new)
        reward += self._compute_shutdown_reward(w_new)
        reward += self._compute_fulfillment_reward(overflow, underflow)
        reward += self._compute_reservation_reward(reserve)
        return reward

    def _compute_cost(self, on_off: np.ndarray, power: np.ndarray) -> np.int64:
        u_new = on_off
        u_curr = self.u
        v_new, w_new = self._reckless_move(u_new, u_curr)
        p_new = u_new * np.minimum(np.maximum(power, self.P_min), self.P_max)
        p_curr = self.p

        # compute cost (raw action)
        UT_violation, DT_violation, UT_cost, DT_cost = self._evaluate_UTDT(u_new, v_new, w_new)
        RampUp_violation, RampDown_violation, RampUp_cost, RampDown_cost = self._evaluate_Ramp(u_new, u_curr,
                                                                                               p_new, p_curr,
                                                                                               v_new, w_new)
        cost = UT_cost + DT_cost + RampUp_cost + RampDown_cost

        # log the violation and cost
        self.env_spec_log['Number of Minimum Up-time Violation'] += np.sum(UT_violation)
        self.env_spec_log['Penalty of Minimum Up-time Violation'] += UT_cost
        self.env_spec_log['Number of Minimum Down-time Violation'] += np.sum(DT_violation)
        self.env_spec_log['Penalty of Minimum Down-time Violation'] += DT_cost
        self.env_spec_log['Number of Ramping Up Violation'] += np.sum(RampUp_violation)
        self.env_spec_log['Penalty of Ramping Up Violation'] += RampUp_cost
        self.env_spec_log['Number of Ramping Down Violation'] += np.sum(RampDown_violation)
        self.env_spec_log['Penalty of Ramping Down Violation'] += RampDown_cost

        return cost

    def _get_terminated(self):
        if self.t >= self.T:
            self.terminated = True
        else:
            self.terminated = False
        return self.terminated

    def _scale_and_round_action(self, on_off, power, angle):
        if torch.is_tensor(on_off):
            action_low = {key: torch.as_tensor(value, device=self._device) for key, value in self.action_low.items()}
            action_high = {key: torch.as_tensor(value, device=self._device) for key, value in self.action_high.items()}
        else:
            action_low = {key: np.array(value) for key, value in self.action_low.items()}
            action_high = {key: np.array(value) for key, value in self.action_high.items()}

        on_off_rounded = (on_off >= 0).float() if torch.is_tensor(on_off) else (on_off >= 0).astype(float)
        power_scaled = (power + 1) / 2 * (action_high["power"] - action_low["power"]) + action_low["power"]
        angle_scaled = (angle + 1) / 2 * (action_high["angle"] - action_low["angle"]) + action_low["angle"]
        return on_off_rounded, power_scaled, angle_scaled

    def step(self, action):
        if self.env_id == 'UC-v0':
            assert len(action) == 2 * self.num_gen  # on_off + power
            on_off = action[:self.num_gen]
            power = action[self.num_gen:]
            if torch.is_tensor(action):
                angle = torch.tensor([0], device=self._device)
            else:
                angle = np.array([0])
        elif self.env_id == 'UC-v1':
            assert len(action) == 2 * self.num_gen + self.num_bus - 1  # on_off + power + angle - 1 (exclude angle 1)
            on_off = action[:self.num_gen]
            power = action[self.num_gen:2 * self.num_gen]
            if torch.is_tensor(action):
                angle = torch.concatenate([torch.zeros(1, device=self._device), action[2 * self.num_gen:]])
            else:
                angle = np.concatenate([np.zeros(1), action[2 * self.num_gen:]])
        else:
            raise NotImplementedError

        if self.scale_action:
            on_off, power, angle = self._scale_and_round_action(on_off, power, angle)

        on_off = on_off.numpy() if torch.is_tensor(on_off) else on_off
        power = power.numpy() if torch.is_tensor(power) else power
        angle = angle.numpy() if torch.is_tensor(angle) else angle
        demand = self.D_forecast
        # now they are at the same time step, t+1

        # compute the cost of raw action
        self.cost += self._compute_cost(on_off, power)

        # repair the action (may fail to repair -> use the partially repaired action and add a large cost)
        # Note the _repair_action function will also update the cost inside the function if irreparable
        (repaired_u_new, repaired_v_new, repaired_w_new,
         repaired_p_new, repaired_pi_new) = self._repair_action(on_off, power, angle)

        # compute the reward
        self.reward = self._compute_reward(repaired_u_new, repaired_v_new, repaired_w_new,
                                           repaired_p_new, repaired_pi_new, demand)

        # update state
        self.t += 1
        self.pi = repaired_pi_new
        self.p_prev = self.p
        self.p = repaired_p_new
        self.u_prev = self.u
        self.u = repaired_u_new
        self.v, self.w = repaired_v_new, repaired_w_new
        self._roll_seq()  # u_seq, v_seq, w_seq
        if self.t < self.T:
            self.D_forecast = self._forecast_demand()
        state = self._get_state()

        return (
            state,
            self.reward - self.cost,  # reward (negative) "-" cost (positive)
            self.terminated,
            self.truncated,
            {}
        )

    @property
    def max_episode_steps(self) -> int:
        return self.T

    def render(self) -> Any:
        print("state:", f"{self._get_state(mode='dict')}")
        print("reward:", f"{self.reward}")
        print("cost:", f"{self.cost}")
        print("specification:", f"{self.env_spec_log}")

    def close(self) -> None:
        return None

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


class ForecastModel:
    def __init__(self, model_type, loc, scale, size, deterministic_demand, nyiso_path='data/'):
        self.model_type = model_type
        self.loc = loc
        self.scale = scale
        self.size = size
        self.deterministic_demand = deterministic_demand
        self.nyiso_path = nyiso_path
        self.model = None
        self.reset()

    def reset(self):
        if self.model_type == "deterministic":
            self.model = self.deterministic_demand
        # elif self.model_type == "nyiso":
        #     # make it 25hr to avoid the error in the last step
        #     forecast_example = np.array(get_random_25hr_forecast(self.nyiso_path)).astype(float)
        #     factor = self.loc / np.mean(forecast_example)
        #     forecast_example = forecast_example * factor + np.random.normal(0, self.scale, forecast_example.shape)
        #     self.model = iter(forecast_example)
        elif self.model_type == "normal":
            # self.model = stats.norm(loc=self.loc, scale=self.scale)
            self.model = np.random.normal(loc=self.loc, scale=self.scale, size=self.size)

    def forecast(self):
        if self.model_type == "deterministic":
            d = self.model[0]
            self.model = np.roll(self.model, shift=-1, axis=0)
            return d
        elif self.model_type == 'normal':
            return np.random.normal(loc=self.loc, scale=self.scale, size=self.size)
        # elif self.model_type == 'nyiso':
        #     return next(self.model)


def assign_env_config(self, kwargs):
    for key, value in kwargs.items():
        setattr(self, key, value)
    if hasattr(self, 'env_config'):
        for key, value in self.env_config.items():
            # Check types based on default settings
            if hasattr(self, key):
                if type(getattr(self,key)) == np.ndarray:
                    setattr(self, key, value)
                else:
                    setattr(self, key,
                        type(getattr(self, key))(value))
            else:
                raise AttributeError(f"{self} has no attribute, {key}")

#
# env = UnitCommitmentMasterEnv(env_id='UC-v0')
# state = env.reset()
# actions = np.load("./opt_action_v0_arr.npy")
# total = 0
# for action in actions:
#     state, reward, terminated, truncated, info = env.step(action)
#     total += reward
# print("Total Reward v0:", total)
#
# env = UnitCommitmentMasterEnv(env_id='UC-v1')
# state = env.reset()
# actions = np.load("./opt_action_v1_arr.npy")
# total = 0
# for t, action in enumerate(actions):
#     state, reward, terminated, truncated, info = env.step(action)
#     total += reward
# print("Total Reward v1:", total)