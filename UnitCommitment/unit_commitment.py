'''
Unit Commitment
Hao Chen
'''
import random
from typing import Any, ClassVar, List, Tuple, Optional, Dict

import torch
import numpy as np

import scipy.stats as stats
from or_gym.utils import assign_env_config
from or_gym.envs.power_system.forecast import get_random_25hr_forecast

import gymnasium as gym
from gymnasium.spaces.utils import flatten_space

import omnisafe
from omnisafe.typing import OmnisafeSpace
from omnisafe.envs.core import CMDP, env_register
from omnisafe.common.logger import Logger


@env_register
class UnitCommitmentMasterEnv(CMDP):
    """
    Problem Description:...
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

    D: demand of bus n at time t, here assume only 1 bus to relax network constraints
    u: on/off status of ith generator at time t
    v: start-up status of ith generator at time t
    w: shut-down status of ith generator at time t
    """

    _support_envs: ClassVar[list[str]] = ['Simple-v0']
    _action_space: OmnisafeSpace
    _observation_space: OmnisafeSpace
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super(UnitCommitmentMasterEnv, self).__init__(env_id)

        self.env_spec_log = {}
        self.verbose = False

        self.horizon = 24
        self._max_episode_steps = self.horizon
        self.num_gen = 5
        self.generators = range(self.num_gen)
        self.t = 0
        self.terminated = False
        self.truncated = False
        self.reward = 0
        self.cost = 0

        # default forecast
        self.model_type = 'normal'
        self.loc = 500
        self.scale = 10
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
        self.C_LS = 10000
        self.C_RP = 100
        self.R = 10

        # default initial values for physical state variables
        self.u0_prev = [1, 0, 0, 0, 0]  # assume only 1st generator is on at t=-1
        self.p0_prev = np.array([300, 0, 0, 0, 0])
        self.u0 = [1, 0, 0, 0, 0]  # assume no change at t=0
        self.p0 = np.array([300, 0, 0, 0, 0])

        self.u_prev = np.array([self.u0_prev[i] for i in self.generators], dtype=int)
        self.p_prev = np.array([self.p0_prev[i] for i in self.generators], dtype=float)
        self.u = np.array([self.u0[i] for i in self.generators], dtype=int)
        self.p = np.array([self.p0[i] for i in self.generators], dtype=float)
        self.v, self.w = self._reckless_move(self.u, self.u_prev)
        # assume no change in the past UT-1 and DT-1 periods
        self.v_seq = {i: np.zeros(self.UT[i]) for i in self.generators}
        self.w_seq = {i: np.zeros(self.DT[i]) for i in self.generators}
        # update the first element of v_seq and w_seq, as v and w at time t (t=0)
        self._roll_seq(self.v, self.w)

        assign_env_config(self, kwargs)

        # some attributes from information
        self.min_demand = np.max(self.P_min)
        self.max_demand = np.sum(self.P_max)
        self.NoMinUTDT = np.array(list(set(np.where(self.UT == 0)[0]).intersection(set(np.where(self.DT == 0)[0]))))

        # initialize belief model/data
        self.D_forecast = 0
        self.forecast_model = ForecastModel(self.model_type, self.loc, self.scale, self.nyiso_path)

        self.reset()

        self.raw_action_space = gym.spaces.Dict({
            "on_off": gym.spaces.MultiBinary(self.num_gen),
            "power": gym.spaces.Box(low=self.P_min, high=self.P_max, dtype=np.float32)
        })
        self._action_space = flatten_space(self.raw_action_space)

        self.raw_observation_space = gym.spaces.Dict(
            {
                "u": gym.spaces.MultiBinary(self.num_gen),
                "v_seq": gym.spaces.MultiBinary(self.UT.sum()),
                "w_seq": gym.spaces.MultiBinary(self.DT.sum()),
                "D_forecast": gym.spaces.Box(low=np.max(self.P_min), high=np.sum(self.P_max), shape=(), dtype=float),
                "p": gym.spaces.Box(low=0, high=self.P_max, dtype=np.float32),
                "t": gym.spaces.Box(low=0, high=self.horizon, shape=(), dtype=int)
            })
        self._observation_space = flatten_space(self.raw_observation_space)

    def _get_state(self) -> np.ndarray:
        obs_dict = {
            "u": self.u,
            "v_seq": self._vectorize_seq(self.v_seq),
            "w_seq": self._vectorize_seq(self.w_seq),
            "D_forecast": self.D_forecast,
            "p": self.p,
            "t": np.array([self.t], dtype=np.int32)
        }
        obs_arr = np.concatenate([self.u,
                                 self._vectorize_seq(self.v_seq),
                                 self._vectorize_seq(self.w_seq),
                                 self.D_forecast,
                                 self.p,
                                 np.array([self.t])])
        return obs_arr

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[torch.Tensor, dict]:
        if seed is not None:
            self.set_seed(seed)
        self.t = 0
        self.terminated = False
        self.truncated = False
        self.reward = 0
        self.cost = 0
        # Bus assumed to be 1 only, drop network constraints
        self.u_prev = np.array([self.u0_prev[i] for i in self.generators], dtype=int)
        self.p_prev = np.array([self.p0_prev[i] for i in self.generators], dtype=float)
        self.u = np.array([self.u0[i] for i in self.generators], dtype=int)
        self.p = np.array([self.p0[i] for i in self.generators], dtype=float)
        self.v, self.w = self._reckless_move(self.u, self.u_prev)
        # assume no change in the past UT-1 and DT-1 periods
        self.v_seq = {i: np.zeros(self.UT[i]) for i in self.generators}
        self.w_seq = {i: np.zeros(self.DT[i]) for i in self.generators}
        # update the first element of v_seq and w_seq, as v and w at time t (t=0)
        self._roll_seq(self.v, self.w)

        # forecast the demand
        self.forecast_model.reset()
        self.D_forecast = self._forecast_net_load()

        return torch.from_numpy(self._get_state()).float(), {}

    def _roll_seq(self, v: np.ndarray, w: np.ndarray):
        for i in self.generators:
            if self.v_seq[i].size > 0:
                self.v_seq[i] = np.roll(self.v_seq[i], 1)
                self.v_seq[i][0] = v[i]
            if self.w_seq[i].size > 0:
                self.w_seq[i] = np.roll(self.w_seq[i], 1)
                self.w_seq[i][0] = w[i]

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

    def _penalize_UTDT(self, on_off: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Penalize the on_off action that violates Minimum Up-Time and Down-Time Constraints
        """
        u_new = on_off
        u_curr = self.u
        v_new, w_new = self._reckless_move(u_new, u_curr)
        # seq has UT or DT terms, t+0, t-1, ..., t-UT+1
        # we need UT or DT terms but, t+1 (new, given by reckless move), t+0 (current), t-1, ..., t-UT+2
        sum_v = v_new + np.array([np.sum(self.v_seq[i][:-1]) for i in self.generators])
        sum_w = w_new + np.array([np.sum(self.w_seq[i][:-1]) for i in self.generators])

        UT_violation = sum_v > u_new
        DT_violation = sum_w > (1 - u_new)

        cost = np.sum(UT_violation) + np.sum(DT_violation)
        return float(cost), UT_violation, DT_violation

    @staticmethod
    def _repair_UTDT(on_off: np.ndarray, UT_violation: np.ndarray, DT_violation: np.ndarray) -> np.ndarray:
        """
        Repair the on_off action that violates Minimum Up-Time and Down-Time Constraints
        """
        on_off[UT_violation] = 0  # the decision, on, implies turn-on and violates UT, so must keep it off
        on_off[DT_violation] = 1  # the decision, off, implies turn-off and violates DT, so must keep it on
        return on_off

    def _penalize_Ramping(self, on_off: np.ndarray, power: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Penalize the action that violates Ramping Constraints
        """
        u_new = on_off
        u_curr = self.u_prev
        p_new = power
        p_curr = self.p_prev

        v_new, w_new = self._reckless_move(u_new, u_curr)

        RampUp_violation = p_new - p_curr > self.RU * u_curr + self.SU * v_new
        RampDown_violation = p_curr - p_new > self.RD * u_new + self.SD * w_new

        cost = np.sum(RampUp_violation) + np.sum(RampDown_violation)
        return float(cost), RampUp_violation, RampDown_violation

    def _compute_reserve(self, on_off: np.ndarray, power: np.ndarray) -> np.ndarray:
        """
        the reserve, r, must satisfy the constraints:
        p_{t+1} + r_{t+1} <= P_max * u_{t+1}
        p_{t+1} + r_{t+1} - p_{t} <= RU * u_{t} + SU * v_{t+1}
        """
        u_new = on_off
        u_curr = self.u_prev
        p_new = power
        p_curr = self.p_prev

        v_new, w_new = self._reckless_move(u_new, u_curr)

        r = np.maximum(
                np.minimum(self.P_max * u_new - p_new,
                           self.RU * u_curr + self.SU * v_new + p_curr - p_new),
                0)
        return r

    @staticmethod
    def _compute_power_slack(power: np.ndarray, demand: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        slack = demand - np.sum(power)
        overflow = np.maximum(0, slack)
        underflow = - np.minimum(0, slack)
        return overflow, underflow

    def _compute_production_reward(self, power):
        return - sum(self.a[i] * (power ** 2) + self.b[i] * power + self.c[i] for i in self.generators)

    def _compute_startup_reward(self, turn_on):
        return - sum(turn_on[i] * self.hot_cost[i] for i in self.generators)

    def _compute_shutdown_reward(self, turn_off):
        return - sum(turn_off[i] * self.C_SD[i] for i in self.generators)

    def _compute_fulfillment_reward(self, overflow, underflow):
        return - self.C_LS * (overflow + underflow)

    def _compute_reservation_reward(self, reserve):
        return - self.C_RP * np.maximum(self.R - np.sum(reserve), 0)

    def _forecast_net_load(self) -> np.ndarray:
        net_load_forecast = max(self.min_demand, min(self.forecast_model.forecast(), self.max_demand))
        return np.array([net_load_forecast])

    def _compute_reward(self, demand) -> float:
        reward = 0
        on_off = self.u
        power = self.p
        turn_on = self.v
        turn_off = self.w

        # compute the reserve
        reserve = self._compute_reserve(on_off, power)
        # compute the power slack
        overflow, underflow = self._compute_power_slack(power, demand)

        # compute the reward
        reward += self._compute_production_reward(power)
        reward += self._compute_startup_reward(turn_on)
        reward += self._compute_shutdown_reward(turn_off)
        reward += self._compute_fulfillment_reward(overflow, underflow)
        reward += self._compute_reservation_reward(reserve)
        return reward

    def _compute_cost(self) -> float:
        cost = 0
        on_off = self.u
        power = self.p
        # penalize the action that violates the Ramping Constraints
        Ramping_cost, RampUp_violation, RampDown_violation = self._penalize_Ramping(on_off, power)
        # cannot be repaired, so leads to truncated
        if np.any(RampUp_violation) or np.any(RampDown_violation):
            self.truncated = True
        cost += Ramping_cost
        return cost

    def _get_terminated(self):
        if self.t >= self.horizon:
            self.terminated = True
        else:
            self.terminated = False
        return self.terminated

    def step(self, action) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        action = action.numpy() if torch.is_tensor(action) else action
        assert len(action) == 2 * self.num_gen  # on_off + power
        on_off = action[:self.num_gen]
        power = action[self.num_gen:]
        demand = self.D_forecast
        # now they are at the same time step, t+1

        # penalize the on_off action that violates Minimum Up-Time and Down-Time Constraints
        UTDT_cost, UT_violation, DT_violation = self._penalize_UTDT(on_off)
        # repair the on_off action that violates Minimum Up-Time and Down-Time Constraints
        on_off = self._repair_UTDT(on_off, UT_violation, DT_violation)

        # clip the power action and apply the on_off action
        power = on_off * np.minimum(np.maximum(power, self.P_min), self.P_max)

        # update state except D_forecast
        self.t += 1
        self.p_prev = self.p
        self.p = power
        self.u_prev = self.u
        self.u = on_off
        self.v, self.w = self._reckless_move(self.u, self.u_prev)
        self._roll_seq(self.v, self.w)
        # use self.x for x at time t+1, use demand for demand at time t+1

        # compute the cost
        self.cost += UTDT_cost
        self.cost += self._compute_cost()

        # compute the reward
        self.reward = self._compute_reward(demand)

        # update D_forecast and get state
        if self.t < self.horizon:
            self.D_forecast = self._forecast_net_load()
        state = self._get_state()

        return (
            torch.from_numpy(state).float(),
            torch.tensor(self.reward, dtype=torch.float32),
            torch.tensor(self.cost, dtype=torch.float32),
            torch.tensor(self.terminated, dtype=torch.bool),
            torch.tensor(self.truncated, dtype=torch.bool),
            {}
        )

    def logg(self, *args):
        if self.verbose:
            print(*args)
        return

    @property
    def max_episode_steps(self) -> int:
        return self.horizon

    def render(self) -> Any:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def close(self) -> None:
        return None

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


class ForecastModel():
    def __init__(self, model_type, loc, scale, nyiso_path='data/'):
        self.model_type = model_type
        self.loc = loc
        self.scale = scale
        self.nyiso_path = nyiso_path
        self.model = None
        self.reset()

    def reset(self):
        if self.model_type == "nyiso":
            # make it 25hr to avoid the error in the last step
            forecast_example = np.array(get_random_25hr_forecast(self.nyiso_path)).astype(float)
            factor = self.loc / np.mean(forecast_example)
            forecast_example = forecast_example * factor + np.random.normal(0, self.scale, forecast_example.shape)
            self.model = iter(forecast_example)
        elif self.model_type == "normal":
            self.model = stats.norm(loc=self.loc, scale=self.scale)

    def forecast(self):
        if self.model_type == 'normal':
            return self.model.rvs()
        elif self.model_type == 'nyiso':
            return next(self.model)


# if __name__ == "__main__":
#     import omnisafe
#     ALGO = "CPO"
#     env_id = "Simple-v0"
#     agent = omnisafe.Agent(ALGO, env_id)
#     agent.learn()