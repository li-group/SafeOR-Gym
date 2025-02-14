'''
Unit Commitment
Hao Chen
'''
import logging
import gymnasium as gym
import numpy as np
import torch
import scipy.stats as stats
import pyomo.environ as pe
from or_gym.utils import assign_env_config

from gymnasium.spaces.utils import flatten_space

import random
from typing import Any, ClassVar
from omnisafe.typing import OmnisafeSpace
from omnisafe.envs.core import CMDP, env_register
from omnisafe.common.logger import Logger


@env_register
class UnitCommitmentMasterEnv(CMDP):
    """
    Problem Description:...
    P_max: maximum power output of ith generator
    P_min: minimum power output of ith generator
    a: power generation cost coefficient, quadratic term
    b: power generation cost coefficient, linear term
    c: power generation cost coefficient, constant term
    UT: minimum up time of ith generator
    DT: minimum down time of ith generator
    RU: maximum limit of ramp up rate of ith generator
    RD: maximum limit of ramp down rate of ith generator
    SU: maximum limit of start up rate of ith generator
    SD: maximum limit of shut down rate of ith generator
    hot_cost: hot start cost of ith generator
    cold_cost: cold start cost of ith generator
    cold_hrs: cold start hour of ith generator
    C_SD: shut down cost of ith generator
    C_LS: load shedding cost
    R: reserve requirement

    D: demand of bus n at time t, here assume only 1 bus to relax network constraints
    u: on/off status of ith generator at time t
    v: start-up status of ith generator at time t
    w: shut-down status of ith generator at time t
    """

    _support_envs: ClassVar[list[str]] = ['Simple-v0']
    _action_space: OmnisafeSpace
    _observation_space: OmnisafeSpace
    metadata: ClassVar[dict[str, int]] = {}
    env_spec_log: dict[str, Any]
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1

    def __init__(self, env_id: str, **kwargs: Any,) -> None:
        super().__init__(env_id)

        self.env_spec_log = {}
        self.verbose = False
        self.pyomo_verbose = False
        if not self.pyomo_verbose:
            logging.getLogger('pyomo.core').setLevel(logging.ERROR)
        self.num_periods = 24
        self._max_episode_steps = self.num_periods
        self.horizon = self.num_periods
        self.num_gen = 5
        self.model_type = 'normal'
        self.forecast_model_params = {'loc': 500, 'scale': 10}
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
        self.R = np.array([10, 0, 0, 0, 0])
        # default initial values for physical state variables
        self.u0_prev = np.array([1, 0, 0, 0, 0])  # assume only 1st generator is on at t=-1
        self.u0 = np.array([1, 0, 0, 0, 0])  # assume no change at t=0
        self.p0_prev = np.array([300, 0, 0, 0, 0])

        assign_env_config(self, kwargs)

        # some attributes from information
        self.min_demand = np.max(self.P_min)
        self.max_demand = np.sum(self.P_max)
        self.v_UT_len = {i: self.UT[i] for i in range(self.num_gen)}
        self.w_DT_len = {i: self.DT[i] for i in range(self.num_gen)}
        self.NoMinUTDT = np.array(list(set(np.where(self.UT == 0)[0]).intersection(set(np.where(self.DT == 0)[0]))))
        # initialize belief model/data
        self.forecast_model = ForecastModel(self.model_type, **self.forecast_model_params)
        # initialize the ED model
        self.init_ED()
        # reset the state, reward, truncated, and terminated
        self.reset()

        self.action_space = gym.spaces.MultiBinary(self.num_gen)
        self._action_space = self.action_space

        self.observation_space = gym.spaces.Dict(
            {
                "u": gym.spaces.MultiBinary(self.num_gen),
                "v_UT_arr": gym.spaces.MultiBinary(self.UT.sum()),
                "w_DT_arr": gym.spaces.MultiBinary(self.DT.sum()),
                "D_forecast": gym.spaces.Box(low=np.max(self.P_min), high=np.sum(self.P_max), shape=(), dtype=float),
                "p": gym.spaces.Box(low=0, high=self.P_max, dtype=np.float32),
                "t": gym.spaces.Box(low=0, high=self.horizon, shape=(), dtype=int)
            })
        self.observation_space = flatten_space(self.observation_space)
        self._observation_space = self.observation_space

    def init_ED(self):
        # ED model
        self.model = pe.ConcreteModel()
        self.model.G_set = pe.RangeSet(self.num_gen)
        # mutable parameters with blank initial values, the real initial values are set by _update_ED() in reset()
        self.model.D_forecast = pe.Param(mutable=True)
        self.model.u = pe.Param(self.model.G_set, mutable=True)
        self.model.v = pe.Param(self.model.G_set, mutable=True)
        self.model.w = pe.Param(self.model.G_set, mutable=True)
        self.model.u_prev = pe.Param(self.model.G_set, mutable=True)
        self.model.p_prev = pe.Param(self.model.G_set, mutable=True)
        # fixed parameters
        self.model.P_max = pe.Param(self.model.G_set, initialize={i + 1: self.P_max[i] for i in range(len(self.P_max))})
        self.model.P_min = pe.Param(self.model.G_set, initialize={i + 1: self.P_min[i] for i in range(len(self.P_min))})
        self.model.a = pe.Param(self.model.G_set, initialize={i + 1: self.a[i] for i in range(len(self.a))})
        self.model.b = pe.Param(self.model.G_set, initialize={i + 1: self.b[i] for i in range(len(self.b))})
        self.model.c = pe.Param(self.model.G_set, initialize={i + 1: self.c[i] for i in range(len(self.c))})
        self.model.RU = pe.Param(self.model.G_set, initialize={i + 1: self.RU[i] for i in range(len(self.RU))})
        self.model.RD = pe.Param(self.model.G_set, initialize={i + 1: self.RD[i] for i in range(len(self.RD))})
        self.model.SU = pe.Param(self.model.G_set, initialize={i + 1: self.SU[i] for i in range(len(self.SU))})
        self.model.SD = pe.Param(self.model.G_set, initialize={i + 1: self.SD[i] for i in range(len(self.SD))})
        self.model.hot_cost = pe.Param(self.model.G_set,
                                       initialize={i + 1: self.hot_cost[i] for i in range(len(self.hot_cost))})
        self.model.cold_cost = pe.Param(self.model.G_set,
                                        initialize={i + 1: self.cold_cost[i] for i in range(len(self.cold_cost))})
        self.model.cold_hrs = pe.Param(self.model.G_set,
                                       initialize={i + 1: self.cold_hrs[i] for i in range(len(self.cold_hrs))})
        self.model.C_SD = pe.Param(self.model.G_set, initialize={i + 1: self.C_SD[i] for i in range(len(self.C_SD))})
        self.model.C_LS = pe.Param(initialize=self.C_LS)
        self.model.R = pe.Param(self.model.G_set, initialize={i + 1: self.R[i] for i in range(len(self.R))})
        # variables
        self.model.p = pe.Var(self.model.G_set)
        self.model.p_bar = pe.Var(self.model.G_set)
        self.model.s_pos = pe.Var(domain=pe.NonNegativeReals)
        self.model.s_neg = pe.Var(domain=pe.NonNegativeReals)
        self.model.s = pe.Var(domain=pe.NonNegativeReals)
        self.model.cp_imp = pe.Var()
        self.model.csu_imp = pe.Var()
        self.model.csd_imp = pe.Var()
        self.model.crelax_imp = pe.Var()
        self.model.c_total_imp = pe.Var()

        # Reserve Requirement
        def reserve_rule(m, i):
            return m.p_bar[i] == m.p[i] + m.R[i]

        self.model.reserve = pe.Constraint(self.model.G_set, rule=reserve_rule)

        # Demand Requirement
        def demand_rule(m):
            return sum(m.p[i] for i in m.G_set) == m.D_forecast

        self.model.demand = pe.Constraint(rule=demand_rule)

        # Generation Bounds
        def p_lb_rule(m, i):
            return m.P_min[i] * m.u[i] <= m.p[i]

        self.model.p_lb = pe.Constraint(self.model.G_set, rule=p_lb_rule)

        def p_ub_rule(m, i):
            return m.p[i] <= m.p_bar[i]

        self.model.p_ub = pe.Constraint(self.model.G_set, rule=p_ub_rule)

        def p_bar_ub_rule(m, i):
            return m.p_bar[i] <= m.P_max[i] * m.u[i]

        self.model.p_bar_ub = pe.Constraint(self.model.G_set, rule=p_bar_ub_rule)

        # Ramping Constraints
        def ramp_up_rule(m, i):
            return m.p_bar[i] - m.p_prev[i] <= m.RU[i] * m.u_prev[i] + m.SU[i] * m.v[i]

        self.model.ramp_up = pe.Constraint(self.model.G_set, rule=ramp_up_rule)

        def ramp_down_rule(m, i):
            return m.p_prev[i] - m.p[i] <= m.RD[i] * m.u[i] + m.SD[i] * m.w[i]

        self.model.ramp_down = pe.Constraint(self.model.G_set, rule=ramp_down_rule)

        # Cost Function and Objective
        def production_cost_rule(m):
            return m.cp_imp == sum(m.a[i] * (m.p[i] ** 2) + m.b[i] * m.p[i] + m.c[i] for i in m.G_set)

        self.model.production_cost = pe.Constraint(rule=production_cost_rule)

        def startup_cost_rule(m):
            return m.csu_imp == sum(m.v[i] * m.hot_cost[i] for i in m.G_set)

        self.model.startup_cost = pe.Constraint(rule=startup_cost_rule)

        def shutdown_cost_rule(m):
            return m.csd_imp == sum(m.w[i] * m.C_SD[i] for i in m.G_set)

        self.model.shutdown_cost = pe.Constraint(rule=shutdown_cost_rule)

        def relaxation_cost_rule(m):
            return m.crelax_imp == m.C_LS * (m.s_pos + m.s_neg)

        self.model.relaxation_cost = pe.Constraint(rule=relaxation_cost_rule)

        def total_cost_rule(m):
            return m.c_total_imp == m.cp_imp + m.csu_imp + m.csd_imp + m.crelax_imp

        self.model.total_cost = pe.Constraint(rule=total_cost_rule)

        self.model.obj = pe.Objective(expr=self.model.c_total_imp, sense=pe.minimize)
        self.solver = pe.SolverFactory('gurobi')
        self.solver.options['NonConvex'] = 2

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict]:
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

    def _forecast_net_load(self):
        net_load_forecast = max(self.min_demand, min(self.forecast_model.forecast(), self.max_demand))
        return net_load_forecast

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
        return new_state, torch.as_tensor(self.reward), torch.as_tensor(self.cost), \
                torch.as_tensor(self.terminated), torch.as_tensor(self.truncated), {}

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

class ForecastModel():
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type

        if self.model_type == 'normal':
            self.model = stats.norm(loc=kwargs.get('loc'), scale=kwargs.get('scale'))
        elif self.model_type == 'np_array':
            self.model = iter(kwargs.get('data'))

    def forecast(self):
        if self.model_type == 'normal':
            return self.model.rvs()
        elif self.model_type == 'np_array':
            return next(self.model)



if __name__ == "__main__":
    import omnisafe
    ALGO = "CPO"
    env_id = "Simple-v0"
    agent = omnisafe.Agent(ALGO, env_id)
    agent.learn()