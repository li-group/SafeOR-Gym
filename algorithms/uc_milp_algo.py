import yaml
import pickle
import numpy as np
import gymnasium as gym
import pyomo.environ as po
from pyomo.opt import SolverFactory
from typing import Any, Dict, Union
from types import SimpleNamespace

from algorithms.base_algo import BaseMILPSolver
from UnitCommitment.Unit_Commitment_Gym import UnitCommitmentMasterEnv
from UnitCommitment.utils import init_model


class UCMILPSolver(BaseMILPSolver):
    '''
    MILP Solver for Unit Commitment Environment

    Args:
        env : Unit Commitment Environment
        solve_horizon : No of time periods to solve the MILP for

    Attributes:
        solve_horizon : No of time periods to solve the MILP for

    '''

    def __init__(self, env: UnitCommitmentMasterEnv, solve_horizon: int = 1, solver_verbose: bool = False):
        super().__init__(env)

        self.env = env
        self.solve_horizon = solve_horizon
        self.solver_verbose = solver_verbose

    def solve_uc_milp(self, observation: Union[Dict[str, Any], np.ndarray]):
        """
        Solve the full UC MILP model using the data available in UnitCommitmentMasterEnv.
        Returns a dictionary with optimal actions and the total objective value.

        """

        print('----- Solving current environment state as a MILP -----')

        # --- Build a minimal args namespace (as expected by MILP builder) ---
        args = SimpleNamespace()
        args.env_id = self.env.env_id
        # --- Load the required UC data ---

        data = None  # implement a data loader

        # --- Build and solve the Pyomo model ---
        model = init_model(args, data)
        solver = SolverFactory("gurobi")
        solver.options['NonConvex'] = 2
        results = solver.solve(model, tee=self.solver_verbose)

        # --- Extract results if optimal ---
        if results.solver.termination_condition != po.TerminationCondition.optimal:
            print(" MILP Solver failed to find an optimal solution.")
            return None

        print("MILP Solver found an optimal solution.")

        # Extract optimal task schedule and batch sizes
        optimal_actions = {
            "on_off": {(t, i): po.value(model.u[t, i]) for i in model.generators for t in model.T_set},
            "power": {(t, i): po.value(model.p[t, i]) for i in model.generators for t in model.T_set},
            "angle": {(t, n): po.value(model.pi[t, n]) for n in model.buses for t in model.T_set if n > 0},
            "objective": po.value(model.obj)
        }

        return optimal_actions

    def get_action(self, observation: Union[Dict[str, Any], np.ndarray]) -> Dict[str, np.ndarray]:
        optimal_action = self.solve_uc_milp(None)

        return optimal_action

#
# if __name__ == '__main__':
#     # Example usage
#     for env_id in ['UC-v0', 'UC-v1']:
#         env = UnitCommitmentMasterEnv(env_id=env_id)
#         solver = UCMILPSolver(env, solve_horizon=24, solver_verbose=False)
#         opt_action = solver.get_action(None)
#         # save the json
#         with open(f'optimal_action_{env_id}.pkl', 'wb') as f:
#             pickle.dump(opt_action, f)
#         print(f"Optimal Action {env_id}:\n\n", opt_action)
#         print(f"Objective Value {env_id}:\n\n", opt_action['objective'])
