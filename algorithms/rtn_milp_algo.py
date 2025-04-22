import yaml
import pickle
import numpy as np
import gynamsium as gym
import pyomo.environ as po
from pyomo.opt import SolverFactory
from typing import Any, Dict, Union
from types import SimpleNamespace

from algorithms.base_algo import BaseMILPSolver
from resource_task_network.environment import RTNEnv
from resource_task_network.utils import init_model

class RTNMILPSolver(BaseMILPSolver):
    '''
    MILP Solver for RTN Environment

    Args:
        env : RTN Environment
        solve_horizon : No of time periods to solve the MILP for

    Attributes:
        solve_horizon : No of time periods to solve the MILP for
    
    '''

    def __init__(self, env : RTNEnv, solve_horizon : int = 1, solver_verbose : bool = False):
        super().__init__(env)

        self.env = env
        self.solve_horizon = solve_horizon
        self.solver_verbose = solver_verbose
        

    def solve_rtn_milp(self, observation : Union[Dict[str, Any], np.ndarray]):
        """
        Solve the full RTN MILP model using the data available in RTNEnv.
        Returns a dictionary with optimal actions and the total profit.
        
        """
        
        print('----- Solving current environment state as a MILP -----')

        # --- Build a minimal args namespace (as expected by MILP builder) ---
        args = SimpleNamespace()
        args.horizon = self.solve_horizon

        # --- Unpack the required RTN data ---
        net = self.env.rtn_graph
        resources_and_task = self.env.rtn_res_tasks
        demand_dict = self.env.demand

        # Build supply_dict for reactants
        reactants = list(resources_and_task[0].keys())
        supply_dict = {
            r: {t: 0.0 for t in range(1, self.solve_horizon + 1)}
            for r in reactants
        }

        # Convert utility cost time series into expected format
        utility_cost_dict = {
            u: {"cost": [self.env.utility_costs[t].get(u, 0.0) for t in range(self.solve_horizon)]}
            for u in resources_and_task[5].keys()
        }

        # --- Build and solve the Pyomo model ---
        model = init_model(args, net, resources_and_task, demand_dict, supply_dict, utility_cost_dict)
        solver = SolverFactory("gurobi")
        results = solver.solve(model, tee = self.solver_verbose)

        # --- Extract results if optimal ---
        if results.solver.termination_condition != po.TerminationCondition.optimal:
            print(" MILP Solver failed to find an optimal solution.")
            return None

        print("MILP Solver found an optimal solution.")

        # Extract optimal task schedule and batch sizes
        optimal_actions = {
            "batch": {(i, t): po.value(model.E[i, t]) for i in model.I for t in model.T},
            "objective": po.value(model.obj)
        }

        return optimal_actions
    

    def get_action(self, observation : Union[Dict[str, Any], np.ndarray]) -> Dict[str, np.ndarray]:
        optimal_action = self.solve_rtn_milp()

        return optimal_action['batch']



