import numpy as np
import tqdm as tqdm
import gymnasium as gym

from typing import Any, Dict, Union
from collections.abc import Sequence

class BaseMILPSolver:
    '''
    Base abstract class for solving MILP of a given environment
    Subclasses are expected to implement the get_action() method based on the specific environment
    
    Args:
        env : Environment to solve MILP for
        solve_horizon : Number of time periods for which the optimal action is to be returned
    
    '''

    def __init__(self, env, solve_horizon : int = 1, solver_verbose : bool = False):
        self.env = env
        self.solve_horizon = solve_horizon
        self.solver_verbose = solver_verbose

    def get_action(self, state : Union[Dict[str, Any], np.ndarray]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        '''
        Returns the action based on the solution of the MILP algorithm

        '''
        return _solve_one_horizon(self, state)
    
    def get_actions(self, state : Union[Dict[str, Any], np.ndarray]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        '''
        Returns the action based on the solution of the MILP algorithm

        '''
        return _solve_full_horizon(self, state)
    
    def reset(self) -> None:
        '''
        Resets the algorithm
        '''
        

class RandomAlgorithm(BaseMILPSolver):
    '''
    Random action

    '''
    def get_action(self, observation : Dict[str, Any]) -> Any:
        ''' Returns a random action '''
        return self.env.action_space.sample()