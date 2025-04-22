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

    def __init__(self, env, solve_horizon : int = 1):
        self.env = env
        self.solve_horizon = solve_horizon

    def get_action(self, observation : Union[Dict[str, Any], np.ndarray]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        '''
        Returns the action based on the solution of the MILP algorithm

        '''
        raise NotImplementedError

    def reset(self) -> None:
        '''
        Resets the algorithm
        '''
        pass

class RandomAlgorithm(BaseMILPSolver):
    '''
    Random action

    '''
    def get_action(self, observation : Dict[str, Any]) -> Any:
        ''' Returns a random action '''
        return self.env.action_space.sample()