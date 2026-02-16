import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import yaml
import json
import pickle
import logging
import random
from typing import Any, ClassVar, List, Tuple, Optional, Dict

import numpy as np
import torch

from omnisafe.envs.core import CMDP, env_register
from omnisafe.common.logger import Logger

from gym_env import STNEnv

@env_register
class SafeSTN(CMDP):
    _support_envs: ClassVar[List[str]] = ['stn-v0']
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(env_id)
        
        self._device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        # Instantiate the environment object
        self._env = STNEnv(env_id = env_id, **kwargs.get('env_init_cfgs', {}))
        # Specify the action space for initialization by the algorithm layer
        self._action_space = self._env.action_space
        # Specify the observation space for initialization by the algorithm layer
        self._observation_space = self._env.observation_space

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[torch.Tensor, dict]:
        # Reset the environment
        obs, info = self._env.reset(seed=seed, options=options)
        
        # Convert the reset observations to a torch tensor.
        return (
            torch.as_tensor(obs, dtype=torch.float32, device=self._device),
            info,
        )

    def step(self, action : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # Read the dynamic information after interacting with the environment
        obs, neg_reward_minus_pos_cost, terminated, truncated, info = self._env.step(action.detach().cpu().numpy(),)

        cost = self._env.cost
        reward = neg_reward_minus_pos_cost + cost

        # Convert dynamic information into torch tensor.
        obs, reward, cost, terminated, truncated = (torch.as_tensor(x, dtype=torch.float32, device=self._device) for x in (obs, reward, cost, terminated, truncated))

        return obs, reward, cost, terminated, truncated, {}

    @property
    def max_episode_steps(self) -> int:
        # Return the maximum number of interaction steps per episode in the environment
        return self._env.T

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        # Release the environment instance after training ends
        self._env.close()

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def spec_log(self, logger: Logger) -> None:
        # Omnisafe method called at the end of each epoch. Averaged values are logged
        for key, value in self.env_spec_log.items():
            logger.store({key: float(value)})
            self.env_spec_log[key] = 0.0

    @property
    def env_spec_log(self):
        return self._env.env_spec_log