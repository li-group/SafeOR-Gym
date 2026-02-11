print("Running Battery Operation OmniSafe Script: Don't Worry")
import warnings
import torch
from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train
import os


import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from gymnasium.utils import seeding
import json
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os, sys
#from utils import *
#from PIL import Image, ImageDraw, ImageFont
from omnisafe.envs.core import CMDP, env_register, env_unregister
from omnisafe.common.logger import Logger
import random
from typing import Any, ClassVar, List, Tuple, Optional, Dict

import torch as th
import yaml
import os
import numpy as np
import omnisafe

from battery_env_gym import BatteryOperationEnv

@env_register
class BatteryOperaionSafe(CMDP):
    _support_envs = ['Battery-v0']
    need_auto_reset_wrapper = True  
    need_time_limit_wrapper = True  
    num_envs = 1

    def __init__(self, env_id: str,
                 **kwargs: Any) -> None:
        super().__init__(env_id)

        self._device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        # Instantiate the environment object
        self._env = BatteryOperationEnv(env_id=env_id, **kwargs.get('env_init_cfgs', {}))
        # Specify the action space for initialization by the algorithm layer
        self._action_space = self._env.action_space
        # Specify the observation space for initialization by the algorithm layer
        self._observation_space = self._env.observation_space

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Reset the environment
        obs, info = self._env.reset(seed=seed, options=options)
        # Convert the reset observations to a torch tensor.
        return (
            torch.as_tensor(obs, dtype=torch.float32, device=self._device),
            info,
        )
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                    dict]:
        # Read the dynamic information after interacting with the environment
        obs, neg_reward_minus_pos_cost, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy(),
        )
        #print(self._env.demand)
        cost = self._env.cost
        reward = self._env.reward
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        return obs, reward, cost, terminated, truncated, {}
    
    @property
    def max_episode_steps(self) -> Optional[int]:
        # Return the maximum number of interaction steps per episode in the environment
        return self._env.T

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        # Release the environment instance after training ends
        self._env.close()

    def set_seed(self, seed: int) -> None:
        random.seed(seed)

    def spec_log(self, logger: Logger) -> None:
        # Omnisafe method called at the end of each epoch. Averaged values are logged
        for key, value in self.env_spec_log.items():
            logger.store({key: float(value)})
            self.env_spec_log[key] = 0.0

    @property
    def env_spec_log(self):
        return self._env.env_spec_log
    
def recurse(eg,current, path=[],):
        if isinstance(current, dict):
            for k, v in current.items():
                recurse(eg,v, path + [str(k)])
        else:
            key = 'env_cfgs:' + ':'.join(path)
            val = current if isinstance(current, list) else [current]
            eg.add(key, val)
