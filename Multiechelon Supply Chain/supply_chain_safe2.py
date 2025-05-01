print("Running Supply Chain OmniSafe Script: Don't Worry")
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

from supply_chain_gym import InvMgmtEnv

@env_register
class SupplyChainSafe(CMDP):
    _support_envs = ['SupplyChain-v0']
    need_auto_reset_wrapper = True  
    need_time_limit_wrapper = True  
    num_envs = 1

    def __init__(self, env_id: str,
                 **kwargs: Any) -> None:
        super().__init__(env_id)

        self._device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        # Instantiate the environment object
        self._env = InvMgmtEnv(env_id=env_id, **kwargs.get('env_init_cfgs', {}))
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

if __name__ == '__main__':
    
    eg = ExperimentGrid(exp_name='Benchmark_SupplyChain')

    # 1) Define your policy lists
    # base_policy            = ['PolicyGradient', 'NaturalPG', 'TRPO', 'PPO']
    # naive_lagrange_policy  = ['PPOLag', 'TRPOLag', 'RCPO']
    # first_order_policy     = ['CUP', 'FOCOPS', 'P3O']
    second_order_policy    = ['CPO', 'PCPO']
    # saute_policy           = ['PPOSaute', 'TRPOSaute']
    # simmer_policy          = ['PPOSimmerPID', 'TRPOSimmerPID']
    # primal_policy          = ['OnCRPO']
    # off_policy             = ['DDPG', 'SAC', 'TD3', 'DDPGLag', 'TD3Lag',
    #                           'SACLag', 'DDPGPID', 'TD3PID', 'SACPID']

    # 2) Which env to run
    eg.add('env_id', ['SupplyChain-v0'])

    # 3) Load & preprocess env-init config from JSON
    script_dir = os.path.dirname(os.path.realpath(__file__))

    cfg_path = os.path.join(script_dir, "config.json")
    # init_cfg = load_config(cfg_path)

    # 4) Wrap under 'env_init_cfgs' for the grid
    # env_config = {'env_init_cfgs': init_cfg}

    # 5) Specify algorithms
    eg.add('algo', second_order_policy)

    # 6) Logger and training settings
    eg.add('logger_cfgs:use_wandb',    [False])
    eg.add('logger_cfgs:use_tensorboard', [True])
    eg.add('train_cfgs:vector_env_nums',   [1])
    eg.add('train_cfgs:torch_threads',     [1])
    # eg.add('env_cfgs', [env_config])
    eg.add('env_cfgs:env_init_cfgs:config_path', [cfg_path])

    # # 7) Inject the preprocessed config and recurse
    # recurse(eg, env_config)

    # 8) The rest of your grid setup
    eg.add('model_cfgs:actor:output_activation', ['tanh'])
    num_steps_per_epoch = 360
    total_epochs        = 1000
    total_steps         = num_steps_per_epoch * total_epochs

    eg.add('algo_cfgs:steps_per_epoch', [num_steps_per_epoch])
    eg.add('train_cfgs:total_steps',    [total_steps])
    # eg.add('logger_cfgs:window_lens',   [int(num_steps_per_epoch / init_cfg['T'])])
    eg.add('seed',                      [0])
    eg.add('train_cfgs:device',         ['cuda:0'])

    # 9) Determine GPU(s)
    available_gpus = list(range(torch.cuda.device_count()))
    gpu_id = [0]
    if gpu_id and not set(gpu_id).issubset(available_gpus):
        warnings.warn('The GPU ID is not available, using CPU instead.', stacklevel=1)
        gpu_id = None

    # 10) Run, analyze, evaluate
    eg.run(train, num_pool=1, gpu_id=gpu_id)
    eg.analyze(parameter='algo', compare_num=1)
    results = eg.evaluate(num_episodes=10)
    print(results)
















