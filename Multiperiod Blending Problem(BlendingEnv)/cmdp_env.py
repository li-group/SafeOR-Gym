import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import warnings
import torch
from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train

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

from gym_env import BlendEnv
@env_register
class Blendenv_safe(CMDP):
    _support_envs = ['Blending-simple']
    need_auto_reset_wrapper = True  
    need_time_limit_wrapper = True  
    num_envs = 1
    def __init__(self, env_id: str,
                 **kwargs: Any) -> None:
        super().__init__(env_id)
        #print(kwargs)
        self._device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        # Instantiate the environment object
        self._env = BlendEnv(env_id=env_id, **kwargs.get('env_init_cfgs', {}))
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
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
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

'''if __name__ == "__main__":
    import omnisafe
    ALGO = "CPO"
    env_id = 'Capacity-Expansion'
    env_config = {'env_init_cfgs':{
    'T': 2,
    'gencap': {"i1": 10},
    'maxgen': {"i1":{ "r1": 1,"r2":1}},
    'installcost': {"generators": {"i1": 10}},
    'demand': {"r1":{"1": 10,"2":10},"r2":{"1": 10,"2":10}}}}
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 60000,
            },
            'algo_cfgs': {
                'steps_per_epoch': 1000,
            'update_iters': 2,
            },
            'logger_cfgs': {'window_lens' : 500},
            'env_cfgs': env_config,
            'model_cfgs': {
                'actor': {
                    'hidden_sizes': [64, 64],
                    'activation': 'relu',
                    'output_activation': 'tanh'}}
    }
    agent = omnisafe.Agent(ALGO, env_id, custom_cfgs=custom_cfgs)
    agent.learn()'''
'''def recurse(eg,current, path=[],):
    if isinstance(current, dict):
        for k, v in current.items():
            recurse(eg,v, path + [str(k)])
    else:
        key = 'env_cfgs:' + ':'.join(path)
        val = current if isinstance(current, list) else [current]
        eg.add(key, val)

if __name__ == '__main__':
    algos =  ['CPO','TRPOLag', 'P3O', 'OnCRPO', 'DDPGLag']
    eg = ExperimentGrid(exp_name='Benchmark_Safety_simple_blend_fin_no')

    mujoco_envs = [
        'Blending-simple'
    ]
    eg.add('env_id', mujoco_envs)

    # Set the device.
    avaliable_gpus = list(range(torch.cuda.device_count()))
    gpu_id = [0]
    # if you want to use CPU, please set gpu_id = None
    # gpu_id = None

    if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
        warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
        gpu_id = None
    num_steps_per_episode = 6
    num_episodes_per_epoch = 100
    num_steps_per_epoch = num_episodes_per_epoch*num_steps_per_episode
    total_epochs = 2000
    total_steps = num_steps_per_epoch*total_epochs
    #eg.add('algo',base_policy+naive_lagrange_policy+first_order_policy+second_order_policy+saute_policy+simmer_policy+primal_policy+off_policy)
    eg.add('algo',algos)
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('logger_cfgs:use_tensorboard', [True])

    eg.add('train_cfgs:vector_env_nums', [1])
    eg.add('train_cfgs:torch_threads', [1])
    env_config = {'env_init_cfgs':{'config_file':'Blending_no_config.json'}}
    recurse(eg, env_config)
    eg.add('model_cfgs:actor:output_activation', ['tanh'])
    eg.add('algo_cfgs:steps_per_epoch', [num_steps_per_epoch])
    eg.add('train_cfgs:total_steps', [total_steps])
    eg.add('logger_cfgs:window_lens', [int(num_episodes_per_epoch)])
    eg.add('seed', [0])
    eg.add('train_cfgs:device', ['cuda:0'])
    # total experiment num must can be divided by num_pool
    # meanwhile, users should decide this value according to their machine
    eg.run(train, num_pool=1, gpu_id=gpu_id)


    eg.analyze(parameter='algo', values=None, compare_num=5)
    #eg.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    a = eg.evaluate(num_episodes=10)

    eg = ExperimentGrid(exp_name='Benchmark_Safety_simple_blend_prop_fin')

    mujoco_envs = [
        'Blending-simple'
    ]
    eg.add('env_id', mujoco_envs)

    # Set the device.
    avaliable_gpus = list(range(torch.cuda.device_count()))
    gpu_id = [0]
    # if you want to use CPU, please set gpu_id = None
    # gpu_id = None

    if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
        warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
        gpu_id = None
    #eg.add('algo',base_policy+naive_lagrange_policy+first_order_policy+second_order_policy+saute_policy+simmer_policy+primal_policy+off_policy)
    eg.add('algo',algos)
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('logger_cfgs:use_tensorboard', [True])

    eg.add('train_cfgs:vector_env_nums', [1])
    eg.add('train_cfgs:torch_threads', [1])
    env_config = {'env_init_cfgs':{'config_file':'Blending_prop_config.json'}}
    recurse(eg, env_config)
    eg.add('model_cfgs:actor:output_activation', ['tanh'])
    eg.add('algo_cfgs:steps_per_epoch', [num_steps_per_epoch])
    eg.add('train_cfgs:total_steps', [total_steps])
    eg.add('logger_cfgs:window_lens', [int(num_episodes_per_epoch)])
    eg.add('seed', [0])
    eg.add('train_cfgs:device', ['cuda:0'])
    # total experiment num must can be divided by num_pool
    # meanwhile, users should decide this value according to their machine
    eg.run(train, num_pool=1, gpu_id=gpu_id)


    eg.analyze(parameter='algo', values=None, compare_num=5)
    #eg.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    a = eg.evaluate(num_episodes=10)

    eg = ExperimentGrid(exp_name='Benchmark_Safety_simple_blend_disable_fin')

    mujoco_envs = [
        'Blending-simple'
    ]
    eg.add('env_id', mujoco_envs)

    # Set the device.
    avaliable_gpus = list(range(torch.cuda.device_count()))
    gpu_id = [0]
    # if you want to use CPU, please set gpu_id = None
    # gpu_id = None

    if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
        warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
        gpu_id = None
    #eg.add('algo',base_policy+naive_lagrange_policy+first_order_policy+second_order_policy+saute_policy+simmer_policy+primal_policy+off_policy)
    eg.add('algo',algos)
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('logger_cfgs:use_tensorboard', [True])

    eg.add('train_cfgs:vector_env_nums', [1])
    eg.add('train_cfgs:torch_threads', [1])
    env_config = {'env_init_cfgs':{'config_file':'Blending_disable_config.json'}}
    recurse(eg, env_config)
    eg.add('model_cfgs:actor:output_activation', ['tanh'])
    eg.add('algo_cfgs:steps_per_epoch', [num_steps_per_epoch])
    eg.add('train_cfgs:total_steps', [total_steps])
    eg.add('logger_cfgs:window_lens', [int(num_episodes_per_epoch)])
    eg.add('seed', [0])
    eg.add('train_cfgs:device', ['cuda:0'])
    # total experiment num must can be divided by num_pool
    # meanwhile, users should decide this value according to their machine
    eg.run(train, num_pool=1, gpu_id=gpu_id)


    eg.analyze(parameter='algo', values=None, compare_num=5)
    #eg.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    a = eg.evaluate(num_episodes=10)'''

    
    

    

