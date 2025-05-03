import os
import sys
import yaml
import pickle
import random
import logging
import warnings
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from typing import Any, ClassVar, List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import pyomo.environ as po

import gymnasium as gym
from gymnasium.spaces.utils import flatten_space

OMNISAFE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'omnisafe'))
if OMNISAFE_PATH not in sys.path:
    sys.path.insert(0, OMNISAFE_PATH)

from omnisafe import Agent
from omnisafe.utils.config import Config
from omnisafe.utils.exp_grid_tools import train
from omnisafe.common.experiment_grid import ExperimentGrid


from environment import SafeRTN


def main(args, env_id):
    custom_cfgs = Config.dict2config({
        'seed' : args.seed,
        'train_cfgs':{
            'device' : 'cpu',
            'total_steps':1000,
        },
        'algo_cfgs':{
            'steps_per_epoch' : 10
        },
        'model_cfgs' : {
            'actor' : {
                'output_activation' : 'tanh'
            }
        },
        'env_cfgs':{
            'env_init_config' : {
            'config_file' : args.env_config,
            'debug' : args.debug,
            'sanitization_cost_weight' : 1.0,
            'cost_coefficient' : 1.0
            }
        }
            
    })
    
    # env = SafeRTN('rtn-v0', **custom_cfgs)
    # env.reset(seed=0)

    while False:
        action = env.action_space.sample()
        action = torch.from_numpy(action)
        obs, reward, cost, terminated, truncated, info = env.step(action)
        print('-' * 20)
        print(f'action : {action}')
        print(f'action shape = {action.shape}')
        print(f'obs: {obs}')
        print(f'Obs space shape : {obs.shape}')
        print(f'reward: {reward}')
        print(f'cost: {cost}')
        print(f'terminated: {terminated}')
        print(f'truncated: {truncated}')
        print('*' * 20)
        if terminated or truncated:
            break
    # env.close()

    #agent = Agent('CPO', 'rtn-v0', custom_cfgs = custom_cfgs)  # pass empty custom_cfgs
    #agent.learn()

    eg = ExperimentGrid(exp_name = 'Benchmark_Safety_rtn_v0')

    base_policy = ['PolicyGradient', 'NaturalPG', 'TRPO', 'PPO']
    naive_lagrange_policy = ['PPOLag', 'TRPOLag', 'RCPO']
    first_order_policy = ['CUP', 'FOCOPS', 'P3O']
    second_order_policy = ['CPO']#, 'PCPO']

    mujoco_envs = [
        'rtn-v0'
    ]
    eg.add('env_id', mujoco_envs)

    available_gpus = list(range(torch.cuda.device_count()))
    gpu_id = [0]
    
    if gpu_id and not set(gpu_id).issubset(available_gpus):
        warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
        gpu_id = None

    T = 10
    STEPS_PER_EPOCH = 80
    TOTAL_EPOCHS = 1000
    TOTAL_STEPS = STEPS_PER_EPOCH * TOTAL_EPOCHS

    eg.add('seed', [args.seed])
    
    eg.add('algo', second_order_policy)
    
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('logger_cfgs:use_tensorboard', [True])
    eg.add('logger_cfgs:window_lens', [int(STEPS_PER_EPOCH / T)])

    eg.add('train_cfgs:vector_env_nums', [1])
    eg.add('train_cfgs:torch_threads', [1])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    eg.add('train_cfgs:device', [device])
    eg.add('train_cfgs:total_steps', [TOTAL_STEPS])
    
    eg.add('model_cfgs:actor:output_activation', ['tanh'])

    eg.add('algo_cfgs:steps_per_epoch', [STEPS_PER_EPOCH])

    eg.add('env_cfgs:env_init_config:config_file', [args.env_config])
    eg.add('env_cfgs:env_init_config:debug', [args.debug])
    eg.add('env_cfgs:env_init_config:sanitization_cost_weight', [1.0])
    eg.add('env_cfgs:env_init_config:cost_coefficient', [1.0])

    eg.run(train, num_pool = 1, gpu_id=gpu_id)
    eg.analyze(parameter='algo', values = None, compare_num = 1)
    a = eg.evaluate(num_episodes = 1)


if __name__ == '__main__':
    ALGO = 'CPO'
    env_id = 'rtn-v0'

    parser = argparse.ArgumentParser()

    parser.add_argument('--env_config', type = str, default = "easy_environment_data.json", help = "Path to yaml file containint environment configuration parameters")
    parser.add_argument('--seed', type = int, default = 10, help = "Seed for reproducability")
    parser.add_argument('--debug', action = "store_true", help = "Enable debugging logging")

    args = parser.parse_args()

    main(args, env_id)

    