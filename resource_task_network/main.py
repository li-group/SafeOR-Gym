import os
import sys
import yaml
import pickle
import random
import logging
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

from environment import RTNEnv


def main(args, env_id):
    custom_cfgs = Config.dict2config({
        'seed' : args.seed,
        'device' : 'cpu',
        'epochs' : 3,
        'steps_per_epoch' : 3,
        'max_ep_len' : 10,
        'use_cost' : True,
        'debug' : args.debug,
        'env_cfgs' : Config.dict2config({
            'env_config' : args.env_config
    })
    })
    
    env = RTNEnv('rtn-v0', **custom_cfgs)
    env.reset(seed=0)

    while True:
        action = env.action_space.sample()
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
    env.close()

    agent = Agent(ALGO, 'rtn-v0')  # pass empty custom_cfgs
    agent.learn()

if __name__ == '__main__':
    ALGO = 'CPO'
    env_id = 'rtn-v0'

    parser = argparse.ArgumentParser()

    parser.add_argument('--env_config', type = str, default = "env_config.yaml", help = "Path to yaml file containint environment configuration parameters")
    parser.add_argument('--seed', type = int, default = 10, help = "Seed for reproducability")
    parser.add_argument('--debug', action = "store_true", help = "Enable debugging logging")

    args = parser.parse_args()

    main(args, env_id)

    