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
from omnisafe.common.experiment_grid import ExperimentGrids


def run_experiments(dir_name,env_id,environment_config_file_path,Steps_per_epoch,T,Total_epochs,output_activation_function = None):
    import dir_name.cmdp_env
    eg = ExperimentGrid(exp_name='Run')
    # Define algorithm categories
    base_policy = ['PolicyGradient', 'NaturalPG', 'TRPO', 'PPO']
    naive_lagrange_policy = ['TRPOLag']
    first_order_policy = ['P3O']
    second_order_policy = ['CPO']
    primal_policy = ['OnCRPO']
    offline_policy = ['DDPGLag']

    # Target environment
    eg.add('env_id', env_id)

    # GPU configuration
    available_gpus = list(range(torch.cuda.device_count()))
    gpu_id = [0]
    if gpu_id and not set(gpu_id).issubset(available_gpus):
        warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
        gpu_id = None

    # Set experiment parameters
    eg.add('seed', [10])
    eg.add('algo', second_order_policy + naive_lagrange_policy + first_order_policy + primal_policy + offline_policy)

    # Logging configuration
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('logger_cfgs:use_tensorboard', [True])
    eg.add('logger_cfgs:window_lens', [int(Steps_per_epoch / T)])

    # Parallelism and device
    eg.add('train_cfgs:vector_env_nums', [1])
    eg.add('train_cfgs:torch_threads', [1])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    eg.add('train_cfgs:device', [device])
    Total_steps = Steps_per_epoch*Total_epochs
    eg.add('train_cfgs:total_steps', [Total_steps])

    if output_activation_function is not None:
        eg.add('model_cfgs:actor:output_activation', ['tanh'])
    
    # Algorithm configuration
    eg.add('algo_cfgs:steps_per_epoch', [Steps_per_epoch])

    # Environment config file and parameters
    eg.add('env_cfgs:env_init_config:config_file', [environment_config_file_path])
    # Run training, analysis, and evaluation
    eg.run(train, num_pool = 1, gpu_id=gpu_id)
    eg.analyze(parameter='algo', values = None, compare_num = 5)
    a = eg.evaluate(num_episodes = 10)
