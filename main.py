import os
import sys
import yaml
import pickle
import random
import logging
import warnings
import argparse
import importlib
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from typing import Any, ClassVar, List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import pyomo.environ as po

import gymnasium as gym
from gymnasium.spaces.utils import flatten_space

OMNISAFE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'omnisafe'))
if OMNISAFE_PATH not in sys.path:
    sys.path.insert(0, OMNISAFE_PATH)

from omnisafe import Agent
from omnisafe.utils.config import Config
from omnisafe.utils.exp_grid_tools import train
from omnisafe.common.experiment_grid import ExperimentGrid


def run_experiments(args):
    importlib.import_module(f"{args.dir_name}.cmdp_env")
    eg = ExperimentGrid(exp_name='Run')

    # Define algorithm categories
    base_policy = ['PolicyGradient', 'NaturalPG', 'TRPO', 'PPO']
    naive_lagrange_policy = ['TRPOLag']
    first_order_policy = ['P3O']
    second_order_policy = ['CPO']
    primal_policy = ['OnCRPO']
    offline_policy = ['DDPGLag']

    # Target environment
    eg.add('env_id', args.env_id)

    # GPU configuration
    available_gpus = list(range(torch.cuda.device_count()))
    gpu_id = [args.gpu_id] if args.gpu_id is not None else None
    if gpu_id and not set(gpu_id).issubset(available_gpus):
        warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
        gpu_id = None

    # Set experiment parameters
    eg.add('seed', [args.seed])
    eg.add('algo', second_order_policy + naive_lagrange_policy + first_order_policy + primal_policy + offline_policy)

    # Logging configuration
    eg.add('logger_cfgs:use_wandb', [args.use_wandb])
    eg.add('logger_cfgs:use_tensorboard', [args.use_tensorboard])
    eg.add('logger_cfgs:window_lens', [int(args.steps_per_epoch / args.T)])

    # Parallelism and device
    eg.add('train_cfgs:vector_env_nums', [args.vector_env_nums])
    eg.add('train_cfgs:torch_threads', [args.torch_threads])

    # Device selection
    if args.device is not None:
        device = args.device
    else:
        device = f"cuda:{args.gpu_id}" if (torch.cuda.is_available() and args.gpu_id is not None) else \
                 ("cuda:0" if torch.cuda.is_available() else "cpu")
    eg.add('train_cfgs:device', [device])

    total_steps = args.steps_per_epoch * args.total_epochs
    eg.add('train_cfgs:total_steps', [total_steps])

    # Model output activation
    # Preserves your current behavior: if provided, force 'tanh'
    if args.output_activation_function is not None:
        eg.add('model_cfgs:actor:output_activation', [args.output_activation_function])

    # Algorithm configuration
    eg.add('algo_cfgs:steps_per_epoch', [args.steps_per_epoch])

    # Environment config file and parameters
    eg.add('env_cfgs:env_init_config:config_file', [args.environment_config_file_path])

    # Run training, analysis, and evaluation
    eg.run(train, num_pool=args.num_pool, gpu_id=gpu_id)
    eg.analyze(parameter='algo', values=None, compare_num=args.compare_num)
    results = eg.evaluate(num_episodes=args.num_episodes)

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run CMDP experiments via ExperimentGrid.")

    # Required args (match run_experiments signature / usage)
    p.add_argument("--dir_name", type=str, required=True, help="Python package/module directory name containing cmdp_env (e.g. 'myproj').")
    p.add_argument("--env_id", type=str, required=True,  help="Environment ID to pass to ExperimentGrid (env_id).")
    p.add_argument("--environment_config_file_path", type=str, required=True, help="Path to environment config file used by env_init_config.")
    p.add_argument("--steps_per_epoch", type=int, required=True, help="Steps per epoch.")
    p.add_argument("--T", type=int, required=True, help="Horizon (used for window length = steps_per_epoch / T).")
    p.add_argument("--total_epochs", type=int, required=True, help="Total number of epochs; total_steps = steps_per_epoch * total_epochs.")

    # Optional knobs (sensible defaults)
    p.add_argument("--output_activation_function", type=str, default=None, help="If set, actor output activation (e.g. 'tanh').")

    p.add_argument("--seed", type=int, default=10, help="Random seed.")
    p.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--use_tensorboard", action="store_true", default=True, help="Enable TensorBoard logging (default: True).")

    p.add_argument("--vector_env_nums", type=int, default=1, help="Number of vector envs.")
    p.add_argument("--torch_threads", type=int, default=1, help="Torch threads.")

    p.add_argument("--device", type=str, default=None, help="Device string (e.g. 'cpu', 'cuda:0'). If omitted, auto-select.")
    p.add_argument("--gpu_id", type=int, default=0, help="GPU id to use (only relevant if CUDA available).")

    p.add_argument("--num_pool", type=int, default=1, help="ExperimentGrid worker pool size.")
    p.add_argument("--compare_num", type=int, default=5, help="Compare_num for eg.analyze.")
    p.add_argument("--num_episodes", type=int, default=10, help="Episodes for evaluation.")

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
    