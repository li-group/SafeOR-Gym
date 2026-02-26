from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type, List
from pathlib import Path
import random
import torch

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import warnings
import torch

import time
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from gymnasium.utils import seeding
import json
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os, sys

from omnisafe.envs.core import CMDP

import random
from typing import Any, ClassVar, List, Tuple, Optional
from omnisafe.envs import env_register

import torch as th
import yaml
import os
import numpy as np
from environments import Env_dict
_PKG_ROOT = Path(__file__).resolve().parent  # .../SafeOR_Gym
_CMDP_CLASS_BY_ENV_ID: Dict[str, Type] = {}

def build_and_register_cmdp_env(
    *,
    base_env_cls: Type,                 # e.g., ASUEnv
    wrapper_class_name: str,            # e.g., "ASU_env_safe"
    support_envs: List[str],            # e.g., ["ASU1"]
):
    """
    Dynamically creates and registers an OmniSafe CMDP wrapper around `base_env_cls`.

    Returns:
        The newly created wrapper class (already decorated with @env_register).
    """

    #@env_register
    class _GeneratedCMDP(CMDP):
        _support_envs = support_envs
        need_auto_reset_wrapper = True  
        need_time_limit_wrapper = True  
        num_envs = 1

        def __init__(self, env_id: str, **kwargs: Any) -> None:
            super().__init__(env_id)

            self._device = kwargs.get(
                "device",
                "cuda" if torch.cuda.is_available() else "cpu",
            )

            # Underlying env gets env_init_cfgs dict merged in
            env_init_cfgs = kwargs.get("env_init_cfgs", {}) or {}
            self._env = base_env_cls(env_id=env_id, **env_init_cfgs)

            self._action_space = self._env.action_space
            self._observation_space = self._env.observation_space

        def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
            obs, info = self._env.reset(seed=seed, options=options)
            return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

        def step(
            self,
            action: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
            # gym env expects numpy
            obs, _, terminated, truncated, info = self._env.step(
                action.detach().cpu().numpy()
            )

            # You said your env stores these on self._env
            cost = self._env.cost
            reward = self._env.reward

            obs, reward, cost, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=self._device)
                for x in (obs, reward, cost, terminated, truncated)
            )

            # Return empty dict to match your current behavior
            return obs, reward, cost, terminated, truncated, {}

        @property
        def max_episode_steps(self) -> Optional[int]:
            return getattr(self._env, "T", None)

        def render(self) -> Any:
            return self._env.render()

        def close(self) -> None:
            self._env.close()

        def set_seed(self, seed: int) -> None:
            random.seed(seed)

        def spec_log(self, logger: "Logger") -> None:
            for key, value in self.env_spec_log.items():
                logger.store({key: float(value)})
                self.env_spec_log[key] = 0.0

        @property
        def env_spec_log(self):
            return self._env.env_spec_log

    # Make the class show up with your chosen name in logs/debugging
    _GeneratedCMDP.__name__ = wrapper_class_name
    _GeneratedCMDP.__qualname__ = wrapper_class_name
    _GeneratedCMDP.__module__ = __name__
    registered_class = env_register(_GeneratedCMDP)
    for _eid in support_envs:
        _CMDP_CLASS_BY_ENV_ID[_eid] = registered_class
    return registered_class

def find_key_by_inner_value(dictionary, g):
    for key, value in dictionary.items():
        if len(value) > 1 and isinstance(value[1], list):
            if g in value[1]:
                return key
    return None
def safeor_make(env_id: str, config_file: str | None = None, **kwargs):
    """
    Instantiate the registered CMDP wrapper for env_id, passing config_file down
    into base env via env_init_cfgs.
    """
    try:
        cmdp_cls = _CMDP_CLASS_BY_ENV_ID[env_id]
    except KeyError as e:
        raise ValueError(
            f"No CMDP registered for env_id='{env_id}'. "
            f"Did you import SafeOR_Gym (or otherwise call build_and_register_cmdp_env) first?"
        ) from e
    if config_file is None:
        dir_name = find_key_by_inner_value(Env_dict,env_id)
        if dir_name is None:
            raise ValueError(
                f"No default config mapping found for env_id='{env_id}'. "
                f"Please pass config_file explicitly."
            )

        # Env_dict[dir_name] is like [a, [b, c], d] -> take d as the filename
        cfg_name = Env_dict[dir_name][2]
        config_file = str(_PKG_ROOT / "envs" / dir_name / cfg_name)
    env_init_cfgs = kwargs.pop("env_init_cfgs", {}) or {}
    env_init_cfgs = {"config_file": config_file, **env_init_cfgs}

    return cmdp_cls(env_id=env_id, env_init_cfgs=env_init_cfgs, **kwargs)
