'''
Unit Commitment
Hao Chen
'''
import random
from typing import Any, ClassVar, List, Tuple, Optional, Dict

import torch
import numpy as np

import scipy.stats as stats
from or_gym.envs.power_system.forecast import get_random_25hr_forecast
from Unit_Commitment_Gym import UnitCommitmentMasterEnv

import gymnasium as gym

import omnisafe
from omnisafe.typing import OmnisafeSpace, DEVICE_CPU
from omnisafe.envs.core import CMDP, env_register
from omnisafe.common.logger import Logger


@env_register
class UnitCommitmentMasterEnvSafe(CMDP):
    _support_envs: ClassVar[list[str]] = ['UC-v0, UC-v1']
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True

    _device = DEVICE_CPU
    _num_envs = 1

    def __init__(self, env_id: str,
                 **kwargs: Any) -> None:
        super().__init__(env_id)
        # Instantiate the environment object
        self._env = UnitCommitmentMasterEnv(env_id=env_id, **kwargs.get('env_cfgs', {}))
        # Specify the action space for initialization by the algorithm layer
        self._action_space = self._env.action_space
        # Specify the observation space for initialization by the algorithm layer
        self._observation_space = self._env.observation_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
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
        obs, reward, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy(),
        )
        # Gymnasium does not explicitly include safety constraints; this is just a placeholder.
        cost = np.zeros_like(reward)#self.env.cost
        # Convert dynamic information into torch tensor.
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        return obs, reward, cost, terminated, truncated, {}

    @property
    def max_episode_steps(self) -> int | None:
        # Return the maximum number of interaction steps per episode in the environment
        return self._env.horizon

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        # Release the environment instance after training ends
        self._env.close()

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


# if __name__ == "__main__":
#     import omnisafe
#     ALGO = "CPO"
#     env_id = "UC-v0"
#     custom_cfgs = {
#         'train_cfgs': {
#             'total_steps': 30,
#         },
#         'algo_cfgs': {
#             'steps_per_epoch': 10,
#             'update_iters': 2,
#         },
#     }
#     agent = omnisafe.Agent(ALGO, env_id, custom_cfgs=custom_cfgs)
#     custom_cfgs.update({'env_cfgs': {'P_max': np.array([456, 130, 130, 80, 55])}})
#     agent = omnisafe.Agent(ALGO, env_id, custom_cfgs=custom_cfgs)
#
#     agent.learn()