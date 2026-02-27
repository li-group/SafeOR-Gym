import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from register_cmdp import build_and_register_cmdp_env,safeor_make,find_key_by_inner_value
from environments import Env_dict
import importlib
from pathlib import Path
from typing import Optional
current_dir_name = os.path.dirname(os.path.abspath(__file__))
for (dir_name) in Env_dict.keys():
    sys.path.insert(0, os.path.join(current_dir_name,"envs", dir_name))
    module = importlib.import_module(f"envs.{dir_name}.gym_env")
    gym_env_class = getattr(module, Env_dict[dir_name][0])
    build_and_register_cmdp_env(base_env_cls=gym_env_class,
    wrapper_class_name=Env_dict[dir_name][0]+'_safe',
    support_envs=Env_dict[dir_name][1])
    sys.path.remove(os.path.join(current_dir_name,"envs", dir_name))
    if 'utils' in sys.modules:
        del sys.modules['utils']

def optimal_simulation_actions(env,solver,tee: bool = True, raise_on_infeasible: bool = True):
    env_gym = env._env
    env_id = env_gym.env_id
    dir_name = find_key_by_inner_value(Env_dict,env_id)
    sys.path.insert(0, os.path.join(current_dir_name,"envs", dir_name))
    module = importlib.import_module(f"envs.{dir_name}.full_space_opt")
    func = getattr(module, 'optimal_simulation')
    return func(env_gym,solver,tee, raise_on_infeasible)
    
    
    
    
def package_root() -> Path:
    """Absolute path to the SafeOR_Gym package directory."""
    return Path(__file__).resolve().parent


def get_default_config_path(env_id: str) -> Path:
    """
    Return the default config file path (as a Path) for a given env_id.
    """
    dir_name = find_key_by_inner_value(Env_dict, env_id)
    if dir_name is None:
        raise ValueError(
            f"No default config mapping found for env_id='{env_id}'. "
            f"Please pass environment_config_file_path explicitly."
        )

    cfg_name = Env_dict[dir_name][2]
    return package_root() / "envs" / dir_name / cfg_name


def resolve_config_path(env_id: str, config_file: Optional[str | Path]) -> Path:
    """
    If config_file is provided, return it as an absolute Path (relative paths resolved from CWD).
    Otherwise return the package default for env_id.
    """
    if config_file is None:
        return get_default_config_path(env_id)
    return Path(config_file).expanduser().resolve()

