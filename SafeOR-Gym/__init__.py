import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from register_cmdp import build_and_register_cmdp_env,safeor_make
from environments import Env_dict
import importlib
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

