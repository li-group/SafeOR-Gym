from register_cmdp import build_and_register_cmdp_env
from environments import Env_dict
import importlib
for (dir) in Env_dict.keys():
    module = importlib.import_module('envs/'+ dir+ '/gym_env')
    gym_env_class = getattr(module, Env_dict[dir][1])
    build_and_register_cmdp_env(base_env_cls=gym_env_class,
    wrapper_class_name=Env_dict[dir][1]+'_safe',
    support_envs=Env_dict[dir][2],
    need_auto_reset_wrapper=True,
    need_time_limit_wrapper=True,
    num_envs=1,)
