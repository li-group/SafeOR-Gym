import os
import yaml
import re
import json
import pickle
import numpy as np
import cloudpickle
import matplotlib.pyplot as plt
import pyomo.environ as po
from pyomo.environ import SolverFactory
from pyomo.repn.standard_repn import generate_standard_repn


env_config_file = "env_config.yaml"
with open(env_config_file, 'r') as f:
    env_config_data = yaml.safe_load(f)

rtn_res_tasks_file = env_config_data.get('res_n_task_file')
with open(rtn_res_tasks_file, 'rb') as f:
    rtn_res_tasks = pickle.load(f)

demand_file = env_config_data.get('demand_file')
with open(demand_file, 'rb') as f:
    demand = pickle.load(f)

elec_cost_file = env_config_data.get('elec_cost_file')
with open(elec_cost_file, 'rb') as f:
    elec_cost = pickle.load(f)

utility_cost_file = env_config_data.get('utility_cost_file')
with open(utility_cost_file, 'rb') as f:
    utility_costs = pickle.load(f)

final_dict = {
    "reactants": 0,
    "intermediates": 0,
    "products": 0,
    "tasks": 0,
    "equipments" : 0,
    "demand": 0,
    "elec_cost": 0,
    "utility_costs": 0
}

reactants = {}
intermediates = {}
products = {}
tasks = {}
equipments = {}

reactant_id = 0
intermediate_id = 0
product_id = 0
task_id = 0
equipment_id = 0

order_list = ['reactants', 'products', 'intermediates', 'tasks', 'equipments']

for idx, item in enumerate(rtn_res_tasks):
    if idx == 3 or idx == 5:
        continue
    if isinstance(item, dict):
        final_dict[order_list[idx]] = {}
        for k, v in item.items():
            final_dict[order_list[idx]][k] = {k1 : float(v1) for k1, v1 in v.items()}

# Format demand
demand_structured = {}
for product, timeseries in demand.items():
    demand_structured[product] = {k : float(v) for k, v in timeseries.items()}

# Format elec_cost
elec_cost_structured = {str(t+1): float(val.item()) if isinstance(val, np.generic) else val
                        for t, val in enumerate(elec_cost)}

# Format utility_costs
utility_structured = {}
for util_name, util_items in utility_costs.items():
    utility_structured[util_name] = {str(t+1): float(val) for t, val in enumerate(util_items['cost'])}

final_dict['demand'] = demand_structured
final_dict['elec_cost'] = elec_cost_structured
final_dict['utility_costs'] = utility_structured

def convert_for_json(obj):
    """
    Recursively convert:
    - lists → indexed dictionaries,
    - NumPy scalars → native Python scalars.
    """
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return {str(i): convert_for_json(v) for i, v in enumerate(obj)}
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

# Example usage
# Assume `task_dict` is already defined (as you said: a standard Python dictionary)

# Convert and clean the task_dict
structured_task_dict = convert_for_json(rtn_res_tasks[3])

final_dict['tasks'] = structured_task_dict

# Write to JSON
with open("structured_environment_data.json", "w") as f:
    json.dump(final_dict, f, indent=4)


