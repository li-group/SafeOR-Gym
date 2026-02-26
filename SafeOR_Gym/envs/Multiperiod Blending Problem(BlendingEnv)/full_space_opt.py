
import sys, os

curr_dir = os.path.abspath(os.getcwd())
sys.path.append(curr_dir)

import numpy as np
import pandas as pd
import random as rd
import datetime, time
import math as m
import argparse
import copy
import json, torch
from pyomo.environ import *

from pyomo.opt import SolverFactory, TerminationCondition

def flatten_dict(dictionary, parent_key='', separator=';'):
    items = []
    for key, value in dictionary.items():
        # Convert tuple keys to a string format before concatenating
        if isinstance(key, tuple):
            key = '_'.join(map(str, key))  # Convert tuple to string format (e.g., ('a', 'b') -> 'a_b')

        new_key = parent_key + separator + str(key) if parent_key else key

        if isinstance(value, dict):
            # If the value is a dictionary (including empty ones), recurse
            if value:
                items.extend(flatten_dict(value, new_key, separator=separator).items())
            #else:
                # Add empty dictionaries as well
                #items.append((new_key, {}))
        else:
            items.append((new_key, value))
    return dict(items)
def flatten_and_track_mappings(dictionary, separator=';'):
    # Flatten the dictionary using the updated flatten_dict function
    flattened_dict = flatten_dict(dictionary, separator=separator)
    
    # Track the mappings of index to the key split by separator
    mappings = []
    for index, (key, value) in enumerate(flattened_dict.items()):
        # Ensure the key is a string before splitting by separator
        mapped_key = key.split(separator)  # Split the string key by the separator
        mappings.append((index, mapped_key))
    
    # Convert the flattened values to a numpy array of float32, but skip non-numeric values
    flattened_values = []
    for value in flattened_dict.values():
        if isinstance(value, (int, float)):  # Only include numeric types
            flattened_values.append(value)
        elif isinstance(value, dict):  # Skip dictionaries
            flattened_values.append(0)  # Or set a default value for empty or nested dictionaries
    
    # Convert the valid numeric values to a numpy array
    flattened_array = np.array(flattened_values).astype("float32")
    
    return flattened_array, mappings
def get_sbp(connections):
    # Function to obtain the list of source, blending, demand tank names from the connections
    
    sources = list(connections["source_blend"].keys())
    
    b_list = list(connections["blend_blend"].keys())
    for b in connections["blend_blend"].keys():
        b_list += connections["blend_blend"][b]
    b_list += list(connections["blend_demand"].keys())
    blenders = list(set(b_list))
    
    p_list = []
    for p in connections["blend_demand"].keys():
        p_list += connections["blend_demand"][p]
    demands = list(set(p_list))
    
    return sources, blenders, demands


def build_optimization_model(env):

    alpha      = env.alpha
    beta       = env.beta
    tau0       = env.tau0 
    delta0     = env.delta0
    sigma      = env.sigma
    sigma_ub   = env.sigma_ub
    sigma_lb   = env.sigma_lb
    s_inv_lb   = env.s_inv_lb
    s_inv_ub   = env.s_inv_ub
    d_inv_lb   = env.d_inv_lb
    d_inv_ub   = env.d_inv_ub
    betaT_d    = env.betaT_d
    betaT_s    = env.betaT_s
    b_inv_ub   = env.b_inv_ub
    b_inv_lb   = env.b_inv_lb
    T          = env.T
    window_len = env.window_len
    properties = env.properties
    action_sample = env.action_sample
    connections = env.connections
    sources, blenders, demands = get_sbp(connections)
    timestamps_act = list(range(1,T+1))
    timestamps_inv = list(range(T+1))
    # Model
    model = ConcreteModel()

    # Sets
    model.sources = Set(initialize=sources)
    model.demands = Set(initialize=demands)
    model.blenders = Set(initialize=blenders)
    model.properties = Set(initialize=properties)
    model.timestamps_inv = Set(initialize=timestamps_inv)
    model.timestamps_act = Set(initialize=timestamps_act)

    # Parameters
    model.alpha = Param(initialize=alpha)
    model.beta = Param(initialize=beta)
    model.s_inv_ub = Param(model.sources, initialize=s_inv_ub)
    #model.tau0 = Param(model.sources, initialize=tau0)
    model.sigma_lb = Param(model.demands, initialize=sigma_lb)
    model.sigma_ub = Param(model.demands, initialize=sigma_ub)
    model.d_inv_ub = Param(model.demands, initialize=d_inv_ub)
    #model.delta0 = Param(model.demands, initialize=delta0)
    model.betaT_s = Param(model.sources, initialize=betaT_s)
    model.betaT_d = Param(model.demands, initialize=betaT_d)
    model.b_inv_ub = Param(model.blenders, initialize=b_inv_ub)
    
    # Decision variables
    # Before flow but after buy
    model.source_inv = Var(model.sources, model.timestamps_inv, domain=NonNegativeReals)
    model.blend_inv = Var(model.blenders, model.timestamps_inv, domain=NonNegativeReals)
    model.demand_inv = Var(model.demands, model.timestamps_inv, domain=NonNegativeReals)

    model.demand_sold = Var(model.demands, model.timestamps_act, domain=NonNegativeReals) # Represents the amount of product sold at each timestep; necessary for objective function
    model.offer_bought = Var(model.sources, model.timestamps_act, domain=NonNegativeReals) # Represents the amount of product sold at each timestep; necessary for objective function

    model.prop_blend_inv = Var(model.properties, model.blenders, model.timestamps_inv, domain=NonNegativeReals)

    model.source_blend_flow = Var(model.sources, model.blenders, model.timestamps_act, domain=NonNegativeReals)
    model.blend_blend_flow = Var(model.blenders, model.blenders, model.timestamps_act, domain=NonNegativeReals)
    model.blend_demand_flow = Var(model.blenders, model.demands, model.timestamps_act, domain=NonNegativeReals)

    model.source_blend_bin = Var(model.sources, model.blenders, model.timestamps_act, domain=Binary)
    model.blend_blend_bin = Var(model.blenders, model.blenders, model.timestamps_act, domain=Binary)
    model.blend_demand_bin = Var(model.blenders, model.demands, model.timestamps_act, domain=Binary)

    # flow = 0 if the pair is not in the dict connections
    def connections_rule0_1(model, s, j, t):
        if j not in connections["source_blend"][s]:
            return model.source_blend_flow[s, j, t] == 0
        else:
            return model.source_blend_flow[s, j, t] >= 0
        
    def connections_rule0_2(model, j, p, t):
        if p not in connections["blend_demand"][j]:
            return model.blend_demand_flow[j, p, t] == 0
        else:
            return model.blend_demand_flow[j, p, t] >= 0
        
    def connections_rule0_3(model, j1, j2, t):
        if j2 not in connections["blend_blend"][j1]:
            return model.blend_blend_flow[j1, j2, t] == 0
        else:
            return model.blend_blend_flow[j1, j2, t] >= 0
    
    model.material_balance_rule0_1 = Constraint(model.sources,  model.blenders, model.timestamps_act, rule=connections_rule0_1)
    model.material_balance_rule0_2 = Constraint(model.blenders, model.demands,  model.timestamps_act, rule=connections_rule0_2)
    model.material_balance_rule0_3 = Constraint(model.blenders, model.blenders, model.timestamps_act, rule=connections_rule0_3)

    # Inventory bounds
    def connections_rule0_1_1(model, j, t):
        return model.blend_inv[j, t] <= model.b_inv_ub[j]

    model.material_balance_rule0_1_1 = Constraint(model.blenders, model.timestamps_inv, rule=connections_rule0_1_1)

    # Cannot buy more than what is available
    def material_balance_rule1_0(model, s, t):
        return model.offer_bought[s, t] <= tau0[s][str(t-1)]

    # Updating source inv before outgoing flows but after buy
    

    # Updating source after outgoing flows and after buy inv
    def material_balance_rule1_1(model, s, t):
        if(t==0):
            return model.source_inv[s, t] == 0
        else:
            return model.source_inv[s, t] == model.source_inv[s, model.timestamps_inv.prev(t)] \
                                            + model.offer_bought[s, t] \
                                            - sum(model.source_blend_flow[s, j, t] for j in model.blenders)

    model.material_balance_rule1_0 = Constraint(model.sources, model.timestamps_act, rule=material_balance_rule1_0)
    model.material_balance_rule1_1 = Constraint(model.sources, model.timestamps_inv, rule=material_balance_rule1_1)

    # Updating blender inventories
    def material_balance_rule2(model, j, t):
        if t == 0:  # Initialize inventory at t=0
            return model.blend_inv[j, t] == 0
        else:
            return model.blend_inv[j, t] == model.blend_inv[j, model.timestamps_inv.prev(t)] \
                                        + sum(model.source_blend_flow[s, j, t] for s  in model.sources) \
                                        + sum(model.blend_blend_flow[jp, j, t] for jp in model.blenders) \
                                        - sum(model.blend_blend_flow[j, jp, t] for jp in model.blenders) \
                                        - sum(model.blend_demand_flow[j, d, t] for d  in model.demands)

    model.material_balance_rule2 = Constraint(model.blenders, model.timestamps_inv, rule=material_balance_rule2)

    # Cannot sell more than what is asked
    def material_balance_rule3_0(model, p, t):
        return model.demand_sold[p, t] <= delta0[p][str(t-1)]

    # Updating demand before sell inv
    def material_balance_rule3_1(model, p, t):
        if t == 0:
            return model.demand_inv[p, t] == 0 # Initialize inventory at t=0
        else:
            return model.demand_inv[p, t] == model.demand_inv[p, model.timestamps_inv.prev(t)] \
                                        + sum(model.blend_demand_flow[j, p, t] for j in model.blenders) \
                                        - model.demand_sold[p, t] 

    model.material_balance_rule3_0 = Constraint(model.demands, model.timestamps_act, rule=material_balance_rule3_0)
    model.material_balance_rule3_1 = Constraint(model.demands, model.timestamps_inv, rule=material_balance_rule3_1)

    M = 90
    # in/out flow constraints
    def material_balance_rule4_1(model, s, j, t):
        return model.source_blend_flow[s, j, t] <= M * model.source_blend_bin[s, j, t]

    def material_balance_rule4_2(model, j1, j2, t):
        return model.blend_blend_flow[j1, j2, t] <= M * model.blend_blend_bin[j1, j2, t]

    def material_balance_rule4_3(model, j, p, t):
        return model.blend_demand_flow[j, p, t] <= M * model.blend_demand_bin[j, p, t]

    model.material_balance_rule4_1 = Constraint(model.sources, model.blenders,  model.timestamps_act, rule=material_balance_rule4_1)
    model.material_balance_rule4_2 = Constraint(model.blenders, model.blenders, model.timestamps_act, rule=material_balance_rule4_2)
    model.material_balance_rule4_3 = Constraint(model.blenders, model.demands,  model.timestamps_act, rule=material_balance_rule4_3)

    # in/out flow constraints
    def material_balance_rule5_1(model, s, j, p, t):
        return model.source_blend_bin[s, j, t] <= 1 - model.blend_demand_bin[j, p, t]

    def material_balance_rule5_2(model, s, j1, j2, t):
        return model.source_blend_bin[s, j1, t] <= 1 - model.blend_blend_bin[j1, j2, t]

    def material_balance_rule5_3(model, j1, j2, p, t):
        return model.blend_blend_bin[j1, j2, t] <= 1 - model.blend_demand_bin[j2, p, t]

    model.material_balance_rule5_1 = Constraint(model.sources, model.blenders, model.demands,  model.timestamps_act, rule=material_balance_rule5_1)
    model.material_balance_rule5_2 = Constraint(model.sources, model.blenders, model.blenders, model.timestamps_act, rule=material_balance_rule5_2)
    model.material_balance_rule5_3 = Constraint(model.blenders, model.blenders, model.demands, model.timestamps_act, rule=material_balance_rule5_3)

    # Quality calculations
    def material_balance_rule6(model, q, j, t):
        if t == 0:
            return model.prop_blend_inv[q, j, t] * model.blend_inv[j, t] == 0 # Initialize empty inventory at t=0
        else:
            return model.prop_blend_inv[q, j, t] * model.blend_inv[j, t] == model.prop_blend_inv[q, j, model.timestamps_inv.prev(t)] * model.blend_inv[j, model.timestamps_inv.prev(t)] \
                                                                            + sum(sigma[s][q] * model.source_blend_flow[s, j, t] for s in model.sources) \
                                                                            + sum(model.prop_blend_inv[q, jp, model.timestamps_inv.prev(t)] * model.blend_blend_flow[jp, j, t] for jp in model.blenders) \
                                                                            - sum(model.prop_blend_inv[q, j,  model.timestamps_inv.prev(t)] * model.blend_blend_flow[j, jp, t] for jp in model.blenders) \
                                                                            - sum(model.prop_blend_inv[q, j,  model.timestamps_inv.prev(t)] * model.blend_demand_flow[j, p, t] for p in model.demands)

    model.material_balance_rule6 = Constraint(model.properties, model.blenders, model.timestamps_inv, rule=material_balance_rule6)

    # Quality constraints
    def material_balance_rule7_1(model, q, p, j, t):
        return sigma_lb[p][q] - M * (1 - model.blend_demand_bin[j, p, t]) <= model.prop_blend_inv[q, j, model.timestamps_inv.prev(t)]

    def material_balance_rule7_2(model, q, p, j, t):
        return sigma_ub[p][q] + M * (1 - model.blend_demand_bin[j, p, t]) >= model.prop_blend_inv[q, j, model.timestamps_inv.prev(t)]

    model.material_balance_rule7_1 = Constraint(model.properties, model.demands, model.blenders, model.timestamps_act, rule=material_balance_rule7_1)
    model.material_balance_rule7_2 = Constraint(model.properties, model.demands, model.blenders, model.timestamps_act, rule=material_balance_rule7_2)
    
    
    def obj_function(model):
        return sum(sum(model.betaT_d[p] * model.demand_sold[p, t] for p in model.demands) for t in model.timestamps_act) \
            - sum(sum(model.betaT_s[s] * model.offer_bought[s, t] for s in model.sources) for t in model.timestamps_act) \
            - sum(sum(
                sum(model.alpha * model.source_blend_bin[s, j, t] + model.beta * model.source_blend_flow[s, j, t] for s in model.sources) \
                + sum(model.alpha * model.blend_blend_bin[j, jp, t] + model.beta * model.blend_blend_flow[j, jp, t] for jp in model.blenders) \
                + sum(model.alpha * model.blend_demand_bin[j, p, t] + model.beta * model.blend_demand_flow[j, p, t] for p in model.demands)
            for t in model.timestamps_act) for j in model.blenders)

    model.obj = Objective(rule=obj_function, sense=maximize)
    return model


def optimal_simulation(env, solver, tee: bool = True, raise_on_infeasible: bool = True):
    m = build_optimization_model(env)
    opt = solver if hasattr(solver, "solve") else SolverFactory(str(solver))
    results = opt.solve(m, tee=tee)

    term = results.solver.termination_condition
    ok = term in (TerminationCondition.optimal, TerminationCondition.locallyOptimal)

    if not ok and raise_on_infeasible:
        raise RuntimeError(
            f"Optimization did not solve to optimality. Termination condition: {term}"
        )
    actions = []
    
    for t in m.timestamps_act:
        action = {}
        action["source_blend"] = {}
        for s in m.sources:
            action["source_blend"][s] = {}
            for j in m.blenders:
                if j not in env.connections["source_blend"][s]:
                    action["source_blend"][s][j] = {}
                else:
                    action["source_blend"][s][j] = 2*m.source_blend_flow[s, j, t].value/env.MAXFLOW-1
        action["blend_blend"] = {}
        for j1 in m.blenders:
            action["blend_blend"][j1] = {}
            for j2 in m.blenders:
                if j2 not in env.connections["blend_blend"][j1]:
                    action["blend_blend"][j1][j2] = {}
                else:
                    action["blend_blend"][j1][j2] = 2*m.blend_blend_flow[j1,j2,t].value/env.MAXFLOW-1
        action["blend_demand"] = {}
        for j in m.blenders:
            action["blend_demand"][j] = {}
            for p in m.demands:
                if p not in env.connections["blend_demand"][j]:
                    action["blend_demand"][j][p] = {}
                else:
                    action["blend_demand"][j][p] = 2*m.blend_demand_flow[j,p,t].value/env.MAXFLOW-1
        action["tau"] = {}
        for s in m.sources:
            try:    
                action["tau"][s] = 2*m.offer_bought[s,t].value/env.tau0[s][str(t-1)]-1
            except:
                action["tau"][s] = 2*m.offer_bought[s,t].value-1
        action["delta"] = {}
        for p in m.demands:
            try:
                action["delta"][p] = 2*m.demand_sold[p,t].value/env.delta0[p][str(t-1)]-1
            except:
                action["delta"][p] = 2*m.demand_sold[p,t].value-1
        action_flatt, mapp = flatten_and_track_mappings(action)
        print(action_flatt)
        actions.append(action_flatt)
    return actions

