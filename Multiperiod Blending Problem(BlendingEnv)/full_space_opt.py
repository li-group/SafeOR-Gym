
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
import json
from pyomo.environ import *
from utils import *

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
alpha = 0
beta = 0


tau0   = {'s1': [10, 10, 10, 0, 0, 0],  's2': [30, 30, 30, 0, 0, 0]}
delta0 = {'p1': [0, 0, 15, 15, 15, 15], 'p2': [0, 0, 15, 15, 15, 15]}
sigma = {"s1":{"q1": 0.06,"q2":0.10}, "s2":{"q1": 0.26,"q2":0.5}} # Source concentrations
sigma_ub = {"p1":{"q1": 0.16,"q2":0.8}, "p2":{"q1": 1,"q2":0.9}} # Demand concentrations UBs
sigma_lb = {"p1":{"q1": 0,"q2":0}, "p2":{"q1": 0,"q2":0}}    # Demand concentrations LBs

s_inv_lb = {'s1': 0, 's2': 0}
s_inv_ub = {'s1': 999, 's2': 999}
d_inv_lb = {'p1': 0, 'p2': 0}
d_inv_ub = {'p1': 999, 'p2': 999}

betaT_d = {'p1': 20, 'p2': 10} # Price of sold products
betaT_s = {'s1': 0, 's2': 0} # Cost of bought products

b_inv_ub = {"j1": 30, "j2": 30, "j3": 30, "j4": 30}
b_inv_lb = {j:0 for j in b_inv_ub.keys()} 

window_len = 2
T = 6

properties = ["q1","q2"]



def solve(tau0 = tau0, delta0 = delta0,
        alpha = alpha,
        beta = beta,
        properties = properties,
        T = T,
        sigma = sigma,
        sigma_ub = sigma_ub,
        sigma_lb = sigma_lb,
        s_inv_ub = s_inv_ub,
        d_inv_ub = d_inv_ub,
        betaT_d = betaT_d,
        betaT_s = betaT_s,
        b_inv_ub = b_inv_ub,
       ):
    
    with open("./data/action_sample_simple_blend.json" ,"r") as f:
        action = f.read()
    action_sample = json.loads(action)
    with open("./data/connections_simple_blend.json" ,"r") as f:
        connections_s = f.readline()
    connections = json.loads(connections_s)
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
    model.tau0 = Param(model.sources, initialize=tau0)
    model.sigma_lb = Param(model.demands, initialize=sigma_lb)
    model.sigma_ub = Param(model.demands, initialize=sigma_ub)
    model.d_inv_ub = Param(model.demands, initialize=d_inv_ub)
    model.delta0 = Param(model.demands, initialize=delta0)
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
        return model.offer_bought[s, t] <= model.tau0[s][t-1]

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
        return model.demand_sold[p, t] <= model.delta0[p][t-1]

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
    # Solve the model
    solver = SolverFactory('gurobi')
    result1 = solver.solve(model, tee=True)
    print(model.obj())
    blenders.sort()
    print(blenders)
    actions = {}
    for t in timestamps_act:
        action = {}
        action["source_blend"] = {}
        for s in sources:
            action["source_blend"][s] = {}
            for j in blenders:
                if j not in connections["source_blend"][s]:
                    action["source_blend"][s][j] = {}
                else:
                    action["source_blend"][s][j] = model.source_blend_flow[s, j, t].value
        action["blend_blend"] = {}
        for j1 in blenders:
            action["blend_blend"][j1] = {}
            for j2 in blenders:
                if j2 not in connections["blend_blend"][j1]:
                    action["blend_blend"][j1][j2] = {}
                else:
                    action["blend_blend"][j1][j2] = model.blend_blend_flow[j1,j2,t].value
        action["blend_demand"] = {}
        for j in blenders:
            action["blend_demand"][j] = {}
            for p in demands:
                if p not in connections["blend_demand"][j]:
                    action["blend_demand"][j][p] = {}
                else:
                    action["blend_demand"][j][p] = model.blend_demand_flow[j,p,t].value
        action["tau"] = {}
        for s in sources:
            action["tau"][s] = model.offer_bought[s,t].value
        action["delta"] = {}
        for p in demands:
            action["delta"][p] = model.demand_sold[p,t].value
        actions[t] = action
    print(properties)
    return actions
    
actions = solve()
print(actions)

