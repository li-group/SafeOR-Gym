import numpy as np
from pyomo.environ import *
#from pyomo import *
#import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.contrib.iis import *
import openpyxl
import random
import numpy as np
import json

problem_data = {
    "config_file" : "gen_trans_config.json",
    "T" : 10,
    "t_init" : 0
}

def build_optimization_model(problem_data = problem_data):
    with open(problem_data["config_file"] ,"r") as f:
        env_config_read = f.read()
    env_config_full = json.loads(env_config_read)
    env_config = env_config_full['env_init_cfgs']
    regions = env_config["demand"].keys()
    generators = env_config["gencap"].keys()
    transmission_lines = env_config['tlcap'].keys()
    trans_dict = {l:tuple(l.split('_')) for l in transmission_lines}
    T = problem_data["T"]
    t_init = problem_data["t_init"]
    time_all = range(t_init, t_init + T + 1)
    time_add = range(t_init + 1, t_init + T + 1)
    num_gen_0 = {(i,r):0 for i in generators for r in regions}
    bin_trans_0 = {l:0 for l in transmission_lines}
    model = ConcreteModel()
    model.num_gen = Var(generators,regions,time_all,domain = Integers)
    model.add_gen = Var(generators,regions,time_all,domain = Integers)
    has_transmission = len(transmission_lines) > 0
    model.num_gen = Var(generators,regions,time_all,domain = Integers)
    model.add_gen = Var(generators,regions,time_all,domain = Integers)
    if has_transmission:
        model.pow_flow = Var(transmission_lines, time_all)  # Reals by default
        model.bin_trans = Var(transmission_lines, time_all, domain=Binary)
        model.bin_trans_add = Var(transmission_lines, time_add, domain=Binary)
    def init_numgen_rule(model, i,r):
        return model.num_gen[i,r,t_init] == num_gen_0[(i,r)]
    model.init_numgen = Constraint(generators,regions, rule=init_numgen_rule)
    def init_addgen_rule(model, i,r):
        return model.add_gen[i,r,t_init] == 0
    
    model.init_numgen = Constraint(generators,regions, rule=init_numgen_rule)
    model.init_addgen = Constraint(generators,regions, rule=init_addgen_rule)
    def lb_numgen_rule(model,i,r,t):
        return model.num_gen[i,r,t]>=0
    model.lb_numgen = Constraint(generators,regions,time_all,rule = lb_numgen_rule)
    def ub_numgen_rule(model,i,r,t):
        return model.num_gen[i,r,t]<=env_config["maxgen"][i][r]
    model.ub_numgen = Constraint(generators,regions,time_all,rule = ub_numgen_rule)
    if has_transmission:
        def init_bin_trans_rule(model, l):
            return model.bin_trans[l,t_init] == bin_trans_0[l]   
        model.init_bintrans = Constraint(transmission_lines, rule=init_bin_trans_rule)
        def lb_powflow_rule(model,l,t):
            return model.pow_flow[l,t]>=-env_config["tlcap"][l]*model.bin_trans[l,t]
        model.lb_powflow = Constraint(transmission_lines,time_all,rule = lb_powflow_rule)
        def ub_powflow_rule(model,l,t):
            return model.pow_flow[l,t]<=env_config["tlcap"][l]*model.bin_trans[l,t]
        model.ub_powflow = Constraint(transmission_lines,time_all,rule = ub_powflow_rule)
        def lb_bin_trans_rule(model,l,t):
            return model.bin_trans[l,t]>=model.bin_trans[l,t-1]
        model.lb_bin_trans = Constraint(transmission_lines,time_add,rule = lb_bin_trans_rule)
        def val_add_trans_rule(model,l,t):
            return model.bin_trans_add[l,t] == model.bin_trans[l,t]-model.bin_trans[l,t-1]
        model.val_add_trans = Constraint(transmission_lines,time_add,rule = val_add_trans_rule)
    def lb_addgen_rule(model,i,r,t):
        return model.add_gen[i,r,t]>=0
    model.lb_addgen = Constraint(generators,regions,time_all,rule = lb_addgen_rule)
    def state_rule(model,i,r,t):
        return model.num_gen[i,r,t]==model.num_gen[i,r,t-1]+model.add_gen[i,r,t]
    model.state = Constraint(generators,regions,time_add,rule = state_rule)
    def demand_check_rule(model, r, t):
        demand_t = env_config["demand"][r].get(str(t), 0) if t <= env_config["T"] else 0
        gen_supply = sum(model.num_gen[i, r, t] * env_config["gencap"][i] for i in generators)

        if not has_transmission:
            return demand_t <= gen_supply

        # Net inflow - outflow for region r
        inflow = sum(model.pow_flow[l, t] for l in transmission_lines if r == trans_dict[l][1])
        outflow = sum(model.pow_flow[l, t] for l in transmission_lines if r == trans_dict[l][0])
        return demand_t <= gen_supply + inflow - outflow

    model.demand_check = Constraint(regions, time_add, rule=demand_check_rule)
    def objective_rule(model):
        gen_cost = -sum(
            model.add_gen[i, r, t] * env_config["installcost"]["generators"][i]
            for i in generators
            for r in regions
            for t in time_all
        )

        if not has_transmission:
            return gen_cost

        trans_cost = -sum(
            model.bin_trans_add[l, t] * env_config["installcost"]["transmission"][l]
            for l in transmission_lines
            for t in time_all
        )
        return gen_cost + trans_cost

    model.obj = Objective(rule=objective_rule, sense=maximize)
    return model

