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
with open("gen_trans_config.json" ,"r") as f:
    env_config_read = f.read()
env_config_full = json.loads(env_config_read)
env_config = env_config_full['env_init_cfgs']
regions = env_config["demand"].keys()
generators = env_config["gencap"].keys()
transmission_lines = env_config['transcap'].keys()
trans_dict = {l:tuple(l.split('_')) for l in transmission_lines}
T = 10
t_init = 0
num_gen_0 = {(i,r):0 for i in generators for r in regions}
bin_trans_0 = {l:0 for l in transmission_lines}
actions = {}

model = ConcreteModel()
model.num_gen = Var(generators,regions,range(t_init,t_init+T+1),domain = Integers)
model.add_gen = Var(generators,regions,range(t_init,t_init+T+1),domain = Integers)
model.pow_flow = Var(transmission_lines,range(t_init,t_init+T+1))
model.bin_trans = Var(transmission_lines,range(t_init,t_init+T+1),domain = Binary)
model.bin_trans_add = Var(transmission_lines,range(t_init+1,t_init+T+1),domain = Binary)
def init_numgen_rule(model, i,r):
    return model.num_gen[i,r,t_init] == num_gen_0[(i,r)]
model.init_numgen = Constraint(generators,regions, rule=init_numgen_rule)
def init_addgen_rule(model, i,r):
    return model.add_gen[i,r,t_init] == 0
def init_bin_trans_rule(model, l):
    return model.bin_trans[l,t_init] == bin_trans_0[l]
model.init_numgen = Constraint(generators,regions, rule=init_numgen_rule)
model.init_addgen = Constraint(generators,regions, rule=init_addgen_rule)
model.init_bintrans = Constraint(transmission_lines, rule=init_bin_trans_rule)
def lb_numgen_rule(model,i,r,t):
    return model.num_gen[i,r,t]>=0
model.lb_numgen = Constraint(generators,regions,range(t_init,t_init+T+1),rule = lb_numgen_rule)
def ub_numgen_rule(model,i,r,t):
    return model.num_gen[i,r,t]<=env_config["maxgen"][i][r]
model.ub_numgen = Constraint(generators,regions,range(t_init,t_init+T+1),rule = ub_numgen_rule)

def lb_powflow_rule(model,l,t):
    return model.pow_flow[l,t]>=-env_config["transcap"][l]*model.bin_trans[l,t]
model.lb_powflow = Constraint(transmission_lines,range(t_init,t_init+T+1),rule = lb_powflow_rule)
def ub_powflow_rule(model,l,t):
    return model.pow_flow[l,t]<=env_config["transcap"][l]*model.bin_trans[l,t]
model.ub_powflow = Constraint(transmission_lines,range(t_init,t_init+T+1),rule = ub_powflow_rule)
def lb_bin_trans_rule(model,l,t):
    return model.bin_trans[l,t]>=model.bin_trans[l,t-1]
model.lb_bin_trans = Constraint(transmission_lines,range(t_init+1,t_init+T+1),rule = lb_bin_trans_rule)


def val_add_trans_rule(model,l,t):
    return model.bin_trans_add[l,t] == model.bin_trans[l,t]-model.bin_trans[l,t-1]
model.val_add_trans = Constraint(transmission_lines,range(t_init+1,t_init+T+1),rule = val_add_trans_rule)

def lb_addgen_rule(model,i,r,t):
    return model.add_gen[i,r,t]>=0
model.lb_addgen = Constraint(generators,regions,range(t_init,t_init+T+1),rule = lb_addgen_rule)
def state_rule(model,i,r,t):
    return model.num_gen[i,r,t]==model.num_gen[i,r,t-1]+model.add_gen[i,r,t]
model.state = Constraint(generators,regions,range(t_init+1,t_init+T+1),rule = state_rule)
def demand_check_rule(model,r,t):
    return (env_config["demand"][r][str(t)] if t<=env_config['T'] else 0)<=sum(model.num_gen[i,r,t]*env_config["gencap"][i] for i in generators) \
+sum(model.pow_flow[l, t] for l in transmission_lines if r in trans_dict[l][1]) \
-sum(model.pow_flow[l, t] for l in transmission_lines if r in trans_dict[l][0])
model.demand_check = Constraint(regions,range(t_init+1,t_init+T+1),rule = demand_check_rule)
def objective_rule(model):
    return -sum(model.add_gen[i,r,t]*env_config["installcost"]["generators"][i] for i in generators for r in regions for t in range(t_init,t_init+T+1)) \
    -sum(model.bin_trans_add[l,t]*env_config["installcost"]["transmission"][l] for l in transmission_lines for t in range(t_init+1,t_init+T+1))
model.obj = Objective(rule=objective_rule, sense=maximize)
solver = SolverFactory('gurobi',options = {'TimeLimit':120,'threads':16,'MIPGap':0.01})  # 'couenne' is a solver for MINLP
#solver.options['IIS'] = 1 
results = solver.solve(model, tee=True)
for t in range(t_init+1,t_init+1+T):
    actions[t] = {'addgen':{(i,r):model.add_gen[i,r,t].value for i in generators for r in regions},
                        'powflow':{l:model.pow_flow[l,t].value for l in transmission_lines}}
print("net_reward",model.obj())
print(actions)