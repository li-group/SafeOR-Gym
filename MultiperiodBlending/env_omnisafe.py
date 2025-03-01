import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from gymnasium.utils import seeding
import json
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os, sys
from utils import *
from PIL import Image, ImageDraw, ImageFont
from omnisafe.envs.core import CMDP, env_register, env_unregister
from omnisafe.common.logger import Logger
import random


def assign_env_config(self, kwargs):
    for key, value in kwargs.items():
        setattr(self, key, value)
    if hasattr(self, 'env_config'):
        for key, value in self.env_config.items():
            # Check types based on default settings
            if hasattr(self, key):
                if type(getattr(self,key)) == np.ndarray:
                    setattr(self, key, value)
                else:
                    setattr(self, key,
                        type(getattr(self, key))(value))
            else:
                raise AttributeError(f"{self} has no attribute, {key}")


def clip(x,a,b):
    if x>b:
        return b
    elif x<a:
        return a
    return x


def flatten_dict(dictionary, parent_key='', separator=';'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

def flatten_and_track_mappings(dictionary, separator=';'):
    flattened_dict = flatten_dict(dictionary, separator=separator)
    mappings = [(index, key.split(separator)) for index, (key, value) in enumerate(flattened_dict.items())]
    flattened_array = np.array([value for key, value in flattened_dict.items()]).astype("float32")
    return flattened_array, mappings

def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value
    return(dic)
    
def reconstruct_dict(flattened_array, mappings, separator=';'):
    reconstructed_dict = {}
    for index, keys in mappings:
        nested_set(reconstructed_dict, keys, flattened_array[index])
    
    return reconstructed_dict



@env_register
class BlendEnv_Omni(CMDP):
    
    _support_envs = ['Blend-base', 'Blend-simple', 'Blend-simplest']  # Supported task names

    need_auto_reset_wrapper = True  # Whether `AutoReset` Wrapper is needed
    need_time_limit_wrapper = True  # Whether `TimeLimit` Wrapper is needed
    
    num_envs = 1
    
    def __init__(self, env_id, **kwargs):
        """
            Initialize the BlendEnv_Omni environment.

            Args:
                env_id (str): Identifier for the environment, which determines the layout. See _support_envs
                alpha (float): Fixed pipeline utilization cost
                beta (float): Proportional pipeline utilization cost
                v (bool): Verbose flag for logging and debugging.
                M (int): Penalty for violating the in/out rule.
                Q (int): Penalty for violating concentration requirements.
                P (int): Penalty for exceeding tank capacity limits.
                B (int): Penalty for exceeding buy/sell bounds.
                Z (int): Reward multiplier emphasizing selling product.
                D (int): Multiplier influencing populating tanks. See self.depths
                L0_pen (float): L0 norm component for violating "P" and "B" penalties
                MAXFLOW (float): Action upper bound (lower bound is 0).
                max_pen_violations (int): Maximum number of penalty violations allowed. 
                                            If set to <999, the episode will truncate early if this number of vialoations is reached.
                illeg_act_handling (str): Strategy for handling illegal actions. "disable" or "prop"
                
            if max_pen_violations == 999 : T=6
            if max_pen_violations <  999 : T=50
        """
        
        super().__init__(env_id)
        self.layout = env_id.split("Blend-")[1]
        print(kwargs)
        
        if kwargs is None or kwargs == {}:
            self.get_default_config()
        
        else:
            self.device = kwargs["dev"] if "render_mode" not in kwargs.keys() else "cpu"
            self.alpha = kwargs["alpha"]
            self.beta = kwargs["beta"]
            try: self.v = kwargs["v"] 
            except: self.v = False# Verbose
            
            self.M = kwargs["M"]            # Negative reward (penalty) constant factor for breaking in/out rule
            self.Q = kwargs["Q"]            # Negative reward (penalty) constant factor for breaking concentrations reqs
            self.P = kwargs["P"]            # Negative reward (penalty) constant factor for breaking tank bounds reqs
            self.B = kwargs["B"]            # Negative reward (penalty) constant factor for breaking buy/sell bounds reqs
            self.Z = kwargs["Z"]            # Positive reward multiplier to emphasize that "selling is good"
            self.D = kwargs["D"]            # Multiplier representing the influence of the depth
            self.L0_pen = kwargs["L0_pen"]   
            
            self.eps = 1e-3         # Tolerance for breaking in/out rule, concentration rule and other "== 0" checks
            
            self.MAXFLOW = kwargs["MAXFLOW"]   
            self.max_pen_violations = kwargs["max_pen_violations"]
            self.illeg_act_handling = kwargs["illeg_act_handling"]
            
            self.random_sd = kwargs["random_sd"]
            # self.random_sd = False
            
            if not self.random_sd:
                self.tau0   = {'s1': [10, 10, 10, 0, 0, 0],  's2': [30, 30, 30, 0, 0, 0]}
                self.delta0 = {'p1': [0, 0, 15, 15, 15, 15], 'p2': [0, 0, 15, 15, 15, 15]}
        
        self.T = 6 if self.max_pen_violations == 999 else 50
        
        self.sigma = {"s1":{"q1": 0.06}, "s2":{"q1": 0.26}} # Source concentrations
        self.sigma_ub = {"p1":{"q1": 0.16}, "p2":{"q1": 1}} # Demand concentrations UBs
        self.sigma_lb = {"p1":{"q1": 0}, "p2":{"q1": 0}}    # Demand concentrations LBs
        
        self.s_inv_lb = {'s1': 0, 's2': 0}
        self.s_inv_ub = {'s1': 999, 's2': 999}
        self.d_inv_lb = {'p1': 0, 'p2': 0}
        self.d_inv_ub = {'p1': 999, 'p2': 999}
        
        self.betaT_d = {'p1': 2, 'p2': 1} # Price of sold products
        self.betaT_s = {'s1': 0, 's2': 0} # Cost of bought products
        
        self.b_inv_ub = {"j1": 30, "j2": 30, "j3": 30, "j4": 30, "j5": 20, "j6": 20, "j7": 20, "j8": 20} 
        self.b_inv_lb = {j:0 for j in self.b_inv_ub.keys()} 
        
        self.forecast_window_len = 6
        
        
        self.connections, self.action_sample = get_jsons(self.layout)
        self.sources, self.blenders, self.demands = get_sbp(self.connections)
        self.properties = ["q1"]
        
        # for s in self.sources:
        #     self.tau0[s].append(0)
        # for p in self.demands:
        #     self.delta0[p].append(0)
        
        self.depths = {"s1": self.D*1, "s2": self.D*1,
                       "j1": self.D*2, "j2": self.D*2, "j3": self.D*2, "j4": self.D*2, 
                       "j5": self.D*3, "j6": self.D*3, "j7": self.D*3, "j8": self.D*3,
                       "p1": self.D*4, "p2": self.D*4}
        
                
        self.env_spec_log = {"Penalties/M": 0, "Penalties/B": 0, "Penalties/P": 0, "Penalties/Q": 0, 
                            "Penalties/n_M": 0, "Penalties/n_B": 0, "Penalties/n_P": 0, "Penalties/n_Q": 0, 
                            "Performances/units_sold": 0, "Performances/units_bought": 0, "Performances/rew_sold": 0, "Performances/rew_depth": 0
                            }
        
        self.reset() # sets state, reward_ep, cost_ep, t, done, NOT env_spec_log

        
        self.flatt_state, self.mapping_obs = flatten_and_track_mappings(self.state)
        self._observation_space = Box(low=0, high=self.MAXFLOW, shape=(self.flatt_state.shape[0],))
        
        self.flatt_act_sample, self.mapping_act = flatten_and_track_mappings(self.action_sample)
        self._action_space = Box(low=0, high=self.MAXFLOW, shape=(len(self.flatt_act_sample),))
        
        
    def step(self, action: th.Tensor):
        """
        The state is kept track of in a human-readable dict "self.state"
        After updating it from the action, we flatten it and return it along with the reward and "done"

        Args:
            action (torch.Tensor): model output
        """
        
        self.t += 1
        
        self.pens_step = {k:0 for k in self.env_spec_log.keys()}
        
        reward = 0
        cost = 0
                
        action = action.clip(0, self.MAXFLOW)
        action = action.tolist()
        action = reconstruct_dict(action, self.mapping_act) # From non-human-readable list to human-readable dict
        action = self.sanitize_action_structure(action)
        
        action, c = self.penalize_action_preflows(action)
        cost += c
        
        cost += self.update_reward1(action)
        
        prev_blend_invs = self.state["blenders"]
        
        for s in self.sources:
            
            # I + t - (x+y) > M: I + at - (x+y) = M => a = (M+(x+y)-I)/t
            # I + t - (x+y) < m: I + t - b(x+y) = m => b = (I+t-m)/(x+y)
            action["tau"][s] = max(0, action["tau"][s])
            outgoing = sum([action["source_blend"][s][j] for j in action["source_blend"][s].keys()])
            newinv = self.state["sources"][s] - outgoing + action["tau"][s]
            # Enforcing bounds
            if newinv > self.s_inv_ub[s] + self.eps: # inv too high -> reduce bought amount
                
                self.logg(f"[PEN] t{self.t}; {s}:\t\t\tbought too much (resulting amount more than source tank UB): {action['tau'][s]} vs {self.state['sources_avail_next_0'][s]}")
                cost += self.B * (self.L0_pen + newinv - self.s_inv_ub[s])
                self.pens_step["Penalties/B"] += self.B * (self.L0_pen + newinv - self.s_inv_ub[s])
                self.pens_step["Penalties/n_B"] += 1
                
                if self.illeg_act_handling == "prop":
                    self.logg(f"{s}: newtau: {self.s_inv_ub[s] + outgoing - self.state['sources'][s]}")
                    action["tau"][s] = self.s_inv_ub[s] + outgoing - self.state["sources"][s]
                
                elif self.illeg_act_handling == "disable": # Remove all outgoing flows
                    self.logg(f"{s}: newtau: 0")
                    action["tau"][s] = 0
                
                
            elif newinv < self.s_inv_lb[s] - self.eps: # inv too low -> reduce outgoing amount
                
                self.logg(f"[PEN] t{self.t}; {s}:\t\t\tbought too little (resulting amount less than source tank LB)")
                cost += self.B * (self.L0_pen + self.s_inv_lb[s] - newinv)
                self.pens_step["Penalties/B"] += self.B * (self.L0_pen + self.s_inv_lb[s] - newinv)
                self.pens_step["Penalties/n_B"] += 1
                
                if self.illeg_act_handling == "prop":
                    b = (self.state["sources"][s] + action["tau"][s] - self.s_inv_lb[s])/outgoing
                    self.logg(f"{s}: b: {b}")
                    for j in action["source_blend"][s].keys():
                        action["source_blend"][s][j] *= b
                        
                elif self.illeg_act_handling == "disable": # Remove all outgoing flows
                    for j in action["source_blend"][s].keys():
                        action["source_blend"][s][j] = 0
                

            # Giving reward depending on depths
            
            incr = self.depths[s] * max(0, newinv - self.state["sources"][s])
            if incr:
                self.logg(f"[INFO] Increased reward by {incr} through tank population in {s}")
            reward += incr
            self.pens_step["Performances/rew_depth"] += incr
            
            # Updating inv
            newinv = self.state["sources"][s] - sum([action["source_blend"][s][j] for j in action["source_blend"][s].keys()]) + action["tau"][s]
            self.state["sources"][s] = clip(newinv, self.s_inv_lb[s], self.s_inv_ub[s])
        
        
        self.logg("Action after processing sources:", action)
        
        for j in self.blenders:
            # Computing inflow and outflow
            in_flow_sources = in_flow_blend = out_flow_blend = out_flow_demands = 0
            for s in self.sources:
                if j in action["source_blend"][s].keys():
                    in_flow_sources += action["source_blend"][s][j]
            for jp in self.blenders:
                if "blend_blend" in action.keys() and j in action["blend_blend"][jp].keys():
                    in_flow_blend += action["blend_blend"][jp][j]
                if "blend_blend" in action.keys() and jp in action["blend_blend"][j].keys():
                    out_flow_blend += action["blend_blend"][j][jp]
            for p in self.demands:
                if p in action["blend_demand"][j].keys():
                    out_flow_demands += action["blend_demand"][j][p]
            
            self.logg(f"{j}: inv: {self.state['blenders'][j]}, in_flow_sources: {in_flow_sources}, in_flow_blend: {in_flow_blend}, out_flow_blend: {out_flow_blend}, out_flow_demands: {out_flow_demands}")
            
            # Enforcing No in and out flow
            if (in_flow_sources + in_flow_blend > self.eps) and (out_flow_blend + out_flow_demands > self.eps):
                self.logg(f"[PEN] t{self.t}; {j}:\t\t\tIn and out flow both non-zero (in: {round(in_flow_sources + in_flow_blend, 2)}, out: {round(out_flow_blend + out_flow_demands, 2)})")
                cost += self.M
                self.pens_step["Penalties/M"] += self.M
                self.pens_step["Penalties/n_M"] += 1
                
                # Choice: we remove all flows. We can also remove only outgoing flows, only incoming flows, or decide based on the tank's position
                # (if the tank is connected to sources, then keep incoming flow, but if it is connected to demands, then keep outgoing flow)
                for s in self.sources:
                    if j in action["source_blend"][s].keys():
                        action["source_blend"][s][j] = 0
                for jp in self.blenders:
                    if "blend_blend" in action.keys() and j in action["blend_blend"][jp].keys():
                        action["blend_blend"][jp][j] = 0
                    if "blend_blend" in action.keys() and jp in action["blend_blend"][j].keys():
                        action["blend_blend"][j][jp] = 0
                for p in self.demands:
                    if p in action["blend_demand"][j].keys():
                        action["blend_demand"][j][p] = 0
                
                continue # Inventory does not change
            
            # else...
            in_flow_sources = max(0, in_flow_sources)
            in_flow_blend = max(0, in_flow_blend)
            out_flow_blend = max(0, out_flow_blend)
            out_flow_demands = max(0, out_flow_demands)
            newinv = self.state["blenders"][j] + in_flow_sources + in_flow_blend - out_flow_blend - out_flow_demands
            
            # I + w+x-(y+z) > M : I + a(w+x) - (y+z) = M  =>  a = (M+y+z-I)/(w+x)
            # I + w+x-(y+z) < m : I + (w+x) - b(y+z) = m  =>  b = (I+w+x-m)/(y+z)
            
            # Enforcing inventory bounds
            # NB: we assume "no in and out" rule is respected
            if newinv > self.b_inv_ub[j] + self.eps: # inv too high -> reduce incoming amount
                self.logg(f"[PEN] t{self.t}; {j}:\t\t\tinventory OOB (resulting amount more than blending tank UB)")
                cost += self.P * (self.L0_pen + newinv - self.b_inv_ub[j])
                self.pens_step["Penalties/P"] += self.P * (self.L0_pen + newinv - self.b_inv_ub[j])
                self.pens_step["Penalties/n_P"] += 1
                
                if self.illeg_act_handling == "prop":
                    a = (self.b_inv_ub[j] + out_flow_blend + out_flow_demands - self.state["blenders"][j])/(in_flow_sources + in_flow_blend)
                    self.logg(f"{j}: a: {a}")
                    
                    for s in self.sources:
                        if j in action["source_blend"][s].keys():
                            action["source_blend"][s][j] *= a
                    for jp in self.blenders:
                        if "blend_blend" in action.keys() and j in action["blend_blend"][jp].keys():
                            action["blend_blend"][jp][j] *= a
                            
                elif self.illeg_act_handling == "disable": # Remove all incoming flows
                    for s in self.sources:
                        if j in action["source_blend"][s].keys():
                            action["source_blend"][s][j] = 0
                    for jp in self.blenders:
                        if "blend_blend" in action.keys() and j in action["blend_blend"][jp].keys():
                            action["blend_blend"][jp][j] = 0
                
                
            elif newinv < self.b_inv_lb[j] - self.eps: # inv too low -> reduce outgoing amount
                self.logg(f"[PEN] t{self.t}; {j}:\t\t\tinventory OOB (resulting amount less than blending tank LB)")
                cost += self.P * (self.L0_pen + self.b_inv_lb[j] - newinv)
                self.pens_step["Penalties/P"] += self.P * (self.L0_pen + self.b_inv_lb[j] - newinv)
                self.pens_step["Penalties/n_P"] += 1
                
                if self.illeg_act_handling == "prop":
                    b = (self.state["blenders"][j] + in_flow_sources + in_flow_blend - self.b_inv_lb[j])/(out_flow_blend + out_flow_demands)
                    self.logg(f"{j}: b: {b}")
                    
                    for jp in self.blenders:
                        if "blend_blend" in action.keys() and jp in action["blend_blend"][j].keys():
                            action["blend_blend"][j][jp] *= b
                    for p in self.demands:
                        if p in action["blend_demand"][j].keys():
                            action["blend_demand"][j][p] *= b
                            
                elif self.illeg_act_handling == "disable": # Remove all outgoing flows
                    for jp in self.blenders:
                        if "blend_blend" in action.keys() and jp in action["blend_blend"][j].keys():
                            action["blend_blend"][j][jp] = 0
                    for p in self.demands:
                        if p in action["blend_demand"][j].keys():
                            action["blend_demand"][j][p] = 0
            
            incr = self.depths[j] * max(0, newinv - self.state["blenders"][j])
            if incr:
                self.logg(f"[INFO] Increased reward by {incr} through tank population in {j}")
            reward += incr
            self.pens_step["Performances/rew_depth"] += incr
            
            # Computing rectified newinv
            in_flow_sources = in_flow_blend = out_flow_blend = out_flow_demands = 0
            for s in self.sources:
                if j in action["source_blend"][s].keys():
                    in_flow_sources += action["source_blend"][s][j]
            for jp in self.blenders:
                if "blend_blend" in action.keys() and j in action["blend_blend"][jp].keys():
                    in_flow_blend += action["blend_blend"][jp][j]
                if "blend_blend" in action.keys() and jp in action["blend_blend"][j].keys():
                    out_flow_blend += action["blend_blend"][j][jp]
            for p in self.demands:
                if p in action["blend_demand"][j].keys():
                    out_flow_demands += action["blend_demand"][j][p]
                    
            newinv = self.state["blenders"][j] + in_flow_sources + in_flow_blend - out_flow_blend - out_flow_demands
            self.state["blenders"][j] = clip(newinv, self.b_inv_lb[j], self.b_inv_ub[j])
            
        # self.logg("Action after processing blenders:", action)
        
        for p in self.demands:
            # Dealing with illegal flows
            # I + (x+y)-d > M: I + a(x+y) - d = M  =>  a = (M+d-I)/(x+y)
            # I + (x+y)-d < m: I + (x+y) - bd = m  =>  b = (I+(x+y)-m)/d
            
            action["delta"][p] = max(0, action["delta"][p])
            incoming = 0
            for jp in self.blenders:
                if p in action["blend_demand"][jp].keys():
                    # print("yyy", jp, p, action["blend_demand"][jp][p])
                    incoming += action["blend_demand"][jp][p]
            
            newinv = self.state["demands"][p] + incoming - action["delta"][p] 
            
            # Enforcing inventory bounds
            if newinv > self.d_inv_ub[p] + self.eps: # inv too high -> reduce incoming amount
                cost += self.B * (self.L0_pen + newinv - self.d_inv_ub[p])
                self.pens_step["Penalties/B"] += self.B * (self.L0_pen + newinv - self.d_inv_ub[p])
                self.logg(f"[PEN] t{self.t}; {p}:\t\t\tsold too little (resulting amount more than demand tank UB)")
                self.pens_step["Penalties/n_B"] += 1
                
                if self.illeg_act_handling == "prop":
                    a = (self.d_inv_ub[p] + action["delta"][p] - self.state["demands"][p])/incoming
                    self.logg(f"{p}: a: {a}")
                    for jp in self.blenders:
                        if p in action["blend_demand"][jp].keys():
                            action["blend_demand"][jp][p] *= a
                            
                elif self.illeg_act_handling == "disable": # Remove all outgoing flows
                    for jp in self.blenders:
                        if p in action["blend_demand"][jp].keys():
                            action["blend_demand"][jp][p] = 0
                            
                
            elif newinv < self.d_inv_lb[p] - self.eps:  # inv too low -> reduce sold amount
                cost += self.B * (self.L0_pen + self.d_inv_lb[p] - newinv)
                self.pens_step["Penalties/B"] += self.B * (self.L0_pen + self.d_inv_lb[p] - newinv)
                self.logg(f"[PEN] t{self.t}; {p}:\t\t\tsold too much (resulting amount less than demand tank LB)")
                self.pens_step["Penalties/n_B"] += 1
                
                if self.illeg_act_handling == "prop":
                    self.logg(f"{p}: {self.state['demands'][p]}, {incoming}, {self.d_inv_lb[p]}")
                    self.logg(f"{p}: newdelta: {self.state['demands'][p] + incoming - self.d_inv_lb[p]}")
                    action["delta"][p] = self.state["demands"][p] + incoming - self.d_inv_lb[p]
                
                elif self.illeg_act_handling == "disable": # Remove all outgoing flows
                    self.logg(f"{p}: newdelta: 0")
                    action["delta"][p] = 0
            
            
            incr = self.depths[p] * max(0, newinv - self.state["demands"][p])
            if incr:
                self.logg(f"[INFO] Increased reward by {incr} through tank population in {p}")
            reward += incr
            self.pens_step["Performances/rew_depth"] += incr
            
            incoming = 0
            for jp in self.blenders:
                if p in action["blend_demand"][jp].keys():
                    incoming += action["blend_demand"][jp][p]
            
            newinv = self.state["demands"][p] + incoming - action["delta"][p] 
            self.state["demands"][p] = clip(newinv, self.d_inv_lb[p], self.d_inv_ub[p])
            
        # Properties                      
        for j in self.blenders:
            for q in self.properties:
                # self.logg(f"\t[INFO10] t{self.t}; {j}; {q}; \t\t\t {self.state['blenders'][j]}")
                
                if self.state["blenders"][j] < self.eps:
                    self.state['properties'][j][q] = 0
                else:
                    in_flow_sources = in_flow_blend = out_flow_blend = out_flow_demands = 0
                    for s in self.sources:
                        if j in action["source_blend"][s].keys():
                            # self.logg(f"[INFO11] t{self.t}; {s}; {q}; \t\t\t {action['source_blend'][s][j]}; {self.sigma[s][q]}")
                            in_flow_sources += action['source_blend'][s][j] * self.sigma[s][q]
                    for jp in self.blenders:
                        if 'blend_blend' in action.keys() and j in action['blend_blend'][jp].keys():
                            # self.logg(f"[INFO12] t{self.t}; {jp}; {q}; \t\t\t {action['blend_blend'][jp][j]}; {self.state['properties'][jp][q]}")
                            in_flow_blend += action['blend_blend'][jp][j] * self.state['properties'][jp][q]
                        if 'blend_blend' in action.keys() and jp in action['blend_blend'][j].keys():
                            # self.logg(f"[INFO13] t{self.t}; {jp}; {q}; \t\t\t {action['blend_blend'][j][jp]}; {self.state['properties'][j][q]}")
                            out_flow_blend += action['blend_blend'][j][jp] * self.state['properties'][j][q]
                    for p in self.demands:
                        if p in action["blend_demand"][j].keys():
                            # self.logg(f"[INFO14] t{self.t}; {p}; {q}; \t\t\t {action['blend_demand'][j][p]}; {self.state['properties'][j][q]}")
                            out_flow_demands += action["blend_demand"][j][p] * self.state['properties'][j][q]

                    # self.logg(f"[INFO15] t{self.t}; {j}; {q}; \t\t\t {in_flow_sources }; { in_flow_blend}; {out_flow_blend}; {out_flow_demands}")
                    # self.logg(f"[INFO2] t{self.t}; {j}; {q}; \t\t\t Previous: {self.state['blenders'][j]}; {self.state['properties'][j][q]}; {in_flow_sources + in_flow_blend - out_flow_blend - out_flow_demands}")
                    self.state['properties'][j][q] = (1/self.state["blenders"][j]) * ( \
                                                    self.state['properties'][j][q] * prev_blend_invs[j] \
                                                    + in_flow_sources + in_flow_blend - out_flow_blend - out_flow_demands
                                                )
                    
                    # self.logg(f"[INFO3] t{self.t}; {j}; {q}; \t\t\t New property value: {self.state['properties'][j][q]}")

        
        r,c = self.update_reward2(action)
        reward += r
        cost += c
        
        for s in self.sources:
            for k in range(self.forecast_window_len):
                self.state[f"sources_avail_next_{k}"][s] = self.tau0[s][k + self.t] if k + self.t < len(self.tau0[s]) else 0
        
        for p in self.demands:
            for k in range(self.forecast_window_len):
                self.state[f"demands_avail_next_{k}"][p] = self.delta0[p][k + self.t] if k + self.t < len(self.delta0[p]) else 0
        
        self.state["t"] = self.t
        
        self.flatt_state, _ = flatten_and_track_mappings(self.state)
        
        if self.t == self.T:
            self.terminated = True
        
        n_violations =  (self.pens_step["Penalties/n_P"] if self.P > 0 else 0) + \
                        (self.pens_step["Penalties/n_B"] if self.B > 0 else 0) + \
                        (self.pens_step["Penalties/n_M"] if self.M > 0 else 0) + \
                        (self.pens_step["Penalties/n_Q"] if self.Q > 0 else 0)
                        
        self.n_violations_ep += n_violations
        # print(self.T)
        # print("VIOLATIONS: ", n_violations, self.n_violations_ep, self.pens_step["Penalties/n_M"], self.M)
        if self.n_violations_ep >= self.max_pen_violations:
            self.truncated = self.terminated = True
            
        self.reward_ep += reward
        self.cost_ep += cost
        
        self.logg(self.reward_ep, self.cost_ep, reward, cost, "\n", n_violations, self.pens_step)
        cost_tensor = th.tensor([self.pens_step["Penalties/B"], 
                                 self.pens_step["Penalties/P"],
                                 self.pens_step["Penalties/Q"], 
                                 self.pens_step["Penalties/M"]]
                                ).reshape((-1,))
        
        for k in self.env_spec_log.keys():
            self.env_spec_log[k] += self.pens_step[k]
            
        # print(self.t, self.T, self.terminated, self.truncated, self.n_violations_ep, self.max_pen_violations)
        
        return th.tensor(self.flatt_state).to(self.device), th.tensor(reward).to(self.device), cost_tensor.sum().to(self.device), th.tensor(self.terminated).to(self.device), \
                th.tensor(self.truncated).to(self.device), {"dict_state": self.state, "pen_tracker": self.pens_step, "terminated": self.terminated, "truncated": self.truncated}
    
        
    def reset(self, seed = 0, options = None):
        self.t = self.reward_ep = self.cost_ep = self.n_violations_ep = 0
        self.get_new_start_state()
        self.truncated = self.terminated = False
        # self.env_spec_log = {"Penalties/M": 0, "Penalties/B": 0, "Penalties/P": 0, "Penalties/Q": 0, 
        #                     "Penalties/n_M": 0, "Penalties/n_B": 0, "Penalties/n_P": 0, "Penalties/n_Q": 0, 
        #                     "Performances/units_sold": 0, "Performances/units_bought": 0, "Performances/rew_sold": 0, "Performances/rew_depth": 0}
        self.flatt_state, _ = flatten_and_track_mappings(self.state)
        return th.tensor(self.flatt_state).to(self.device), {"dict_state": self.state, "terminated": self.terminated, "truncated": self.truncated}
    
    def get_new_start_state(self):
        self.state = {
            "sources": {s:0 for s in self.sources},
            "blenders": {b:0 for b in self.blenders},
            "demands": {p:0 for p in self.demands},
            'properties': {b: {q:0 for q in self.properties} for b in self.blenders}
        }
        
        if self.random_sd:
            self.tau0   = {s: [(np.random.binomial(1, 0.7) * np.random.normal(15, 3)).__round__(1) for _ in range(self.T + 1)] for s in ["s1", "s2"]}
            self.delta0 = {p: [(np.random.binomial(1, 0.7) * np.random.normal(15, 3)).__round__(1) for _ in range(self.T + 1)] for p in ["p1", "p2"]}
    
        for k in range(self.forecast_window_len):
            self.state[f"sources_avail_next_{k}"] = {s: self.tau0[s][k]   if k < len(self.tau0[s]) else 0 for s in self.sources}
            self.state[f"demands_avail_next_{k}"] = {p: self.delta0[p][k] if k < len(self.delta0[p]) else 0 for p in self.demands}
            
        self.state["t"] = self.t
    

    def update_reward1(self, action):
        # Args: action (dict): See action_sample.json .
        Q_float = Q_bin = 0
        if "blend_blend" in action.keys():
            L = ["source_blend", "blend_blend", "blend_demand"]
        else:
            L = ["source_blend", "blend_demand"]
            
        for k in L:
            for tank1 in action[k].keys():
                for tank2 in action[k][tank1].keys():
                    Q_float += action[k][tank1][tank2]
                    Q_bin   += 1 if action[k][tank1][tank2] > 0 else 0 
                    
        return self.alpha * Q_bin + self.beta * Q_float
        
        
    def update_reward2(self, action):
        units_sold = units_bought = 0
        reward = cost = 0
        
        for p in self.demands:
            units_sold += action["delta"][p]
            reward += self.betaT_d[p] * action["delta"][p] * self.Z
            
        self.pens_step["Performances/rew_sold"] += reward
        for s in self.sources:
            units_bought += action["tau"][s]
            cost += self.betaT_s[s] * action["tau"][s]
            
        for j in self.blenders:
            cost += self.penalty_in_out_flow(j, action)
            for q in self.properties:
                for p in self.demands:
                    cost += self.penalty_quality(p, q, j, action)
        
        self.pens_step["Performances/units_sold"] += units_sold
        self.pens_step["Performances/units_bought"] += units_bought
        
        return reward, cost
        
        
    def penalty_quality(self, p, q, j, action):
        if (self.state['properties'][j][q] < self.sigma_lb[p][q] - self.eps or self.state['properties'][j][q] > self.sigma_ub[p][q] + self.eps) \
                and (p in action["blend_demand"][j].keys() and action["blend_demand"][j][p] > 0):
            self.logg(f"[PEN] t{self.t}; {p}; {q}; {j}:\t\t\tSold qualities out of bounds ({self.state['properties'][j][q]})")
            self.pens_step["Penalties/n_Q"] += 1
            self.pens_step["Penalties/Q"] += self.Q
            return self.Q
        return 0
    
    
    def penalty_in_out_flow(self, j, action):
        sum_in = sum_out = 0
        if "blend_blend" in action.keys():
            for jp in self.blenders:
                sum_in  += action["blend_blend"][jp][j] if j in action["blend_blend"][jp].keys() else 0
                sum_out += action["blend_blend"][j][jp] if jp in action["blend_blend"][j].keys() else 0
        
        for s in self.sources:
            sum_in  += action["source_blend"][s][j] if j in action["source_blend"][s].keys() else 0
        
        for p in self.demands:
            sum_out += action["blend_demand"][j][p] if p in action["blend_demand"][j].keys() else 0
            
        if sum_in > self.eps and sum_out > self.eps: # /!\
            self.logg(f"[PEN] t{self.t}; {j}:\t\t\tIn and out flow both non-zero (in: {round(sum_in, 2)}, out:{round(sum_out, 2)})")
            self.pens_step["Penalties/M"] += self.M
            self.pens_step["Penalties/n_M"] += 1
            return self.M
        
        return 0
    
    
    def sanitize_action_structure(self, action):
        """Normalize model action if needed

        Args:
            action (dict): Action dict
        """
        if "blend_blend" not in action.keys():
            return(action)
        
        for j in self.blenders:
            if j not in action["blend_blend"].keys():
                action["blend_blend"][j] = {}
            if j not in action["blend_demand"].keys():
                action["blend_demand"][j] = {}
        return(action)
        
        
    def penalize_action_preflows(self, action):
        """Add Penalty if the action is illegal (before flows are processed).
        Includes penalties related to the model proposing to buy/sell more product than the demands/sources allow (not inventory).

        Args:
            action (dict)
            pen (bool, optional): Set to False to disable penalties. Defaults to True.
        """
        cost = 0
        # Add penalty and log if trying to buy too much product
        for s in self.sources:
            if action["tau"][s] > self.state["sources_avail_next_0"][s]:
                cost += self.B * (self.L0_pen + action["tau"][s] - self.state["sources_avail_next_0"][s]) # incur penalty
                self.pens_step["Penalties/B"] += self.B * (self.L0_pen + action["tau"][s] - self.state["sources_avail_next_0"][s])
                self.logg(f"[PEN] t{self.t}; {s}:\t\t\tbought too much (more than supply): {action['tau'][s]} vs {self.state['sources_avail_next_0'][s]}")
                self.pens_step["Penalties/n_B"] += 1
                action["tau"][s] = self.state["sources_avail_next_0"][s]
        
        # Add penalty and log if trying to sell too much product (more than available demand or more than available inventory)
        for p in self.demands:
            if action["delta"][p] > self.state["demands_avail_next_0"][p]:
                cost += self.B * (self.L0_pen + action["delta"][p] - self.state["demands_avail_next_0"][p]) # incur penalty
                self.pens_step["Penalties/B"] += self.B * (self.L0_pen + action["delta"][p] - self.state["demands_avail_next_0"][p])
                self.logg(f"[PEN] t{self.t}; {p}:\t\t\tsold too much (more than demand)")
                self.pens_step["Penalties/n_B"] += 1
                action["delta"][p] = self.state["demands_avail_next_0"][p]
        
        return action, cost
        
    def render(self, action = None):
        # Load the base image
        img = Image.open(f"img/env_{self.layout}.png")
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype("arial.ttf", 12)  # Adjust font and size as needed
        font = ImageFont.load_default()
        draw.text((70,70), f"t = {self.t}", fill="black", font=font)
        
        # Define positions for each tank
        if self.layout == "base":
            positions = {
                "sources": {"s1": (140, 170), "s2": (140, 243)},
                "blenders": {"j1": (243, 97), "j2": (243, 170), "j3": (243, 243), "j4": (243, 314), 
                            "j5": (332, 97), "j6": (332, 170), "j7": (332, 243), "j8": (332, 314)},
                "demands": {"p1": (424, 170), "p2": (424, 243)}
            }
        elif self.layout == "simple":
            positions = {
                "sources": {"s1": (116, 170), "s2": (116, 240)},
                "blenders": {"j1": (217, 98), "j2": (217, 170), "j3": (217, 240), "j4": (217, 309)},
                "demands": {"p1": (311, 170), "p2": (311, 240)}
            }
            
        else:
            return

        # Draw inventory values for each tank
        for tank_type in ["sources", "blenders", "demands"]:
            for tank, pos in positions[tank_type].items():
                value = self.state[tank_type][tank]
                draw.text((pos[0] + 3, pos[1]), f"{value:.1f}", fill=color_gradient(value), font=font)

        # Draw available values for sources and demands
        
        for s, pos in positions["sources"].items():
            value = self.state["sources_avail_next_0"][s]
            draw.text((pos[0] - 70, pos[1] - 20), f"Avail: {value:.1f}", fill="black", font=font)
            if action is not None:
                bought = action["tau"][s]
                draw.text((pos[0] - 70, pos[1] + 20), f"Bought: {bought:.1f}", fill="blue", font=font)

        for d, pos in positions["demands"].items():
            value = self.state["demands_avail_next_0"][d]
            draw.text((pos[0] + 40, pos[1] - 20), f"Avail: {value:.1f}", fill="black", font=font)
            if action is not None:
                sold = action["delta"][d]
                draw.text((pos[0] + 40, pos[1] + 20), f"Sold: {sold:.1f}", fill="blue", font=font)

        return np.array(img)
        
    def logg(self, *args):
        if self.v:
            print(*args)
        return
    
    def spec_log(self, logger: Logger) -> None: # Called at the end of each epoch
        for key, value in self.env_spec_log.items():
            logger.store({key: value})
            self.env_spec_log[key] = 0.0
    
    def close(self):
        pass
    
    def set_seed(self, seed: int):
        random.seed(seed)
        
    def get_default_config(self):
        self.T = 6
        self.alpha = 0
        self.beta = 0
        self.v = False
        self.M = 100
        self.Q = 100
        self.P = 1
        self.B = 1
        self.Z = 100
        self.D = 0.1
        self.L0_pen = 1
        self.eps = 1e-3
        self.MAXFLOW = 500
        self.max_pen_violations = 999
        self.illeg_act_handling = "prop"
        self.tau0   = {'s1': [10, 10, 10, 0, 0, 0],  's2': [30, 30, 30, 0, 0, 0]}
        self.delta0 = {'p1': [0, 0, 15, 15, 15, 15], 'p2': [0, 0, 15, 15, 15, 15]}
        self.random_sd = False
        self.device = "cpu"
        
    @property
    def max_episode_steps(self) -> None:
        """The max steps per episode."""
        return self.T


