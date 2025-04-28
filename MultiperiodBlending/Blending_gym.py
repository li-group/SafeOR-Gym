import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from gymnasium.utils import seeding
import json
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import copy
import os, sys
from utils import assign_env_config,flatten_and_track_mappings,reconstruct_dict,convert_dict_to_tuple_keys,flatten_dict,convert_tuple_keys_to_string,get_jsons,get_sbp,convert_string_keys_to_tuple,clip
from typing import Any, ClassVar, List, Tuple, Optional, Dict
import random
class BlendEnv(gym.Env):
    '''
    The multiperiod blending problem is a mixed integer nonlinear programming optimization problem that involves involves blending streams in blenders to obtain 
    products across different time periods. The problem involves balancing resource availability and property specifications to meet demand fluctuations 
    or maximize profit by selling the output. 
    Problem Setup
        - We have a set of source streams, output/demand streams and blenders. We know the maximum amount of source that can be bought as well as the maximum amount of outputs
        that can be sold
        - We buy a set of source streams
        - Each of the source streams has a certain set of properties and flows into an inventory vessel.
        - The streams are then extracted from these inventory vessels into a blender where they are blended in an appropriate way.
        - Then output/demand streams are extracted from the blenders in such a way that the properties lie within a specified range and are stored in an inventory vessel.
        - We then extract the output/demand streams from these inventory vessels and sell it
        - Constraints include:
            - Maximum amount of supply bought or output sold
            - Inventory bounds
            - In and out rule: There cannot be simultaneous inflow and outflow in the blenders
            - Properties of outputs should be within the bounds
        - Violating these constraints results in penalties. Costs are incurred for flow of streams as well as units bought. Revenue is incurred for selling output/demand
        - The agent's objective is to maximize revenue-costs - penalties.

    State space:
        The state space is a dictionary with the following keys:
            - sources: A dictionary with keys being the source streams and the values being the amount of materials in the corresponding inventory
            - blenders: A dictionary with keys being the blenders and the values being the amount of materials in the corresponding inventory
            - demand : A dictionary with keys being the output/demand streams and the values being the amount of materials in the corresponding inventory
            - properties: A dictionary with keys as tuples (j,q) where j is the blender and q is the property and the values being the 
            value of the properties of the materials in the blender
            - source_avail_next: the schedule of source for the current and next few steps
            - source_demand_next: the schedule of demand for the current and next few steps
            - t: The current time period.
        All values are flattened into a single array for the observation space.
    Action space:
        The action space is a dictionary with the following keys:
            - source_blend: A nested dictionary of the form {s:{b:value}} with s being a source stream, b being a blender and value being the amount of materials moved
            from the source inventory and blender
            - blend_blend: A nested dictionary of the form {b1:{b2:value}} with b1 and b2 being distinct blenders and value being the amount of materials moved
            between the blenders
            - blend_demand: A nested dictionary of the form {b:{p:value}} with b being a blender and p being a demand stream and value being the amount of materials moved
            from the blender to the demand inventory
            - tau: A dictionary with keys as s where s is a source steam and value being the amount of products bought
            - delta: A dictionary with keys as p where p is a demand steam and value being the amount of products sold
        In practice, the values in action space are values between 0 and the maximum flow. For the sake of training the agent, we use a flattened action space with 
        values between -1 and 1. The values are then scaled to the range between 0 and the max_flow. 
    Transition to next state:
        The inventories and the properites are updated based on the flows. We then update the future schedule and the time appropriately.
    Cost:
        The cost is the penalties incurred for violating the constraints.  
    Reward:
        The reward is the net profit of the operations
    Starting State:
        We start with no initial inventories and properties, with the schedule for the window length and time being 0. 
    Termination:
        The episode terminates when the time period t reaches T.
    '''
    
    def __init__(self, env_id: str,
                 **kwargs: Any) -> None:
        """
            Initialize the BlendEnv_Omni environment.

            Args:
                env_id (str): Identifier for the environment
                env_spec_log: Components to be noted
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
                illeg_act_handling (str): Strategy for handling illegal actions. "disable" or "prop"
                tau0: Maximum possible units bought at each time period
                delta0: Maximum possible units sold at each time period
                sigma: Source concentrations
                sigma_lb: Lower bound on demand concentration
                sigma_ub: Upper bound on demand concentration
                s_inv_lb: Lower bound on source inventory
                s_inv_ub: Upper bound on source inventory
                d_inv_lb: Lower bound on demand inventory
                d_inv_ub: Upper bound on demand inventory
                b_inv_lb: Lower bound on blender inventory
                b_inv_ub: Upper bound on blender inventory
                forecast_window_len: Window for the forecast of maximum possible units bought and sold included in the state
                T: Number of time periods
        """
        super().__init__()
        self.env_id = env_id
        self._device = kwargs.get('device', 'cuda' if th.cuda.is_available() else 'cpu')
        self.env_spec_log = {"Penalties/M": 0, "Penalties/B": 0, "Penalties/P": 0, "Penalties/Q": 0, 
                            "Penalties/n_M": 0, "Penalties/n_B": 0, "Penalties/n_P": 0, "Penalties/n_Q": 0, 
                            "Performances/units_sold": 0, "Performances/units_bought": 0, "Performances/rew_sold": 0, "Performances/rew_depth": 0
                            }
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
        self.illeg_act_handling = "prop"
        self.tau0   = {'s1': [10, 10, 10, 0, 0, 0],  's2': [30, 30, 30, 0, 0, 0]}
        self.delta0 = {'p1': [0, 0, 15, 15, 15, 15], 'p2': [0, 0, 15, 15, 15, 15]}
        self.random_sd = False
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
        
        self.forecast_window_len = 2
        self.T = 6
        assign_env_config(self, kwargs)
        with open("./action_sample_base_blend.json" ,"r") as f:
            action = f.read()
        self.action_sample = json.loads(action)
        with open("./connections_base_blend.json" ,"r") as f:
            connections_s = f.readline()
        self.connections = json.loads(connections_s)
        self.sources, self.blenders, self.demands = get_sbp(self.connections)
        self.properties = ["q1"]
        self.depths = {"s1": self.D*1, "s2": self.D*1,
                       "j1": self.D*2, "j2": self.D*2, "j3": self.D*2, "j4": self.D*2, 
                       "j5": self.D*3, "j6": self.D*3, "j7": self.D*3, "j8": self.D*3,
                       "p1": self.D*4, "p2": self.D*4}
        for s in self.sources: #To see
            self.tau0[s].append(0)
        for p in self.demands:
            self.delta0[p].append(0)
        self.reset() # sets state, reward, t, done
        
        self.flatt_state, self.mapping_obs = flatten_and_track_mappings(self.state)
        self.observation_space = Box(low=0, high=self.MAXFLOW, shape=(self.flatt_state.shape[0],))
        
        self.flatt_act_sample, self.mapping_act = flatten_and_track_mappings(self.action_sample)
        low_list = []
        high_list = []
        for i in range(len(self.flatt_act_sample)):
            low_list.append(0)
            high_list.append(self.MAXFLOW)
        self.action_low = np.array(low_list)
        self.action_high = np.array(high_list)
        self.action_space = Box(low=0, high=self.MAXFLOW, shape=(len(self.flatt_act_sample),))
    
    def get_new_start_state(self):
        #Function to get a new start state. 
        self.state = {
            "sources": {s:0 for s in self.sources},
            "blenders": {b:0 for b in self.blenders},
            "demands": {p:0 for p in self.demands},
            'properties': {b: {q:0 for q in self.properties} for b in self.blenders}
        }
        
        for k in range(self.forecast_window_len):
            self.state[f"sources_avail_next_{k}"] = {s: self.tau0[s][k]   if k < len(self.tau0[s]) else 0 for s in self.sources}
            self.state[f"demands_avail_next_{k}"] = {p: self.delta0[p][k] if k < len(self.delta0[p]) else 0 for p in self.demands}
            
        self.state["t"] = self.t
    
    def reset(self, seed=0,options = None):
        ## Reset the environment to the initial state
        self.t = self.reward_ep = 0
        self.cost_ep = 0
        self.reward = 0
        self.cost = 0
        self.get_new_start_state()
        self.truncated = self.terminated = False
        self.flatt_state, _ = flatten_and_track_mappings(self.state)
        return self.flatt_state, {"dict_state": self.state, "terminated": self.terminated, "truncated": self.truncated}
    
    def compute_reward_flowarc(self, action):
        # Function to calculate reward associated with the flow between the different inventories. It involves a negative of fixed and variable cost
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
                    
        reward = -(self.alpha * Q_bin + self.beta * Q_float)
        return reward
        
    def compute_reward_sold_bought(self, action):
        # Function to caluclate reward from selling and buying units. It is calculated as the net profit from units sold and bought.
        units_sold = units_bought = 0
        reward = 0
        
        for p in self.demands:
            units_sold += action["delta"][p]
            reward += self.betaT_d[p] * action["delta"][p] * self.Z
            
        self.pens_step["Performances/rew_sold"] += reward
        for s in self.sources:
            units_bought += action["tau"][s]
            reward -= self.betaT_s[s] * action["tau"][s]

        
        self.pens_step["Performances/units_sold"] += units_sold
        self.pens_step["Performances/units_bought"] += units_bought
        return reward
        
    def penalty_property(self,action):
        #Function to calculate the penalty of violating the property bounds
        cost = 0
        for j in self.blenders:
            for q in self.properties:
                for p in self.demands:
                    if (self.state['properties'][j][q] < self.sigma_lb[p][q] - self.eps or self.state['properties'][j][q] > self.sigma_ub[p][q] + self.eps) \
                            and (p in action["blend_demand"][j].keys() and action["blend_demand"][j][p] > 0):
                        #self.logg(f"[PEN] t{self.t}; {p}; {q}; {j}:\t\t\tSold qualities out of bounds ({self.state['properties'][j][q]})")
                        self.pens_step["n_Q"] += 1
                        self.pens_step["Q"] -= self.Q
                        cost+= self.Q
        return cost
    

    
    def sanitize_action_structure(self, action):
        #Function to santize action structure to account for missing flow arcs and maintain structure
        if "blend_blend" not in action.keys():
            return(action)
        
        for j in self.blenders:
            if j not in action["blend_blend"].keys():
                action["blend_blend"][j] = {}
            if j not in action["blend_demand"].keys():
                action["blend_demand"][j] = {}
        return(action)
        
    def check_illegal_actions_preflow_cost(self, action):
        #Function to check if the actions violate the maximum possible buying and selling before flows are processed. If there are violations, the actions are adjusted 
        # and a penalty is calculated for the violation.
        cost = 0
        # Add penalty and log if trying to buy too much product
        for s in self.sources:
            if action["tau"][s] > self.state["sources_avail_next_0"][s]:
                cost += self.B * (self.L0_pen + action["tau"][s] - self.state["sources_avail_next_0"][s]) # incur penalty
                self.pens_step["Penalties/B"] += self.B * (self.L0_pen + action["tau"][s] - self.state["sources_avail_next_0"][s])
                #self.logg(f"[PEN] t{self.t}; {s}:\t\t\tbought too much (more than supply): {action['tau'][s]} vs {self.state['sources_avail_next_0'][s]}")
                self.pens_step["Penalties/n_B"] += 1
                action["tau"][s] = self.state["sources_avail_next_0"][s]
        
        # Add penalty and log if trying to sell too much product (more than available demand or more than available inventory)
        for p in self.demands:
            if action["delta"][p] > self.state["demands_avail_next_0"][p]:
                cost += self.B * (self.L0_pen + action["delta"][p] - self.state["demands_avail_next_0"][p]) # incur penalty
                self.pens_step["Penalties/B"] += self.B * (self.L0_pen + action["delta"][p] - self.state["demands_avail_next_0"][p])
                #self.logg(f"[PEN] t{self.t}; {p}:\t\t\tsold too much (more than demand)")
                self.pens_step["Penalties/n_B"] += 1
                action["delta"][p] = self.state["demands_avail_next_0"][p]
        
        return action, cost
    def check_source_inv_bounds(self,action):
        # Function to calculte the violation of source inventory bounds based on action and adjust the actions
        cost = 0
        for s in self.sources:
            
            # I + t - (x+y) > M: I + at - (x+y) = M => a = (M+(x+y)-I)/t
            # I + t - (x+y) < m: I + t - b(x+y) = m => b = (I+t-m)/(x+y)
            action["tau"][s] = max(0, action["tau"][s])
            outgoing = sum([action["source_blend"][s][j] for j in action["source_blend"][s].keys()])
            newinv = self.state["sources"][s] - outgoing + action["tau"][s]
            # Enforcing bounds
            if newinv > self.s_inv_ub[s] + self.eps: # inv too high -> reduce bought amount
                
                #self.logg(f"[PEN] t{self.t}; {s}:\t\t\tbought too much (resulting amount more than source tank UB): {action['tau'][s]} vs {self.state['sources_avail_next_0'][s]}")
                cost += self.B * (self.L0_pen + newinv - self.s_inv_ub[s])
                self.pens_step["Penalties/B"] += self.B * (self.L0_pen + newinv - self.s_inv_ub[s])
                self.pens_step["Penalties/n_B"] += 1
                
                if self.illeg_act_handling == "prop":
                    #self.logg(f"{s}: newtau: {self.s_inv_ub[s] + outgoing - self.state['sources'][s]}")
                    action["tau"][s] = self.s_inv_ub[s] + outgoing - self.state["sources"][s]
                
                elif self.illeg_act_handling == "disable": # Remove all outgoing flows
                    #self.logg(f"{s}: newtau: 0")
                    action["tau"][s] = 0
                
                
            elif newinv < self.s_inv_lb[s] - self.eps: # inv too low -> reduce outgoing amount
                
                #self.logg(f"[PEN] t{self.t}; {s}:\t\t\tbought too little (resulting amount less than source tank LB)")
                cost += self.B * (self.L0_pen + self.s_inv_lb[s] - newinv)
                self.pens_step["Penalties/B"] += self.B * (self.L0_pen + self.s_inv_lb[s] - newinv)
                self.pens_step["Penalties/n_B"] += 1
                
                if self.illeg_act_handling == "prop":
                    b = (self.state["sources"][s] + action["tau"][s] - self.s_inv_lb[s])/outgoing
                    #self.logg(f"{s}: b: {b}")
                    for j in action["source_blend"][s].keys():
                        action["source_blend"][s][j] *= b
                        
                elif self.illeg_act_handling == "disable": # Remove all outgoing flows
                    for j in action["source_blend"][s].keys():
                        action["source_blend"][s][j] = 0
        return action,cost      
    def update_source_inv(self,action):
        #Function to update source inventory
        for s in self.sources:
            newinv = self.state["sources"][s] - sum([action["source_blend"][s][j] for j in action["source_blend"][s].keys()]) + action["tau"][s]
            self.state["sources"][s] = clip(newinv, self.s_inv_lb[s], self.s_inv_ub[s])
    def update_blender_inv(self,action):
        #Function to update blender inventory
        for j in self.blenders:
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
    def check_in_out_rule(self,action):
        #Function to calculate penalty for the violation of in and out rule and adjust the actions
        cost = 0
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
            
            #self.logg(f"{j}: inv: {self.state['blenders'][j]}, in_flow_sources: {in_flow_sources}, in_flow_blend: {in_flow_blend}, out_flow_blend: {out_flow_blend}, out_flow_demands: {out_flow_demands}")
            
            # Enforcing No in and out flow
            if (in_flow_sources + in_flow_blend > self.eps) and (out_flow_blend + out_flow_demands > self.eps):
                #self.logg(f"[PEN] t{self.t}; {j}:\t\t\tIn and out flow both non-zero (in: {round(in_flow_sources + in_flow_blend, 2)}, out: {round(out_flow_blend + out_flow_demands, 2)})")
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
        return action,cost
    def check_blender_inv_bounds(self,action):
        # Function to calculate the penalty for the violation of blender inventory bounds based on action and adjust the actions
        cost = 0
        for j in self.blenders:
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
            in_flow_sources = max(0, in_flow_sources)
            in_flow_blend = max(0, in_flow_blend)
            out_flow_blend = max(0, out_flow_blend)
            out_flow_demands = max(0, out_flow_demands)
            newinv = self.state["blenders"][j] + in_flow_sources + in_flow_blend - out_flow_blend - out_flow_demands
            if newinv > self.b_inv_ub[j] + self.eps: # inv too high -> reduce incoming amount
                #self.logg(f"[PEN] t{self.t}; {j}:\t\t\tinventory OOB (resulting amount more than blending tank UB)")
                cost += self.P * (self.L0_pen + newinv - self.b_inv_ub[j])
                self.pens_step["Penalties/P"] += self.P * (self.L0_pen + newinv - self.b_inv_ub[j])
                self.pens_step["Penalties/n_P"] += 1
                
                if self.illeg_act_handling == "prop":
                    a = (self.b_inv_ub[j] + out_flow_blend + out_flow_demands - self.state["blenders"][j])/(in_flow_sources + in_flow_blend)
                    #self.logg(f"{j}: a: {a}")
                    
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
                #self.logg(f"[PEN] t{self.t}; {j}:\t\t\tinventory OOB (resulting amount less than blending tank LB)")
                cost += self.P * (self.L0_pen + self.b_inv_lb[j] - newinv)
                self.pens_step["Penalties/P"] += self.P * (self.L0_pen + self.b_inv_lb[j] - newinv)
                self.pens_step["Penalties/n_P"] += 1
                
                if self.illeg_act_handling == "prop":
                    b = (self.state["blenders"][j] + in_flow_sources + in_flow_blend - self.b_inv_lb[j])/(out_flow_blend + out_flow_demands)
                    #self.logg(f"{j}: b: {b}")
                    
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
        return action,cost
    def check_demand_inv_bounds(self,action):
        # Function to calculate penalty for the violation of demand inventory bounds based on action and adjust the actions
        cost = 0
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
                #self.logg(f"[PEN] t{self.t}; {p}:\t\t\tsold too little (resulting amount more than demand tank UB)")
                self.pens_step["Penalties/n_B"] += 1
                
                if self.illeg_act_handling == "prop":
                    a = (self.d_inv_ub[p] + action["delta"][p] - self.state["demands"][p])/incoming
                    #self.logg(f"{p}: a: {a}")
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
                #self.logg(f"[PEN] t{self.t}; {p}:\t\t\tsold too much (resulting amount less than demand tank LB)")
                self.pens_step["Penalties/n_B"] += 1
                
                if self.illeg_act_handling == "prop":
                    #self.logg(f"{p}: {self.state['demands'][p]}, {incoming}, {self.d_inv_lb[p]}")
                    #self.logg(f"{p}: newdelta: {self.state['demands'][p] + incoming - self.d_inv_lb[p]}")
                    action["delta"][p] = self.state["demands"][p] + incoming - self.d_inv_lb[p]
                
                elif self.illeg_act_handling == "disable": # Remove all outgoing flows
                    #self.logg(f"{p}: newdelta: 0")
                    action["delta"][p] = 0
        return action,cost
    def update_demand_inv(self,action):
        #Function to update demand inventory
        for p in self.demands:
            incoming = 0
            for jp in self.blenders:
                if p in action["blend_demand"][jp].keys():
                    incoming += action["blend_demand"][jp][p]
            
            newinv = self.state["demands"][p] + incoming - action["delta"][p] 
            self.state["demands"][p] = clip(newinv, self.d_inv_lb[p], self.d_inv_ub[p])
    def update_properties(self,action,prev_blend_invs,prev_properties):
        #Function to update the properties in the blenders
        for j in self.blenders:
            for q in self.properties:
                # #self.logg(f"\t[INFO10] t{self.t}; {j}; {q}; \t\t\t {self.state['blenders'][j]}")
                
                if self.state["blenders"][j] < self.eps:
                    self.state['properties'][j][q] = 0
                else:
                    in_flow_sources = in_flow_blend = out_flow_blend = out_flow_demands = 0
                    for s in self.sources:
                        if j in action["source_blend"][s].keys():
                            # #self.logg(f"[INFO11] t{self.t}; {s}; {q}; \t\t\t {action['source_blend'][s][j]}; {self.sigma[s][q]}")
                            in_flow_sources += action['source_blend'][s][j] * self.sigma[s][q]
                    for jp in self.blenders:
                        if 'blend_blend' in action.keys() and j in action['blend_blend'][jp].keys():
                            # #self.logg(f"[INFO12] t{self.t}; {jp}; {q}; \t\t\t {action['blend_blend'][jp][j]}; {self.state['properties'][jp][q]}")
                            in_flow_blend += action['blend_blend'][jp][j] * prev_properties[jp][q]
                        if 'blend_blend' in action.keys() and jp in action['blend_blend'][j].keys():
                            # #self.logg(f"[INFO13] t{self.t}; {jp}; {q}; \t\t\t {action['blend_blend'][j][jp]}; {self.state['properties'][j][q]}")
                            out_flow_blend += action['blend_blend'][j][jp] * prev_properties[j][q]
                    for p in self.demands:
                        if p in action["blend_demand"][j].keys():
                            # #self.logg(f"[INFO14] t{self.t}; {p}; {q}; \t\t\t {action['blend_demand'][j][p]}; {self.state['properties'][j][q]}")
                            out_flow_demands += action["blend_demand"][j][p] * prev_properties[j][q]

                    # #self.logg(f"[INFO15] t{self.t}; {j}; {q}; \t\t\t {in_flow_sources }; { in_flow_blend}; {out_flow_blend}; {out_flow_demands}")
                    # #self.logg(f"[INFO2] t{self.t}; {j}; {q}; \t\t\t Previous: {self.state['blenders'][j]}; {self.state['properties'][j][q]}; {in_flow_sources + in_flow_blend - out_flow_blend - out_flow_demands}")
                    self.state['properties'][j][q] = (1/self.state["blenders"][j]) * ( \
                                                    prev_properties[j][q] * prev_blend_invs[j] \
                                                    + in_flow_sources + in_flow_blend - out_flow_blend - out_flow_demands
                                                )
    def update_avail_schedule(self):
        #FUnction to update the future sources and demand available in the state
        for s in self.sources:
            for k in range(self.forecast_window_len):
                self.state[f"sources_avail_next_{k}"][s] = self.tau0[s][k + self.t] if k + self.t < len(self.tau0[s]) else 0
        
        for p in self.demands:
            for k in range(self.forecast_window_len):
                self.state[f"demands_avail_next_{k}"][p] = self.delta0[p][k + self.t] if k + self.t < len(self.delta0[p]) else 0
    def step(self, action_scaled: th.Tensor):
        action_scaled = th.as_tensor(action_scaled, device=self._device)
        low_torch = th.as_tensor(self.action_low, dtype=action_scaled.dtype, device=self._device)
        high_torch = th.as_tensor(self.action_high, dtype=action_scaled.dtype, device=self._device)
        action = (action_scaled + 1) / 2 * (high_torch - low_torch) + low_torch #Scale the action
        action = action_scaled
        self.t += 1
        self.pens_step = {k:0 for k in self.env_spec_log.keys()}
        self.reward = 0
        self.cost = 0
        action = action.clip(0, self.MAXFLOW)
        action = action.tolist()
        action = reconstruct_dict(action, self.mapping_act) # From non-human-readable list to human-readable dict
        action = self.sanitize_action_structure(action)
        action, illegal_preflow_cost = self.check_illegal_actions_preflow_cost(action)
        flow_reward = self.compute_reward_flowarc(action)
        prev_state = copy.deepcopy(self.state)
        prev_blend_invs = prev_state["blenders"]
        prev_properties = prev_state["properties"]
        action,source_bound_cost = self.check_source_inv_bounds(action)
        self.update_source_inv(action)
        #self.logg("Action after processing sources:", action)
        action,in_out_cost = self.check_in_out_rule(action)
        action,blender_bound_cost = self.check_blender_inv_bounds(action)
        self.update_blender_inv(action)
        action,demand_bound_cost = self.check_demand_inv_bounds(action)
        self.update_demand_inv(action)
        self.update_properties(action,prev_blend_invs,prev_properties)
        sold_bought_reward = self.compute_reward_sold_bought(action)
        cost_properties = self.penalty_property(action)
        self.cost+=illegal_preflow_cost+source_bound_cost+in_out_cost+blender_bound_cost+demand_bound_cost+cost_properties
        self.reward+=flow_reward+sold_bought_reward
        self.cost_ep+=self.cost
        self.reward_ep+=self.reward
        self.update_avail_schedule()
        self.state["t"] = self.t
        
        self.flatt_state, _ = flatten_and_track_mappings(self.state)
        
        if self.t == self.T:
            self.terminated = True
        for k in self.env_spec_log.keys():
            self.env_spec_log[k] += self.pens_step[k]
        
        return th.tensor(self.flatt_state).to(self._device), th.tensor(self.reward-self.cost).to(self._device), th.tensor(self.terminated).to(self._device), \
                th.tensor(self.truncated).to(self._device), {"dict_state": self.state, "pen_tracker": self.pens_step, "terminated": self.terminated, "truncated": self.truncated}
    
    @property
    def max_episode_steps(self) -> int:
        return self.T
    
    def render(self, mode='human'):
        print("state:",f"{self.state}")
        print("Specification:",f"{self.env_spec_log}")

    def close(self) -> None:
        return None

    def set_seed(self, seed: int) -> None:
        random.seed(seed)

act_1 = {
    "source_blend": {
        "s1": {
            "j1": 11,
            "j2": 0,
            "j3": 0,
            "j4": 0
        },
        "s2": {
            "j1": 10,
            "j2": 0,
            "j3": 20,
            "j4": 0
        }
    },
    "blend_blend": {
        "j1": {
            "j5": 0,
            "j6": 0,
            "j7": 0,
            "j8": 0
        },
        "j2": {
            "j5": 0,
            "j6": 0,
            "j7": 0,
            "j8": 0
        },
        "j3": {
            "j5": 0,
            "j6": 0,
            "j7": 0,
            "j8": 0
        },
        "j4": {
            "j5": 0,
            "j6": 0,
            "j7": 0,
            "j8": 0
        },
        "j5": {},
        "j6": {},
        "j7": {},
        "j8": {}
    },
    "blend_demand": {
        "j1": {},
        "j2": {},
        "j3": {},
        "j4": {},
        "j5": {
            "p1": 0,
            "p2": 0
        },
        "j6": {
            "p1": 0,
            "p2": 0
        },
        "j7": {
            "p1": 0,
            "p2": 0
        },
        "j8": {
            "p1": 0,
            "p2": 0
        }
    },
    "tau": {
        "s1": 11,
        "s2": 30
    },
    "delta": {
        "p1": 0,
        "p2": 0
    }
}

act_2 = {
    "source_blend": {
        "s1": {
            "j1": 0,
            "j2": 10,
            "j3": 0,
            "j4": 0
        },
        "s2": {
            "j1": 0,
            "j2": 10,
            "j3": 0,
            "j4": 20
        }
    },
    "blend_blend": {
        "j1": {
            "j5": 15,
            "j6": 0,
            "j7": 0,
            "j8": 0
        },
        "j2": {
            "j5": 0,
            "j6": 0,
            "j7": 0,
            "j8": 0
        },
        "j3": {
            "j5": 0,
            "j6": 0,
            "j7": 15,
            "j8": 0
        },
        "j4": {
            "j5": 0,
            "j6": 0,
            "j7": 0,
            "j8": 0
        },
        "j5": {},
        "j6": {},
        "j7": {},
        "j8": {}
    },
    "blend_demand": {
        "j1": {},
        "j2": {},
        "j3": {},
        "j4": {},
        "j5": {
            "p1": 0,
            "p2": 0
        },
        "j6": {
            "p1": 0,
            "p2": 0
        },
        "j7": {
            "p1": 0,
            "p2": 0
        },
        "j8": {
            "p1": 0,
            "p2": 0
        }
    },
    "tau": {
        "s1": 10,
        "s2": 30
    },
    "delta": {
        "p1": 0,
        "p2": 0
    }
}

act_3 = {
    "source_blend": {
        "s1": {
            "j1": 10,
            "j2": 0,
            "j3": 0,
            "j4": 0
        },
        "s2": {
            "j1": 10,
            "j2": 0,
            "j3": 20,
            "j4": 0
        }
    },
    "blend_blend": {
        "j1": {
            "j5": 0,
            "j6": 0,
            "j7": 0,
            "j8": 0
        },
        "j2": {
            "j5": 0,
            "j6": 20,
            "j7": 0,
            "j8": 0
        },
        "j3": {
            "j5": 0,
            "j6": 0,
            "j7": 0,
            "j8": 0
        },
        "j4": {
            "j5": 0,
            "j6": 0,
            "j7": 0,
            "j8": 20
        },
        "j5": {},
        "j6": {},
        "j7": {},
        "j8": {}
    },
    "blend_demand": {
        "j1": {},
        "j2": {},
        "j3": {},
        "j4": {},
        "j5": {
            "p1": 15,
            "p2": 0
        },
        "j6": {
            "p1": 0,
            "p2": 0
        },
        "j7": {
            "p1": 0,
            "p2": 15
        },
        "j8": {
            "p1": 0,
            "p2": 0
        }
    },
    "tau": {
        "s1": 10,
        "s2": 30
    },
    "delta": {
        "p1": 15,
        "p2": 15
    }
}

act_4 = {
    "source_blend": {
        "s1": {
            "j1": 0,
            "j2": 0,
            "j3": 0,
            "j4": 0
        },
        "s2": {
            "j1": 0,
            "j2": 0,
            "j3": 0,
            "j4": 0
        }
    },
    "blend_blend": {
        "j1": {
            "j5": 15,
            "j6": 0,
            "j7": 0,
            "j8": 0
        },
        "j2": {
            "j5": 0,
            "j6": 0,
            "j7": 0,
            "j8": 0
        },
        "j3": {
            "j5": 0,
            "j6": 0,
            "j7": 15,
            "j8": 0
        },
        "j4": {
            "j5": 0,
            "j6": 0,
            "j7": 0,
            "j8": 0
        },
        "j5": {},
        "j6": {},
        "j7": {},
        "j8": {}
    },
    "blend_demand": {
        "j1": {},
        "j2": {},
        "j3": {},
        "j4": {},
        "j5": {
            "p1": 0,
            "p2": 0
        },
        "j6": {
            "p1": 15,
            "p2": 0
        },
        "j7": {
            "p1": 0,
            "p2": 0
        },
        "j8": {
            "p1": 0,
            "p2": 15
        }
    },
    "tau": {
        "s1": 0,
        "s2": 0
    },
    "delta": {
        "p1": 15,
        "p2": 15
    }
}

act_5 = {
    "source_blend": {
        "s1": {
            "j1": 0,
            "j2": 0,
            "j3": 0,
            "j4": 0
        },
        "s2": {
            "j1": 0,
            "j2": 0,
            "j3": 0,
            "j4": 0
        }
    },
    "blend_blend": {
        "j1": {
            "j5": 0,
            "j6": 10,
            "j7": 0,
            "j8": 0
        },
        "j2": {
            "j5": 0,
            "j6": 0,
            "j7": 0,
            "j8": 0
        },
        "j3": {
            "j5": 0,
            "j6": 0,
            "j7": 0,
            "j8": 10
        },
        "j4": {
            "j5": 0,
            "j6": 0,
            "j7": 0,
            "j8": 0
        },
        "j5": {},
        "j6": {},
        "j7": {},
        "j8": {}
    },
    "blend_demand": {
        "j1": {},
        "j2": {},
        "j3": {},
        "j4": {},
        "j5": {
            "p1": 15,
            "p2": 0
        },
        "j6": {
            "p1": 0,
            "p2": 0
        },
        "j7": {
            "p1": 0,
            "p2": 15
        },
        "j8": {
            "p1": 0,
            "p2": 0
        }
    },
    "tau": {
        "s1": 0,
        "s2": 0
    },
    "delta": {
        "p1": 15,
        "p2": 15
    }
}
        
        
act_6 = {
    "source_blend": {
        "s1": {
            "j1": 0,
            "j2": 0,
            "j3": 0,
            "j4": 0
        },
        "s2": {
            "j1": 0,
            "j2": 0,
            "j3": 0,
            "j4": 0
        }
    },
    "blend_blend": {
        "j1": {
            "j5": 0,
            "j6": 0,
            "j7": 0,
            "j8": 0
        },
        "j2": {
            "j5": 0,
            "j6": 0,
            "j7": 0,
            "j8": 0
        },
        "j3": {
            "j5": 0,
            "j6": 0,
            "j7": 0,
            "j8": 0
        },
        "j4": {
            "j5": 0,
            "j6": 0,
            "j7": 0,
            "j8": 0
        },
        "j5": {},
        "j6": {},
        "j7": {},
        "j8": {}
    },
    "blend_demand": {
        "j1": {},
        "j2": {},
        "j3": {},
        "j4": {},
        "j5": {
            "p1": 0,
            "p2": 0
        },
        "j6": {
            "p1": 15,
            "p2": 0
        },
        "j7": {
            "p1": 0,
            "p2": 0
        },
        "j8": {
            "p1": 0,
            "p2": 15
        }
    },
    "tau": {
        "s1": 0,
        "s2": 0
    },
    "delta": {
        "p1": 15,
        "p2": 15
    }
}

act_list = [act_1, act_2, act_3, act_4, act_5, act_6]
#act_list = [act_1]
M, Q, P, B, Z, D = 100, 0, 1, 1, 1, 0
env = BlendEnv(M=M, Q=Q, P=P, B=B, Z=Z, D=D,env_id='Blending-simple')
episode_rewards = []
obs = env.reset()
obs, obs_dict = obs
rew_tot = 0
for action in act_list:
    action_flatt, mapp = flatten_and_track_mappings(action)
    obs, reward, done, term, _ = env.step(action_flatt)
    print("After step:",env.t)
    env.render()
    rew_tot += reward
print(rew_tot)


