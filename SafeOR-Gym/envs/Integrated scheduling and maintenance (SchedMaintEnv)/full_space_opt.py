import pyomo.environ as pyo
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyomo.environ import value
from pyomo.environ import *
from scipy.spatial import ConvexHull
import logging
import os

class optimize_GASU:

    def __init__(self, state_horizon, action_horizon, compressors):

        self.state_horizon = state_horizon
        self.action_horizon = action_horizon
        self.model = pyo.ConcreteModel()
        self._initialize_compressors(compressors)
        self._define_sets()
        self._define_parameters()
        self._define_variables()
        self._define_binary_variables()
        self._define_constraints()
        self._define_objective()

        """
        To solve model, we would need to:
            1. One time update
                :update_external_purchase_price(external_purchase_price)
                :update_external_purchase_capacity(external_purchase_capacity)
            2. Based on state
                :update_information_state(demand_array, price_array)
                :update_mode_and_switch_history(compressor_observation)
        """
    
    def reset(self):
        self.model = pyo.ConcreteModel()  
        self._initialize_compressors()
        self._define_parameters()
        self._define_variables()
        self._define_binary_variables()
        self._define_constraints()
        self._define_objective()
        self._define_sets()        

    def _initialize_compressors(self, compressors):
        self.compressors = compressors
        self.max_mttf = max(comp["mttf"] for comp in self.compressors.values())  # Minimum history needed
        
    def _define_sets(self): 
        self.model.T = pyo.RangeSet(-self.max_mttf, self.state_horizon, doc = "Total horizon")
        self.model.T_bar = pyo.RangeSet(1, self.state_horizon, doc = "Scheduling horizon")    
        self.model.T_z = pyo.RangeSet(-self.max_mttf, -1, doc = "History horizon")
        self.model.T_BC = pyo.RangeSet(0, self.state_horizon, doc = "Scheduling horizon with boundary condition")
        self.model.TR = pyo.Set(initialize=[('W', 'M'), ('M', 'W')], doc="Transition set")
        self.model.C = pyo.Set(initialize=self.compressors.keys(), doc = "Compressors")
        modes = ['W', 'M']  # W is working, M is maintenance
        self.model.M = pyo.Set(initialize=modes, doc = "Modes: W, M")

    def _define_parameters(self):

        ## 1. FIXED PARAMETERS
        # Assuming self.compressors is defined as shown
        cap_dict = {cid: data["capacity"] for cid, data in self.compressors.items()}
        cap_param_dict = {
            (cid, mode): cap_dict[cid] if mode == "W" else 0
            for cid in self.model.C
            for mode in self.model.M
        }
        mttf_dict = {cid: data["mttf"] for cid, data in self.compressors.items()}
        mttr_dict = {cid: data["mttr"] for cid, data in self.compressors.items()}
        mntr_dict = {cid: data["mntr"] for cid, data in self.compressors.items()}
        spen_dict = {cid: data["specific_energy"] for cid, data in self.compressors.items()} 
        self.model.CAP = pyo.Param(self.model.C, self.model.M, initialize=cap_param_dict, doc="Compressor capacity by mode")
        self.model.MTTF = pyo.Param(self.model.C, initialize=mttf_dict, doc="Mean Time To Failure, days")
        self.model.MTTR = pyo.Param(self.model.C, initialize=mttr_dict, doc="Mean Time To Repair, days")
        self.model.MNTR = pyo.Param(self.model.C, initialize=mntr_dict, doc="Minimum No Repair Time, days")
        self.model.SPEN = pyo.Param(self.model.C, initialize=spen_dict, doc="Specific Energy Consumption, KWh/ton")
        
        ## 2. MUTABLE PARAMETERS
        self.model.D = pyo.Param(self.model.T_bar, initialize=0, mutable=True, doc="Demand")
        self.model.ELP = pyo.Param(self.model.T_bar, initialize=0, mutable=True, doc="Electricty Price")
        self.model.EXPP = pyo.Param(initialize=0, mutable=True, doc="External Purchase Price, $/MWh")
        self.model.EXPC = pyo.Param(initialize=0, mutable=True, doc="External Purchase Capacity, ton/day")
        # Mode and Switch History Parameters
        self.model.y_i = pyo.Param(self.model.C, self.model.M, initialize=0, mutable=True, doc="Mode history")
        self.model.z_h = pyo.Param(self.model.C, self.model.M, self.model.M, self.model.T_z, initialize=0, mutable=True, doc="Switch history")

    def decode_observation(self, flatt_state):
        n = len(self.compressors)
        S = self.state_horizon

        return {
            "demand": flatt_state[:S],
            "electricity_price": flatt_state[S:2*S],
            "TLCM": flatt_state[2*S:2*S+n],
            "TSLM": flatt_state[2*S+n:2*S+2*n],
            "CDM": flatt_state[2*S+2*n:2*S+3*n].astype(int)
        }

    def update_state(self, flatt_state):
        """
        Update the model with the current state.
        :param state: A dictionary containing the current state of the environment.
        """
        state = self.decode_observation(flatt_state)
        demand_array = state['demand']
        price_array = state['electricity_price']
        
        comp_tlcm = state['TLCM']
        comp_tslm = state['TSLM']
        comp_cdm = state['CDM']
    
        self._update_information_state(demand_array, price_array)
        self._update_mode_and_switch_history(comp_tlcm, comp_tslm, comp_cdm)

    def _update_external_purchase_price(self, external_purchase_price):
        self.model.EXPP.set_value(external_purchase_price) 

    def _update_external_purchase_capacity(self, external_purchase_capacity):
        self.model.EXPC.set_value(external_purchase_capacity + 2) 
    
    def _update_information_state(self, demand_array, price_array):
        for t in self.model.T_bar:
            self.model.D[t].set_value(float(demand_array[t - 1]))
            self.model.ELP[t].set_value(float(price_array[t - 1]))  

    def _update_mode_and_switch_history(self, comp_tlcm, comp_tslm, comp_cdm):
        # Optimizer takes decision for z_0 itself.
        self.max_mttf = max(comp["mttf"] for comp in self.compressors.values())  # Minimum history needed
        comp_ids = list(self.compressors.keys()) 

        for i, cid in enumerate(comp_ids):
            c_tlcm = comp_tlcm[i]
            c_tslm = comp_tslm[i]
            c_cdm = comp_cdm[i]

            mttr = self.compressors[cid]["mttr"]
            mttf = self.compressors[cid]["mttf"]
            # t_max = -mttf + 1  # old
            t_max = -mttf  # new

            # === CASE 1: Compressor currently under maintenance or has completed maintenance
            if (c_tlcm > 0) or (c_tslm == 0 and c_tslm == 0):
                self.model.y_i[cid, 'M'].set_value(1)
                self.model.y_i[cid, 'W'].set_value(0)
                t_switch = mttr - c_tlcm

                for t in range(t_max, 0):
                    if t == -t_switch:
                        self.model.z_h[cid, 'W', 'M', t].set_value(1)
                        self.model.z_h[cid, 'M', 'W', t].set_value(0)
                    else:
                        self.model.z_h[cid, 'W', 'M', t].set_value(0)
                        self.model.z_h[cid, 'M', 'W', t].set_value(0)
            
            # === CASE 2: Compressor is working
            if c_tslm > 0:
                self.model.y_i[cid, 'W'].set_value(1)
                self.model.y_i[cid, 'M'].set_value(0)
                t_switch = c_tslm

                for t in range(t_max, 0):
                    if t == -t_switch:
                        self.model.z_h[cid, 'W', 'M', t].set_value(0)
                        self.model.z_h[cid, 'M', 'W', t].set_value(1)
                    else:
                        self.model.z_h[cid, 'W', 'M', t].set_value(0)
                        self.model.z_h[cid, 'M', 'W', t].set_value(0)
                
            # === Clear transition history if t_max > max_mttf
            if t_max > self.max_mttf:
                for t in range(-self.max_mttf, t_max):
                    self.model.z_h[cid, 'W', 'M', t].set_value(0)
                    self.model.z_h[cid, 'M', 'W', t].set_value(0)
             
    def _define_variables(self):
        '''
        __NOTE__: z_(c, mm', t) @t=0  z_(c, mm', t) would be derived from the scheduling decision, i.e., 
          .. Based on y_1 decison taken by optimizer:
          .. then that would be captured in z_0. Since transition index lags the mode decision by 1 time step
        '''
        self.model.ELC = pyo.Var(self.model.T_bar, within=pyo.NonNegativeReals, doc="Electricity Consumption, KWh/day")
        self.model.EXPQ = pyo.Var(self.model.T_bar, within=pyo.NonNegativeReals, doc="External Purchase Quantity, ton/day")
        # self.model.EXPQ = pyo.Var(self.model.T_bar, within=pyo.UnitInterval, doc="Fraction of Max External Purchase Quantity, ton/day")
        self.model.PD_bar = pyo.Var(self.model.C, self.model.M, self.model.T_bar, within=pyo.NonNegativeReals, doc="Mode specfic production, ton/day")
        self.model.PD = pyo.Var(self.model.C, self.model.T_bar, within=pyo.NonNegativeReals, doc="Total production, ton/day")
        self.model.R = pyo.Var(self.model.C, self.model.M, self.model.T_bar, within=pyo.NonNegativeReals, doc="Ramp rate")

    def _define_binary_variables(self):
        self.model.y = pyo.Var(self.model.C, self.model.M, self.model.T_BC, within=pyo.Binary, doc="Mode decision")
        self.model.z = pyo.Var(self.model.C, self.model.M, self.model.M, self.model.T, within=pyo.Binary, doc="Switch decision")
    
    def _define_constraints(self):

        # Demand Satisfaction Constraint
        def demand_satisfaction_rule(model, t):
            return sum(model.PD[c, t] for c in model.C) + model.EXPQ[t]  == model.D[t]
        self.model.demand_satisfaction = pyo.Constraint(self.model.T_bar, rule=demand_satisfaction_rule)

        # Production Constraint
        def production_rule(model, c, t):
            return model.PD[c, t] == sum(model.PD_bar[c, m, t] for m in model.M)
        self.model.production = pyo.Constraint(self.model.C, self.model.T_bar, rule=production_rule)

        # Mode Production Constraint
        def mode_production_rule(model, c, m, t):
            return model.PD_bar[c, m, t] == model.R[c, m, t] * model.CAP[c, m]
        self.model.mode_production = pyo.Constraint(self.model.C, self.model.M, self.model.T_bar, rule=mode_production_rule)

        # Ramp Rate Constraint
        def ramp_rate_rule(model, c, m, t):
            return model.R[c, m, t] <= model.y[c, m, t]
        self.model.ramp_rate = pyo.Constraint(self.model.C, self.model.M, self.model.T_bar, rule=ramp_rate_rule)

        # Energy Consumption Constraint
        def energy_consumption_rule(model, t):
            return model.ELC[t] == sum(model.SPEN[c] * model.PD[c, t] for c in model.C)     # KWh/t * ton/day
        self.model.energy_consumption = pyo.Constraint(self.model.T_bar, rule=energy_consumption_rule)
        
        ###### ###### ###### ###### ###### ######
        # Boundary conditions
        # Boundary condition for y
        def y_boundary_rule(model, c, m):
         return model.y[c, m, 0] == model.y_i[c, m]
        self.model.y_boundary = pyo.Constraint(self.model.C, self.model.M, rule=y_boundary_rule)

        # Boundary condition for z
        def switch_history_rule(model, c, m1, m2, tz):
         if (m1, m2) in model.TR:
               return model.z[c, m1, m2, tz] == model.z_h[c, m1, m2, tz]
         else:
               return pyo.Constraint.Skip
        self.model.switch_history = pyo.Constraint(self.model.C, self.model.M, self.model.M, self.model.T_z, rule = switch_history_rule)
        # self.model.T_z = pyo.RangeSet(-self.max_mttf+1, -1, doc = "History horizon")

        ###### ###### ###### ###### ###### ######
        # Transition constraints

        # 1 Active mode rule
        def sum_active_mode_rule(model, c, t):
            return sum(model.y[c, m, t] for m in model.M) == 1
        self.model.sum_active_mode = pyo.Constraint(self.model.C, self.model.T_bar, rule=sum_active_mode_rule)

        # Switch variable constraint
        def transitions_combined_rule(model, c, m, t):
         return sum(model.z[c, m1, m , t-1] for (m1, m2) in model.TR if m2 == m) - sum(model.z[c, m, m2, t-1] for (m1, m2) in model.TR if m1 == m) == model.y[c, m, t] - model.y[c, m, t-1]
        self.model.transitions_combined = pyo.Constraint(self.model.C, self.model.M, self.model.T_bar, rule = transitions_combined_rule)

        # one transition constraint
        def one_transition_per_timestep_rule(model, c, t):
            return sum(model.z[c, m1, m2, t] for m1 in model.M for m2 in model.M if m1 != m2) <= 1
        self.model.one_transition_per_timestep = pyo.Constraint(self.model.C, self.model.T_BC, rule=one_transition_per_timestep_rule)
        
        # minimum time no repair constraint
        def minimum_time_no_repair_rule(model, c, m, t):
            return model.y[c, 'W', t] >= sum(model.z[c, 'M', 'W', t - k] for k in range(1, model.MNTR[c] + 1))
        self.model.minimum_time_no_repair = pyo.Constraint(self.model.C, self.model.M, self.model.T_BC, rule=minimum_time_no_repair_rule) 

        # minimum time to repair constraint
        def minimum_repair_ON_time(model, c, m, t):
            return model.y[c, 'M', t] >= sum(model.z[c, 'W', 'M', t - k] for k in range(1, model.MTTR[c] + 1))
        self.model.minimum_repair_ON_time = pyo.Constraint(self.model.C, self.model.M, self.model.T_bar, rule=minimum_repair_ON_time)

        # pre-defined transition from 'if M started --> W'
        def fixed_transition_rule(model, c, t):
            return model.z[c, 'W', 'M', t - model.MTTR[c]] == model.z[c, 'M', 'W', t]
        self.model.fixed_transition = pyo.Constraint(self.model.C, self.model.T_BC, rule=fixed_transition_rule)

        def maximum_working_time_rule(model, c, t):
            t_limit = t + model.MTTF[c]
            if t_limit+1 in model.T_bar:
                return model.y[c, 'W', t_limit+1] <= 1 - model.z[c, 'M', 'W', t]
            else:
                return pyo.Constraint.Skip
        self.model.max_working_time = pyo.Constraint(self.model.C, self.model.T, rule=maximum_working_time_rule)

    def _define_objective(self):
        # Objective function
        def objective_rule(model):
            return sum(model.ELC[t] * model.ELP[t] for t in model.T_bar) + sum(model.EXPQ[t] * model.EXPP for t in model.T_bar)
        self.model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
        
    def solve(self, solver_name='gurobi', tee=False):
        solver = pyo.SolverFactory('gurobi')
        results = solver.solve(self.model, tee=False)
        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            logging.info("Optimal solution found.")
        else:
            logging.warning("Solver did not find an optimal solution.")
        
        objective_value = pyo.value(self.model.objective)
        return objective_value

    def encode_action(self, action_dict):
        """
        Converts a structured action dictionary into a flat NumPy array
        compatible with the flat Box action space.
        """
        maintenance_action = np.array(action_dict["maintenance_action"], dtype=np.float32)  # shape (n,)
        production_rate = np.array(action_dict["production_rate"], dtype=np.float32)        # shape (n,)
        external_purchase = np.array(action_dict["external_purchase"], dtype=np.float32)    # shape (1,) or scalar

        # Ensure external_purchase is a 1-element array
        if external_purchase.ndim == 0:
            external_purchase = np.array([external_purchase], dtype=np.float32)

        flat_action = np.concatenate([maintenance_action, production_rate, external_purchase])
        return flat_action

    def _fetch_optimal_action_for_action_horizon(self):
        action = {}
        t = 1  # Day 1

        # --- 1. Maintenance action: y[c, 'M', t] == 1 means compressor is in maintenance
        maintenance_action = [
            int(pyo.value(self.model.y[c, 'M', t])) for c in self.model.C
        ]
        action["maintenance_action"] = np.array(maintenance_action, dtype=np.int32)

        # --- 2. Production rate: use ramp variable R[c, 'W', t] (normalized between 0–1)
        production_rate = [
            float(pyo.value(self.model.R[c, 'W', t])) for c in self.model.C
        ]
        action["production_rate"] = np.array(production_rate, dtype=np.float32)

        # --- 3. External purchase
        ext_purchase = float(pyo.value(self.model.EXPQ[t]))
        ext_purchase = ext_purchase/pyo.value(self.model.EXPC)
        action["external_purchase"] = np.array([ext_purchase], dtype=np.float32)
        
        flat_action = self.encode_action(action)
        return flat_action
    
    def action_horizon_cost(self):  
        t = 1  # action horizon is 1
        energy_cost = pyo.value(self.model.ELC[t]) * pyo.value(self.model.ELP[t])
        purchase_cost = pyo.value(self.model.EXPQ[t]) * pyo.value(self.model.EXPP) 
        cost = energy_cost + purchase_cost
        return cost
    
    def _plots(self):
        for c in self.model.C:
            fig, ax = plt.subplots(figsize=(10, 4))
            time_points = list(self.model.T_BC)

            # Plot mode indicator lines for context (step aligned with highlight)
            for m in self.model.M:
                y_values = [value(self.model.y[c, m, t]) for t in time_points]
                ax.step(time_points, y_values, label=f'Mode: {m}', where='pre')

            # Highlight working and maintenance periods
            for m in self.model.M:
                mode_color = 'green' if m == 'W' else 'red'

                for t in time_points[1:]:  # start from t=1 to get t-1
                    y_val = value(self.model.y[c, m, t])
                    if y_val == 1:
                        ax.axvspan(t - 1, t, color=mode_color, alpha=0.2)

            ax.set_xlabel('Time')
            ax.set_ylabel('Mode (0 or 1)')
            ax.set_title(f'Mode Switching Plot — Compressor {c}')
            ax.set_ylim(-0.1, 1.1)
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.show()

        ### PLOT 2: External Purchase Quantity Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        expq_values = [value(self.model.EXPQ[t]) for t in self.model.T_bar]
        ax.plot(self.model.T_bar, expq_values, label='External Purchase Quantity', color='orange')
        ax.set_xlabel('Time')
        ax.set_ylabel('External Purchase Quantity')
        ax.set_title('External Purchase Quantity Plot')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()

        ### PLOT 3: Production Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for c in self.model.C:
            pd_values = [value(self.model.PD[c, t]) for t in self.model.T_bar]
            ax.plot(self.model.T_bar, pd_values, label=f'Compressor {c}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Production')
        ax.set_title('Production Plot')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        ### PLOT 4: State Demand
        fig, ax = plt.subplots(figsize=(10, 6))
        demand_values = [value(self.model.D[t]) for t in self.model.T_bar]
        ax.plot(self.model.T_bar, demand_values, label='Demand', color='purple')
        ax.set_xlabel('Time')
        ax.set_ylabel('Demand')
        ax.set_title('Demand Plot')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()

        ### Plot 5: Ramp Rate Plot (0 when not in 'W' mode)
        fig, ax = plt.subplots(figsize=(10, 6))

        for c in self.model.C:
            ramp_rate_values = []

            for t in self.model.T_bar:
                y_w = pyo.value(self.model.y[c, 'W', t])
                if y_w == 1: 
                    ramp = pyo.value(self.model.R[c, 'W', t])
                else:
                    ramp = 0.0  # not working
                ramp_rate_values.append(ramp)

            ax.plot(self.model.T_bar, ramp_rate_values, label=f'Compressor {c}')

        ax.set_xlabel('Time')
        ax.set_ylabel('Ramp Rate (only when in W mode)')
        ax.set_title('Ramp Rate Profile — W mode only')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

