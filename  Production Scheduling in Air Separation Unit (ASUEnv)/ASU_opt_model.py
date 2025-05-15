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

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'asu_config.json')
# Load the configuration mapping from a JSON file
with open(config_path, 'r') as config_file:
   ASU_DATA_FILES = json.load(config_file)

class optimize_ASU:
        
   def __init__(self, asu_name, lookahead):
      
      self.name = asu_name
      self.lookahead = lookahead  # number of days to look ahead
      # Get the data file path from the configuration
      data_file = ASU_DATA_FILES.get(asu_name)
      if data_file is None:
         raise ValueError(f"Unknown ASU identifier: {asu_name}")
      
      # Load the ASU data from the JSON file
      with open(data_file, 'r') as file:
         self.loaded_data = json.load(file)
      
      # self.loaded_data = loaded_data
      self.model = pyo.ConcreteModel()
      self._define_sets()
      self._define_parameters()
      self._define_variables()
      self._define_binary_variables()
      self._define_objective()
      self._define_constraints()

   def _reset_model(self):
      """
      Reset the model to its initial state.
      """
      # Reset the model by re-initializing it
      self.model = pyo.ConcreteModel()
      self._define_sets()
      self._define_parameters()
      self._define_variables()
      self._define_binary_variables()
      self._define_objective()
      self._define_constraints()
              
   def _define_sets(self):
      """
      Define the sets used in the optimization model.
      """
      theta_minon = self.loaded_data['theta_minon']
      theta_minoff = self.loaded_data['theta_minoff']
      theta_startup = self.loaded_data['theta_startup']
      total_hours = self.loaded_data['total_hours']
      interval_hours = self.loaded_data['interval_hours']
      # total_hours = total_hours[0]
      total_hours = (1 + self.lookahead) * 24
      interval_hours = interval_hours[0]
      self.time_values = np.arange(0, total_hours, interval_hours)

      ### Data preprocessing to yield Convex Hull ###
      liq_prod_data = self.loaded_data['liq_prod_data']
          # Convert liq_prod_data to a list of tuples
      points = [(liq_prod_data['LIN'][i], liq_prod_data['LOX'][i], liq_prod_data['LAR'][i]) for i in liq_prod_data['LIN']]
      # Compute the convex hull
      points_np = np.array(points)                         # Convert list of tuples to a NumPy array
      hull = ConvexHull(points_np)
      self.extreme_points_liqp = points_np[hull.vertices]       # Extract the vertices (extreme points)
      row_liqprod, colliq_prod = self.extreme_points_liqp.shape
      vertices = {
        'OFF': [1],
        'Liquid_SU': [1],
        'Liquid_Prod': list(range(1, row_liqprod + 1))
      }

      ########################################################################

      self.model.I = pyo.Set(initialize=['LIN', 'LOX', 'LAR'])                   # Products
      self.model.M = pyo.Set(initialize=['OFF', 'Liquid_SU', 'Liquid_Prod'])     # Operating Modes
      self.model.J = pyo.Set(self.model.M, within=pyo.NonNegativeIntegers, initialize=lambda model, m: vertices[m])
      self.model.MJI = pyo.Set(dimen=3, initialize=[(m, j, i) for m in self.model.M for j in list(self.model.J[m]) for i in self.model.I])
      self.model.SQ = pyo.Set(initialize=[      # SQ = Pre-defined sequences of mode transitions
         ('OFF','Liquid_SU','Liquid_Prod')
      ])
      self.model.TR = pyo.Set(initialize=[      # TR = Possible mode-to-mode transition
         ('OFF','Liquid_SU'),
         ('Liquid_SU','Liquid_Prod'),
         ('Liquid_Prod','OFF'),
      ])
      self.model.DTR = pyo.Set(initialize=[     # DTR = Diallowed transitions
         ('OFF','Liquid_Prod'),
         ('Liquid_Prod','Liquid_SU'),
         ('Liquid_SU','OFF'),
      ]) 
      
      # Time Period   
      self.model.T_bar = pyo.Set(initialize=list(range(1, len(self.time_values)+1 )), doc="Time Horizon")  # 1 to 192 - Decisions to make at each time    # point for subsequent 1 hour interval                                                    
      self.model.T = pyo.Set(initialize=list(range(-5 +1 , len(self.time_values)+1 )))       # -5 to 192      Whole Time Horizon
      self.model.T_BC = pyo.Set(initialize=list(range(0 , len(self.time_values)+1)))      #  0 t0 192 :to define mode at y[-1] 
      self.model.T_zhistory = pyo.Set(initialize=list(range(-4 , 0)))   # Previosuly -5 to -1, Since min OFF was 6 Hours.    
      self.model.tf = pyo.Param(initialize=len(self.time_values))  # Final time period
      self.model.W = pyo.Set(initialize=list(range(1, 2+self.lookahead)))  # Weeks      
      # Function to map each day to a set of time periods
      def day_to_time_set(model, w):
        start = 24 * (w - 1) + 1
        end = 24 * w
        return list(range(start, end + 1))
      # Create a 2D set for days and corresponding time periods
      self.model.W_T = pyo.Set(self.model.W, initialize=day_to_time_set, within=self.model.T_bar)

      self.model.theta_minon = pyo.Set(initialize = theta_minon) # theta_minon = list(range(0, 5))
      self.model.theta_minoff = pyo.Set(initialize = theta_minoff) # theta_minoff = list(range(0, 5))   ## Previosu: theta_minoff = list(range(0, 6)) 
      self.model.theta_startup = pyo.Set(initialize = theta_startup)  # theta_startup = list(range(0, 2))
      self.cardinality_startup = len(self.model.theta_startup)

   def _define_parameters(self):
      """ Define the parameters used in the optimization model. """

      delta = self.loaded_data['delta']
      gamma_fetch = self.loaded_data['gamma']
      gamma = {}
      for key, value in gamma_fetch.items():
         # Identify the product by checking if the key ends with one of the product names.
         for prod in self.model.I:
            if key.endswith(prod):
                     # Remove the underscore and product from the end to extract the mode.
                     mode = key[:-len(prod)-1]  # subtract length of product and underscore
                     gamma[(mode, prod)] = value
                     break
      IV_u = self.loaded_data['IV_u']
      IV_l = self.loaded_data['IV_l']
      IV_f = self.loaded_data['IV_f']
      y_initial = self.loaded_data['y_i']
      IV_i = self.loaded_data['IV_i']

      z_history_str_keys = self.loaded_data['z_history']
      z_history = {}
      for key, value in z_history_str_keys.items():
         # For each valid (m1, m2) in switch_elements, check if the key starts with their joined string.
         for m1, m2 in self.model.TR:
         # for m1, m2 in switch_elements:
            prefix = f"{m1}_{m2}_"
            if key.startswith(prefix):
                  # Extract the time stamp t from the remainder of the key
                  t_str = key[len(prefix):]
                  try:
                     t = int(t_str)
                  except ValueError:
                     raise ValueError(f"Cannot convert time stamp part '{t_str}' to int in key '{key}'")
                  # Use a tuple key as originally defined
                  z_history[(m1, m2, t)] = value
                  break
      
      product_index = {'LIN': 0, 'LOX': 1, 'LAR': 2}
      def v_init(b, m, j, i):   # Defining the extreme points for each mode
         if m == 'Liquid_Prod' and i in b.I:
               # Use product_index to get the appropriate column index for i
               return self.extreme_points_liqp[j - 1, product_index[i]]
         return 0
      
      #### MUTABLE PARAMETERS: El, D, y_i, IV_i, z_history 
      self.model.alpha_EP = pyo.Param(self.model.T_bar,initialize = 0, mutable = True, doc="specific costs for electricity at time t")
      self.model.D = pyo.Param(self.model.I,self.model.T_bar,initialize=0, mutable = True)
      self.model.y_i = pyo.Param(self.model.M, initialize=y_initial, mutable = True, domain=pyo.Binary)
      self.model.z_h = pyo.Param(self.model.TR, self.model.T_zhistory, initialize = z_history, mutable = True, domain=pyo.Binary)
      self.model.IV_i = pyo.Param(self.model.I, initialize=IV_i, mutable = True, domain=pyo.NonNegativeReals)

      ####    NON-MUTABLE PARAMETERS: gamma, delta, IV_u, IV_l, IV_f    ####
      self.model.v = pyo.Param(self.model.MJI, initialize=v_init)
      self.model.delta = pyo.Param(self.model.M, initialize=delta)
      self.model.gamma = pyo.Param(self.model.M, self.model.I, initialize=gamma)
      self.model.IV_u = pyo.Param(self.model.I, initialize=IV_u)
      self.model.IV_l = pyo.Param(self.model.I, initialize=IV_l)
      self.model.IV_f = pyo.Param(self.model.I, initialize=IV_f)

   def _define_binary_variables(self):
      # Now add the new variables with the desired domain
      self.model.add_component('y', pyo.Var(self.model.M, self.model.T_BC, domain=pyo.Binary, initialize=0))
      self.model.add_component('z', pyo.Var(self.model.M, self.model.M, self.model.T, domain=pyo.Binary, initialize=0))

   def _define_variables(self):
      """
      Define the decision variables.
      """
      # self.model.y = pyo.Var(self.model.M, self.model.T_BC, domain=pyo.Binary)
      # self.model.z = pyo.Var(self.model.M, self.model.M, self.model.T, domain=pyo.Binary)

      self.model.EC = pyo.Var(self.model.T_bar, domain=pyo.NonNegativeReals)
      self.model.EP = pyo.Var(self.model.T_bar, domain=pyo.NonNegativeReals)
      self.model.IV = pyo.Var(self.model.I, self.model.T_BC,  domain=pyo.NonNegativeReals)
      self.model.PD = pyo.Var(self.model.I, self.model.T_bar,domain=pyo.NonNegativeReals)
      self.model.PD_bar = pyo.Var(self.model.M, self.model.I, self.model.T_bar, domain=pyo.NonNegativeReals)
      self.model.SL = pyo.Var(self.model.I, self.model.T_bar, domain=pyo.NonNegativeReals)
      # self.model.llambda = pyo.Var(self.model.M, self.model.J[m], self.model.T_bar, domain=pyo.NonNegativeReals)
      index_set = [(m, j, t) for m in self.model.M
                     for j in self.model.J[m]
                     for t in self.model.T_bar]
      self.model.llambda = pyo.Var(index_set, domain=pyo.UnitInterval)
      # self.model.llambda = pyo.Var(self.model.M, {m: self.model.J[m] for m in self.model.M},   # Ensure J[m] is correctly referenced
      #       self.model.T_bar, domain=pyo.UnitInterval)

   def _define_objective(self):
      """
      Define the objective function.
      Example: maximize total production or profit.
      """
      def compute_elec_cost(model):
         """Inner function to compute electricity cost."""
         return sum(model.alpha_EP[t] * model.EP[t] for t in model.T_bar)

      self.model.objective = pyo.Objective(rule=compute_elec_cost, sense=pyo.minimize)

   def _define_constraints(self):
      """
      Define the model constraints.  """

         # 1. MASS BALANCE

      def tank_balance_rule(model, i, t):
         return model.IV[i,t] == model.IV[i,t-1] + model.PD[i, t] - model.SL[i, t] 
      self.model.tank_balance = pyo.Constraint(self.model.I, self.model.T_bar, rule = tank_balance_rule)

      # Lower limit on Storage (Liquid)
      def tank_storage_lower_limit_rule(model, i ,t):
         return model.IV_l[i] <= model.IV[i, t]
      self.model.tank_storage_lower_limit = pyo.Constraint(self.model.I, self.model.T_bar, rule = tank_storage_lower_limit_rule)

      # Upper limit on Storage (Liquid)
      def tank_storage_upper_limit_rule(model, i, t):
         return model.IV[i, t] <= model.IV_u[i]
      self.model.tank_storage_upper_limit = pyo.Constraint(self.model.I, self.model.T_bar, rule = tank_storage_upper_limit_rule)

      def meeting_daily_liquid_demand_rule(model, i, w):       # write in terms of only model.D
         return sum(model.SL[i, t] for t in model.W_T[w]) == sum(model.D[i, t] for t in model.W_T[w])
      self.model.meeting_daily_liquid_demand = pyo.Constraint(self.model.I, self.model.W, rule = meeting_daily_liquid_demand_rule)

      # ship only after 24 hours of production
      def ship_only_after_24hours_rule(model, i, w):
         return model.SL[i, w*24] == model.D[i, w*24]
      self.model.ship_only_after_24hours = pyo.Constraint(self.model.I, self.model.W, rule = ship_only_after_24hours_rule)

      ###### ###### ###### ###### ###### ######
      # 2. ENERGY BALANCE
      def energy_balance_rule(model, t):
         return model.EC[t] == model.EP[t]
      self.model.energy_balance = pyo.Constraint(self.model.T_bar, rule = energy_balance_rule)

      ###### ###### ###### ###### ###### ######
      # 3. Feasible Region Hull Reformulation

      # Production for prod i in a time period as a sum of production over all possible modes
      def product_production_rule(model, i, t):
         return model.PD[i ,t] == sum(model.PD_bar[m, i, t] for m in model.M)
      self.model.product_production = pyo.Constraint(self.model.I, self.model.T_bar,  rule = product_production_rule)

      # Production as per Convex Hulls of Feasible Regions
      def production_combined_rule(model, m, i, t):
         return model.PD_bar[m , i ,t] ==  sum(model.llambda[m , j, t]*model.v[m, j, i] for j in list(model.J[m]))
      self.model.production_combined = pyo.Constraint(self.model.M, self.model.I, self.model.T_bar, rule = production_combined_rule)

      # Convex constraint on Lambda_mode[j, t] : sum = 1   (Defined for only 4 modes, in which some production is happening)
      def sum_llambda_combined_rule(model, m, t):
         return sum(model.llambda[m, j, t] for j in list(model.J[m])) == model.y[m, t]
      self.model.sum_llambda_combined = pyo.Constraint(self.model.M, self.model.T_bar, rule = sum_llambda_combined_rule)

      # Electricity Consumption in Time period, t
      def electricity_consumption_interval_rule(model, t):
         return model.EC[t] == sum(model.delta[m]*model.y[m, t] + sum(model.gamma[m, i]*model.PD_bar[m, i, t] for i in model.I) for m in model.M)
      self.model.electricity_consumption_interval = pyo.Constraint(self.model.T_bar, rule=electricity_consumption_interval_rule)

      # One mode active at a time, t
      def sum_activemode_one_rule(model, t):
         return sum(model.y[m, t] for m in model.M) == 1 
      self.model.sum_activemode_one = pyo.Constraint(self.model.T_bar, rule = sum_activemode_one_rule)

      ###### ###### ###### ###### ###### ######
      # 3. Boundary conditions

      def initial_storage_rule(model, i):
         return model.IV[i, 0] == model.IV_i[i]
      self.model.initial_storage = pyo.Constraint(self.model.I, rule = initial_storage_rule)

      def final_storage_rule(model, i):
         return model.IV[i, model.tf.value] >= model.IV_f[i] 
      self.model.final_storage = pyo.Constraint(self.model.I, rule = final_storage_rule)

      def switch_history_rule(model, m1, m2, tz):
         if (m1, m2) in model.TR:
               return model.z[m1, m2, tz] == model.z_h[m1, m2, tz]
         else:
               return pyo.Constraint.Skip
      self.model.switch_history = pyo.Constraint(self.model.M, self.model.M, self.model.T_zhistory, rule = switch_history_rule)

      def y_boundary_rule(model, m):
         return model.y[m, 0] == model.y_i[m]
      self.model.y_boundary = pyo.Constraint(self.model.M, rule = y_boundary_rule)

      ###### ###### ###### ###### ###### ######
      # 4. Transition constraints

      ###### 4.1 Switch Variable Constraints ######
      def transitions_combined_rule(model, m, t):
         return sum(model.z[m1, m , t-1] for (m1, m2) in model.TR if m2 == m) - sum(model.z[m, m2, t-1] for (m1, m2) in model.TR if m1 == m) == model.y[m, t] - model.y[m, t-1]
      self.model.transitions_combined = pyo.Constraint(self.model.M, self.model.T_bar, rule = transitions_combined_rule)

      def one_transition_per_timestep_rule(model, t):
         return sum(model.z[m1, m2, t] for m1 in model.M for m2 in model.M if m1 != m2) <= 1
      self.model.one_transition_per_timestep = pyo.Constraint(self.model.T_bar, rule=one_transition_per_timestep_rule)

      # ###### 4.2 Forbidden Transitions ######
      def forbidden_transitions_rule(model, m1, m2, t):
         if (m1, m2) in model.DTR:
               return model.z[m1, m2, t] == 0
         else:
               return pyo.Constraint.Skip
      self.model.forbidden_transitions = pyo.Constraint(self.model.M, self.model.M, self.model.T, rule = forbidden_transitions_rule)    # do  .DTR
   
   def update_state(self, state):
      """
      Update the optimization model parameters using the state from the environment.
      
      Args:
         state (dict): A dictionary with keys:
            'electricity_prices': NumPy array of shape (24*(1+self.lookahead_days),)
            'demand': NumPy array of shape (len(self.products), 24*(1+self.lookahead_days))
            'IV': NumPy array of shape (len(self.products),)
      
      This function updates self.model.D and self.model.alpha_EP with the values from state.
      Assumes self.model.T_bar contains time indices 1, 2, ..., total_hours.
      """
      # Update demand: iterate over each product and time period.
      for i, prod in enumerate(self.model.I):
         for t in self.model.T_bar:  # assuming T_bar = 1,2,...,total_hours
               # Convert t to 0-indexed when accessing the state array.
               demand_value = state['demand'][i, t - 1]
               self.model.D[prod, t].set_value(demand_value)
         
      # Update electricity prices for each time period.
      for t in self.model.T_bar:
         price_value = state['electricity_prices'][t - 1]
         self.model.alpha_EP[t].set_value(price_value)

      for i, prod in enumerate(self.model.I):
         inv_value = state['IV'][i]
         self.model.IV_i[prod].set_value(inv_value)


   def solve(self, solver_name='gurobi', tee=False):
      """
      Solve the optimization model.
      
      :param solver_name: Name of the solver to use (default: 'glpk').
      :param tee: Whether to display solver output.
      :return: The result of the solver.
      """
      solver = pyo.SolverFactory(solver_name)
      result = solver.solve(self.model, tee=tee)

      objective_value = pyo.value(self.model.objective)
   
      return objective_value

   def day1_cost(self):
      day1_cost = pyo.value(sum(self.model.alpha_EP[t] * self.model.EP[t] for t in range(1, 25)))
      return day1_cost
   
   def _update_asu_parameters(self):
      """
      Update the mutable parameters internal to the model.

      """
      # Inventory history
      for i in self.model.I:
         prod_q = self.model.IV[i, 24].value
         self.model.IV_i[i].set_value(prod_q)
      
      # Mode history
      for m in self.model.M:
         prev_mode = self.model.y[m, 24].value
         if prev_mode == 1:
            self.model.y_i[m].set_value(1)
         else:
            self.model.y_i[m].set_value(0)

      # Switch history
      for m1, m2 in self.model.TR:
         for t in self.model.T_zhistory:
            val = self.model.z[m1, m2, 24 + t].value
            self.model.z_h[m1, m2, t].set_value(val)
   
   def extract_optimal_lambda(self):
 
      lambda_values = {}
      m = 'Liquid_Prod'
      for t in range(1,25):  # assuming T_bar covers t=1,...,24
         lambda_values[t] = {}
         for j in self.model.J[m]:
            total_lambda = value(self.model.llambda[m, j, t])
            lambda_values[t][j] = total_lambda
      return lambda_values

   def _get_plots(self):
      """
      Plot the mode switching over time.
      """
      plt.figure(figsize=(10, 6))
      time = list(self.model.T_bar)
      off = [self.model.y['OFF', t]() for t in time]
      liquid_su = [self.model.y['Liquid_SU', t]() for t in time]
      liquid_prod = [self.model.y['Liquid_Prod', t]() for t in time]
      plt.plot(time, off, label='OFF')
      plt.plot(time, liquid_su, label='Liquid_SU')    
      plt.plot(time, liquid_prod, label='Liquid_Prod')
      plt.xlabel('Time Period')
      plt.ylabel('Switch Variable')
      plt.legend()

      # Shipping Plot
      plt.figure(figsize=(10, 6))
      for i in self.model.I:
         demand = [self.model.SL[i, t]() for t in time]
         plt.plot(time, demand, label=f'Shipped Product {i}')
      plt.xlabel('Time Period')
      plt.ylabel('Product Shipping by the  ASU over the week')
      plt.legend()
      plt.show()

      plt.figure(figsize=(10, 6))
      for product, demands in self.demand.items():
          periods = sorted(demands.keys())
          values = [demands[p] for p in periods]
          plt.plot(periods, values, marker='o', label=product)

      plt.xlabel("Time Period (Day)")
      plt.ylabel("Demand")
      plt.title("Demand Over the Week for Each Product")
      plt.legend()
      plt.show()

      # Plot self.model.IV over time
      plt.figure(figsize=(10, 6))
      time = sorted(list(self.model.T_BC))
      for product in self.model.I:
         inv = [self.model.IV[product, t]() for t in time]
         plt.plot(time, inv, marker='o', label=f'Inventory {product}')
      plt.xlabel('Time Period')
      plt.ylabel('Inventory Level')
      plt.title('Inventory Level over Time')
      plt.legend()
      plt.show()

      plt.figure(figsize=(10, 6))
      time = list(self.model.T_bar)
      prices = [self.model.alpha_EP[t]() for t in time]
      plt.plot(time, prices, label='Electricity Prices')
      plt.xlabel('Time Period')
      plt.ylabel('Electricity Price')
      plt.title('Electricity Prices Over Time')
      plt.legend()
      plt.grid()
      plt.show()

