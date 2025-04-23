import pyomo.environ as pyo

def create_energy_storage_model(env):

    model = pyo.ConcreteModel()

    # Define Sets
    model.T = pyo.RangeSet(env.time_periods)                  # Time periods
    model.L = pyo.Set(initialize=env.TransmissionLines)       # Transmission Lines (each element could be a tuple (n_from, n_to))
    model.G = pyo.Set(initialize=env.Generators)              # Generators (all generators)
    model.N = pyo.Set(initialize=env.Buses)                   # Buses
    model.L_off = pyo.Set(initialize=env.DeEnergizedLines)     # Set of de-energized lines
    
    # For linking generators to buses we distinguish:
    # (a) gen_bus: parameter for the bus at which a given generator is located
    # (b) G_at_bus: set of generators at a given bus (for use in the power balance)
    model.gen_bus = pyo.Param(model.G, initialize=env.BusGeneratorLink)
    model.G_at_bus = pyo.Set(model.N, initialize=env.GeneratorsAtBus)
    
    # For power demand at buses (rename to avoid conflict with power flow variable)
    model.demand = pyo.Param(model.N, model.T, initialize=env.PowerDemandAtBus)
    
    # For each bus: sets of lines with this bus as the “to” and “from” bus
    model.L_to = pyo.Set(model.N, initialize=env.LinesToN)
    model.L_from = pyo.Set(model.N, initialize=env.LinesFromN)

    # For Each Transmission Line
    model.b = pyo.Param(model.L, initialize=env.LineSusceptance)   # Susceptance
    model.p_limit = pyo.Param(model.L, initialize=env.LinePowerFlowLimit)  # Power flow limit
    model.r = pyo.Param(model.L, model.T, initialize=env.WildfireRisk)   # Wildfire Risk (unitless)
    model.delta_up = pyo.Param(model.L, initialize=env.LineUpperVoltageAngle)  # Voltage angle upper bound
    model.delta_low = pyo.Param(model.L, initialize=env.LineLowerVoltageAngle) # Voltage angle lower bound

    # For Each Generator: bounds are defined above; cost will be in objective.
    model.g_up = pyo.Param(model.G, initialize=env.GeneratorUpperLimit)
    model.g_low = pyo.Param(model.G, initialize=env.GeneratorLowerLimit)
    
    # Battery parameters for each bus
    model.e_up = pyo.Param(model.N, initialize=env.BatteryUpLimit)          # Maximum energy capacity per battery (p.u.)
    model.e_low = pyo.Param(model.N, initialize=env.BatteryLowLimit)        # Minimum energy capacity per battery (p.u.)
    model.E_o = pyo.Param(model.N, initialize=env.BatteryInitialCharge)     # Initial charge (p.u.)
    model.p_c_low = pyo.Param(model.N, initialize=env.LowerBatteryChargeRate)  # Lower bound on charging rate per battery (p.u.)
    model.p_c_up = pyo.Param(model.N, initialize=env.UpperBatteryChargeRate)  # Upper bound on charging rate per battery (p.u.)
    # Battery discharge rates (must be provided in data)
    model.p_d_low = pyo.Param(model.N, initialize=env.LowerBatteryDischargeRate)
    model.p_d_up = pyo.Param(model.N, initialize=env.UpperBatteryDischargeRate)
    
    model.eff = pyo.Param(initialize=env.ChargeEfficiency)      # Charging/discharging efficiency (e)
    model.h = pyo.Param(initialize=env.CarryOverRate)             # Hourly carryover rate (to model self-discharge)
    model.x_max = pyo.Param(initialize=env.MaxBatteriesAllowed)     # Maximum batteries allowed at a bus
    model.x_total = pyo.Param(initialize=env.TotalBatteries)          # Maximum total batteries in the network

    # Generator cost (polynomial) parameters: assume degree is provided
    model.J = pyo.RangeSet(0, env.PolynomialDegree - 1)  
    # Parameter c[g,j] is the coefficient for generator g for term j
    model.c = pyo.Param(model.G, model.J, initialize=env.GeneratorCost)
    
    # Cost penalties for load shedding and slack generation
    model.Kls = pyo.Param(initialize=env.Kls)
    model.Kslack = pyo.Param(initialize=env.Kslack)
    
    # Define Variables

    # Generator power output for each generator and time period, bounded by generator limits.
    def power_bounds(model, g, t):
        return (model.g_low[g], model.g_up[g])
    model.power = pyo.Var(model.G, model.T, bounds=power_bounds)
    
    # Slack generation at bus (to ensure feasibility), with a high upper bound.
    model.gslack = pyo.Var(model.N, model.T, bounds=(0, 10000))
    
    # Voltage angle at each bus (in radians)
    model.theta = pyo.Var(model.N, model.T, domain=pyo.Reals)
    
    # Load shedding at bus; bounded between 0 and the demand.
    def load_shed_bounds(model, n, t):
        return (0, model.demand[n, t])
    model.p_ls = pyo.Var(model.N, model.T, bounds=load_shed_bounds)
    
    # Power flow on transmission lines.
    def power_flow_bounds(model, n_fr, n_to, t):
        # If line l is de-energized, enforce zero flow.
        l = (n_fr, n_to)
        if l in model.L_off:
            return (0, 0)
        else:
            return (-model.p_limit[l], model.p_limit[l])
    model.p_l = pyo.Var(model.L, model.T, bounds=power_flow_bounds)
    
    # Battery-related variables:
    # Number of batteries placed at each bus (investment decision; assumed continuous here; round as needed)
    model.x = pyo.Var(model.N, bounds=(0, model.x_max))
    # Charging rate at bus n and time t
    model.p_c = pyo.Var(model.N, model.T, domain=pyo.Reals)
    # Discharging rate at bus n and time t
    model.p_d = pyo.Var(model.N, model.T, domain=pyo.Reals)
    # State-of-charge (SOC) of batteries at bus n at time t; note index over T, not "t"
    model.E = pyo.Var(model.N, model.T, domain=pyo.Reals)

    # Define Constraints

    # --- Voltage Angle Bounds ---
    def voltage_angle_low_cons(model, n_fr, n_to, t):
        l = (n_fr, n_to)
        if l not in model.L_off:
            return model.theta[n_to, t] - model.theta[n_fr, t] >= model.delta_low[l]
        else:
            return pyo.Constraint.Skip
    model.voltage_angle_cons_low = pyo.Constraint(model.L, model.T, rule=voltage_angle_low_cons)

    def voltage_angle_up_cons(model, n_fr, n_to, t):
        l = (n_fr, n_to)
        if l not in model.L_off:
            return model.theta[n_to, t] - model.theta[n_fr, t] <= model.delta_up[l]
        else:
            return pyo.Constraint.Skip
    model.voltage_angle_cons_up = pyo.Constraint(model.L, model.T, rule=voltage_angle_up_cons)

    # --- Power Flow Constraints (using B-θ DC approximation) ---
    def power_flow_low_cons(model, n_fr, n_to, t):
        l = (n_fr, n_to)
        if l not in model.L_off:
            return model.p_l[l, t] <= -model.b[l] * (model.theta[n_fr, t] - model.theta[n_to, t])
        else:
            return pyo.Constraint.Skip
    model.power_flow_cons_low = pyo.Constraint(model.L, model.T, rule=power_flow_low_cons)

    def power_flow_up_cons(model, n_fr, n_to, t):
        l = (n_fr, n_to)
        if l not in model.L_off:
            return model.p_l[l, t] >= -model.b[l] * (model.theta[n_fr, t] - model.theta[n_to, t])
        else:
            return pyo.Constraint.Skip
    model.power_flow_cons_up = pyo.Constraint(model.L, model.T, rule=power_flow_up_cons)

    # --- Total Battery Placement Constraint ---
    # Ensure that the sum of batteries installed across all buses does not exceed x_total.
    def total_batteries_cons(model):
        return sum(model.x[n] for n in model.N) <= model.x_total
    model.Total_Batteries_Cons = pyo.Constraint(rule=total_batteries_cons)

    # --- State-of-Charge (SOC) Balance Constraint ---
    def SOC_Balance_Cons(model, n, t):
        if t == model.T.first():
            return model.E[n, t] == model.h * model.E_o[n] + model.eff * model.p_c[n, t] - (1 / model.eff) * model.p_d[n, t]
        else:
            return model.E[n, t] == model.h * model.E[n, t-1] + model.eff * model.p_c[n, t] - (1 / model.eff) * model.p_d[n, t]
    model.SOC_Balance_Cons = pyo.Constraint(model.N, model.T, rule=SOC_Balance_Cons)

    # --- SOC Bounds Constraint ---
    def SOC_Bounds_low_Cons(model, n, t):
        return model.E[n, t] >= model.x[n] * model.e_low[n]
    model.SOC_Bounds_low_Cons = pyo.Constraint(model.N, model.T, rule=SOC_Bounds_low_Cons)

    def SOC_Bounds_up_Cons(model, n, t):
        return model.E[n, t] <= model.x[n] * model.e_up[n]
    model.SOC_Bounds_up_Cons = pyo.Constraint(model.N, model.T, rule=SOC_Bounds_up_Cons)

    # --- Battery Charge Rate Constraints ---
    def Charge_low_Cons(model, n, t):
        return model.p_c[n, t] >= model.p_c_low[n] * model.x[n]
    model.Charge_low_Cons = pyo.Constraint(model.N, model.T, rule=Charge_low_Cons)

    def Charge_up_Cons(model, n, t):
        return model.p_c[n, t] <= model.p_c_up[n] * model.x[n]
    model.Charge_up_Cons = pyo.Constraint(model.N, model.T, rule=Charge_up_Cons)

    # --- Battery Discharge Rate Constraints ---
    def Discharge_low_Cons(model, n, t):
        return model.p_d[n, t] >= model.p_d_low[n] * model.x[n]
    model.Discharge_low_Cons = pyo.Constraint(model.N, model.T, rule=Discharge_low_Cons)

    def Discharge_up_Cons(model, n, t):
        return model.p_d[n, t] <= model.p_d_up[n] * model.x[n]
    model.Discharge_up_Cons = pyo.Constraint(model.N, model.T, rule=Discharge_up_Cons)

    # --- Power Balance Constraint ---
    # For each bus and time period, the net injection must match the difference between flows leaving and entering.
    # Incorporates generator outputs, slack, demand, load shedding, battery charging and discharging.
    def Power_Balance_Cons(model, n, t):
        generation = sum(model.power[g, t] for g in model.G_at_bus[n])
        flows = sum(model.p_l[l, t] for l in model.L_to[n]) - sum(model.p_l[l, t] for l in model.L_from[n])
        return generation + model.gslack[n, t] - model.demand[n, t] + model.p_ls[n, t] - model.p_c[n, t] + model.p_d[n, t] == flows
    model.Power_Balance_Cons = pyo.Constraint(model.N, model.T, rule=Power_Balance_Cons)

    # --- Objective Function ---
    # The total operating cost is the sum over time of:
    # (a) Generation cost (modeled as a polynomial function for each generator),
    # (b) Load-shedding penalty (Kls per unit of unmet demand),
    # (c) Slack generation penalty (Kslack per unit).
    def objective_rule(model):
        gen_cost = sum(model.c[g, j] * model.power[g, t]**j for t in model.T for g in model.G for j in model.J)
        load_shed_cost = sum(model.Kls * model.p_ls[n, t] for t in model.T for n in model.N)
        slack_cost = sum(model.Kslack * model.gslack[n, t] for t in model.T for n in model.N)
        return gen_cost + load_shed_cost + slack_cost
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return model
