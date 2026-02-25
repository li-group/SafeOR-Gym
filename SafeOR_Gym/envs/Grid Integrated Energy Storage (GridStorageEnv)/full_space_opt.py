import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition


def create_energy_storage_model(env):
    """
    Create the Pyomo optimization model for the energy storage / DC power flow problem.
    
    Parameters
    ----------
    env : BatteryOperationEnv
        The gymnasium environment instance containing network structure and parameters.
    
    Returns
    -------
    model : pyo.ConcreteModel
        The Pyomo model ready for solving.
    """
    model = pyo.ConcreteModel()

    # =========================================================================
    # Sets
    # =========================================================================
    model.T = pyo.RangeSet(1, env.time_periods)  # Time periods (1-indexed)
    model.L = pyo.Set(initialize=env.TransmissionLines)  # Transmission Lines (tuples)
    model.G = pyo.Set(initialize=env.Generators)  # Generators
    model.N = pyo.Set(initialize=env.Buses)  # Buses
    model.L_off = pyo.Set(initialize=env.DeEnergizedLines)  # De-energized lines

    # Generator-bus mapping
    model.gen_bus = pyo.Param(model.G, initialize=env.BusGeneratorLink)
    model.G_at_bus = pyo.Set(model.N, initialize=env.GeneratorsAtBus)

    # Lines connected to each bus
    model.L_to = pyo.Set(model.N, initialize=env.LinesToN)
    model.L_from = pyo.Set(model.N, initialize=env.LinesFromN)

    # =========================================================================
    # Parameters
    # =========================================================================
    # Demand at each bus and time
    model.demand = pyo.Param(model.N, model.T, initialize=env.PowerDemandAtBus)

    # Line parameters
    model.b = pyo.Param(model.L, initialize=env.LineSusceptance)
    model.p_limit = pyo.Param(model.L, initialize=env.LinePowerFlowLimit)
    model.delta_up = pyo.Param(model.L, initialize=env.LineUpperVoltageAngle)
    model.delta_low = pyo.Param(model.L, initialize=env.LineLowerVoltageAngle)

    # Generator limits
    model.g_up = pyo.Param(model.G, initialize=env.GeneratorUpperLimit)
    model.g_low = pyo.Param(model.G, initialize=env.GeneratorLowerLimit)

    # Battery parameters
    model.e_up = pyo.Param(model.N, initialize=env.BatteryUpLimit)
    model.e_low = pyo.Param(model.N, initialize=env.BatteryLowLimit)
    model.E_o = pyo.Param(model.N, initialize=env.BatteryInitialCharge)
    model.p_c_low = pyo.Param(model.N, initialize=env.LowerBatteryChargeRate)
    model.p_c_up = pyo.Param(model.N, initialize=env.UpperBatteryChargeRate)
    model.p_d_low = pyo.Param(model.N, initialize=env.LowerBatteryDischargeRate)
    model.p_d_up = pyo.Param(model.N, initialize=env.UpperBatteryDischargeRate)

    # Efficiency and carryover
    model.eff = pyo.Param(initialize=env.ChargeEfficiency)
    model.h = pyo.Param(initialize=env.CarryOverRate)

    # Generator cost coefficients
    model.J = pyo.RangeSet(0, env.PolynomialDegree - 1)
    model.c = pyo.Param(model.G, model.J, initialize=env.GeneratorCost)

    # Penalty costs
    model.Kls = pyo.Param(initialize=env.Kls)
    model.Kslack = pyo.Param(initialize=env.Kslack)

    # =========================================================================
    # Variables
    # =========================================================================
    # Generator power output
    def power_bounds(model, g, t):
        return (model.g_low[g], model.g_up[g])

    model.power = pyo.Var(model.G, model.T, bounds=power_bounds)

    # Slack generation at each bus
    model.gslack = pyo.Var(model.N, model.T, bounds=(0, env.smax))

    # Voltage angles (radians)
    model.theta = pyo.Var(model.N, model.T, domain=pyo.Reals)

    # Load shedding
    def load_shed_bounds(model, n, t):
        return (0, model.demand[n, t])

    model.p_ls = pyo.Var(model.N, model.T, bounds=load_shed_bounds)

    # Power flow on transmission lines
    def power_flow_bounds(model, n_fr, n_to, t):
        l = (n_fr, n_to)
        if l in model.L_off:
            return (0, 0)
        else:
            return (-model.p_limit[l], model.p_limit[l])

    model.p_l = pyo.Var(model.L, model.T, bounds=power_flow_bounds)

    # Battery charging rate
    def charge_bounds(m, n, t):
        return (m.p_c_low[n], m.p_c_up[n])

    model.p_c = pyo.Var(model.N, model.T, bounds=charge_bounds)

    # Battery discharging rate
    def discharge_bounds(m, n, t):
        return (m.p_d_low[n], m.p_d_up[n])

    model.p_d = pyo.Var(model.N, model.T, bounds=discharge_bounds)

    # State-of-charge (SOC)
    model.E = pyo.Var(model.N, model.T, domain=pyo.Reals)

    # =========================================================================
    # Constraints
    # =========================================================================
    # Reference bus angle = 0
    ref_bus = env.Buses[0]

    def reference_bus_rule(m, t):
        return m.theta[ref_bus, t] == 0

    model.ReferenceBusAngle = pyo.Constraint(model.T, rule=reference_bus_rule)

    # Voltage angle bounds
    def voltage_angle_low_cons(model, n_fr, n_to, t):
        l = (n_fr, n_to)
        if l not in model.L_off:
            return (
                model.theta[n_to, t] - model.theta[n_fr, t] >= model.delta_low[l]
            )
        else:
            return pyo.Constraint.Skip

    model.voltage_angle_cons_low = pyo.Constraint(
        model.L, model.T, rule=voltage_angle_low_cons
    )

    def voltage_angle_up_cons(model, n_fr, n_to, t):
        l = (n_fr, n_to)
        if l not in model.L_off:
            return model.theta[n_to, t] - model.theta[n_fr, t] <= model.delta_up[l]
        else:
            return pyo.Constraint.Skip

    model.voltage_angle_cons_up = pyo.Constraint(
        model.L, model.T, rule=voltage_angle_up_cons
    )

    # DC Power flow: f = B * (θ_from - θ_to)
    def power_flow_low_cons(model, n_fr, n_to, t):
        l = (n_fr, n_to)
        if l not in model.L_off:
            return (
                model.p_l[l, t]
                <= model.b[l] * (model.theta[n_fr, t] - model.theta[n_to, t])
            )
        else:
            return pyo.Constraint.Skip

    model.power_flow_cons_low = pyo.Constraint(
        model.L, model.T, rule=power_flow_low_cons
    )

    def power_flow_up_cons(model, n_fr, n_to, t):
        l = (n_fr, n_to)
        if l not in model.L_off:
            return (
                model.p_l[l, t]
                >= model.b[l] * (model.theta[n_fr, t] - model.theta[n_to, t])
            )
        else:
            return pyo.Constraint.Skip

    model.power_flow_cons_up = pyo.Constraint(
        model.L, model.T, rule=power_flow_up_cons
    )

    # SOC balance: E[t] = h*E[t-1] + eff*p_c[t] - (1/eff)*p_d[t]
    def SOC_Balance_Cons(model, n, t):
        prev = model.E[n, t - 1] if t > model.T.first() else model.E_o[n]
        return (
            model.E[n, t]
            == model.h * prev
            + model.eff * model.p_c[n, t]
            - (1 / model.eff) * model.p_d[n, t]
        )

    model.SOC_Balance_Cons = pyo.Constraint(model.N, model.T, rule=SOC_Balance_Cons)

    # SOC bounds
    def SOC_low(model, n, t):
        return model.E[n, t] >= model.e_low[n]

    def SOC_high(model, n, t):
        return model.E[n, t] <= model.e_up[n]

    model.SOC_low = pyo.Constraint(model.N, model.T, rule=SOC_low)
    model.SOC_high = pyo.Constraint(model.N, model.T, rule=SOC_high)

    # Power balance at each bus
    # Generation + slack - demand + load_shed - charge + discharge = net_flow
    def Power_Balance_Cons(model, n, t):
        generation = sum(model.power[g, t] for g in model.G_at_bus[n])
        flows = sum(model.p_l[l, t] for l in model.L_to[n]) - sum(
            model.p_l[l, t] for l in model.L_from[n]
        )
        return (
            generation
            + model.gslack[n, t]
            - model.demand[n, t]
            + model.p_ls[n, t]
            - model.p_c[n, t]
            + model.p_d[n, t]
            == flows
        )

    model.Power_Balance_Cons = pyo.Constraint(model.N, model.T, rule=Power_Balance_Cons)

    # =========================================================================
    # Objective: Minimize total operating cost
    # =========================================================================
    def obj_rule(m):
        # Generation cost (polynomial)
        gen_cost = sum(
            m.c[g, j] * m.power[g, t] ** j
            for t in m.T
            for g in m.G
            for j in m.J
        )
        # Load shedding and slack penalties
        penalty_cost = sum(
            m.Kls * m.p_ls[n, t] + m.Kslack * m.gslack[n, t]
            for t in m.T
            for n in m.N
        )
        return gen_cost + penalty_cost

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model


def optimal_simulation(
    env,
    solver="gurobi",
    tee: bool = False,
    raise_on_infeasible: bool = True,
):
    """
    Solve the deterministic energy storage optimization and return the optimal
    action sequence for the full horizon.
        
    Example
    -------
    >>> env = BatteryOperationEnv(config_file='config.json')
    >>> env.reset(seed=42)
    >>> actions = optimal_simulation(env, solver='gurobi')
    >>> for t in range(env.T):
    ...     obs, reward, done, truncated, info = env.step(actions[t])
    """
    T = env.T
    G = env.G
    N = env.N

    # Action dimension: G + 3*N + (N-1)
    # [pg (G), cn (N), pdn (N), ln (N), theta_vars (N-1)]
    action_dim = G + 3 * N + (N - 1)

    # Create and solve the model
    model = create_energy_storage_model(env)

    opt = solver if hasattr(solver, "solve") else SolverFactory(str(solver))
    results = opt.solve(model, tee=tee)

    # Check termination condition
    term = results.solver.termination_condition
    ok = term in (TerminationCondition.optimal, TerminationCondition.locallyOptimal)
    if (not ok) and raise_on_infeasible:
        raise RuntimeError(
            f"Optimization did not solve to optimality. Termination condition: {term}"
        )

    # Extract optimal values
    # Arrays to hold physical action values
    opt_pg = np.zeros((T, G), dtype=np.float32)
    opt_cn = np.zeros((T, N), dtype=np.float32)
    opt_pdn = np.zeros((T, N), dtype=np.float32)
    opt_ln = np.zeros((T, N), dtype=np.float32)
    opt_theta = np.zeros((T, N), dtype=np.float32)

    generators = list(env.Generators)
    buses = list(env.Buses)

    for t in range(1, T + 1):
        t_idx = t - 1  # Convert to 0-indexed

        # Generator power
        for g_idx, g in enumerate(generators):
            val = pyo.value(model.power[g, t])
            opt_pg[t_idx, g_idx] = 0.0 if val is None else float(val)

        # Battery charge, discharge, load shedding, angles
        for n_idx, n in enumerate(buses):
            val_c = pyo.value(model.p_c[n, t])
            val_d = pyo.value(model.p_d[n, t])
            val_ls = pyo.value(model.p_ls[n, t])
            val_theta = pyo.value(model.theta[n, t])

            opt_cn[t_idx, n_idx] = 0.0 if val_c is None else float(val_c)
            opt_pdn[t_idx, n_idx] = 0.0 if val_d is None else float(val_d)
            opt_ln[t_idx, n_idx] = 0.0 if val_ls is None else float(val_ls)
            opt_theta[t_idx, n_idx] = 0.0 if val_theta is None else float(val_theta)

    # Convert physical actions to normalized [-1, 1] actions
    # Environment scaling: a_phys = 0.5 * (a_norm + 1) * (amax - amin) + amin
    # Inverse: a_norm = 2 * (a_phys - amin) / (amax - amin) - 1

    amin = env.amin
    amax = env.amax

    raw_actions = np.zeros((T, action_dim), dtype=np.float32)

    for t_idx in range(T):
        # Concatenate physical actions in the same order as env expects
        # [pg (G), cn (N), pdn (N), ln (N), theta_vars (N-1)]
        theta_vars = opt_theta[t_idx, 1:]  # Exclude reference bus (index 0)

        a_phys = np.concatenate(
            [
                opt_pg[t_idx],
                opt_cn[t_idx],
                opt_pdn[t_idx],
                opt_ln[t_idx],
                theta_vars,
            ]
        )

        # Scale to [-1, 1]
        denom = amax - amin
        # Avoid division by zero
        safe_denom = np.where(np.abs(denom) > 1e-12, denom, 1.0)
        a_norm = 2.0 * (a_phys - amin) / safe_denom - 1.0

        # Handle zero-range dimensions
        a_norm[np.abs(denom) <= 1e-12] = 0.0

        raw_actions[t_idx] = a_norm

    # Clip to ensure within bounds (numerical safety)
    raw_actions = np.clip(raw_actions, -1.0, 1.0)

    return raw_actions


def get_optimal_value(env, solver="gurobi", tee: bool = False):
    """
    Solve the optimization and return only the optimal objective value.
    
    """
    model = create_energy_storage_model(env)

    opt = solver if hasattr(solver, "solve") else SolverFactory(str(solver))
    results = opt.solve(model, tee=tee)

    term = results.solver.termination_condition
    if term not in (TerminationCondition.optimal, TerminationCondition.locallyOptimal):
        return float("inf")

    return pyo.value(model.obj)


def simulate_with_optimal_policy(env, solver="gurobi", seed=None, tee=False):
    """
    Reset the environment, compute optimal actions, and simulate the full episode.
    
    """
    # Reset environment
    obs, info = env.reset(seed=seed)

    # Get optimal actions
    actions = optimal_simulation(env, solver=solver, tee=tee)

    # Simulate
    observations = [obs.cpu().numpy() if hasattr(obs, "cpu") else obs]
    rewards = []
    total_reward = 0.0
    total_cost = 0.0

    for t in range(env.T):
        obs, reward, done, truncated, info = env.step(actions[t])

        reward_val = reward.item() if hasattr(reward, "item") else float(reward)
        rewards.append(reward_val)
        total_reward += reward_val
        total_cost += env.cost

        obs_np = obs.cpu().numpy() if hasattr(obs, "cpu") else obs
        observations.append(obs_np)

        if done:
            break

    return {
        "total_reward": total_reward,
        "total_cost": total_cost,
        "actions": actions,
        "rewards": rewards,
        "observations": observations,
    }