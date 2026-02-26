import numpy as np
import pyomo.environ as pe
from pyomo.opt import SolverFactory, TerminationCondition


def create_model(env):
    num_gen = env.num_gen
    T = env.T
    num_bus = env.num_bus
    gen_bus = env.gen_bus
    bus_gen = env.bus_gen
    num_line = env.num_line
    line_bus = env.line_bus
    B = env.B
    F_max = env.F_max
    F_min = env.F_min
    Pi_max = env.Pi_max
    Pi_min = env.Pi_min
    pi0 = env.pi0
    deterministic_demand = env.deterministic_demand

    horizon = range(1, T + 1)
    generators = range(num_gen)
    buses = range(num_bus)
    lines = range(num_line)
    line_from_bus = {}
    line_to_bus = {}
    for line in lines:
        line_from_bus[line] = line_bus[line][0]
        line_to_bus[line] = line_bus[line][1]

    from_bus_lines = {i: [] for i in buses}
    to_bus_lines = {i: [] for i in buses}
    for line, (from_bus, to_bus) in line_bus.items():
        from_bus_lines[from_bus].append(line)
        to_bus_lines[to_bus].append(line)

    P_max = env.P_max
    P_min = env.P_min
    a = env.a
    b = env.b
    c = env.c
    UT = env.UT
    DT = env.DT
    RU = env.RU
    RD = env.RD
    SU = env.SU
    SD = env.SD
    hot_cost = env.hot_cost
    cold_cost = env.cold_cost
    cold_hrs = env.cold_hrs
    C_SD = env.C_SD
    C_LS = env.C_LS
    C_RP = env.C_RP
    R = env.R

    u0_seq = env.u0_seq
    v0_seq = {}
    w0_seq = {}
    for i in generators:
        u_diff_seq = u0_seq[i][:-1] - u0_seq[i][1:]
        v0_seq.update({i: np.maximum(0, u_diff_seq[:UT[i]])})
        w0_seq.update({i: - np.minimum(0, u_diff_seq[:DT[i]])})
    v_prev_set = [(-t, i) for i in generators for t in range(UT[i])]
    w_prev_set = [(-t, i) for i in generators for t in range(DT[i])]

    v_prev = {}
    w_prev = {}
    for i in generators:
        for t in range(UT[i]):
            v_prev.update({(-t, i): v0_seq[i][t]})
        for t in range(DT[i]):
            w_prev.update({(-t, i): w0_seq[i][t]})

    u0 = env.u0
    p0 = env.p0

    model = pe.ConcreteModel()
    model.T_set = pe.Set(initialize=horizon)
    model.generators = pe.Set(initialize=generators)
    model.buses = pe.Set(initialize=buses)
    model.lines = pe.Set(initialize=lines)
    model.bus_gen = pe.Set(model.buses, initialize=bus_gen)
    model.from_bus_lines = pe.Set(model.buses, initialize=from_bus_lines)
    model.to_bus_lines = pe.Set(model.buses, initialize=to_bus_lines)
    model.v_prev_set = pe.Set(initialize=v_prev_set)
    model.w_prev_set = pe.Set(initialize=w_prev_set)


    model.demand = pe.Param(model.T_set, model.buses,
                            initialize={(t, n): deterministic_demand[t-1, n] for t in horizon for n in buses}, mutable=True)
    model.s_pos = pe.Var(model.T_set, model.buses, domain=pe.NonNegativeReals)
    model.s_neg = pe.Var(model.T_set, model.buses, domain=pe.NonNegativeReals)
    model.u = pe.Var(model.T_set, model.generators, domain=pe.Binary)
    model.v = pe.Var(model.T_set, model.generators, domain=pe.Binary)
    model.w = pe.Var(model.T_set, model.generators, domain=pe.Binary)
    model.u_prev = pe.Param([0], model.generators, initialize={(0, i): u0[i] for i in generators})
    model.v_prev = pe.Param(model.v_prev_set, initialize=v_prev)
    model.w_prev = pe.Param(model.w_prev_set, initialize=w_prev)
    model.p = pe.Var(model.T_set, model.generators, domain=pe.NonNegativeReals)
    model.p_bar = pe.Var(model.T_set, model.generators, domain=pe.NonNegativeReals)
    model.p_prev = pe.Param([0], model.generators, initialize={(0, i): p0[i] for i in generators})
    model.pi_prev = pe.Param([0], model.buses, initialize={(0, i): pi0[i] for i in buses})
    model.r = pe.Var(model.T_set, model.generators, domain=pe.NonNegativeReals)
    model.sr = pe.Var(model.T_set, domain=pe.NonNegativeReals)
    model.pi = pe.Var(model.T_set, model.buses, domain=pe.Reals)
    model.f = pe.Var(model.T_set, model.lines, domain=pe.Reals)

    model.production_cost = pe.Var(model.T_set)
    model.startup_cost = pe.Var(model.T_set)
    model.shutdown_cost = pe.Var(model.T_set)
    model.load_shedding_cost = pe.Var(model.T_set)
    model.reserve_penalty_cost = pe.Var(model.T_set)
    model.total_cost = pe.Var(model.T_set)

    model.P_max = pe.Param(model.generators, initialize={i: P_max[i] for i in generators})
    model.P_min = pe.Param(model.generators, initialize={i: P_min[i] for i in generators})
    model.a = pe.Param(model.generators, initialize={i: a[i] for i in generators})
    model.b = pe.Param(model.generators, initialize={i: b[i] for i in generators})
    model.c = pe.Param(model.generators, initialize={i: c[i] for i in generators})
    model.UT = pe.Param(model.generators, initialize={i: UT[i] for i in generators})
    model.DT = pe.Param(model.generators, initialize={i: DT[i] for i in generators})
    model.RU = pe.Param(model.generators, initialize={i: RU[i] for i in generators})
    model.RD = pe.Param(model.generators, initialize={i: RD[i] for i in generators})
    model.SU = pe.Param(model.generators, initialize={i: SU[i] for i in generators})
    model.SD = pe.Param(model.generators, initialize={i: SD[i] for i in generators})
    model.hot_cost = pe.Param(model.generators, initialize={i: hot_cost[i] for i in generators})
    model.cold_cost = pe.Param(model.generators, initialize={i: cold_cost[i] for i in generators})
    model.cold_hrs = pe.Param(model.generators, initialize={i: cold_hrs[i] for i in generators})
    model.C_SD = pe.Param(model.generators, initialize={i: C_SD[i] for i in generators})
    model.C_LS = pe.Param(initialize=C_LS)
    model.C_RP = pe.Param(initialize=C_RP)
    model.R = pe.Param(initialize=R)
    model.B = pe.Param(model.lines, initialize={l: B[l] for l in lines})
    model.F_max = pe.Param(model.lines, initialize={l: F_max[l] for l in lines})
    model.F_min = pe.Param(model.lines, initialize={l: F_min[l] for l in lines})
    model.Pi_max = pe.Param(model.buses, initialize={n: Pi_max[n] for n in buses})
    model.Pi_min = pe.Param(model.buses, initialize={n: Pi_min[n] for n in buses})


    # Minimum Up and Down Time Constraints
    def uvw_rule(m, t, i):
        if t > 1:
            return m.u[t, i] - m.u[t - 1, i] == m.v[t, i] - m.w[t, i]
        else:
            return m.u[t, i] - m.u_prev[t - 1, i] == m.v[t, i] - m.w[t, i]

    def min_up_rule(m, t, i):
        if t >= m.UT[i]:
            return sum(m.v[tau, i] for tau in range(t - m.UT[i] + 1, t)) <= m.u[t, i]
        else:
            return sum(m.v_prev[tau, i] for tau in range(t - m.UT[i] + 1, 1)) + sum(
                m.v[tau, i] for tau in range(1, t)) <= m.u[t, i]

    def min_down_rule(m, t, i):
        if t >= m.DT[i]:
            return sum(m.w[tau, i] for tau in range(t - m.DT[i] + 1, t)) <= 1 - m.u[t, i]
        else:
            return sum(m.w_prev[tau, i] for tau in range(t - m.DT[i] + 1, 1)) + sum(
                m.w[tau, i] for tau in range(1, t)) <= 1 - m.u[t, i]

    model.uvw = pe.Constraint(model.T_set, model.generators, rule=uvw_rule)
    model.min_up = pe.Constraint(model.T_set, model.generators, rule=min_up_rule)
    model.min_down = pe.Constraint(model.T_set, model.generators, rule=min_down_rule)

    # Reserve Requirement

    def reserve_rule(m, t):
        return sum(m.r[t, i] for i in m.generators) + m.sr[t] >= m.R

    def rp_rule(m, t, i):
        return m.r[t, i] == m.p_bar[t, i] - m.p[t, i]

    model.reserve = pe.Constraint(model.T_set, rule=reserve_rule)
    model.rp = pe.Constraint(model.T_set, model.generators, rule=rp_rule)

    # Generation Bounds
    def p_lb_rule(m, t, i):
        return m.P_min[i] * m.u[t, i] <= m.p[t, i]

    def p_ub_rule(m, t, i):
        return m.p[t, i] <= m.p_bar[t, i]

    def p_bar_ub_rule(m, t, i):
        return m.p_bar[t, i] <= m.P_max[i] * m.u[t, i]

    model.p_lb = pe.Constraint(model.T_set, model.generators, rule=p_lb_rule)
    model.p_ub = pe.Constraint(model.T_set, model.generators, rule=p_ub_rule)
    model.p_bar_ub = pe.Constraint(model.T_set, model.generators, rule=p_bar_ub_rule)

    # Ramping Constraints
    def ramp_up_rule(m, t, i):
        if t > 1:
            return m.p_bar[t, i] - m.p[t - 1, i] <= m.RU[i] * m.u[t - 1, i] + m.SU[i] * m.v[t, i]
        else:
            return m.p_bar[t, i] - m.p_prev[t - 1, i] <= m.RU[i] * m.u_prev[t - 1, i] + m.SU[i] * m.v[t, i]

    def ramp_down_rule(m, t, i):
        if t > 1:
            return m.p[t - 1, i] - m.p[t, i] <= m.RD[i] * m.u[t, i] + m.SD[i] * m.w[t, i]
        else:
            return m.p_prev[t - 1, i] - m.p[t, i] <= m.RD[i] * m.u[t, i] + m.SD[i] * m.w[t, i]

    model.ramp_up = pe.Constraint(model.T_set, model.generators, rule=ramp_up_rule)
    model.ramp_down = pe.Constraint(model.T_set, model.generators, rule=ramp_down_rule)

    # Cost Function and Objective
    def production_cost_rule(m, t):
        return m.production_cost[t] == sum(m.a[i] * (m.p[t, i] ** 2) + m.b[i] * m.p[t, i] + m.c[i] for i in m.generators)

    def startup_cost_rule(m, t):
        return m.startup_cost[t] == sum(m.v[t, i] * m.hot_cost[i] for i in m.generators)

    def shutdown_cost_rule(m, t):
        return m.shutdown_cost[t] == sum(m.w[t, i] * m.C_SD[i] for i in m.generators)

    def load_shedding_cost_rule(m, t):
        return m.load_shedding_cost[t] == sum(m.C_LS * (m.s_pos[t, n]) for n in m.buses)

    def reserve_penalty_cost_rule(m, t):
        return m.reserve_penalty_cost[t] == m.C_RP * m.sr[t]

    def total_cost_rule(m, t):
        return (m.total_cost[t] == m.production_cost[t] + m.startup_cost[t] + m.shutdown_cost[t] +
                m.load_shedding_cost[t] + m.reserve_penalty_cost[t])

    model.pc = pe.Constraint(model.T_set, rule=production_cost_rule)
    model.suc = pe.Constraint(model.T_set, rule=startup_cost_rule)
    model.sdc = pe.Constraint(model.T_set, rule=shutdown_cost_rule)
    model.lsc = pe.Constraint(model.T_set, rule=load_shedding_cost_rule)
    model.rpc = pe.Constraint(model.T_set, rule=reserve_penalty_cost_rule)
    model.tc = pe.Constraint(model.T_set, rule=total_cost_rule)

    # Network Constraints
    def balance_rule(m, t, n):
        return (sum(m.p[t, i] for i in m.bus_gen[n])
                + sum(m.f[t, k] for k in m.to_bus_lines[n])
                - sum(m.f[t, k] for k in m.from_bus_lines[n])
                + m.s_pos[t, n] - m.s_neg[t, n] == m.demand[t, n])

    def power_flow_rule(m, t, l):
        return m.f[t, l] == m.B[l] * (m.pi[t, line_from_bus[l]] - m.pi[t, line_to_bus[l]])

    def f_lb_rule(m, t, l):
        return m.F_min[l] <= m.f[t, l]

    def f_ub_rule(m, t, l):
        return m.f[t, l] <= m.F_max[l]

    def pi_lb_rule(m, t, n):
        return m.Pi_min[n] <= m.pi[t, n]

    def pi_ub_rule(m, t, n):
        return m.pi[t, n] <= m.Pi_max[n]

    def zero_first_pi_rule(m, t):
        return m.pi[t, 0] == 0

    model.balance = pe.Constraint(model.T_set, model.buses, rule=balance_rule)
    model.power_flow = pe.Constraint(model.T_set, model.lines, rule=power_flow_rule)
    model.f_lb = pe.Constraint(model.T_set, model.lines, rule=f_lb_rule)
    model.f_ub = pe.Constraint(model.T_set, model.lines, rule=f_ub_rule)
    model.pi_lb = pe.Constraint(model.T_set, model.buses, rule=pi_lb_rule)
    model.pi_ub = pe.Constraint(model.T_set, model.buses, rule=pi_ub_rule)
    model.zero_first_pi = pe.Constraint(model.T_set, rule=zero_first_pi_rule)

    model.obj = pe.Objective(expr=sum(model.total_cost[t] for t in model.T_set), sense=pe.minimize)
    return model



def optimal_simulation(env, solver, tee: bool = False, raise_on_infeasible: bool = True):
    """Solve the UC Pyomo model and return the optimal action sequence for the horizon."""
    horizon = int(env.T)
    m = create_model(env)

    opt = solver if hasattr(solver, "solve") else SolverFactory(str(solver))
    results = opt.solve(m, tee=tee)

    term = results.solver.termination_condition
    ok = term in (TerminationCondition.optimal, TerminationCondition.locallyOptimal)
    if (not ok) and raise_on_infeasible:
        raise RuntimeError(
            f"Optimization did not solve to optimality. Termination condition: {term}"
        )

    num_gen = env.num_gen
    num_bus = env.num_bus
    include_angle = num_bus > 1

    u = np.zeros((horizon, num_gen), dtype=np.float32)
    p = np.zeros((horizon, num_gen), dtype=np.float32)
    pi = np.zeros((horizon, num_bus), dtype=np.float32) if include_angle else None

    for t in range(1, horizon + 1):
        t_idx = t - 1
        for i in range(num_gen):
            u_val = pe.value(m.u[t, i])
            p_val = pe.value(m.p[t, i])
            u[t_idx, i] = 0.0 if u_val is None else float(u_val)
            p[t_idx, i] = 0.0 if p_val is None else float(p_val)
        if include_angle:
            for n in range(num_bus):
                pi_val = pe.value(m.pi[t, n])
                pi[t_idx, n] = 0.0 if pi_val is None else float(pi_val)

    if env.scale_action:
        raw_on_off = (u >= 0.5).astype(np.float32) * 2.0 - 1.0

        p_min = np.asarray(env.P_min, dtype=np.float32)
        p_max = np.asarray(env.P_max, dtype=np.float32)
        denom_p = p_max - p_min
        safe_denom_p = np.where(denom_p > 1e-12, denom_p, 1.0)
        raw_power = 2.0 * (p - p_min) / safe_denom_p - 1.0
        raw_power = np.clip(raw_power, -1.0, 1.0).astype(np.float32)
        raw_power[:, denom_p <= 1e-12] = 0.0

        if include_angle:
            pi_min = np.asarray(env.Pi_min, dtype=np.float32)
            pi_max = np.asarray(env.Pi_max, dtype=np.float32)
            denom_pi = pi_max - pi_min
            safe_denom_pi = np.where(denom_pi > 1e-12, denom_pi, 1.0)
            raw_pi = 2.0 * (pi - pi_min) / safe_denom_pi - 1.0
            raw_pi = np.clip(raw_pi, -1.0, 1.0).astype(np.float32)
            raw_pi[:, denom_pi <= 1e-12] = 0.0
            angle_actions = raw_pi[:, 1:]
    else:
        raw_on_off = u
        raw_power = p
        if include_angle:
            angle_actions = pi[:, 1:]

    if include_angle:
        action_dim = 2 * num_gen + (num_bus - 1)
    else:
        action_dim = 2 * num_gen

    raw_actions = np.zeros((horizon, action_dim), dtype=np.float32)
    raw_actions[:, :num_gen] = raw_on_off
    raw_actions[:, num_gen:2 * num_gen] = raw_power
    if include_angle:
        raw_actions[:, 2 * num_gen:] = angle_actions

    return raw_actions
