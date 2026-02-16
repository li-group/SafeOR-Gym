import numpy as np
import pyomo.environ as pe
import json
import os


def build_optimization_model(args):

    config_path = getattr(args, 'config_path', None)
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'unit_commitment_config.json')

    with open(config_path, 'r') as f:
        cfg = json.load(f)

    base_cfg = cfg.get('common', {})
    if args.env_id in cfg:
        env_cfg = cfg[args.env_id]
    elif 'common' in cfg:
        raise ValueError(f"Config missing section for env_id '{args.env_id}'")
    else:
        env_cfg = cfg

    cfg = {**base_cfg, **env_cfg}

    def _int_keyed_dict(d):
        return {int(k): v for k, v in d.items()}

    if 'gen_bus' in cfg:
        cfg['gen_bus'] = _int_keyed_dict(cfg['gen_bus'])
    if 'bus_gen' in cfg:
        cfg['bus_gen'] = {int(k): v for k, v in cfg['bus_gen'].items()}
    if 'line_bus' in cfg:
        cfg['line_bus'] = {int(k): tuple(v) for k, v in cfg['line_bus'].items()}
    if 'u0_seq' in cfg:
        cfg['u0_seq'] = {int(k): np.array(v, dtype=float) for k, v in cfg['u0_seq'].items()}

    float_array_keys = {
        'B', 'F_max', 'F_min', 'Pi_max', 'Pi_min', 'pi0', 'loc', 'scale',
        'deterministic_demand', 'P_max', 'P_min', 'a', 'b', 'c', 'RU', 'RD',
        'SU', 'SD', 'hot_cost', 'cold_cost', 'C_SD', 'u0_prev', 'u0',
        'p0_prev', 'p0'
    }
    int_array_keys = {'UT', 'DT', 'cold_hrs'}

    for key in float_array_keys:
        if key in cfg:
            cfg[key] = np.array(cfg[key], dtype=float)
    for key in int_array_keys:
        if key in cfg:
            cfg[key] = np.array(cfg[key], dtype=int)

    for key in ['T', 'num_gen', 'num_bus', 'num_line']:
        if key in cfg:
            cfg[key] = int(cfg[key])

    num_gen = cfg['num_gen']
    T = cfg['T']
    num_bus = cfg['num_bus']
    gen_bus = cfg['gen_bus']
    bus_gen = cfg['bus_gen']
    num_line = cfg['num_line']
    line_bus = cfg['line_bus']
    B = cfg['B']
    F_max = cfg['F_max']
    F_min = cfg['F_min']
    Pi_max = cfg['Pi_max']
    Pi_min = cfg['Pi_min']
    pi0 = cfg['pi0']
    deterministic_demand = cfg['deterministic_demand']

    horizon = range(1, T + 1)
    generators = range(num_gen)
    buses = range(num_bus)
    lines = range(num_line)
    # line_from_bus = {k: [] for k in lines}
    # line_to_bus = {k: [] for k in lines}
    # for line in lines:
    #     line_from_bus[line].append(line_bus[line][0])
    #     line_to_bus[line].append(line_bus[line][1])
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

    P_max = cfg['P_max']
    P_min = cfg['P_min']
    a = cfg['a']
    b = cfg['b']
    c = cfg['c']
    UT = cfg['UT']
    DT = cfg['DT']
    RU = cfg['RU']
    RD = cfg['RD']
    SU = cfg['SU']
    SD = cfg['SD']
    hot_cost = cfg['hot_cost']
    cold_cost = cfg['cold_cost']
    cold_hrs = cfg['cold_hrs']
    C_SD = cfg['C_SD']
    C_LS = cfg['C_LS']
    C_RP = cfg['C_RP']
    R = cfg['R']

    u0_seq = cfg['u0_seq']
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

    u0 = cfg['u0']
    p0 = cfg['p0']

    model = pe.ConcreteModel()
    model.T_set = pe.Set(initialize=horizon)
    model.generators = pe.Set(initialize=generators)
    model.buses = pe.Set(initialize=buses)
    model.lines = pe.Set(initialize=lines)
    model.bus_gen = pe.Set(model.buses, initialize=bus_gen)
    # model.line_from_bus = pe.Set(model.lines, initialize=lambda m, l: line_bus[l][0])
    # model.line_to_bus = pe.Set(model.lines, initialize=lambda m, l: line_bus[l][0])
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




