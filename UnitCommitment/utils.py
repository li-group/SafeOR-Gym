import numpy as np
import pyomo.environ as pe


def init_model(args, data):
    num_gen = 5
    T = 24
    if args.env_id == 'UC-v0':
        num_bus = 1
        gen_bus = {i: 0 for i in range(num_gen)}
        bus_gen = {0: [0, 1, 2, 3, 4]}
        num_line = 0
        line_bus = {}
        B = np.array([20])
        F_max = np.array([0])
        F_min = np.array([0])
        Pi_max = np.array([0])
        Pi_min = np.array([0])
        pi0 = np.array([0])
        deterministic_demand = np.array([[362.], [191.], [303.], [263.], [416.], [302.],
                                              [328.], [234.], [357.], [266.], [333.], [325.],
                                              [343.], [285.], [290.], [329.], [245.], [305.],
                                              [311.], [254.], [385.], [214.], [197.], [227.]])
    elif args.env_id == 'UC-v1':
        num_bus = 4
        gen_bus = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}
        bus_gen = {0: [0], 1: [1], 2: [2], 3: [3, 4]}
        num_line = 5
        line_bus = {0: (0, 1), 1: (0, 2), 2: (1, 2), 3: (1, 3), 4: (2, 3)}
        B = np.array([20, 20, 20, 20, 20])
        F_max = np.array([100, 100, 100, 100, 100])
        F_min = np.array([-100, -100, -100, -100, -100])
        Pi_max = np.array([0, 0.2, 0.2, 0.2])  # angle_1 = 0
        Pi_min = np.array([0, -0.2, -0.2, -0.2])  # angle_1 = 0
        pi0 = np.array([0, 0, 0, 0])
        deterministic_demand = np.array([[226., 57., 52., 59.], [233., 57., 65., 77.],
                                              [258., 59., 48., 78.], [272., 67., 55., 67.],
                                              [247., 55., 65., 65.], [232., 65., 69., 63.],
                                              [213., 57., 59., 69.], [245., 71., 60., 74.],
                                              [243., 59., 72., 61.], [263., 63., 55., 56.],
                                              [291., 60., 55., 72.], [235., 58., 59., 73.],
                                              [234., 59., 67., 65.], [253., 47., 54., 63.],
                                              [267., 52., 47., 55.], [223., 58., 57., 72.],
                                              [239., 67., 62., 67.], [260., 60., 56., 63.],
                                              [234., 61., 62., 76.], [241., 59., 54., 84.],
                                              [298., 63., 63., 76.], [235., 55., 52., 80.],
                                              [273., 63., 75., 80.], [276., 54., 73., 70.]])

    else:
        raise ValueError(f"Unknown env_id: {args.env_id}")

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

    P_max = np.array([455, 130, 130, 80, 55])
    P_min = np.array([150, 20, 20, 20, 55])
    a = np.array([0.00048, 0.00200, 0.00211, 0.00712, 0.00413])
    b = np.array([16.19, 16.60, 16.50, 22.26, 25.92])
    c = np.array([1000, 700, 680, 370, 660])
    UT = np.array([8, 5, 5, 3, 1])
    DT = np.array([8, 5, 5, 3, 1])
    RU = np.array([300, 85, 85, 55, 55])
    RD = np.array([300, 85, 85, 55, 55])
    SU = np.array([300, 85, 85, 55, 55])
    SD = np.array([300, 85, 85, 55, 55])
    hot_cost = np.array([4500, 550, 560, 170, 30])
    cold_cost = np.array([9000, 1100, 1120, 340, 60])
    cold_hrs = np.array([5, 4, 4, 2, 0])
    C_SD = np.array([0, 0, 0, 0, 0])
    C_LS = 100
    C_RP = 100
    R = 10

    u0_seq = {0: np.ones(8 + 1),  # assume only 1st generator is on
              1: np.zeros(5 + 1),
              2: np.zeros(5 + 1),
              3: np.zeros(3 + 1),
              4: np.zeros(1 + 1)}  # assume no change happened from - [max(UT, DT)+1] to 0
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

    u0 = np.array([1, 0, 0, 0, 0])
    p0 = np.array([300, 0, 0, 0, 0])

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


# class args:
#     def __init__(self, env_id):
#         self.env_id = env_id
#
# if __name__ == "__main__":
#     args_instance = args(env_id='UC-v1')
#     data = None  # Replace with actual data if needed
#     model = init_model(args_instance, data)
#
#     solver = pe.SolverFactory("gurobi")
#     solver.options['NonConvex'] = 2
#     results = solver.solve(model, False)
#
#     action = {'on_off': {}, 'power': {}, 'angle': {}}
#     action_arr = np.zeros((24, 5+5+3))
#     for t in range(1, 25):
#         for i in range(5):
#             action['on_off'][t, i] = model.u[t, i].value
#             action['power'][t, i] = model.p[t, i].value
#             action_arr[t-1, i] = model.u[t, i].value
#             action_arr[t-1, i + 5] = model.p[t, i].value
#         for n in range(1, 4):
#             action['angle'][t, n] = model.pi[t, n].value
#             action_arr[t-1, 9 + n] = model.pi[t, n].value
#     # save action arr
#     np.save('opt_action_v1_arr.npy', action_arr)
#     print(f"optimal cost v1: {model.obj()}")
#
#     args_instance = args(env_id='UC-v0')
#     data = None  # Replace with actual data if needed
#     model = init_model(args_instance, data)
#
#     action = {'on_off': {}, 'power': {}, 'angle': {}}
#     action_arr = np.zeros((24, 5+5))
#     for t in range(1, 25):
#         for i in range(5):
#             action['on_off'][t, i] = model.u[t, i].value
#             action['power'][t, i] = model.p[t, i].value
#             action_arr[t-1, i] = model.u[t, i].value
#             action_arr[t-1, i + 5] = model.p[t, i].value
#     # save action arr
#     np.save('opt_action_v0_arr.npy', action_arr)
#     print(f"optimal cost v0: {model.obj()}")







