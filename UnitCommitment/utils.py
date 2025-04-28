import numpy as np
import pyomo.environ as pe


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
    C_LS = 10000
    C_RP = 100
    R = 10

    u0_seq = {0: np.ones(8 + 1),  # assume only 1st generator is on
              1: np.zeros(5 + 1),
              2: np.zeros(5 + 1),
              3: np.zeros(3 + 1),
              4: np.zeros(1 + 1)}  # assume no change happened from - [max(UT, DT)+1] to 0
    # assume only 1st generator is on
    u0_prev = np.array([1, 0, 0, 0, 0])
    u0 = np.array([1, 0, 0, 0, 0])
    self.v, self.w = self._reckless_move(self.u, self.u_prev)
    self.v_seq, self.w_seq = self._u2vw_seq(self.u_seq)

    self.p_prev = self.p0_prev = np.array([300, 0, 0, 0, 0])
    self.p = self.p0 = np.array([300, 0, 0, 0, 0])



    generators = range(num_gen)
    buses = range(num_bus)
    lines = range(num_line)
    from_bus = []
    to_bus = []
    for line in lines:
        from_bus.append(line_bus[line][0])
        to_bus.append(line_bus[line][1])
    from_bus_lines = {i: [] for i in buses}
    to_bus_lines = {i: [] for i in buses}
    for line, (from_bus, to_bus) in line_bus.items():
        from_bus_lines[from_bus].append(line)
        to_bus_lines[to_bus].append(line)

    model = pe.ConcreteModel()
    model.T_set = pe.Set(initialize=range(T))
    model.generators = pe.Set(initialize=generators)
    model.buses = pe.Set(initialize=buses)
    model.lines = pe.Set(initialize=lines)
    model.from_bus_lines = pe.Set(model.buses, initialize=from_bus_lines)
    model.to_bus_lines = pe.Set(model.buses, initialize=to_bus_lines)

    p_prev_init = {(0, i + 1): 0 if i > 0 else 300 for i in range(num_gen)}
    u_prev_init = {(0, i + 1): 0 if i > 0 else 1 for i in range(num_gen)}
    v_prev_set = [(-t, i + 1) for i in range(num_gen) for t in range(UT[i])]
    w_prev_set = [(-t, i + 1) for i in range(num_gen) for t in range(DT[i])]
    v_prev_init = {idx: 0 for idx in v_prev_set}  # assume no change made in the past
    w_prev_init = {idx: 0 for idx in w_prev_set}  # assume no change made in the past



    model.v_prev_set = pe.Set(initialize=v_prev_set)
    model.w_prev_set = pe.Set(initialize=w_prev_set)







    model.D_forecast = pe.Param(model.T_set,
                                initialize={t + 1: D_forecast[t] for t in range(len(D_forecast))}, mutable=True)
    model.s_pos = pe.Var(model.T_set, domain=pe.NonNegativeReals)
    model.s_neg = pe.Var(model.T_set, domain=pe.NonNegativeReals)
    model.s = pe.Var(model.T_set, domain=pe.NonNegativeReals)
    model.u = pe.Var(model.T_set, model.G_set, domain=pe.Binary)
    model.v = pe.Var(model.T_set, model.G_set, domain=pe.Binary)
    model.w = pe.Var(model.T_set, model.G_set, domain=pe.Binary)
    model.u_prev = pe.Param([0], model.G_set, initialize=u_prev_init)
    model.v_prev = pe.Param(model.v_prev_set, initialize=v_prev_init, mutable=True)
    model.w_prev = pe.Param(model.w_prev_set, initialize=w_prev_init)
    model.p = pe.Var(model.T_set, model.G_set)
    model.p_bar = pe.Var(model.T_set, model.G_set)
    model.p_prev = pe.Param([0], model.G_set, initialize=p_prev_init)

    model.cp_imp = pe.Var()
    model.csu_imp = pe.Var()
    model.csd_imp = pe.Var()
    model.crelax_imp = pe.Var()
    model.c_total_imp = pe.Var()

    model.P_max = pe.Param(model.G_set, initialize={i + 1: P_max[i] for i in range(len(P_max))})
    model.P_min = pe.Param(model.G_set, initialize={i + 1: P_min[i] for i in range(len(P_min))})
    model.a = pe.Param(model.G_set, initialize={i + 1: a[i] for i in range(len(a))})
    model.b = pe.Param(model.G_set, initialize={i + 1: b[i] for i in range(len(b))})
    model.c = pe.Param(model.G_set, initialize={i + 1: c[i] for i in range(len(c))})
    model.UT = pe.Param(model.G_set, initialize={i + 1: UT[i] for i in range(len(UT))})
    model.DT = pe.Param(model.G_set, initialize={i + 1: DT[i] for i in range(len(DT))})
    model.RU = pe.Param(model.G_set, initialize={i + 1: RU[i] for i in range(len(RU))})
    model.RD = pe.Param(model.G_set, initialize={i + 1: RD[i] for i in range(len(RD))})
    model.SU = pe.Param(model.G_set, initialize={i + 1: SU[i] for i in range(len(SU))})
    model.SD = pe.Param(model.G_set, initialize={i + 1: SD[i] for i in range(len(SD))})
    model.hot_cost = pe.Param(model.G_set,
                              initialize={i + 1: hot_cost[i] for i in range(len(hot_cost))})
    model.cold_cost = pe.Param(model.G_set,
                               initialize={i + 1: cold_cost[i] for i in range(len(cold_cost))})
    model.cold_hrs = pe.Param(model.G_set,
                              initialize={i + 1: cold_hrs[i] for i in range(len(cold_hrs))})
    model.C_SD = pe.Param(model.G_set, initialize={i + 1: C_SD[i] for i in range(len(C_SD))})
    model.C_LS = pe.Param(initialize=C_LS)
    model.R = pe.Param(model.T_set, model.G_set,
                       initialize={(t + 1, i + 1): R[t, i] for t in range(num_periods) for i in range(num_gen)})

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

    model.uvw = pe.Constraint(model.T_set, model.G_set, rule=uvw_rule)
    model.min_up = pe.Constraint(model.T_set, model.G_set, rule=min_up_rule)
    model.min_down = pe.Constraint(model.T_set, model.G_set, rule=min_down_rule)

    # Reserve Requirement
    def reserve_rule(m, t, i):
        return m.p_bar[t, i] == m.p[t, i] + m.R[t, i]

    model.reserve = pe.Constraint(model.T_set, model.G_set, rule=reserve_rule)

    # Generation Bounds
    def p_lb_rule(m, t, i):
        return m.P_min[i] * m.u[t, i] <= m.p[t, i]

    def p_ub_rule(m, t, i):
        return m.p[t, i] <= m.p_bar[t, i]

    def p_bar_ub_rule(m, t, i):
        return m.p_bar[t, i] <= m.P_max[i] * m.u[t, i]

    model.p_lb = pe.Constraint(model.T_set, model.G_set, rule=p_lb_rule)
    model.p_ub = pe.Constraint(model.T_set, model.G_set, rule=p_ub_rule)
    model.p_bar_ub = pe.Constraint(model.T_set, model.G_set, rule=p_bar_ub_rule)

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

    model.ramp_up = pe.Constraint(model.T_set, model.G_set, rule=ramp_up_rule)
    model.ramp_down = pe.Constraint(model.T_set, model.G_set, rule=ramp_down_rule)

    # Cost Function and Objective
    def production_cost_rule(m):
        return m.cp_imp == sum(
            m.a[i] * (m.p[t, i] ** 2) + m.b[i] * m.p[t, i] + m.c[i] for i in m.G_set for t in m.T_set)

    def startup_cost_rule(m):
        return m.csu_imp == sum(m.v[t, i] * m.hot_cost[i] for i in m.G_set for t in m.T_set)

    def shutdown_cost_rule(m):
        return m.csd_imp == sum(m.w[t, i] * m.C_SD[i] for i in m.G_set for t in m.T_set)

    def relaxation_cost_rule(m):
        return m.crelax_imp == sum(m.C_LS * (m.s_pos[t] + m.s_neg[t]) for t in m.T_set)

    def total_cost_rule(m):
        return m.c_total_imp == m.cp_imp + m.csu_imp + m.csd_imp + m.crelax_imp

    model.production_cost = pe.Constraint(rule=production_cost_rule)
    model.startup_cost = pe.Constraint(rule=startup_cost_rule)
    model.shutdown_cost = pe.Constraint(rule=shutdown_cost_rule)
    model.relaxation_cost = pe.Constraint(rule=relaxation_cost_rule)
    model.total_cost = pe.Constraint(rule=total_cost_rule)

    # Network Constraints
    def slack_rule(m, t):
        return m.s[t] == m.s_pos[t] - m.s_neg[t]

    def demand_rule(m, t):
        return sum(m.p[t, i] for i in m.G_set) + m.s[t] == m.D_forecast[t]

    model.slack = pe.Constraint(model.T_set, rule=slack_rule)
    model.demand = pe.Constraint(model.T_set, rule=demand_rule)

    model.obj = pe.Objective(expr=model.c_total_imp, sense=pe.minimize)
    solver = pe.SolverFactory('gurobi')
    solver.options['NonConvex'] = 2
    solver.solve(model, tee=False)

    actions = np.zeros((num_periods, num_gen))
    p = np.zeros((num_periods, num_gen))
    rewards = np.zeros(num_periods)
    for t in range(num_periods):
        for i in range(num_gen):
            actions[t, i] = model.u[t + 1, i + 1].value
            p[t, i] = model.p[t + 1, i + 1].value

    for t in range(num_periods):
        print(f"a{t}: {actions[t]}")
        print(f"p{t + 1}: {p[t]}")
        production_cost = sum(a[i] * (p[t, i] ** 2) + b[i] * p[t, i] + c[i] for i in range(num_gen))
        startup_cost = sum(hot_cost[i] * model.v[t + 1, i + 1].value for i in range(num_gen))
        relaxation_cost = C_LS * (model.s_pos[t + 1].value + model.s_neg[t + 1].value)
        total_cost = production_cost + startup_cost + relaxation_cost
        print(f"r{t + 1}: {total_cost}")

    return RTN





