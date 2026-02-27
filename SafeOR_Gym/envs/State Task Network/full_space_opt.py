import json
import numpy as np
import pyomo.environ as po
from pyomo.opt import SolverFactory, TerminationCondition

def build_optimization_model(config_file: str, horizon: int) -> po.ConcreteModel:
    with open(config_file, "r") as f:
        data = json.load(f)

    R_dict  = data["reactants"]
    IM_dict = data["intermediates"]
    P_dict  = data["products"]
    J_dict  = data["equipments"]
    U_costs = data["utility_costs"]
    tasks   = data["tasks"]
    demand_raw = data["demand"]

    R_list = list(R_dict)
    M_list = list(IM_dict)
    P_list = list(P_dict)
    S_list = R_list + M_list + P_list
    J_list = list(J_dict)
    U_list = list(U_costs)

    # time series (env uses string keys "1".."T")
    demand = {p: [demand_raw[p].get(str(t), 0.0) for t in range(1, horizon + 1)]  for p in demand_raw}
    util_cost = {u: [U_costs[u].get(str(t), 0.0) for t in range(1, horizon + 1)] for u in U_costs}

    # --- Precompute mappings exactly like env uses ---
    # equipment used by each task (your tasks have 1 equipment)
    equip_of = {i: next(iter(tasks[i]["equipments"].values())) for i in tasks}

    # batch bounds depend on that equipment key
    Vmin = {i: tasks[i]["Vmin"][equip_of[i]] for i in tasks}
    Vmax = {i: tasks[i]["Vmax"][equip_of[i]] for i in tasks}

    # equipment return delay = max over p dict (env uses last_tau = max(tau.values()))
    tau_eq = {i: int(max(tasks[i]["p"].values())) for i in tasks}

    # per-produced-resource delay: only for resources with positive coeff in stoich (prod_dist/int_prod_dist)
    out_delay = {}
    for i, attr in tasks.items():
        pmap = attr["p"]
        for s in list(attr.get("prod_dist", {}).keys()) + list(attr.get("int_prod_dist", {}).keys()):
            if s not in pmap:
                raise KeyError(f"Task {i} produces {s} but has no p[{s}] delay.")
            out_delay[(i, s)] = int(pmap[s])

    # stoichiometry (keep your sign convention: consumption terms are negative)
    raw_cons = {(i, r): -tasks[i]["raw_dist"][r] for i in tasks for r in tasks[i]["raw_dist"]}
    int_cons = {(i, r): -tasks[i]["int_react_dist"][r] for i in tasks for r in tasks[i]["int_react_dist"]}
    prod_out = {(i, r):  tasks[i]["prod_dist"][r] for i in tasks for r in tasks[i]["prod_dist"]}
    int_out  = {(i, r):  tasks[i]["int_prod_dist"][r] for i in tasks for r in tasks[i]["int_prod_dist"]}

    # utilities
    util_rate = {}
    for i, attr in tasks.items():
        for u, uf in attr.get("utilities", {}).items():
            util_rate[(i, u)] = uf

    # --- Model ---
    m = po.ConcreteModel()
    m.I = po.Set(initialize=list(tasks.keys()))
    m.R = po.Set(initialize=R_list)
    m.M = po.Set(initialize=M_list)
    m.P = po.Set(initialize=P_list)
    m.S = po.Set(initialize=S_list)
    m.J = po.Set(initialize=J_list)
    m.U = po.Set(initialize=U_list)
    m.RJ = po.Set(initialize=S_list + J_list)

    m.T0 = po.RangeSet(0, horizon)
    m.T  = po.RangeSet(1, horizon)

    # Params
    def _X0(_m, r):
        if r in R_dict:  return R_dict[r]["X0"]
        if r in IM_dict: return IM_dict[r]["X0"]
        if r in P_dict:  return P_dict[r]["X0"]
        return J_dict[r]["X0"]

    m.X0   = po.Param(m.RJ, initialize=_X0)
    m.Xmin = po.Param(m.RJ, initialize={
        r: (R_dict.get(r, IM_dict.get(r, P_dict.get(r, J_dict.get(r)))))["Xmin"]
        for r in m.RJ
    })
    m.Xmax = po.Param(m.RJ, initialize={
        r: (R_dict.get(r, IM_dict.get(r, P_dict.get(r, J_dict.get(r)))))["Xmax"]
        for r in m.RJ
    })

    m.Vmin = po.Param(m.I, initialize=Vmin)
    m.Vmax = po.Param(m.I, initialize=Vmax)
    m.tau_eq = po.Param(m.I, initialize=tau_eq)

    # demand is negative in your JSON; keep it like your code
    m.demand = po.Param(m.P, m.T, initialize={
        (p, t): demand.get(p, [0.0]*horizon)[t-1] for p in m.P for t in m.T
    })

    m.Ucost = po.Param(m.U, m.T, initialize={
        (u, t): util_cost[u][t-1] for u in m.U for t in m.T
    })

    m.Price = po.Param(m.P, initialize={p: P_dict[p]["cost"] for p in m.P})
    m.RawCost = po.Param(m.R, initialize={r: R_dict[r]["cost"] for r in m.R})

    # Vars
    # NOTE: env allows reactants < 0, so make X Reals and enforce nonneg only for non-reactants/equipment
    m.X  = po.Var(m.RJ, m.T0, domain=po.Reals)
    m.N  = po.Var(m.I,  m.T,  domain=po.Binary)
    m.E  = po.Var(m.I,  m.T,  domain=po.NonNegativeReals)
    m.F  = po.Var(m.U,  m.T,  domain=po.NonNegativeReals)

    m.Sl = po.Var(m.P,  m.T,  domain=po.NonNegativeReals)  # unmet
    m.Ss = po.Var(m.P,  m.T,  domain=po.NonNegativeReals)  # sold

    # reactant deficit to match env reward term: deficit = -min(X,0)
    m.Buy = po.Var(m.R,  m.T,  domain=po.NonNegativeReals)

    # initial
    for r in m.RJ:
        m.X[r, 0].fix(m.X0[r])

    # Objective: env reward aggregated over horizon (minus constant terms)
    def _obj(m):
        revenue = sum(m.Ss[p, t] * m.Price[p] for p in m.P for t in m.T)
        pen = 1.5 * sum(m.Sl[p, t] * m.Price[p] for p in m.P for t in m.T)
        util = sum(m.F[u, t] for u in m.U for t in m.T)
        react = sum(m.Buy[r,t] * m.RawCost[r] for r in m.R for t in m.T)
        return revenue - util - pen + react

    m.obj = po.Objective(rule=_obj, sense=po.maximize)

    # --- Constraints ---

    m.Balance = po.ConstraintList()
    for s in m.S:
        for t in m.T:
            expr = m.X[s, t] - m.X[s, t-1]

            # immediate consumption at t (raw_cons/int_cons are negative)
            expr -= sum(raw_cons.get((i, s), 0.0) * m.E[i, t] for i in m.I)
            expr -= sum(int_cons.get((i, s), 0.0) * m.E[i, t] for i in m.I)

            # delayed arrivals at t (prod_out/int_out are positive)
            for i in m.I:
                if (i, s) in prod_out:
                    d = out_delay[(i, s)]
                    if t - d >= 1:
                        expr -= prod_out[(i, s)] * m.E[i, t - d]
                if (i, s) in int_out:
                    d = out_delay[(i, s)]
                    if t - d >= 1:
                        expr -= int_out[(i, s)] * m.E[i, t - d]

            # Buy adds inventory => -Buy on LHS
            if s in m.R:
                expr -= m.Buy[s, t]

            # Sales ship product => +Ss on LHS
            if s in m.P:
                expr += m.Ss[s, t]

            m.Balance.add(expr == 0)

    # 2) Equipment balances: return after tau_eq (max p-values)
    m.EquipBal = po.ConstraintList()
    for j in m.J:
        for t in m.T:
            cons = sum(m.N[i, t] for i in m.I if equip_of[i] == j)
            ret  = sum(m.N[i, t - m.tau_eq[i]] for i in m.I
                       if equip_of[i] == j and (t - m.tau_eq[i]) >= 1)
            m.EquipBal.add(m.X[j, t] == m.X[j, t-1] - cons + ret)

    # 3) Batch-size / activation coupling (avoid “N=1,E=0” degeneracy)
    eps = 1e-3
    m.BatchUB = po.ConstraintList()
    m.BatchLB = po.ConstraintList()
    for i in m.I:
        for t in m.T:
            m.BatchLB.add(m.E[i, t] >= max(eps, m.Vmin[i]) * m.N[i, t])
            m.BatchUB.add(m.E[i, t] <= m.Vmax[i] * m.N[i, t])

    # 4) Utility cost (exactly like env: sum uf*price*batch)
    m.UtilBal = po.ConstraintList()
    for u in m.U:
        for t in m.T:
            expr = sum(util_rate.get((i, u), 0.0) * m.E[i, t] for i in m.I) * m.Ucost[u, t]
            m.UtilBal.add(m.F[u, t] == expr)

    # 5) Sales-demand link (your sign convention: demand is negative)
    m.SalesDemandLink = po.ConstraintList()
    for p in m.P:
        for t in m.T:
            m.SalesDemandLink.add(m.Ss[p, t] + m.Sl[p, t] == -m.demand[p, t])

    # 7) Non-reactant & equipment bounds (prevents env clamp cost)
    m.Bounds = po.ConstraintList()
    for r in list(IM_dict.keys()) + list(P_dict.keys()) + list(J_dict.keys()):
        for t in m.T0:
            m.Bounds.add(m.X[r, t] >= m.Xmin[r])
            m.Bounds.add(m.X[r, t] <= m.Xmax[r])
    
    m.ReactantNonneg = po.ConstraintList()
    for r in m.R:
        for t in m.T0:
            m.ReactantNonneg.add(m.X[r,t] >= 0.0)
            m.ReactantNonneg.add(m.X[r,t] <= m.Xmax[r])

    return m

def optimal_simulation(env, solver, tee: bool = False, raise_on_fail: bool = True):
    """
    Solve the STN Pyomo model and return the optimal flattened action sequence.

    Parameters
    ----------
    env : STNEnv
        Must have: env.config_file, env.T, env.task_names, env.equipments,
                   env.min_batch (num_tasks x num_eq), env.max_batch (num_tasks x num_eq).
    solver : str or Pyomo solver instance
        e.g. "gurobi", "cbc", "glpk" or an OptSolver with .solve(...)
    tee : bool
        Print solver output.
    raise_on_fail : bool
        Raise if not optimal.

    Returns
    -------
    raw_actions_flat : np.ndarray of shape (T, num_tasks*num_eq)
        Actions in [-1,1] to feed directly to env.step(raw_actions_flat[t]).
    batch_actions_mat : np.ndarray of shape (T, num_tasks, num_eq)
        Batch sizes placed on the chosen equipment column per task.
    results : SolverResults
        Pyomo results object.
    """
    horizon = int(env.T)
    m = build_optimization_model(env.config_file, horizon)

    opt = solver if hasattr(solver, "solve") else SolverFactory(str(solver))
    results = opt.solve(m, tee=tee)

    term = results.solver.termination_condition
    ok = term in (TerminationCondition.optimal, TerminationCondition.locallyOptimal)
    if (not ok) and raise_on_fail:
        raise RuntimeError(f"Optimization did not solve to optimality. Termination: {term}")

    task_names = list(env.task_names)
    eq_names = list(env.equipments)
    num_tasks = len(task_names)
    num_eq = len(eq_names)

    # Build mapping: each task -> (single) equipment used in the Pyomo model.
    # Your build_optimization_model picks: eq = next(iter(tasks[i]['equipments'].values()))
    # We recover that from the env's loaded tasks_dict to ensure consistent mapping.
    task_to_eq_idx = {}
    for i, task in enumerate(task_names):
        eq_used = next(iter(env.tasks_dict[task]["equipments"].values()))
        task_to_eq_idx[task] = eq_names.index(eq_used)

    # Extract optimal E[i,t] and place it into the corresponding (task, eq_used) entry.
    batch_actions_mat = np.zeros((horizon, num_tasks, num_eq), dtype=np.float32)

    for t in range(1, horizon + 1):
        for i, task in enumerate(task_names):
            if task not in m.I:
                continue
            e_val = po.value(m.E[task, t])
            if e_val is None or abs(e_val) < 1e-3:
                e_val = 0.0

            eq_idx = task_to_eq_idx[task]
            batch_actions_mat[t - 1, i, eq_idx] = float(e_val)

    # Convert batch sizes -> raw actions in [-1,1] with env's per-(task,equip) min/max.
    min_b = np.asarray(env.min_batch, dtype=np.float32)  # (num_tasks, num_eq)
    max_b = np.asarray(env.max_batch, dtype=np.float32)  # (num_tasks, num_eq)
    denom = (max_b - min_b)
    raw_actions_mat = np.zeros_like(batch_actions_mat, dtype=np.float32)

    valid = denom > 1e-3  # robust

    for t in range(horizon):
        for i in range(num_tasks):
            for e in range(num_eq):
                b = float(batch_actions_mat[t, i, e])

                if not valid[i, e]:
                    # forbidden slot: must be exactly 0.0 raw action
                    raw_actions_mat[t, i, e] = 0.0
                    continue

                # clip batch into feasible range for safety
                b = float(np.clip(b, min_b[i, e], max_b[i, e]))

                r = 2.0 * (b - min_b[i, e]) / denom[i, e] - 1.0
                r = float(np.clip(r, -1.0, 1.0))

                raw_actions_mat[t, i, e] = r

    raw_actions_flat = raw_actions_mat.reshape(horizon, num_tasks * num_eq)
    print(raw_actions_flat)

    return raw_actions_flat