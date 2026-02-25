import json
import numpy as np
import pyomo.environ as po
from pyomo.opt import SolverFactory, TerminationCondition

def build_optimisation_model(config_file: str, horizon: int):
    # 1) LOAD JSON
    with open(config_file, 'r') as f:
        data = json.load(f)

    R_dict     = data['reactants']
    IM_dict    = data['intermediates']
    P_dict     = data['products']
    J_dict     = data['equipments']
    U_costs    = data['utility_costs']
    tasks      = data['tasks']
    demand_raw = data['demand']

    # 2) FLATTEN SET LISTS
    S_list = list(R_dict) + list(IM_dict) + list(P_dict)
    J_list = list(J_dict)
    U_list = list(U_costs)

    # 3) BUILD ZERO‐BASED DEMAND & UTILITY TIME SERIES
    demand = {
        p: [demand_raw[p].get(str(t), 0.0) for t in range(1, horizon+1)]
        for p in demand_raw
    }
    util_cost = {
        u: [U_costs[u].get(str(t), 0.0) for t in range(1, horizon+1)]
        for u in U_costs
    }

    # 4) MODEL & SETS
    m = po.ConcreteModel()
    m.I   = po.Set(initialize=tasks.keys())
    m.S   = po.Set(initialize=S_list)
    m.J   = po.Set(initialize=J_list)
    m.U   = po.Set(initialize=U_list)

    m.RJ  = po.Set(initialize=S_list + J_list)
    m.RJU = po.Set(initialize=S_list + J_list + U_list)

    m.T0  = po.RangeSet(0, horizon)
    m.T   = po.RangeSet(1, horizon)

    # 5) PARAMETERS
    m.tau  = po.Param(m.I, initialize={i: tasks[i]['tau']  for i in m.I})
    m.Vmin = po.Param(m.I, initialize={i: tasks[i]['Vmin'] for i in m.I})
    m.Vmax = po.Param(m.I, initialize={i: tasks[i]['Vmax'] for i in m.I})

    def _X0_init(m, r):
        if   r in R_dict:  return R_dict[r]['X0']
        elif r in IM_dict: return IM_dict[r]['X0']
        elif r in P_dict:  return P_dict[r]['X0']
        else:              return J_dict[r]['X0']
    m.X0   = po.Param(m.RJ, initialize=_X0_init)
    m.Xmin = po.Param(m.RJ, initialize={
        r: (R_dict.get(r, IM_dict.get(r, P_dict.get(r, J_dict.get(r)))))['Xmin']
        for r in m.RJ
    })
    m.Xmax = po.Param(m.RJ, initialize={
        r: (R_dict.get(r, IM_dict.get(r, P_dict.get(r, J_dict.get(r)))))['Xmax']
        for r in m.RJ
    })

    m.Cost    = po.Param(m.S, initialize={
        r: R_dict.get(r, {}).get('cost', 0.0) for r in m.S
    })
    m.InvCost = po.Param(m.S, initialize={
        r: (R_dict.get(r, IM_dict.get(r, P_dict.get(r, {})))).get('inventory_cost', 0.0)
        for r in m.S
    })
    m.Price   = po.Param(m.S, initialize={
        r: P_dict.get(r, {}).get('cost', 0.0) for r in m.S
    })

    m.demand  = po.Param(m.S, m.T, initialize={
        (p, t): demand.get(p, [0.0]*horizon)[t-1] if p in P_dict else 0.0
        for p in m.S for t in m.T
    })
    m.Ucost   = po.Param(m.U, m.T, initialize={
        (u, t): util_cost[u][t-1] for u in m.U for t in m.T
    })

    # stoichiometry
    raw_cons = {(i,r): -tasks[i]['raw_dist'][r]
                for i in tasks for r in tasks[i]['raw_dist']}
    int_cons = {(i,r): -tasks[i]['int_react_dist'][r]
                for i in tasks for r in tasks[i]['int_react_dist']}
    prod_out = {(i,r):  tasks[i]['prod_dist'][r]
                for i in tasks for r in tasks[i]['prod_dist']}
    int_out  = {(i,r):  tasks[i]['int_prod_dist'][r]
                for i in tasks for r in tasks[i]['int_prod_dist']}

    # 6) VARIABLES
    m.X  = po.Var(m.RJ, m.T0, domain=po.NonNegativeReals)
    m.N  = po.Var(m.I,  m.T,  domain=po.Binary)
    m.E  = po.Var(m.I,  m.T,  domain=po.NonNegativeReals)
    m.F  = po.Var(m.U,  m.T,  domain=po.NonNegativeReals)
    m.Sl = po.Var(m.S,  m.T,  domain=po.NonNegativeReals)  # unmet‐demand slack
    m.Ss = po.Var(m.S, m.T, domain=po.NonNegativeReals)  # sales

    # fix t=0 inventories
    for r in m.RJ:
        m.X[r,0].fix(m.X0[r])

    # 7) OBJECTIVE
    def _obj(m):
        rev   = sum(m.Ss[p,t] * m.Price[p] for p in P_dict for t in m.T)
        pen   = 1.5 * sum(m.Sl[p,t] * m.Price[p] for p in P_dict for t in m.T)
        util  = sum(m.F[u,t] for u in m.U for t in m.T)
        slack = sum(m.Sl[r,t] * m.Cost[r] for r in m.S for t in m.T)
        return rev - util - pen + slack
    m.obj = po.Objective(rule=_obj, sense=po.maximize)

    # 8) CONSTRAINTS
    # 8a) material‐state balance
    m.Balance = po.ConstraintList()
    for s in m.S:
        for t in m.T:
            expr = m.X[s,t] - m.X[s,t-1]
            expr += sum(raw_cons.get((i,s),0)*m.E[i,t] for i in m.I)
            expr += sum(int_cons.get((i,s),0)*m.E[i,t] for i in m.I)
            expr += sum(prod_out.get((i,s),0)*m.E[i,t-m.tau[i]]
                        for i in m.I if t-m.tau[i]>=1)
            expr += sum(int_out.get((i,s),0)*m.E[i,t-m.tau[i]]
                        for i in m.I if t-m.tau[i]>=1)
            # negative demand reduces inventory, slack adds back unmet
            expr += m.demand[s,t] + m.Sl[s,t]
            m.Balance.add(expr == 0)

    # 8b) batch‐size / start coupling
    m.BatchLB = po.ConstraintList()
    m.BatchUB = po.ConstraintList()
    for i in m.I:
        for t in m.T:
            m.BatchLB.add(m.E[i,t] >= m.Vmin[i] * m.N[i,t])
            m.BatchUB.add(m.E[i,t] <= m.Vmax[i] * m.N[i,t])

    # 8c) equipment availability
    m.EquipAvail = po.ConstraintList()
    for j in m.J:
        for t in m.T:
            cons = sum(m.N[i,t]
                       for i in m.I
                       if j in tasks[i]['equipments'].values())
            ret  = sum(m.N[i,t-m.tau[i]]
                       for i in m.I
                       if t-m.tau[i]>=1 and j in tasks[i]['equipments'].values())
            m.EquipAvail.add(
                m.X[j,t] == m.X[j,t-1] - cons + ret
            )

    # 8d) utility use
    m.UtilBal = po.ConstraintList()
    for u in m.U:
        users = [i for i in m.I if u in tasks[i]['utilities']]
        rates = {i: tasks[i]['utilities'][u] for i in users}
        for t in m.T:
            m.UtilBal.add(
                m.F[u,t] == sum(rates[i]*m.E[i,t] for i in users) * m.Ucost[u,t]
            )

    m.SalesCap = po.ConstraintList()
    for p in P_dict:
        for t in m.T:
            # sum of all positive nu for producing p at time t
            prod_capacity = sum(
                prod_out.get((i,p), 0.0) * m.E[i, t-m.tau[i]]
                for i in m.I if t-m.tau[i] >= 1
            )
            m.SalesCap.add( m.Ss[p,t] <= prod_capacity )


    return m#, R_dict, P_dict, IM_dict, J_dict

def optimal_simulation(env, solver, tee: bool = False, raise_on_infeasible: bool = True):
    """
    Solve the RTN Pyomo model and return the optimal action sequence for the whole horizon.

    Parameters
    ----------
    env : RTNEnv
        An instantiated environment (must have env.config_file, env.T, env.task_names,
        env.min_batch, env.max_batch).
    solver : str or pyomo.opt.base.solvers.OptSolver
        Either a solver name (e.g. "gurobi", "cbc", "glpk") or a Pyomo solver instance.
    tee : bool
        If True, prints solver output.
    raise_on_infeasible : bool
        If True, raise an error when the model is not solved to optimality.

    Returns
    -------
    raw_actions : np.ndarray, shape (T, num_tasks)
        Actions in [-1, 1] suitable to pass directly into env.step(action).
        (If a task is not run at time t, raw action is 0.0.)
    batch_actions : np.ndarray, shape (T, num_tasks)
        The corresponding optimal batch sizes E[i,t] (what the optimizer chose).
    results : pyomo.opt.results.SolverResults
        The solver results object.
    """
    horizon = int(env.T)
    m = build_optimisation_model(env.config_file, horizon)

    opt = solver if hasattr(solver, "solve") else SolverFactory(str(solver))
    results = opt.solve(m, tee=tee)

    term = results.solver.termination_condition
    ok = term in (TerminationCondition.optimal, TerminationCondition.locallyOptimal)

    if not ok and raise_on_infeasible:
        raise RuntimeError(
            f"Optimization did not solve to optimality. Termination condition: {term}"
        )

    # 2) Establish a consistent task order that matches the env action vector
    # env.task_names is built from tasks_dict.keys() in RTNEnv.__init__ (dict order preserved).
    task_names = list(env.task_names)
    num_tasks = len(task_names)

    # 3) Extract optimal batch sizes from Pyomo: E[i,t]
    # IMPORTANT: m.I is a Pyomo Set; to avoid ordering issues, we index by task name explicitly.
    batch_actions = np.zeros((horizon, num_tasks), dtype=np.float32)

    for t in range(1, horizon + 1):
        for k, task in enumerate(task_names):
            # If model didn't include task for some reason, keep 0
            if task not in m.I:
                batch_actions[t - 1, k] = 0.0
                continue

            e_val = po.value(m.E[task, t])
            if e_val is None:
                e_val = 0.0

            # Numerically clean tiny values
            if abs(e_val) < 1e-8:
                e_val = 0.0

            batch_actions[t - 1, k] = float(e_val)

    # 4) Convert batch sizes to raw actions in [-1,1] so env scaling reproduces them:
    # scaled = 0.5*(raw + 1)*(max-min) + min  =>  raw = 2*(scaled-min)/(max-min) - 1
    min_b = np.asarray(env.min_batch, dtype=np.float32)
    max_b = np.asarray(env.max_batch, dtype=np.float32)
    denom = (max_b - min_b)

    raw_actions = np.zeros_like(batch_actions, dtype=np.float32)

    for t in range(horizon):
        for k in range(num_tasks):
            b = batch_actions[t, k]

            # Your env treats abs(raw) <= 1e-3 as "task not initiated".
            # So if optimizer says batch=0, set raw=0 (this bypasses scaling).
            if b <= 0.0:
                raw_actions[t, k] = 0.0
                continue

            # Guard against zero range (shouldn't happen, but safe)
            if denom[k] <= 1e-12:
                raw_actions[t, k] = 0.0
                continue

            r = 2.0 * (b - min_b[k]) / denom[k] - 1.0
            # Clip to [-1,1] to be safe w.r.t. numerical issues
            raw_actions[t, k] = float(np.clip(r, -1.0, 1.0))

            # Extra safety: avoid accidentally being treated as "inactive"
            if abs(raw_actions[t, k]) <= 1e-3:
                # If batch is positive but raw is near zero due to min/max weirdness,
                # nudge slightly away from 0.
                raw_actions[t, k] = 0.01 if raw_actions[t, k] >= 0 else -0.01

    return raw_actions