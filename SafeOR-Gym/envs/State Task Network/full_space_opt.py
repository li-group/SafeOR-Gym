import json
import numpy as np
import pyomo.environ as po
from pyomo.opt import SolverFactory, TerminationCondition

def build_optimization_model(config_file: str, horizon: int) -> po.ConcreteModel:
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
    P_list = list(P_dict)
    J_list = list(J_dict)
    U_list = list(U_costs)

    # 3) ZERO-BASED DEMAND & UTILITY SERIES
    demand = {
        p: [demand_raw[p].get(str(t), 0.0) for t in range(1, horizon+1)]
        for p in demand_raw
    }
    util_cost = {
        u: [U_costs[u].get(str(t), 0.0) for t in range(1, horizon+1)]
        for u in U_costs
    }

    # 4) PRECOMPUTE TASK DURATIONS τ_i
    tau_i = { i: next(iter(attr['p'].values())) for i, attr in tasks.items() }

    # 5) STOICHIOMETRIC COEFFS
    raw_cons = { (i,r): -attr['raw_dist'][r]
                 for i,attr in tasks.items() for r in attr['raw_dist'] }
    int_cons = { (i,r): -attr['int_react_dist'][r]
                 for i,attr in tasks.items() for r in attr['int_react_dist'] }
    prod_out = { (i,r):  attr['prod_dist'][r]
                 for i,attr in tasks.items() for r in attr['prod_dist'] }
    int_out  = { (i,r):  attr['int_prod_dist'][r]
                 for i,attr in tasks.items() for r in attr['int_prod_dist'] }

    # 6) BUILD MODEL & SETS
    m = po.ConcreteModel()
    m.I = po.Set(initialize=tasks.keys())    # tasks
    m.S = po.Set(initialize=S_list)          # all states
    m.P = po.Set(initialize=P_list)          # product states
    m.J = po.Set(initialize=J_list)          # equipment
    m.U = po.Set(initialize=U_list)          # utilities

    # combined sets
    m.RJ  = po.Set(initialize=S_list+J_list)
    m.RJU = po.Set(initialize=S_list+J_list+U_list)

    # time sets
    m.T0 = po.RangeSet(0, horizon)
    m.T  = po.RangeSet(1, horizon)

    # 7) PARAMETERS
    # 7.1) processing time
    m.tau = po.Param(m.I, initialize=tau_i)

    # 7.2) batch bounds (single-equipment per task)
    def _vmin(m,i):
        eq = next(iter(tasks[i]['equipments'].values()))
        return tasks[i]['Vmin'][eq]
    def _vmax(m,i):
        eq = next(iter(tasks[i]['equipments'].values()))
        return tasks[i]['Vmax'][eq]

    m.Vmin = po.Param(m.I, initialize=_vmin)
    m.Vmax = po.Param(m.I, initialize=_vmax)

    # 7.3) inventories
    def _X0(m,r):
        if   r in R_dict:  return R_dict[r]['X0']
        elif r in IM_dict: return IM_dict[r]['X0']
        elif r in P_dict:  return P_dict[r]['X0']
        else:              return J_dict[r]['X0']
    m.X0   = po.Param(m.RJ, initialize=_X0)
    m.Xmin = po.Param(m.RJ, initialize={
        r:(R_dict.get(r, IM_dict.get(r, P_dict.get(r, J_dict.get(r)))))
           ['Xmin'] for r in m.RJ
    })
    m.Xmax = po.Param(m.RJ, initialize={
        r:(R_dict.get(r, IM_dict.get(r, P_dict.get(r, J_dict.get(r)))))
           ['Xmax'] for r in m.RJ
    })

    # 7.4) cost/price
    m.Cost    = po.Param(m.S, initialize={r:R_dict.get(r,{}).get('cost',0.0) for r in m.S})
    m.InvCost = po.Param(m.S, initialize={
        r:(R_dict.get(r, IM_dict.get(r, P_dict.get(r, {}))))
           .get('inventory_cost',0.0) for r in m.S
    })
    m.Price   = po.Param(m.S, initialize={r:P_dict.get(r,{}).get('cost',0.0)   for r in m.S})

    # 7.5) time-series
    m.demand = po.Param(m.S, m.T, initialize={
        (p,t): demand.get(p,[0.0]*horizon)[t-1] if p in P_dict else 0.0
        for p in m.S for t in m.T
    })
    m.Ucost  = po.Param(m.U, m.T, initialize={
        (u,t): util_cost[u][t-1] for u in m.U for t in m.T
    })

    # 8) VARIABLES
    m.X  = po.Var(m.RJ, m.T0, domain=po.NonNegativeReals)
    m.N  = po.Var(m.I,  m.T,  domain=po.Binary)
    m.E  = po.Var(m.I,  m.T,  domain=po.NonNegativeReals)
    m.F  = po.Var(m.U,  m.T,  domain=po.NonNegativeReals)
    m.Sl = po.Var(m.P,  m.T,  domain=po.NonNegativeReals)  # slack only for products
    m.Ss = po.Var(m.P,  m.T,  domain=po.NonNegativeReals)  # sales only for products

    # fix initial stocks
    for r in m.RJ:
        m.X[r,0].fix(m.X0[r])

    # 9) OBJECTIVE
    def _obj(m):
        rev   = sum(m.Ss[p,t] * m.Price[p] for p in m.P for t in m.T)
        pen   = 1.5 * sum(m.Sl[p,t] * m.Price[p] for p in m.P for t in m.T)
        util  = sum(m.F[u,t] for u in m.U for t in m.T)
        slack = sum(m.Sl[p,t] * m.Cost[p] for p in m.P for t in m.T)
        return rev - util - pen + slack

    m.obj = po.Objective(rule=_obj, sense=po.maximize)

    # 10) CONSTRAINTS

    # 10a) material-balance
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
            if s in m.P:
                expr += m.demand[s,t] + m.Sl[s,t] #+ m.Ss[s,t]
            m.Balance.add(expr == 0)

    # 10b) batch-size / start coupling
    m.BatchLB = po.ConstraintList()
    m.BatchUB = po.ConstraintList()
    for i in m.I:
        for t in m.T:
            m.BatchLB.add(m.E[i,t] >= m.Vmin[i] * m.N[i,t])
            m.BatchUB.add(m.E[i,t] <= m.Vmax[i] * m.N[i,t])

    # 10c) equipment availability
    m.EquipAvail = po.ConstraintList()
    for j in m.J:
        for t in m.T:
            cons = sum(m.N[i,t]
                       for i in m.I
                       if j in tasks[i]['equipments'].values())
            ret  = sum(m.N[i,t-m.tau[i]]
                       for i in m.I
                       if t-m.tau[i]>=1 and j in tasks[i]['equipments'].values())
            m.EquipAvail.add(m.X[j,t] == m.X[j,t-1] - cons + ret)

    # 10d) utility-use balance
    m.UtilBal = po.ConstraintList()
    for u in m.U:
        users = [i for i in m.I if u in tasks[i]['utilities']]
        rates = {i:tasks[i]['utilities'][u] for i in users}
        for t in m.T:
            m.UtilBal.add(
                m.F[u,t] == sum(rates[i]*m.E[i,t] for i in users) * m.Ucost[u,t]
            )

    # 10e) sales-capacity
    m.SalesCap = po.ConstraintList()
    for p in m.P:
        for t in m.T:
            cap = sum(prod_out.get((i,p),0)*m.E[i,t-m.tau[i]]
                      for i in m.I if t-m.tau[i]>=1)
            m.SalesCap.add(m.Ss[p,t] <= cap)

    
    return m #, R_dict, P_dict, IM_dict, J_dict

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
            if e_val is None or abs(e_val) < 1e-8:
                e_val = 0.0

            eq_idx = task_to_eq_idx[task]
            batch_actions_mat[t - 1, i, eq_idx] = float(e_val)

    # Convert batch sizes -> raw actions in [-1,1] with env's per-(task,equip) min/max.
    min_b = np.asarray(env.min_batch, dtype=np.float32)  # (num_tasks, num_eq)
    max_b = np.asarray(env.max_batch, dtype=np.float32)  # (num_tasks, num_eq)
    denom = (max_b - min_b)

    raw_actions_mat = np.zeros_like(batch_actions_mat, dtype=np.float32)

    for t in range(horizon):
        for i in range(num_tasks):
            for e in range(num_eq):
                b = batch_actions_mat[t, i, e]

                # If optimizer didn't select this (task,equip), keep it inactive:
                if b <= 0.0:
                    raw_actions_mat[t, i, e] = 0.0
                    continue

                # Safety: avoid division by zero if min=max (shouldn't happen but protect anyway)
                if denom[i, e] <= 1e-12:
                    raw_actions_mat[t, i, e] = 0.0
                    continue

                r = 2.0 * (b - min_b[i, e]) / denom[i, e] - 1.0
                r = float(np.clip(r, -1.0, 1.0))

                # Avoid being interpreted as "inactive" by sanitize_action's threshold
                if abs(r) <= 1e-3:
                    r = 0.01 if r >= 0 else -0.01

                raw_actions_mat[t, i, e] = r

    # Flatten in the same order as unflatten_action_vector expects.
    # Given your use: unflatten_action_vector(action, num_tasks, len(equipments)),
    # the natural assumption is row-major reshape: (num_tasks*num_eq,) -> (num_tasks, num_eq)
    raw_actions_flat = raw_actions_mat.reshape(horizon, num_tasks * num_eq)

    return raw_actions_flat