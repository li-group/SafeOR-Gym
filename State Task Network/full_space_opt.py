import json
import pyomo.environ as po

def build_stn_model(config_file: str, horizon: int) -> po.ConcreteModel:
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

    
    return m, R_dict, P_dict, IM_dict, J_dict


model, R_dict, P_dict, IM_dict, J_dict = build_stn_model("hard_environment_data.json", horizon = 30)
#model.pprint()
solver = po.SolverFactory('gurobi')
results = solver.solve(model, tee = True)

rev   = sum(po.value(model.Ss[p,t]) * po.value(model.Price[p]) for p in P_dict for t in model.T)
pen   = 1.5 * sum(po.value(model.Sl[p,t]) * po.value(model.Price[p]) for p in P_dict for t in model.T)
util  = sum(po.value(model.F[u,t]) for u in model.U for t in model.T)
slack = sum(po.value(model.Sl[r,t]) * po.value(model.Cost[r]) for r in model.P for t in model.T)

print(f'Revenue : {rev}, util : {util}, pen : {pen}, slack : {slack}')