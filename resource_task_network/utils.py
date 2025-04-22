import pyomo.environ as po
import networkx as nx

def init_model(args, net, resources_and_task, demand_dict, supply_dict, utility_cost):
    R, P, IM, I, J, U = resources_and_task

    RTN = po.ConcreteModel()

    set_variables(args, net, RTN, resources_and_task)
    param_variables(net, RTN, resources_and_task, demand_dict, supply_dict)
    var_variables(RTN)
    define_objective(RTN, R, P)
    add_constraints(net, RTN, R, P, I, utility_cost)

    return RTN


def set_variables(args, net, RTN, resources_and_task):
    R, P, IM, I, J, U = resources_and_task
    max_tau = max([net.nodes[i]['tau'] for i in I])
    resources = list(R.keys()) + list(P.keys()) + list(IM.keys())
    equipments = list(J.keys())
    utilities = list(U.keys())
    resources_equipments = resources + equipments
    resources_utilities = resources + utilities
    resources_equipments_utilities = resources + equipments + utilities

    RTN.I = po.Set(initialize=list(I.keys()))
    RTN.R = po.Set(initialize=resources)
    RTN.U = po.Set(initialize=utilities)
    RTN.RJ = po.Set(initialize=resources_equipments)
    RTN.RJU = po.Set(initialize=resources_equipments_utilities)

    RTN.T0 = po.Set(initialize=range(args.horizon + 1))
    RTN.T = po.Set(initialize=range(1, args.horizon + 1))

    RTN.Ir = po.Set(RTN.RJ, initialize={r: list(net.successors(r)) + list(net.predecessors(r)) for r in RTN.RJ})
    RTN.Iu = po.Set(RTN.U, initialize={u: list(net.successors(u)) for u in RTN.U})

    idx = [(i, r, theta)
           for i in I
           for r in resources_equipments
           for theta in range(max_tau + 1)
           if r in nx.all_neighbors(net, i) and theta <= net.nodes[i]['tau']]
    RTN.idx = po.Set(dimen=3, initialize=idx)


def param_variables(net, RTN, resources_and_task, demand_dict, supply_dict):
    R, P, IM, I, J, U = resources_and_task
    max_tau = max([net.nodes[i]['tau'] for i in I])
    mu_init = {}
    nu_init = {}

    for idx in RTN.idx:
        try:
            mu_init[idx] = net.edges[(idx[0], idx[1])]['mu'][idx[2]]
            nu_init[idx] = net.edges[(idx[0], idx[1])]['nu'][idx[2]]
        except:
            mu_init[idx] = net.edges[(idx[1], idx[0])]['mu'][idx[2]]
            nu_init[idx] = net.edges[(idx[1], idx[0])]['nu'][idx[2]]

    RTN.mu = po.Param(RTN.idx, initialize=mu_init)
    RTN.nu = po.Param(RTN.idx, initialize=nu_init)

    RTN.tau = po.Param(RTN.I, initialize={i: net.nodes[i]['tau'] for i in RTN.I})
    RTN.Vmax = po.Param(RTN.I, initialize={i: net.nodes[i]['Vmax'] for i in RTN.I})
    RTN.Vmin = po.Param(RTN.I, initialize={i: net.nodes[i]['Vmin'] for i in RTN.I})

    RTN.X0 = po.Param(RTN.RJ, initialize={r: net.nodes[r]['X0'] for r in RTN.RJ})
    RTN.Xmax = po.Param(RTN.RJ, initialize={r: net.nodes[r]['Xmax'] for r in RTN.RJ})
    RTN.Xmin = po.Param(RTN.RJ, initialize={r: net.nodes[r]['Xmin'] for r in RTN.RJ})

    RTN.ResourceCost = po.Param(RTN.R, initialize={r: net.nodes[r]['cost'] for r in RTN.R})
    RTN.InventoryCost = po.Param(RTN.R, initialize={r: net.nodes[r]['inventory_cost'] for r in RTN.R})

    RTN.Uf = po.Param(RTN.U, initialize={u: net.nodes[u]['Uf'] for u in RTN.U})

    RTN.Pi = po.Param(RTN.RJ, RTN.T, initialize={
        (r, t): demand_dict[r][t] if r in P else 0
        for r in RTN.RJ for t in RTN.T
    })


def set_initial_resource_levels(RTN):
    for r in RTN.RJ:
        RTN.X[r, 0].fix(RTN.X0[r])


def var_variables(RTN):
    RTN.X = po.Var(RTN.RJ, RTN.T0, domain=po.NonNegativeReals)
    set_initial_resource_levels(RTN)
    RTN.N = po.Var(RTN.I, RTN.T, domain=po.Binary)
    RTN.E = po.Var(RTN.I, RTN.T, domain=po.NonNegativeReals)
    RTN.F = po.Var(RTN.U, RTN.T, domain=po.NonNegativeReals)
    RTN.Sl = po.Var(RTN.RJ, RTN.T, domain=po.NonNegativeReals)
    RTN.S = po.Var(RTN.R, RTN.T, domain=po.NonNegativeReals)


def define_objective(RTN, R, P):
    sales = sum(RTN.S[r, t] * RTN.ResourceCost[r] for r in RTN.R if r in P for t in RTN.T)
    costs = sum(RTN.Sl[r, t] * RTN.ResourceCost[r] for r in RTN.R if r in R for t in RTN.T)
    inventory_cost = sum(RTN.X[r, t] * RTN.InventoryCost[r] for r in RTN.R for t in RTN.T)
    penalty_cost = sum(1.5 * RTN.Sl[r, t] * RTN.ResourceCost[r] for r in RTN.R if r in P for t in RTN.T)
    cost_util = sum(RTN.F[u, t] for u in RTN.U for t in RTN.T)

    RTN.obj = po.Objective(expr=sales - cost_util + inventory_cost - penalty_cost + costs, sense=po.maximize)


def add_constraints(net, RTN, R, P, I, utility_cost):
    RTN.Balance = po.ConstraintList()
    RTN.Sales = po.ConstraintList()
    RTN.Slacks = po.ConstraintList()
    RTN.ResourceLB = po.ConstraintList()
    RTN.ResourceUB = po.ConstraintList()
    RTN.BatchLB = po.ConstraintList()
    RTN.BatchUB = po.ConstraintList()
    RTN.UCon = po.ConstraintList()

    max_tau = max([net.nodes[i]['tau'] for i in I])

    for t in RTN.T:
        for r in RTN.RJ:
            RTN.Balance.add(
                RTN.X[r, t] == RTN.X[r, t - 1]
                + sum(RTN.mu[i, r, theta] * RTN.N[i, t - theta]
                      + RTN.nu[i, r, theta] * RTN.E[i, t - theta]
                      for i in RTN.Ir[r]
                      for theta in range(max_tau + 1)
                      if theta <= RTN.tau[i] and t - theta >= 1)
                + RTN.Pi[r, t] + RTN.Sl[r, t]
            )

            if r in P:
                RTN.Sales.add(RTN.S[r, t] == -RTN.Sl[r, t] - RTN.Pi[r, t])
            elif r in R:
                continue
            else:
                RTN.Slacks.add(RTN.Sl[r, t] == 0)

            RTN.ResourceLB.add(RTN.Xmin[r] <= RTN.X[r, t])
            RTN.ResourceUB.add(RTN.X[r, t] <= RTN.Xmax[r])

        for u in RTN.U:
            RTN.UCon.add(
                RTN.F[u, t] == utility_cost[u]['cost'][t - 1] * sum(RTN.E[i, t] for i in RTN.Iu[u])
            )

        for i in RTN.I:
            RTN.BatchLB.add(RTN.E[i, t] - RTN.Vmin[i] * RTN.N[i, t] >= 0)
            RTN.BatchUB.add(RTN.E[i, t] <= RTN.Vmax[i] * RTN.N[i, t])