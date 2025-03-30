import pyomo.environ as po

def init_stn_model(args, net, resources_and_task, demand_dict, supply_dict, utility_cost):
    S, T, E, U = resources_and_task  # States, Tasks, Equipments, Utilities
    STN = po.ConcreteModel()

    # Sets
    STN.TASKS = po.Set(initialize=list(T.keys()))
    STN.STATES = po.Set(initialize=list(S.keys()))
    STN.EQUIPMENTS = po.Set(initialize=list(E.keys()))
    STN.UTILITIES = po.Set(initialize=list(U.keys()))

    STN.T0 = po.Set(initialize=range(args.horizon + 1))
    STN.T = po.Set(initialize=range(1, args.horizon + 1))

    max_tau = max(T[i]['tau'] for i in T)
    STN.tau = po.Param(STN.TASKS, initialize={i: T[i]['tau'] for i in T})

    # State-task connectivity
    STN.InputStates = po.Set(STN.TASKS, initialize={i: list(T[i]['inputs'].keys()) for i in T})
    STN.OutputStates = po.Set(STN.TASKS, initialize={i: list(T[i]['outputs'].keys()) for i in T})
    STN.TasksFromState = po.Set(STN.STATES, initialize={
        s: [i for i in T if s in T[i]['inputs']] for s in S
    })
    STN.TasksToState = po.Set(STN.STATES, initialize={
        s: [i for i in T if s in T[i]['outputs']] for s in S
    })

    # Parameters
    STN.X0 = po.Param(STN.STATES, initialize={s: S[s]['X0'] for s in S})
    STN.Xmin = po.Param(STN.STATES, initialize={s: S[s]['Xmin'] for s in S})
    STN.Xmax = po.Param(STN.STATES, initialize={s: S[s]['Xmax'] for s in S})
    STN.Vmin = po.Param(STN.TASKS, initialize={i: T[i]['Vmin'] for i in T})
    STN.Vmax = po.Param(STN.TASKS, initialize={i: T[i]['Vmax'] for i in T})

    # Utility usage factors (per task per utility)
    STN.Uf = po.Param(
        STN.TASKS,
        STN.UTILITIES,
        initialize={(i, u): T[i]['utilities'].get(u, 0.0) for i in T for u in U},
        default=0.0
    )

    # Demand
    STN.Demand = po.Param(STN.STATES, STN.T, initialize={
        (s, t): demand_dict.get(s, {}).get(t, 0.0) for s in S for t in STN.T
    }, default=0.0)

    # Variables
    STN.X = po.Var(STN.STATES, STN.T0, domain=po.NonNegativeReals)   # Inventory
    STN.E = po.Var(STN.TASKS, STN.T, domain=po.NonNegativeReals)     # Batch size
    STN.N = po.Var(STN.TASKS, STN.T, domain=po.Binary)               # Task active
    STN.F = po.Var(STN.UTILITIES, STN.T, domain=po.NonNegativeReals) # Utility cost

    # Fix initial inventory
    for s in S:
        STN.X[s, 0].fix(S[s]['X0'])

    # Objective: minimize total utility cost
    STN.obj = po.Objective(
        expr=sum(STN.F[u, t] for u in STN.UTILITIES for t in STN.T),
        sense=po.minimize
    )

    # Constraints
    STN.MaterialBalance = po.ConstraintList()
    STN.UtilityBalance = po.ConstraintList()
    STN.BatchBounds = po.ConstraintList()
    STN.InventoryBounds = po.ConstraintList()

    for t in STN.T:
        for s in STN.STATES:
            inflow = sum(
                T[i]['outputs'][s] * STN.E[i, t - STN.tau[i]]
                for i in STN.TasksToState[s] if t - STN.tau[i] >= 1
            ) if s in STN.TasksToState else 0

            outflow = sum(
                -T[i]['inputs'][s] * STN.E[i, t]
                for i in STN.TasksFromState[s]
            ) if s in STN.TasksFromState else 0

            STN.MaterialBalance.add(
                STN.X[s, t] == STN.X[s, t - 1] + inflow + outflow - STN.Demand[s, t]
            )

            STN.InventoryBounds.add(STN.Xmin[s] <= STN.X[s, t])
            STN.InventoryBounds.add(STN.X[s, t] <= STN.Xmax[s])

        for i in STN.TASKS:
            STN.BatchBounds.add(STN.E[i, t] >= STN.Vmin[i] * STN.N[i, t])
            STN.BatchBounds.add(STN.E[i, t] <= STN.Vmax[i] * STN.N[i, t])

        for u in STN.UTILITIES:
            cost_t = utility_cost[u]['cost'][t - 1]
            STN.UtilityBalance.add(
                STN.F[u, t] == cost_t * sum(STN.Uf[i, u] * STN.E[i, t] for i in STN.TASKS)
            )

    return STN
