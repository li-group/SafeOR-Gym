import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
import pprint


def create_model(env, demand, bigm=10000):
    # Model definition

    # Use env.T for time horizon
    T = env.T

    dem_param = {
        (t, (j, k)): float(demand[(j, k)][t - 1])
        for (j, k) in env.retailer_routes
        for t in range(1, T + 1)
    }

    model = pyo.ConcreteModel()

    # Sets definitions
    model.all_routes = pyo.Set(initialize=list(env.unit_price.keys()))
    model.Markets = pyo.Set(initialize=range(0, env.num_markets))
    model.Retailers = pyo.Set(
        initialize=range(env.num_markets, env.num_markets + env.num_retailers)
    )
    model.Distributors = pyo.Set(
        initialize=range(
            env.num_markets + env.num_retailers,
            env.num_markets + env.num_retailers + env.num_distributors,
        )
    )
    model.Producers = pyo.Set(
        initialize=range(
            env.num_markets + env.num_retailers + env.num_distributors,
            env.num_markets
            + env.num_retailers
            + env.num_distributors
            + env.num_producers,
        )
    )
    model.RawDistributors = pyo.Set(
        initialize=range(
            env.num_markets
            + env.num_retailers
            + env.num_distributors
            + env.num_producers,
            env.num_markets
            + env.num_retailers
            + env.num_distributors
            + env.num_producers
            + env.num_raw_distributors,
        )
    )
    model.Time_periods = pyo.RangeSet(1, T)
    model.Time_periods2 = pyo.RangeSet(2, T)  # Excluding the first time period
    model.Main = pyo.Set(
        initialize=range(
            env.num_markets,
            env.num_distributors + env.num_producers + env.num_retailers + 1,
        )
    )
    model.Main2 = pyo.Set(
        initialize=range(
            env.num_markets,
            env.num_distributors
            + env.num_producers
            + env.num_retailers
            + env.num_raw_distributors
            + 1,
        )
    )
    model.Reordering_routes = pyo.Set(
        initialize=[key for key in env.unit_price if key[0] not in model.Retailers]
    )
    model.Ret_market_routes = pyo.Set(
        initialize=[key for key in env.unit_price if key[0] in model.Retailers]
    )

    # Variable definitions
    def reordering_route_capacity_bounds(model, t, i, j):
        return (0, env.reordering_route_capacity.get((i, j)))

    def inv_capacity_bounds(model, t, r):
        return (0, env.inv_capacity.get((r)))

    model.a = pyo.Var(
        model.Time_periods,
        model.Reordering_routes,
        domain=pyo.NonNegativeReals,
        bounds=reordering_route_capacity_bounds,
    )
    model.s_d = pyo.Var(
        model.Time_periods2, model.Ret_market_routes, domain=pyo.NonNegativeReals
    )
    model.s_o = pyo.Var(
        model.Time_periods,
        model.Main,
        domain=pyo.NonNegativeReals,
        bounds=inv_capacity_bounds,
    )
    model.s_p = pyo.Var(
        model.Time_periods, model.Reordering_routes, domain=pyo.NonNegativeReals
    )
    model.u = pyo.Var(
        model.Time_periods2, model.Ret_market_routes, domain=pyo.NonNegativeReals
    )
    model.r = pyo.Var(model.Time_periods, model.Main, domain=pyo.Reals)
    model.a_p = pyo.Var(
        model.Time_periods, model.Reordering_routes, domain=pyo.Reals
    )

    model.sr = pyo.Var(model.Time_periods, model.Main, domain=pyo.Reals)
    model.pc = pyo.Var(model.Time_periods, model.Main, domain=pyo.Reals)
    model.oc = pyo.Var(model.Time_periods, model.Producers, domain=pyo.Reals)
    model.up = pyo.Var(model.Time_periods2, model.Retailers, domain=pyo.Reals)
    model.hc = pyo.Var(model.Time_periods, model.Main, domain=pyo.Reals)

    # Parameter definitions
    model.p = pyo.Param(model.all_routes, initialize=env.unit_price)
    model.jin = pyo.Param(model.Main, initialize=env.j_in, within=pyo.Any)
    model.jout = pyo.Param(
        model.Main | model.RawDistributors, initialize=env.j_out, within=pyo.Any
    )
    model.o = pyo.Param(model.Producers, initialize=env.operating_cost)
    model.v = pyo.Param(model.Producers, initialize=env.production_yield)
    model.b = pyo.Param(
        model.Ret_market_routes, initialize=env.unfulfilled_utility_penalty
    )
    model.h = pyo.Param(model.Main, initialize=env.inventory_holding_cost)
    model.g = pyo.Param(model.Reordering_routes, initialize=env.material_holding_cost)
    model.lead_times = pyo.Param(model.Reordering_routes, initialize=env.lead_times)
    model.d = pyo.Param(
        model.Time_periods, model.Ret_market_routes, initialize=dem_param, mutable=True
    )
    model.init_inv = pyo.Param(model.Main, initialize=env.initial_inv, mutable=True)

    # Objective function
    model.obj = pyo.Objective(
        expr=(
            sum(model.r[t, j] for t in model.Time_periods for j in model.Main)
        ),
        sense=pyo.maximize,
    )

    # Constraints
    def profit_rule(model, t, j):
        if j in model.Producers:
            return (
                model.r[t, j]
                == model.sr[t, j] - model.pc[t, j] - model.oc[t, j] - model.hc[t, j]
            )
        elif j in model.Retailers:
            if t == 1:
                return model.r[t, j] == -model.pc[t, j] - model.hc[t, j]
            else:
                return (
                    model.r[t, j]
                    == model.sr[t, j]
                    - model.pc[t, j]
                    - model.up[t, j]
                    - model.hc[t, j]
                )
        else:
            return (
                model.r[t, j]
                == model.sr[t, j] - model.pc[t, j] - model.hc[t, j]
            )

    model.cons1 = pyo.Constraint(model.Time_periods, model.Main, rule=profit_rule)

    def sr_rule(model, t, j):
        if j in model.Producers or j in model.Distributors:
            return model.sr[t, j] == sum(
                model.p[(j, k)] * model.a[(t, j, k)] for k in model.jout[j]
            )
        elif j in model.Retailers and t >= 2:
            return model.sr[t, j] == sum(
                model.p[(j, k)] * model.s_d[(t, j, k)] for k in model.jout[j]
            )
        else:
            return pyo.Constraint.Skip

    model.cons2 = pyo.Constraint(model.Time_periods, model.Main, rule=sr_rule)

    def pc_rule(model, t, j):
        return model.pc[t, j] == sum(
            model.p[k, j] * model.a[t, k, j] for k in model.jin[j]
        )

    model.cons3 = pyo.Constraint(model.Time_periods, model.Main, rule=pc_rule)

    def oc_rule(model, t, j):
        return model.oc[t, j] == (model.o[j] / model.v[j]) * sum(
            model.a[t, j, k] for k in model.jout[j]
        )

    model.cons4 = pyo.Constraint(model.Time_periods, model.Producers, rule=oc_rule)

    def up_rule(model, t, j):
        return model.up[t, j] == sum(
            model.b[j, k] * model.u[t, j, k] for k in model.jout[j]
        )

    model.cons5 = pyo.Constraint(model.Time_periods2, model.Retailers, rule=up_rule)

    def hc_rule(model, t, j):
        return model.hc[t, j] == model.h[j] * model.s_o[t, j] + sum(
            model.g[k, j] * model.s_p[t, k, j] for k in model.jin[j]
        )

    model.cons6 = pyo.Constraint(model.Time_periods, model.Main, rule=hc_rule)

    def so_rule(model, t, j):
        if j in model.Producers:
            if t == 1:
                return (
                    model.s_o[t, j]
                    == model.init_inv[j]
                    + sum(model.a_p[t, k, j] for k in model.jin[j])
                    - (1 / model.v[j])
                    * sum(model.a[t, j, k] for k in model.jout[j])
                )
            else:
                return (
                    model.s_o[t, j]
                    == model.s_o[t - 1, j]
                    + sum(model.a_p[t, k, j] for k in model.jin[j])
                    - (1 / model.v[j])
                    * sum(model.a[t, j, k] for k in model.jout[j])
                )
        elif j in model.Distributors:
            if t == 1:
                return (
                    model.s_o[t, j]
                    == model.init_inv[j]
                    + sum(model.a_p[t, k, j] for k in model.jin[j])
                    - sum(model.a[t, j, k] for k in model.jout[j])
                )
            else:
                return (
                    model.s_o[t, j]
                    == model.s_o[t - 1, j]
                    + sum(model.a_p[t, k, j] for k in model.jin[j])
                    - sum(model.a[t, j, k] for k in model.jout[j])
                )
        else:
            if t == 1:
                return (
                    model.s_o[t, j]
                    == model.init_inv[j]
                    + sum(model.a_p[t, k, j] for k in model.jin[j])
                )
            else:
                return (
                    model.s_o[t, j]
                    == model.s_o[t - 1, j]
                    + sum(model.a_p[t, k, j] for k in model.jin[j])
                    - sum(model.s_d[t, j, k] for k in model.jout[j])
                )

    model.cons7 = pyo.Constraint(model.Time_periods, model.Main, rule=so_rule)

    model.cons8 = pyo.ConstraintList()

    def ap_rule(model, t, j):
        for k in model.jin[j]:
            if t - model.lead_times[k, j] < 1:
                model.cons8.add(model.a_p[t, k, j] == 0)
            elif t - model.lead_times[k, j] == 1:
                model.cons8.add(model.a_p[t, k, j] == model.a[1, k, j])
            else:
                model.cons8.add(
                    model.a_p[t, k, j] == model.a[t - model.lead_times[k, j], k, j]
                )

    model.cons9 = pyo.ConstraintList()

    def sp_rule(model, t, j):
        for k in model.jin[j]:
            if t == 1:
                model.cons9.add(
                    model.s_p[t, k, j] == -model.a_p[t, k, j] + model.a[t, k, j]
                )
            else:
                model.cons9.add(
                    model.s_p[t, k, j]
                    == model.s_p[t - 1, k, j] - model.a_p[t, k, j] + model.a[t, k, j]
                )

    for t in model.Time_periods:
        for j in model.Main:
            ap_rule(model, t, j)
            sp_rule(model, t, j)

    def inv_req2(model, t, j):
        if j in model.Producers:
            return (
                sum(model.a[t, j, k] for k in model.jout[j])
                <= model.s_o[t, j] * model.v[j]
            )
        elif j in model.Distributors:
            return (
                sum(model.a[t, j, k] for k in model.jout[j]) <= model.s_o[t, j]
            )
        else:
            return pyo.Constraint.Skip

    model.cons11 = pyo.Constraint(model.Time_periods, model.Main, rule=inv_req2)

    def market_sales1(model, t, j):
        return (
            sum(model.s_d[t, j, k] for k in model.jout[j]) <= model.s_o[t - 1, j]
        )

    model.cons12 = pyo.Constraint(
        model.Time_periods2, model.Retailers, rule=market_sales1
    )

    model.cons13 = pyo.ConstraintList()

    def market_sales2(model, t, j):
        for k in model.jout[j]:
            if t == 2:
                model.cons13.add(model.s_d[t, j, k] <= model.d[t - 1, j, k])
            else:
                model.cons13.add(
                    model.s_d[t, j, k] <= model.d[t - 1, j, k] + model.u[t - 1, j, k]
                )

    model.cons14 = pyo.ConstraintList()

    def unf_rule(model, t, j):
        for k in model.jout[j]:
            if t == 2:
                model.cons14.add(
                    model.u[t, j, k] == model.d[t - 1, j, k] - model.s_d[t, j, k]
                )
            else:
                model.cons14.add(
                    model.u[t, j, k]
                    == model.u[t - 1, j, k]
                    + model.d[t - 1, j, k]
                    - model.s_d[t, j, k]
                )

    for t in model.Time_periods2:
        for j in model.Retailers:
            market_sales2(model, t, j)
            unf_rule(model, t, j)

    return model


def optimal_simulation(
    env,
    solver="gurobi",
    tee: bool = False,
    raise_on_infeasible: bool = True,
):
    """
    Solve the original formulation and return scaled [-1,1] actions
    for the env's step() function.
    """
    T = env.T
    reordering_routes = env.reordering_routes
    num_routes = len(reordering_routes)

    demand = env.demand
    model = create_model(env, demand)

    opt = solver if hasattr(solver, "solve") else SolverFactory(str(solver))
    results = opt.solve(model, tee=tee)

    term = results.solver.termination_condition
    ok = term in (TerminationCondition.optimal, TerminationCondition.locallyOptimal)
    if (not ok) and raise_on_infeasible:
        raise RuntimeError(
            f"Optimization did not solve to optimality. Termination condition: {term}"
        )

    # Extract optimal reorder quantities from model
    # model.a[t, (i,j)] — 1-indexed time
    optimal_reorders = np.zeros((T, num_routes), dtype=np.float32)
    for t in range(1, T + 1):
        t_idx = t - 1
        for route_idx, rt in enumerate(reordering_routes):
            val = pyo.value(model.a[t, rt])
            optimal_reorders[t_idx, route_idx] = 0.0 if val is None else float(val)

    # Inverse of env scaling:
    #   env: action_dict[rt] = (raw_action[i] + 1) * 0.5 * capacity[rt]
    #   inverse: raw_action[i] = 2 * (order / capacity) - 1
    raw_actions = np.zeros((T, num_routes), dtype=np.float32)
    for route_idx, rt in enumerate(reordering_routes):
        capacity = env.reordering_route_capacity[rt]
        if capacity > 1e-12:
            raw_actions[:, route_idx] = (
                2.0 * optimal_reorders[:, route_idx] / capacity - 1.0
            )
        else:
            raw_actions[:, route_idx] = 0.0

    raw_actions = np.clip(raw_actions, -1.0, 1.0)

    print(f"Optimal objective: {pyo.value(model.obj):.4f}")
    print(f"Time periods: {T}, Routes: {num_routes}")

    return raw_actions


def get_optimal_value(env, solver="gurobi", tee: bool = False):
    """
    Solve the optimization and return only the optimal objective value.
    """
    demand = env.demand
    model = create_model(env, demand)

    opt = solver if hasattr(solver, "solve") else SolverFactory(str(solver))
    results = opt.solve(model, tee=tee)

    term = results.solver.termination_condition
    if term not in (TerminationCondition.optimal, TerminationCondition.locallyOptimal):
        return float("-inf")

    return pyo.value(model.obj)


def simulate_with_optimal_policy(env, solver="gurobi", seed=None, tee=False):
    """
    Reset the environment, compute optimal actions, simulate the full episode,
    and compare model objective vs env simulation reward.
    """
    obs, info = env.reset(seed=seed)

    raw_actions = optimal_simulation(env, solver=solver, tee=tee)
    opt_obj = get_optimal_value(env, solver=solver, tee=tee)

    observations = [obs.cpu().numpy() if hasattr(obs, "cpu") else obs]
    rewards = []
    total_reward = 0.0
    total_cost = 0.0

    for t in range(env.T):
        obs, reward, done, truncated, info = env.step(raw_actions[t])

        reward_val = reward.item() if hasattr(reward, "item") else float(reward)
        rewards.append(reward_val)
        total_reward += reward_val
        total_cost += env.cost

        obs_np = obs.cpu().numpy() if hasattr(obs, "cpu") else obs
        observations.append(obs_np)

        if done:
            break

    print(f"\n--- Comparison ---")
    print(f"Model objective:      {opt_obj:.4f}")
    print(f"Env simulation total: {total_reward:.4f}")
    print(f"Gap:                  {abs(opt_obj - total_reward):.4f}")

    return {
        "opt_obj": opt_obj,
        "total_reward": total_reward,
        "total_cost": total_cost,
        "actions": raw_actions,
        "rewards": rewards,
        "observations": observations,
    }