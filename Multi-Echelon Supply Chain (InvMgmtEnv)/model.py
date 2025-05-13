import pyomo.environ as pyo
import pprint

def create_model(env, demand, bigm = 10000):
    # Model definition

    # Use env.T for time horizon
    T = env.T

    dem_param = {
        (t, (j, k)): float(demand[(j, k)][t-1])
        for (j, k) in env.retailer_routes
        for t in range(1, T + 1)
    }

    model = pyo.ConcreteModel()

    # Sets definitions
    model.all_routes = pyo.Set(initialize=list(env.unit_price.keys()))
    model.Markets = pyo.Set(initialize=range(0, env.num_markets))
    model.Retailers = pyo.Set(initialize=range(env.num_markets, env.num_markets + env.num_retailers))
    model.Distributors = pyo.Set(initialize=range(env.num_markets + env.num_retailers, env.num_markets + env.num_retailers + env.num_distributors))
    model.Producers = pyo.Set(initialize=range(env.num_markets + env.num_retailers + env.num_distributors, env.num_markets + env.num_retailers + env.num_distributors + env.num_producers))
    model.RawDistributors = pyo.Set(initialize=range(env.num_markets + env.num_retailers + env.num_distributors + env.num_producers, env.num_markets + env.num_retailers + env.num_distributors + env.num_producers + env.num_raw_distributors))
    model.Time_periods = pyo.RangeSet(1, T)
    model.Time_periods2 = pyo.RangeSet(2, T) # Excluding the first time period
    model.Main = pyo.Set(initialize=range(env.num_markets, env.num_distributors + env.num_producers + env.num_retailers + 1))
    model.Main2 = pyo.Set(initialize=range(env.num_markets, env.num_distributors + env.num_producers + env.num_retailers + env.num_raw_distributors + 1))
    model.Reordering_routes = pyo.Set(initialize=[key for key in env.unit_price if key[0] not in model.Retailers])
    model.Ret_market_routes = pyo.Set(initialize=[key for key in env.unit_price if key[0] in model.Retailers])

    # Variable definitions
    def reordering_route_capacity_bounds(model, t, i, j):
        return (0, env.reordering_route_capacity.get((i,j)))

    def inv_capacity_bounds(model, t, r):
        return (0, env.inv_capacity.get((r)))

    model.a = pyo.Var(model.Time_periods, model.Reordering_routes, domain=pyo.NonNegativeReals, bounds = reordering_route_capacity_bounds) # Reorder quantites req by node 'k' to node 'j' at the beg.. of time period 't'
    model.s_d = pyo.Var(model.Time_periods2, model.Ret_market_routes, domain=pyo.NonNegativeReals) # The amount retailer 'j' sells to market 'k' at time period 't'
    model.s_o = pyo.Var(model.Time_periods, model.Main, domain=pyo.NonNegativeReals, bounds = inv_capacity_bounds) # On-hand inventory at node 'j' prior to realizing demand at time period 't'
    model.s_p = pyo.Var(model.Time_periods, model.Reordering_routes, domain=pyo.NonNegativeReals) # In-transit pipeline inventory b/w 'j' and 'k' prior to demand realization at time peirod 't'
    model.u = pyo.Var(model.Time_periods2, model.Ret_market_routes, domain=pyo.NonNegativeReals) # Unfulfilled demand at retailer 'j' associated with market 'k' in period 't-1'
    model.r = pyo.Var(model.Time_periods, model.Main, domain=pyo.Reals) # Profit at node 'j' at time period 't'
    model.a_p = pyo.Var(model.Time_periods, model.Reordering_routes, domain=pyo.Reals) # Pipeline inventory arriving at node 'j' from node 'k' in period 't'

    model.sr = pyo.Var(model.Time_periods, model.Main, domain= pyo.Reals) # Sales revenue
    model.pc = pyo.Var(model.Time_periods, model.Main, domain = pyo.Reals) # Procrurement costs
    model.oc = pyo.Var(model.Time_periods, model.Producers, domain=pyo.Reals) # Operating costs
    model.up = pyo.Var(model.Time_periods2, model.Retailers, domain = pyo.Reals) # Unfulfilled demand penalties
    model.hc = pyo.Var(model.Time_periods, model.Main, domain = pyo.Reals) # Inventory holding costs

    # Parameter definitions
    model.p = pyo.Param(model.all_routes, initialize=env.unit_price)
    model.jin = pyo.Param(model.Main, initialize=env.j_in, within = pyo.Any)
    model.jout = pyo.Param(model.Main|model.RawDistributors, initialize=env.j_out, within = pyo.Any)
    model.o = pyo.Param(model.Producers, initialize=env.operating_cost)
    model.v = pyo.Param(model.Producers, initialize=env.production_yield)
    model.b = pyo.Param(model.Ret_market_routes, initialize=env.unfulfilled_utility_penalty)
    model.h = pyo.Param(model.Main, initialize=env.inventory_holding_cost)
    model.g = pyo.Param(model.Reordering_routes, initialize=env.material_holding_cost)
    model.lead_times = pyo.Param(model.Reordering_routes, initialize=env.lead_times)
    # model.c = pyo.Param(model.Producers, initialize=env.capacity)
    model.d = pyo.Param(model.Time_periods, model.Ret_market_routes, initialize=dem_param, mutable = True)
    model.init_inv = pyo.Param(model.Main, initialize=env.initial_inv, mutable = True)

    # Objective function
    model.obj = pyo.Objective(
        expr=(
            sum(model.r[t, j] 
                for t in model.Time_periods 
                for j in model.Main 
            )
        ),
        sense=pyo.maximize
    )

    # Constraints
    def profit_rule(model, t, j):
        if j in model.Producers:
            return model.r[t,j] == model.sr[t,j] - model.pc[t,j] - model.oc[t,j] - model.hc[t,j]
        elif j in model.Retailers:
            if t == 1:
                return model.r[t,j] == - model.pc[t,j] - model.hc[t,j]
            else:
                return model.r[t,j] == model.sr[t,j] - model.pc[t,j] - model.up[t,j] - model.hc[t,j]
        else:
            return model.r[t,j] == model.sr[t,j] - model.pc[t,j] - model.hc[t,j]

    model.cons1 = pyo.Constraint(model.Time_periods, model.Main, rule=profit_rule)

    def sr_rule(model, t, j):
        if j in model.Producers or j in model.Distributors:
            return model.sr[t,j] == sum(model.p[(j,k)]*model.a[(t,j,k)] for k in model.jout[j])
        elif j in model.Retailers and t>=2:
            return model.sr[t,j] == sum(model.p[(j,k)]*model.s_d[(t,j,k)] for k in model.jout[j])
        else:
            return pyo.Constraint.Skip

    model.cons2 = pyo.Constraint(model.Time_periods, model.Main, rule=sr_rule)

    def pc_rule(model, t, j):
        return model.pc[t,j] == sum(model.p[k,j]*model.a[t,k,j] for k in model.jin[j]) 

    model.cons3 = pyo.Constraint(model.Time_periods, model.Main, rule=pc_rule)

    def oc_rule(model, t, j):
        return model.oc[t,j] == (model.o[j]/model.v[j])*sum(model.a[t,j,k] for k in model.jout[j])

    model.cons4 = pyo.Constraint(model.Time_periods, model.Producers, rule = oc_rule)

    def up_rule(model, t, j):
        return model.up[t,j] == sum(model.b[j,k]*model.u[t,j,k] for k in model.jout[j])

    model.cons5 = pyo.Constraint(model.Time_periods2, model.Retailers, rule = up_rule)

    def hc_rule(model, t, j):
        return model.hc[t,j] == model.h[j]*model.s_o[t,j] + sum(model.g[k,j]*model.s_p[t,k,j] for k in model.jin[j])

    model.cons6 = pyo.Constraint(model.Time_periods, model.Main, rule = hc_rule)

    def so_rule(model, t, j):
        if j in model.Producers:
            if t ==1:
                return model.s_o[t,j] == model.init_inv[j] + sum(model.a_p[t,k,j] for k in model.jin[j]) - (1/model.v[j])*sum(model.a[t,j,k] for k in model.jout[j])
            else:
                return model.s_o[t,j] == model.s_o[t-1,j] + sum(model.a_p[t,k,j] for k in model.jin[j]) - (1/model.v[j])*sum(model.a[t,j,k] for k in model.jout[j])
        elif j in model.Distributors:
            if t ==1:
                return model.s_o[t,j] == model.init_inv[j] + sum(model.a_p[t,k,j] for k in model.jin[j]) - sum(model.a[t,j,k] for k in model.jout[j])
            else:
                return model.s_o[t,j] == model.s_o[t-1,j] + sum(model.a_p[t,k,j] for k in model.jin[j]) - sum(model.a[t,j,k] for k in model.jout[j])
        else:
            if t ==1:
                return model.s_o[t,j] == model.init_inv[j] + sum(model.a_p[t,k,j] for k in model.jin[j])
            else:
                return model.s_o[t,j] == model.s_o[t-1,j] + sum(model.a_p[t,k,j] for k in model.jin[j]) - sum(model.s_d[t,j,k] for k in model.jout[j])

    model.cons7 = pyo.Constraint(model.Time_periods, model.Main, rule=so_rule)

    model.cons8 = pyo.ConstraintList()

    def ap_rule(model, t, j):
        for k in model.jin[j]:
            if t - model.lead_times[k, j] < 1:
                # Add the constraint directly to the model's ConstraintList
                model.cons8.add(model.a_p[t, k, j] == 0)
            elif t - model.lead_times[k, j] == 1:
                # Add the constraint directly to the model's ConstraintList
                model.cons8.add(model.a_p[t, k, j] == model.a[1, k, j])
            else:
                # Add the constraint directly to the model's ConstraintList
                model.cons8.add(model.a_p[t, k, j] == model.a[t - model.lead_times[k, j], k, j])
            
    model.cons9 = pyo.ConstraintList()

    def sp_rule(model, t, j):
        for k in model.jin[j]:
            if t == 1:
                # Add the constraint directly to the model's ConstraintList
                model.cons9.add(model.s_p[t, k, j] == -model.a_p[t, k, j] + model.a[t, k, j])
            else:
                # Add the constraint directly to the model's ConstraintList
                model.cons9.add(model.s_p[t, k, j] == model.s_p[t-1, k, j] - model.a_p[t, k, j] + model.a[t, k, j])

    # Loop over all time periods and retailers (Main) to apply the rule
    for t in model.Time_periods:
        for j in model.Main:
            ap_rule(model, t, j)
            sp_rule(model, t, j)

    # def inv_req1(model, t, j):
        # return sum(model.a[t,j,k] for k in model.jout[j]) <= model.c[j]

    # model.cons10 = pyo.Constraint(model.Time_periods, model.Producers, rule=inv_req1)

    def inv_req2(model, t, j):
        if j in model.Producers:
            return sum(model.a[t,j,k] for k in model.jout[j]) <= model.s_o[t,j]* model.v[j]
        elif j in model.Distributors:
            return sum(model.a[t,j,k] for k in model.jout[j]) <= model.s_o[t,j]
        else:
            return pyo.Constraint.Skip
        
    model.cons11 = pyo.Constraint(model.Time_periods, model.Main, rule=inv_req2)

    def market_sales1(model, t, j):
        return sum(model.s_d[t,j,k] for k in model.jout[j]) <= model.s_o[t-1,j]

    model.cons12 = pyo.Constraint(model.Time_periods2, model.Retailers, rule=market_sales1)

    model.cons13 = pyo.ConstraintList()

    def market_sales2(model, t, j):
        for k in model.jout[j]:
            if t == 2:
                # Add the constraint directly to the model's ConstraintList
                model.cons13.add(model.s_d[t, j, k] <= model.d[t-1, j, k])
            else:
                # Add the constraint directly to the model's ConstraintList
                model.cons13.add(model.s_d[t, j, k] <= model.d[t-1, j, k] + model.u[t-1, j, k])        

    model.cons14 = pyo.ConstraintList()

    def unf_rule(model, t, j):
        for k in model.jout[j]:
            if t == 2:
                # Add the constraint directly to the model's ConstraintList
                model.cons14.add(model.u[t, j, k] == model.d[t-1, j, k] - model.s_d[t, j, k])
            else:
                # Add the constraint directly to the model's ConstraintList
                model.cons14.add(model.u[t, j, k] == model.u[t-1, j, k] + model.d[t-1, j, k] - model.s_d[t, j, k])

    # Loop over all time periods and retailers to apply the rule
    for t in model.Time_periods2:
        for j in model.Retailers:
            market_sales2(model, t, j)
            unf_rule(model, t, j)
    
    return model