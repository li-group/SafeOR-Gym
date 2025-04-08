State Representation
The environment state is defined by the following components:

Inventory: A real-valued vector representing the quantities of all resources (reactants, intermediates, products) currently in storage.

Demand: A vector indicating the external demand for final products at the current time step.

Pending Outputs: A flattened vector representing future inflows of products due to task completion delays. Each task has an associated processing time (tau), and products are added to the inventory only after this delay.

Time (t): Internally tracked simulation step, incremented after each transition.

The observation space is a flattened Box space combining the above components (excluding t).

Action Representation
The action space is a real-valued Box of dimension equal to the number of tasks. Each component specifies the batch size for a corresponding task. Batch sizes must lie within predefined lower and upper bounds for each task.

A batch size below a small threshold (e.g., 1e-4) is treated as a zero action (i.e., the task is not executed). This approach allows a continuous action space without requiring a separate binary trigger vector.

Transition Dynamics
Each environment transition involves the following sequence:

Action Parsing: The batch vector is extracted and thresholded.

Stoichiometric Updates:

Reactant and intermediate consumption is applied immediately.

Product generation is scheduled for future time steps via a delayed_production_queue, indexed by processing time tau.

Inventory Update: Includes both immediate consumption and the arrival of previously scheduled product outputs.

Demand Fulfillment: If the action is feasible, product demands are subtracted from inventory to reflect order fulfillment.

Feasibility Verification:

Equipment capacities are not exceeded.

Resource inventories remain within specified lower and upper bounds.

Demand for products is satisfied (partial fulfillment is permitted, but shortfalls may be penalized).

Cost and Reward Computation:

Utility cost is computed based on time-indexed utility prices and task-specific usage rates.

Reward is defined as revenue from sales minus total utility cost.

Termination: The episode ends if the time horizon is reached or if an infeasible action is taken.

Reward and Cost Structure
The reward at each step is defined as:

Reward = min(inventory,demand)⋅unit price−utility cost
Sales revenue is computed based on the fulfilled portion of demand and per-unit product revenue.

Utility cost is proportional to the batch size and the utility usage factor, multiplied by the current utility price.

For infeasible actions, the environment imposes a large negative reward:

Reward = 10^-6 - utility_cost
and flags the transition as truncated.

Feasibility Constraints
The following constraints are checked prior to applying a transition:

Equipment Constraints: The total usage of each equipment type must not exceed its capacity.

Inventory Bounds: All resources must remain within predefined minimum and maximum inventory levels.

Demand Constraints: For products, inventory must be sufficient to meet current demand (optional: partial fulfillment allowed).

If any constraint is violated, the action is rejected, and the environment remains in the same state.


