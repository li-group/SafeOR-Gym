State Representation
Each observation from the environment is composed of the following components:

State Storage (state_storage): A real-valued vector indicating the inventory level of all resources (raw materials, intermediates, and final products) at the current time step.

Demand: A vector representing the external demand for final products at the current time step.

Pending Outputs (pending_outputs): A flattened vector of size (max_tau × num_resources) tracking the materials scheduled to arrive due to task completions in future periods, where tau is the processing time for each task.

Time (t): Internally tracked but not flattened into the observation. Used for time-indexed operations like delayed deliveries and utility prices.

The observation space is a flattened Box space built from a dictionary.

Action Representation
The action space is defined as a Dict of:

Run (run): A binary vector (MultiBinary) indicating whether each task is triggered at the current time step.

Batch Size (batch): A real-valued vector specifying the batch size for each task, constrained within [Vmin, Vmax].

The action vector is flattened for compatibility with RL algorithms. Batch values are zeroed out when the corresponding run entry is inactive.

Transition Dynamics
At each simulation step:

Task Execution:

If a task is run, the corresponding reactants are immediately consumed from storage.

The products are not added immediately but are scheduled for delivery after a task-specific delay (tau), using a delayed_queue.

Inventory Update:

Materials from previously scheduled outputs (if any) are added to inventory at the current time step.

The resulting inventory is validated against minimum and maximum bounds.

Demand Fulfillment:

If feasible, the current product demand is fulfilled (via inventory reduction), based on available quantities.

Feasibility Check:

Ensures resource inventories remain within bounds after applying task execution and delivery updates.

If infeasible, the state is unchanged, and a heavy penalty is applied.

Reward and Cost:

Utility costs are computed based on task usage, batch size, and current utility prices.

Revenue is earned for fulfilling product demands.

The reward is computed as:

Reward = Revenue − Utility Cost

Termination:

The episode terminates when the time step t reaches the defined horizon.

Reward and Cost Functions
Utility Cost:

Revenue:

Reward:

Reward 𝑡 = Revenue 𝑡 − Cost 𝑡
​
 
For infeasible actions, a penalty is applied:

Reward = −10^6 , Cost = 10^6
Reward=−10^6,Cost=10^6
 
Key Features
Delayed Material Handling: Tasks produce outputs after a delay (tau), handled via a future delivery queue.

Flat Action and Observation Spaces: Enables compatibility with continuous control RL algorithms.

Multi-Period Demand: External demand varies over time, encouraging temporal planning.

Utility-Based Costing: Task operation cost is influenced by utility usage and prices that evolve over time.

