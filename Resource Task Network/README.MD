The state space is defined as follows:
1. Current inventory
2. Demand at that time period
3. Time period it is currently at
4. Product/intermediates currently being processed

The action space is defined as follows:
1. The batch size of each task to be processed at that time period

After each action, the transition function/step function adds the task output in the buffer (as processing time might not be 1). At the same time, the inventories are updated if any ongoing task is finished which also frees the equipment which was being used.

The reward is calculated based on the price obtained for selling the product, buying raw material and running the task and utilities. The feasibility is computed based on the bounds of the batch size of the task and the inventory capacities of different materials. Infeasible actions are penalised with a large penalty. 

