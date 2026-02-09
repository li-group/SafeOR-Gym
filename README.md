# SafeOR-Gym: A Benchmark Suite for Safe Reinforcement Learning Algorithms on Practical Operations Research Problems

SafeOR-Gym is a benchmark suite of Gym-compatible environments for safe reinforcement learning (SafeRL) in industrially relevant operations research (OR) problems. It is designed to evaluate SafeRL algorithms on realistic, structured, and safety-critical decision-making problems commonly encountered in industrial planning and real-time control.

This suite includes nine environments that model some well-known and challenging problems such as unit commitment, plant scheduling, resource allocation, supply chain logistics, and energy system operations. Each environment integrates strict constraints and planning horizons—making them ideal for testing the safety, robustness, and feasibility performance of RL agents. SafeOR-Gym is natively compatible with the OmniSafe framework, providing out-of-the-box support for constraint-handling algorithms, parallel training, and standardized benchmarking.

The key contributions of this project:


- A modular suite of nine OR-inspired SafeRL environments with varying structures, horizons, and complexities.

- Ready-to-use integration with OmniSafe, enabling immediate use of a large number of SafeRL algorithms.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Environments](#environments)
- [Benchmarking Setup (ExperimentGrid)](#benchmarking-setup-experimentgrid)
- [Cite us](#cite-us)
- [License](#license)
- [How to Contribute](#how-to-contribute)

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch ≥ 1.10
- (Optional) Gurobi / CPLEX for optimization

### Install

Simply run the bash file requirements.sh. The file can be modified to change the environment name or use any other package manager.

```bash
bash requirements.sh
```
Note: This repository uses a modified [version](https://github.com/li-group/omnisafe/tree/main) of [OmniSafe](https://github.com/PKU-Alignment/omnisafe), which includes a few minor changes. The core algorithm implementations remain unchanged.

## Environments

- **Production Scheduling in Air Separation Unit (ASUEnv)**: Optimize liquid production to minimize electricity and production costs, while fulfilling demand and respecting unit capacities across time.
- **Generation and Transmission Expansion Planning (GTEPEnv)**: Plan capacity expansion in power systems under long-term investment and operational constraints.
- **Grid Integrated Energy Storage (GridStorageEnv)**: Manage storage dispatch in a grid setting with price arbitrage and safety limits.
- **Integrated Scheduling and Maintenance**: Jointly optimize production schedules and maintenance windows under equipment availability constraints.
- **Multi-Echelon Supply Chain (InvMgmtEnv)**: Simulate inventory dynamics across multiple tiers of a supply chain network.
- **Multiperiod Blending Problem (BlendingEnv)**: Solve a multi-time-step blending optimization under ratio, availability, and demand constraints.
- **Resource Task Network**: Schedule resource-consuming tasks across time with bounded inventories and task delays.
- **State Task Network**: Model discrete-time transitions of material states via tasks executed on shared units.
- **Unit Commitment**: Optimize on/off decisions for generators over time while meeting demand and respecting ramping and reserve constraints.


Each environment has its own folder containing the relevant code. To run and benchmark an environment, execute the corresponding script located within its folder.

| Environment                                  | Main Script File                  |
|--------------------------------------------  |-----------------------------------|
| Production Scheduling in Air Separation Unit | `ASU_safe.py`                     |
| Generation & Transmission Expansion          | `gen_transmission_exp_safe.py`    |
| Grid Integrated Energy Storage               | `battery_env_safe.py`             |
| Integrated Scheduling and Maintenance        | `ISM_safe.py`                     |
| Multi-Echelon Supply Chain                   | `supply_chain_safe.py`            |
| MultiPeriod Blending                         | `Blending_safe.py`                |
| Resource Task Network                        | `main.py`                         |
| State Task Network                           | `main.py`                         |
| Unit Commitment                              | `Unit_Commitment_Safe.py`         |

## Benchmarking Setup (ExperimentGrid)

This repository uses `ExperimentGrid` to manage and evaluate SafeRL training experiments. Below is an overview of its key functionality and a sample code block for running benchmarks.

### Key Features

- **Algorithms**: Choose from multiple categories including first-order, second-order, primal, and offline SafeRL methods.
- **Environments**: Select environment(s) and config paths using `env_id` and `env_cfgs:env_init_config`.
- **Epoch Control**: Define `STEPS_PER_EPOCH` and `TOTAL_EPOCHS` to set training time.
- **Logging**: Enable logging with TensorBoard or Weights & Biases (WandB).
- **Parallelism**: Set `vector_env_nums` and `torch_threads` for environment and CPU parallelism.
- **Device**: Automatically uses GPU if available, otherwise falls back to CPU.
- **Evaluation**: Includes automated training, comparison analysis, and evaluation post-training.

---

###  Benchmarking Code

```python
eg = ExperimentGrid(exp_name='Benchmark_Safety_rtn_v0')

# Define algorithm categories
base_policy = ['PolicyGradient', 'NaturalPG', 'TRPO', 'PPO']
naive_lagrange_policy = ['TRPOLag']
first_order_policy = ['P3O']
second_order_policy = ['CPO']
primal_policy = ['OnCRPO']
offline_policy = ['DDPGLag']

# Target environment
mujoco_envs = ['rtn-v0']
eg.add('env_id', mujoco_envs)

# GPU configuration
available_gpus = list(range(torch.cuda.device_count()))
gpu_id = [0]
if gpu_id and not set(gpu_id).issubset(available_gpus):
    warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
    gpu_id = None

# Training configuration
T = 30
STEPS_PER_EPOCH = T * 128
TOTAL_EPOCHS = 100
TOTAL_STEPS = STEPS_PER_EPOCH * TOTAL_EPOCHS

# Set experiment parameters
eg.add('seed', [10])
eg.add('algo', second_order_policy + naive_lagrange_policy + first_order_policy + primal_policy + offline_policy)

# Logging configuration
eg.add('logger_cfgs:use_wandb', [False])
eg.add('logger_cfgs:use_tensorboard', [True])
eg.add('logger_cfgs:window_lens', [int(STEPS_PER_EPOCH / T)])

# Parallelism and device
eg.add('train_cfgs:vector_env_nums', [1])
eg.add('train_cfgs:torch_threads', [1])
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
eg.add('train_cfgs:device', [device])
eg.add('train_cfgs:total_steps', [TOTAL_STEPS])

# Model configuration
eg.add('model_cfgs:actor:output_activation', ['tanh'])

# Algorithm configuration
eg.add('algo_cfgs:steps_per_epoch', [STEPS_PER_EPOCH])

# Environment config file and parameters
eg.add('env_cfgs:env_init_config:config_file', [ENVIRONMENT_CONFIG_FILE_PATH])
eg.add('env_cfgs:env_init_config:debug', [False])
eg.add('env_cfgs:env_init_config:sanitization_cost_weight', [1.0])
eg.add('env_cfgs:env_init_config:cost_coefficient', [1.0])

# Run training, analysis, and evaluation
eg.run(train, num_pool = 1, gpu_id=gpu_id)
eg.analyze(parameter='algo', values = None, compare_num = 5)
a = eg.evaluate(num_episodes = 10)
```
## Cite us
<a name="citation"></a>
Cite us ❤️
```bibtex
@article{ramanujam2025safeor,
  title={SafeOR-Gym: A Benchmark Suite for Safe Reinforcement Learning Algorithms on Practical Operations Research Problems},
  author={Ramanujam, Asha and Elyoumi, Adam and Chen, Hao and Kompalli, Sai Madhukiran and Ahluwalia, Akshdeep Singh and Pal, Shraman and Papageorgiou, Dimitri J and Li, Can},
  journal={arXiv preprint arXiv:2506.02255},
  year={2025}
}
```


## License

This repository is licensed under the [MIT License](LICENSE).

## How to Contribute

We welcome contributions and bug reports. If you'd like to extend the environments or benchmark new algorithms for any environment, please open a pull request or issue.

### Creating New Environments

SafeOR-Gym follows a modular design pattern that makes it easy to create new environments. Below is a step-by-step guide using the Multi-Echelon Supply Chain environment as an example.

#### 1. Base Gym Environment Structure

Create a standard Gymnasium environment by inheriting from `gym.Env`. Here's the basic structure:

```python
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict

class InvMgmtEnv(gym.Env):
    def __init__(self, env_id: str = 'InvMgmt-v0', **kwargs):
        super().__init__()
        
        # Load configuration
        config_path = kwargs.pop('config_path', None)
        raw_cfg = self.load_config(config_path)
        assign_env_config(self, raw_cfg)
        
        # Define observation and action spaces
        self.observation_space = Box(low=low_obs, high=high_obs, shape=(obs_dim,))
        self.action_space = Box(low=-1.0, high=1.0, shape=(act_dim,))
        
    def reset(self, seed=None, options=None):
        # Initialize environment state
        return self._get_state(), {}
        
    def step(self, action):
        # Execute one time step
        # Return: observation, reward, terminated, truncated, info
        pass
```

#### 2. State Space Definition

Define your state representation with clear structure. For the supply chain example:

```python
def _get_state(self, mode='arr'):
    """Return current state as dict or flattened array"""
    state_dict = {
        'on_hand_inventory': {node: inventory_level},
        'pipeline_inventory': {(i,j): [transit_quantities]},
        'sales': {(retailer, market): units_sold},
        'backlog': {(retailer, market): unfulfilled_demand},
        'demand_window': {(retailer, market): [future_demands]},
        't': current_time_period
    }
    
    if mode == 'dict':
        return state_dict
    else:
        # Flatten for neural network input
        flat_obs, mapping = flatten_and_track_mappings(state_dict)
        return flat_obs
```

#### 3. Action Space Design

Design continuous or discrete action spaces based on your problem:

```python
# Continuous actions scaled from [-1,1] to actual ranges
def decode_action(self, raw_action):
    action_dict = {}
    for i, route in enumerate(self.reordering_routes):
        # Scale from [-1,1] to [0, capacity]
        scaled_value = (raw_action[i] + 1.0) * 0.5 * self.route_capacity[route]
        action_dict[route] = max(0.0, scaled_value)  # Ensure non-negative
    return action_dict
```

#### 4. Constraint Handling

Implement constraint violations as costs for safe RL:

```python
def check_action_bounds_cost(self, action_dict):
    """Check action constraints and calculate penalties"""
    penalty = 0.0
    for route, value in action_dict.items():
        # Lower bound constraint
        if value < 0.0:
            penalty += abs(value) * self.penalty_factors['action']
            action_dict[route] = 0.0
        
        # Upper bound constraint (capacity)
        if value > self.route_capacity[route]:
            excess = value - self.route_capacity[route]
            penalty += excess * self.penalty_factors['action']
            action_dict[route] = self.route_capacity[route]
    
    return action_dict, penalty

def check_obs_bounds_cost(self, observation):
    """Check state constraints and calculate penalties"""
    penalty = 0.0
    for i, value in enumerate(observation):
        category, _ = self.obs_mapping[i]
        if category in self.penalty_factors:
            # Inventory capacity constraints
            if value > self.obs_space.high[i]:
                excess = value - self.obs_space.high[i]
                penalty += excess * self.penalty_factors[category]
    
    return penalty
```

#### 5. Safe RL Wrapper

Create a CMDP wrapper for integration with OmniSafe:

```python
from omnisafe.envs.core import CMDP, env_register

@env_register
class SupplyChainSafe(CMDP):
    _support_envs = ['SupplyChain-v0']
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True

    def __init__(self, env_id: str, **kwargs):
        super().__init__(env_id)
        self._env = InvMgmtEnv(env_id=env_id, **kwargs.get('env_init_cfgs', {}))
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action.cpu().numpy())
        cost = self._env.cost  # Constraint violations
        return (torch.tensor(obs), torch.tensor(reward), 
                torch.tensor(cost), torch.tensor(terminated), 
                torch.tensor(truncated), info)
```

#### 6. Configuration Files

Use JSON configuration files to make environments easily customizable:

```json
{
    "T": 30,
    "num_markets": 1,
    "num_retailers": 1,
    "num_distributors": 2,
    "initial_inv": {"1": 100, "2": 120},
    "inventory_holding_cost": {"1": 0.04, "2": 0.03},
    "reordering_route_capacity": {"(2,1)": 500, "(3,1)": 400},
    "penalty_factors": {
        "action": 10.0,
        "inventory": 5.0,
        "pipeline": 3.0
    }
}
```

#### 7. Integration with Benchmarking

Add your environment to the benchmarking framework:

```python
# In your main benchmarking script
eg = ExperimentGrid(exp_name='Benchmark_YourEnvironment')
eg.add('env_id', ['YourEnv-v0'])
eg.add('env_cfgs:env_init_config:config_file', ['/path/to/config.json'])

# Run experiments
eg.run(train, num_pool=1)
```

#### Key Design Principles

1. **Modular Structure**: Separate the base environment from the safe RL wrapper
2. **Configuration-Driven**: Use JSON/YAML files for environment parameters
3. **Constraint-Aware**: Implement constraints as costs, not hard boundaries
4. **Scalable State/Action**: Design spaces that can handle different problem sizes
5. **Clear Documentation**: Document state variables, action meanings, and constraints

#### Testing Your Environment

1. Test basic functionality: `env.reset()`, `env.step()`, action/observation spaces
2. Verify constraint handling: Check that violations produce appropriate costs
3. Run with SafeRL algorithms: Ensure compatibility with OmniSafe
4. Validate against domain knowledge: Verify behavior matches expected OR problem dynamics

For more detailed examples, examine the existing environments in the repository, particularly the Multi-Echelon Supply Chain (`supply_chain_gym.py`, `supply_chain_safe.py`) and configuration files.

### Customizing Constraints

#### Modifying Existing Constraints

You can easily modify constraints without changing code by editing configuration files:

```json
{
    "penalty_factors": {
        "action": 10.0,        // Penalty for action bound violations
        "inventory": 5.0,      // Penalty for inventory capacity violations
        "pipeline": 3.0        // Penalty for pipeline constraints
    },
    "reordering_route_capacity": {
        "(2,1)": 500,          // Maximum reorder quantity for route (2,1)
        "(3,1)": 400
    },
    "inv_capacity": {
        "1": 1000,             // Maximum inventory at node 1
        "2": 800
    }
}
```

#### Adding New Constraint Types

Extend constraint handling by modifying the base environment:

```python
def check_custom_constraints(self, action, state):
    """Add domain-specific constraints"""
    penalty = 0.0
    
    # Example: Production rate constraints
    for producer in self.producers:
        production_rate = action.get(f'produce_{producer}', 0)
        if production_rate > self.max_production_rate[producer]:
            penalty += (production_rate - self.max_production_rate[producer]) ** 2
    
    # Example: Resource availability constraints
    total_resource_usage = sum(action.values())
    if total_resource_usage > self.available_resources:
        penalty += (total_resource_usage - self.available_resources) * 100
    
    return penalty
```


#### Community Contributions

We encourage researchers to contribute:

1. **New Environments**: Submit OR problems from different domains
2. **Algorithm Implementations**: Add new SafeRL algorithms to the benchmark
3. **Evaluation Metrics**: Propose domain-specific performance measures
4. **Real-World Validation**: Provide datasets or case studies for validation

For contributing guidelines, see our [GitHub repository](https://github.com/li-group/SafeOR-Gym) and submit pull requests with detailed documentation.


