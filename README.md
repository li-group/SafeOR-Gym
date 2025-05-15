# SafeOR-Gym: A Benchmark Suite for Safe Reinforcement Learning Algorithms on Practical Operations Research Problems

SafeOR-Gym is a benchmark suite of Gym-compatible environments for safe reinforcement learning (SafeRL) in industrially relevant operations research (OR) problems. It is designed to evaluate SafeRL algorithms on realistic, structured, and safety-critical decision-making problems commonly encountered in industrial planning and real-time control.

This suite includes nine environments that model some well-known and challenging problems such as unit commitment, plant scheduling, resource allocation, supply chain logistics, and energy system operations. Each environment integrates strict constraints and planning horizons—making them ideal for testing the safety, robustness, and feasibility performance of RL agents. SafeOR-Gym is natively compatible with the OmniSafe framework, providing out-of-the-box support for constraint-handling algorithms, parallel training, and standardized benchmarking.

The key contributions of this project:


- A modular suite of nine OR-inspired SafeRL environments with varying structures, horizons, and complexities.

- Ready-to-use integration with OmniSafe, enabling immediate use of a large number of SafeRL algorithms.

---

## Table of Contents

- [Installation](#installation)
- [Environments](#environments)
- [Benchmarked Algorithms](#benchmarked-algorithms)
- [Citing This Work](#citing-this-work)
- [License](#license)
- [Contributions](#contributions)
- [Contact](#contact)

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch ≥ 1.10
- (Optional) Gurobi / CPLEX for optimization

### Install

```bash
git clone https://github.com/li-group/SafeOR-Gym.git
cd SafeOR-Gym
conda env create safe_rl_env
conda activate safe_rl_env
pip install pyomo
cd omnisafe
pip install setuptools==65.5.1
pip install .
cd ..
```
Note: This repository uses a modified version of [OmniSafe](https://github.com/PKU-Alignment/omnisafe), which includes a few minor changes. The core algorithm implementations remain unchanged.

## Environments

- **Production Scheduling in Air Separation Unit (ASUEnv)**: Optimize cryogenic gas separation with flow, temperature, and purity constraints.
- **Generation and Transmission Expansion Planning (GTEPEnv)**: Plan capacity expansion in power systems under long-term investment and operational constraints.
- **Grid Integrated Energy Storage (GridStorageEnv)**: Manage storage dispatch in a grid setting with price arbitrage and safety limits.
- **Integrated Scheduling and Maintenance**: Jointly optimize production schedules and maintenance windows under equipment availability constraints.
- **Multi-Echelon Supply Chain (InvMgmtEnv)**: Simulate inventory dynamics across multiple tiers of a supply chain network.
- **Multiperiod Blending Problem (BlendingEnv)**: Solve a multi-time-step blending optimization under ratio, availability, and demand constraints.
- **Resource Task Network**: Schedule resource-consuming tasks across time with bounded inventories and task delays.
- **State Task Network**: Model discrete-time transitions of material states via tasks executed on shared units.
- **Unit Commitment**: Optimize on/off decisions for generators over time while meeting demand and respecting ramping and reserve constraints.


## Benchmarking Environments

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

### Benchmarking Setup (ExperimentGrid)

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



## Citing this work
If you find SafeOR-Gym useful, please cite it in your publication

> ```bibtex
> @article{,
>   author  = {THE LI GROUP},
>   title   = {SafeOR-Gym: A Benchmark Suite for Safe Reinforcement Learning Algorithms on Practical Operations Research Problems},
>   journal = {Coming soon},
>   year    = {2025},
>   url     = {}
> }
> ```

## License

This repository is licensed under the [MIT License](LICENSE).

## Contributions

We welcome contributions and bug reports. If you'd like to extend the environments or benchmark a new algorithms for any environment, please open a pull request or issue.

## Contact

For questions, reach out to [canli.pse@gmail.com](mailto:canli.pse@gmail.com)
