# SafeOR-Gym: A Benchmark Suite for Safe Reinforcement Learning Algorithms on Practical Operations Research Problems

SafeOR-Gym is a benchmark suite of Gym-compatible environments for safe reinforcement learning (SafeRL) in industrially relevant operations research (OR) problems. It is designed to evaluate SafeRL algorithms on realistic, structured, and safety-critical decision-making problems commonly encountered in industrial planning and real-time control.

This suite includes nine environments that model some well-known and challenging problems such as unit commitment, plant scheduling, resource allocation, supply chain logistics, and energy system operations. Each environment integrates combinatorial structure, strict constraints, and long planning horizons—making them ideal for testing the safety, robustness, and feasibility performance of RL agents. SafeOR-Gym is natively compatible with the OmniSafe framework, providing out-of-the-box support for constraint-handling algorithms, parallel training, and standardized benchmarking.

The key contributions of this project:


- A modular suite of nine OR-inspired SafeRL environments with varying structures, horizons, and complexities.

- Ready-to-use integration with OmniSafe, enabling immediate use of a large number of SafeRL algorithms.

- Empirical evaluation tools to assess safety violations, constraint satisfaction, and policy performance under structured conditions.


---

## Table of Contents

- [Installation](#installation)
- [Environments](#environments)
- [Benchmarked Algorithms](#benchmarked-algorithms)
- [Citing This Work](#citing-this-work)

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

## Environments

- **Air Separation Unit**: Optimize cryogenic gas separation with flow, temperature, and purity constraints.
- **Generation and Transmission Expansion Planning (GTEPEnv)**: Plan capacity expansion in power systems under long-term investment and operational constraints.
- **Grid Integrated Energy Storage (GridStorageEnv)**: Manage storage dispatch in a grid setting with price arbitrage and safety limits.
- **Integrated Scheduling and Maintenance**: Jointly optimize production schedules and maintenance windows under equipment availability constraints.
- **Multi-Echelon Supply Chain (InvMgmtEnv)**: Simulate inventory dynamics across multiple tiers of a supply chain network.
- **Multiperiod Blending Problem (BlendingEnv)**: Solve a multi-time-step blending optimization under ratio, availability, and demand constraints.
- **Resource Task Network**: Schedule resource-consuming tasks across time with bounded inventories and task delays.
- **State Task Network**: Model discrete-time transitions of material states via tasks executed on shared units.
- **Unit Commitment**: Optimize on/off decisions for generators over time while meeting demand and respecting ramping and reserve constraints.


## Benchmarked Algorithms

- **CPO**
- **TRPOLag**
- **P3O**
- **OnCRPO**
- **DDPGLag**



### Citing this work
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

