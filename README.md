# SafeOR-Gym: A Benchmark Suite for Safe Reinforcement Learning Algorithms on Practical Operations Research Problems

SafeOR-Gym is a benchmark suite of Gym-compatible environments for safe reinforcement learning (SafeRL) in industrially relevant operations research (OR) problems. It is designed to evaluate SafeRL algorithms on realistic, structured, and safety-critical decision-making problems commonly encountered in industrial planning and real-time control.

This suite includes nine environments that model some well-known and challenging problems such as unit commitment, plant scheduling, resource allocation, supply chain logistics, and energy system operations. Each environment integrates combinatorial structure, strict constraints, and long planning horizons—making them ideal for testing the safety, robustness, and feasibility performance of RL agents. SafeOR-Gym is natively compatible with the OmniSafe framework, providing out-of-the-box support for constraint-handling algorithms, parallel training, and standardized benchmarking.

The key contributions of this project:


- A modular suite of nine OR-inspired SafeRL environments with varying structures, horizons, and complexities.

- Ready-to-use integration with OmniSafe, enabling immediate use of a large number of SafeRL algorithms.

- Empirical evaluation tools to assess safety violations, constraint satisfaction, and policy performance under structured conditions.

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

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Folder Structure](#folder-structure)
- [Implemented Algorithms](#implemented-algorithms)
- [Environments](#environments)
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

