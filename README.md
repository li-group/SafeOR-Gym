# SafeOR-Gym: A Benchmark Suite for Safe Reinforcement Learning Algorithms on Practical Operations Research Problems

SafeOR-Gym is a benchmark suite of Gym-compatible environments for safe reinforcement learning (SafeRL) in industrially relevant operations research (OR) problems. It is designed to evaluate SafeRL algorithms on realistic, structured, and safety-critical decision-making problems commonly encountered in industrial planning and real-time control.

This suite includes nine environments that model some well-known and challenging problems such as unit commitment, plant scheduling, resource allocation, supply chain logistics, and energy system operations. Each environment integrates combinatorial structure, strict constraints, and long planning horizons—making them ideal for testing the safety, robustness, and feasibility performance of RL agents. SafeOR-Gym is natively compatible with the OmniSafe framework, providing out-of-the-box support for constraint-handling algorithms, parallel training, and standardized benchmarking.

The key contributions of this project:


- A modular suite of nine OR-inspired SafeRL environments with varying structures, horizons, and complexities.

- Ready-to-use integration with OmniSafe, enabling immediate use of a large number of SafeRL algorithms.

- Empirical evaluation tools to assess safety violations, constraint satisfaction, and policy performance under structured conditions.

If you find SafeOR-Gym useful, please cite it in your publication

> ```bibtex
> @article{JMLR:v25:23-0681,
>   author  = {Jiaming Ji and Jiayi Zhou and Borong Zhang and Juntao Dai and Xuehai Pan and Ruiyang Sun},
>   title   = {OmniSafe: An Infrastructure for Accelerating Safe Reinforcement Learning Research},
>   journal = {Journal of Machine Learning Research},
>   year    = {2024},
>   volume  = {25},
>   number  = {285},
>   pages   = {1--6},
>   url     = {http://jmlr.org/papers/v25/23-0681.html}
> }
> ```


