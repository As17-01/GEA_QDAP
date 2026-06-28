# GEA-QDAP — Genetic Algorithm for the Generalized Quadratic Assignment Problem

A Python implementation of ten metaheuristics for the Generalized Quadratic Assignment
Problem (GQAP): a textbook standard GA, this project's own enhanced GA (GEA) and its
self-adaptive variant, simulated annealing, particle swarm optimization, two GA hybrids
(with PSO and with SA), and three GEA "modernization" scenarios that each make a single
operator adaptive. See [PROBLEM_DESCRIPTION.md](PROBLEM_DESCRIPTION.md) for the
mathematical formulation and [ALGORITHMS.md](ALGORITHMS.md) for a full description of
every algorithm, its citation, and the shared framework (operators, repair, selection)
they're all built on.

## Features

- **Ten algorithms**, all sharing the same permutation representation, problem-instance
  format, and run loop — see [ALGORITHMS.md](ALGORITHMS.md) for the full list and how
  each one works
- **Diversity-aware selection** (GEA family): parents are chosen via a
  diversity-weighted roulette wheel; survivors are kept via an elite + diversity/cost
  hybrid score that decays toward pure cost-based selection over the run
- **Memetic local search** (GEA family): top elite individuals get a hill-climbing
  polish pass each generation (auto-disabled on large instances where it isn't worth
  the cost)
- **Capacity repair**: infeasible offspring/mutations are repaired by reassigning
  overloaded jobs to facilities with spare capacity, either greedily (`GreedyRepair`,
  `StandardGA`'s default) or with randomized sampling for diversity (`RFRepair`, the
  GEA family's default)
- **Reproducible runs**: a single `seed_all()` call seeds both NumPy's RNG and the
  separate RNG used inside the `numba`-jitted operators

## Installation

This project uses [Poetry](https://python-poetry.org/):

```bash
poetry install
```

Core dependencies: `numpy`, `scipy`, `numba`, `pandas`, `openpyxl`, `matplotlib`,
`hydra-core`, `optuna`.

## Project structure

```
src/
├── algos/
│   ├── base.py            # BaseGA: shared init, selection, repair, local search, run loop
│   ├── ga_standard.py     # StandardGA: textbook Holland (1992) baseline
│   ├── ga_gea.py          # GEA: diversity selection, repair, immigrants, local search
│   ├── ga_adaptive.py     # AdaptiveGA: GEA + lambda-scaled, performance-adaptive rates
│   ├── ga_sa.py           # SimulatedAnnealing: population of independent annealing chains
│   ├── ga_pso.py          # ParticleSwarm: discrete PSO via crossover/mutation moves
│   ├── ga_hybrid_gapso.py # HybridGAPSO: rank-split PSO refinement + GA regeneration
│   ├── ga_hybrid_gasa.py  # HybridGASA: GA offspring filtered by an SA acceptance rule
│   ├── gea_scenario_1.py  # GEAScenario1: GEA + adaptive Robust-Chromosome crossover
│   ├── gea_scenario_2.py  # GEAScenario2: GEA + adaptive Directed Mutation
│   ├── gea_scenario_3.py  # GEAScenario3: GEA + adaptive Gene Injection
│   └── logger.py          # Iteration/timing/operator-stat reporting
├── operators/
│   ├── crossover.py       # One-point / two-point / uniform / greedy crossover
│   ├── mutations.py       # Swap, reversion, insertion, big-swap, random, scramble, ...
│   └── repair.py          # GreedyRepair / RFRepair capacity-constraint repair
├── costs.py                # cost_function_perm(_delta), evaluate_permutation(_delta_batch)
├── selection.py             # DiversitySelector: diversity-weighted parent/survivor selection
├── seeding.py               # seed_all: seeds both NumPy's and numba's RNGs
└── data/
    ├── models.py           # Model (problem instance) and Individual dataclasses
    └── model_loader.py     # Parses MATLAB-style .m dataset files

datasets/  # Benchmark instances (e.g. c201535, T1-T14)
scripts/
├── run.py             # Hydra-driven benchmark runner (any algorithm config)
├── tune_algorithm.py  # Optuna tuning of GEA's rate/stagnation knobs
├── tune_components.py # Optuna tuning of GEA's repair/selector knobs
└── conf/              # One Hydra config per algorithm (standard.yaml, gea.yaml, ...)
```

## Usage

```python
from src.data.model_loader import load_model
from src.algos.ga_gea import GEA
from src.algos.ga_adaptive import AdaptiveGA
from src.seeding import seed_all

model = load_model("c201535")

seed_all(42)  # reproducible across NumPy and numba-jitted operators

ga = GEA(
    model,
    population_size=350,
    iterations=1000,
    crossover_rate=0.7,
    mutation_rate=0.3,
)
best = ga.run(time_limit=1000)  # seconds

print(f"Best cost: {best.cost:.6f}")
print(f"Assignment (facility per job): {best.permutation.tolist()}")
```

Every other algorithm in `src/algos/` follows the same `Algorithm(model, **params).run(time_limit=...)`
shape; see [ALGORITHMS.md](ALGORITHMS.md) for each one's specific constructor
parameters.

### Benchmark runner

```bash
python3 scripts/run.py --config-name=gea          # or standard / adaptive / sa / pso /
                                                    # hybrid_ga_pso / hybrid_ga_sa /
                                                    # gea_scenario_1 / gea_scenario_2 /
                                                    # gea_scenario_3
```

Runs the configured algorithm across every dataset listed in `scripts/conf/datasets/common.yaml`
and writes per-dataset cost/runtime statistics (mean/median/min/max/std) to
`scripts/results/<algo>.json`. Any config field can be overridden on the command line,
e.g. `python3 scripts/run.py --config-name=adaptive ga.population_size=500 run.runs=10`.

### Tuning

```bash
python3 scripts/tune_algorithm.py    # tunes GEA's crossover_rate/mutation_rate/stagnation_limit/immigrant_rate
python3 scripts/tune_components.py   # tunes GEA's repair_class/selector knobs
```

Both run an Optuna TPE search against a fixed baseline and write the best candidate
back into `scripts/conf/gea.yaml` / `scripts/conf/components/common.yaml`.
