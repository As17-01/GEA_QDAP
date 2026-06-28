# GEA-QDAP — Genetic Algorithm for the Generalized Quadratic Assignment Problem

A Python implementation of two genetic algorithm variants — a classic/standard GA and a
self-adaptive GA — for solving the Generalized Quadratic Assignment Problem (GQAP). See
[PROBLEM_DESCRIPTION.md](PROBLEM_DESCRIPTION.md) for the mathematical formulation.

## Features

- **Two GA variants**: `StandardGA` (fixed crossover/mutation rates) and `AdaptiveGA`
  (rates are scaled each generation by lambda multipliers that adapt to how much
  improvement each operator actually produced)
- **Diversity-aware selection**: parents are chosen via a diversity-weighted roulette
  wheel; survivors are kept via an elite + diversity/cost hybrid score that decays
  toward pure cost-based selection over the run
- **Memetic local search**: top elite individuals get a hill-climbing polish pass each
  generation (auto-disabled on large instances where it isn't worth the cost)
- **Capacity repair**: infeasible offspring/mutations are repaired by reassigning
  overloaded jobs to facilities with spare capacity, either greedily or with
  randomized sampling for diversity
- **Reproducible runs**: a single `seed_all()` call seeds both NumPy's RNG and the
  separate RNG used inside the `numba`-jitted operators

## Installation

This project uses [Poetry](https://python-poetry.org/):

```bash
poetry install
```

Core dependencies: `numpy`, `scipy`, `numba`, `pandas`, `openpyxl`, `matplotlib`.

## Project structure

```
src/
├── algos/
│   ├── ga_core.py        # BaseGA: shared init, selection, repair, local search, run loop
│   ├── ga_standard.py     # StandardGA: fixed crossover/mutation rates
│   ├── ga_adaptive.py     # AdaptiveGA: lambda-scaled, performance-adaptive rates
│   ├── ga_logging.py      # Iteration/timing/operator-stat reporting
│   ├── costs.py           # cost_function_perm, evaluate_permutation, diversity metric
│   └── utils.py           # seed_all, clone_individual, the `timed` profiling decorator
├── operators/
│   ├── crossover.py       # One-point / two-point crossover
│   ├── mutations.py       # Swap, reversion, insertion, big-swap, random-reassignment
│   └── repair.py          # GreedyRepair / RFRepair capacity-constraint repair
└── data/
    ├── models.py          # Model (problem instance) and Individual dataclasses
    └── model_loader.py    # Parses MATLAB-style .m dataset files

datasets/        # Full benchmark instances (e.g. c201535, T1-T14)
debug_datasets/  # Smaller instances used by the test harness
scripts/
└── run.py  # Sequential standard-vs-adaptive benchmark runner
```

## Usage

```python
from src.data.model_loader import load_model
from src.algos.ga_standard import StandardGA
from src.algos.ga_adaptive import AdaptiveGA
from src.algos.utils import seed_all

model = load_model("c201535")  # looked up under debug_datasets/<name>.m

seed_all(42)  # reproducible across NumPy and numba-jitted operators

ga = StandardGA(
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

`AdaptiveGA` takes the same constructor arguments plus `alpha`, `lambda_min`,
`lambda_max`, and `epsilon` to control how aggressively operator rates adapt.

### Comparison test runner

```bash
python3 scripts/run.py
```

Runs the configured algorithm(s) across every dataset under `debug_datasets/` and
writes per-dataset cost statistics (mean/median/min/max/std) to `results.json`.

## How the adaptive algorithm works

1. Both crossover and mutation start with `lambda = 1.0`.
2. Each generation, every offspring/mutant's normalized improvement over the current
   best (`delta = (parent_cost - child_cost) / (parent_cost + epsilon)`) is accumulated
   per operator.
3. `lambda_new = clip(lambda_old + alpha * delta_avg, lambda_min, lambda_max)`.
4. The number of crossovers/mutations performed next generation is
   `base_rate * population_size * lambda`.

An operator that is currently producing improvements gets used more; one that isn't
gets throttled back — without any manual per-instance rate tuning.

## Selection and repair

- **Parent selection**: roulette wheel weighted by a softmax over each individual's
  Hamming-distance diversity relative to the population.
- **Survivor selection**: the best half of the pool survives as elites; the rest is
  chosen by `diversity_weight * diversity_score + cost_weight * cost_score`, with
  `diversity_weight` decaying linearly from 0.7 to 0.2 over the run so late generations
  can converge rather than being held apart by the diversity term.
- **Repair**: any offspring/mutant that violates a facility's capacity has its
  heaviest job evicted and reassigned to a facility with available slack — either the
  cheapest option (`GreedyRepair`) or a randomized subsample of candidates
  (`RFRepair`, the default, used for diversity).
