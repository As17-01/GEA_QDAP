# Algorithms

This project solves the Generalized Quadratic Assignment Problem (GQAP, see
[PROBLEM_DESCRIPTION.md](PROBLEM_DESCRIPTION.md)) with ten metaheuristics, all sharing
the same solution representation, problem instance format, and a common scaffolding
class (`BaseGA`). This document describes that shared scaffolding first, then each
algorithm built on top of it, then the tuning and experimental procedures.

---

## Shared framework (`src/algos/base.py`)

### Representation

A candidate solution (`Individual`, in `src/data/models.py`) is an integer array
`permutation` of length `J` (number of jobs), where `permutation[j]` is the index of
the facility assigned to job `j`. Alongside it, every `Individual` carries:

- `cost`: the GQAP objective value (`inf` if the assignment violates any facility's
  capacity)
- `cvar`: per-facility remaining capacity slack (`b_i - load_i`), used by some
  operators as a feasibility-robustness signal

A problem instance (`Model`) holds the assignment-cost matrix `cij`, resource-usage
matrix `aij`, facility capacities `bi`, facility-distance matrix `DIS`, and
job-interaction matrix `F`.

### Cost evaluation (`src/costs.py`)

`cost_function_perm` computes a permutation's cost from scratch (assignment cost +
quadratic interaction cost), returning `inf` if any facility is over capacity.
`cost_function_perm_delta` computes the same cost incrementally from a known-feasible
baseline cost, touching only the positions that changed — O(K·J) instead of O(J²)
when K changed positions is small. Every algorithm routes offspring through the
batched version (`evaluate_permutation_delta_batch`), each child paired with whichever
parent it is "closest to" so the delta stays cheap.

### Operators

All GA-based algorithms in this project share the same operator sets, dispatched
uniformly at random by `choose_crossover` / `choose_mutation`. Operator consistency
across algorithms is a design requirement: each algorithm is only distinguished by
*how* it applies and combines these operators, not *which* operators it has access to.

**Crossover** (`src/operators/crossover.py`) — four operators:

| Operator | Mechanism |
|---|---|
| `crossover_one_point` | Splits both parents at one random point; each child is one parent's prefix + the other's suffix. |
| `crossover_two_point` | Splices a random middle segment from one parent into the other's copy, and vice versa. |
| `crossover_uniform` | Per-gene coin flip decides which parent each child inherits from at each position. |
| `crossover_greedy` | Per-gene, child 1 takes whichever parent's assignment is cheaper (`cij`); child 2 takes the complementary gene, spanning the same diversity as a random crossover. |

**Mutation** (`src/operators/mutations.py`) — nine operators, including the five
primary discrete moves (swap, big swap, insertion, reversion, random) plus four
additional operators:

| Operator | Category | Mechanism |
|---|---|---|
| `mutation_swap` | Primary | Swaps two adjacent genes. |
| `mutation_big_swap` | Primary | Swaps two genes at arbitrary (non-adjacent) positions. |
| `mutation_insertion` | Primary | Moves a random segment to a different position. |
| `mutation_reversion` | Primary | Reverses a random contiguous segment. |
| `mutation_random` | Primary | Reassigns 1–5 random genes to brand-new random facilities (fresh genetic material). |
| `mutation_scramble` | Extended | Shuffles the values within a random window of 3–5 genes. |
| `mutation_cyclic_shift` | Extended | Cyclically rotates 3 random genes. |
| `mutation_greedy_reassign` | Directed | Finds the single worst-assigned job and reassigns it to the cheapest feasible facility. |
| `mutation_migration` | Directed | Moves the priciest job out of the most-loaded facility into the facility with the most spare capacity. |

Mutation is applied **per-individual as a discrete operator**: one randomly chosen
operator is applied to the whole individual with probability `mutation_rate`.
Iterating over every gene with a small per-gene probability is not used — it raises
complexity unnecessarily and conflates operator application with gene-level probability.

### Repair (`src/repair.py`)

Every crossover and mutation can produce an infeasible permutation (a facility over
capacity). Repair iteratively evicts an overloaded facility's job and reassigns it
until feasible:

- **`GreedyRepair`**: deterministic — always picks the worst capacity violation and
  reassigns to the cheapest feasible facility. Used by `StandardGA` and `SA`.
- **`RFRepair`** (default for the GEA family): picks from a random subsample of
  candidates instead of the single greedy best, trading exactness for population
  diversity.

### Selection (`src/selection.py`, `DiversitySelector`)

Used by the GEA family; not used by `StandardGA` or `SA`.

- **Parent selection**: roulette wheel weighted by each individual's mean Hamming
  distance to the rest of the population — more distinctive individuals are more
  likely to be picked as parents.
- **Survivor selection**: the cheapest `elite_fraction` of the merged pool (parents +
  offspring + mutants + immigrants, deduplicated) survives as elites; the rest is
  ranked by `diversity_weight · diversity_score + cost_weight · cost_score`, with
  `diversity_weight` decaying linearly over the run so later generations converge.

### Local search (memetic polish)

Each generation, the top 3 individuals receive a per-job hill-climbing pass (try every
facility for each job, keep whichever lowers cost). Auto-disabled above `J = 50` jobs
where the O(J·I) cost per pass dominates runtime.

### Run loop

`run(time_limit)` initializes the population, then repeatedly calls the subclass's
`step()`, re-sorts by cost, polishes elites, and stops once `time_limit` (wall-clock
seconds) or `iterations` is reached.

---

## 1. `StandardGA` (`src/algos/ga_standard.py`)

> Holland, J. H. (1992). Genetic algorithms. *Scientific American*, 267(1), 66–73.

The textbook simple genetic algorithm, kept as close to Holland's original description
as the capacity-constrained encoding allows. It deliberately **does not** use any of
this project's own GEA enhancements:

- **Selection**: plain fitness-proportionate (roulette wheel) selection on raw cost
  (`fitness = 1 / (cost + ε)`) — no diversity weighting.
- **Crossover**: one operator chosen uniformly at random from the four operators above
  (`choose_crossover`), applied with probability `crossover_rate` per mating pair;
  otherwise the pair passes through unchanged.
- **Mutation**: with probability `mutation_rate`, one operator chosen uniformly at
  random from the nine operators above is applied to the individual as a whole — no
  per-gene iteration.
- **Replacement**: full generational replacement with single-individual elitism
  (`elitism_count`, default `1`) so the best solution is never lost to randomness.
- No diversity selection, no `RFRepair` (uses deterministic `GreedyRepair`), no
  stagnation immigrants, no memetic local search.

This is the literal baseline the rest of the algorithms are compared against.

> **Design notes:** Uses all four crossover operators (`choose_crossover`) and all nine
> mutation operators (`choose_mutation`), including the five primary discrete mutation
> operators: swap, big swap, insertion, reversion, and random. Mutation is applied as a
> discrete per-individual operator, not as a per-gene Bernoulli trial.

---

## 2. `GEA` (`src/algos/ga_gea.py`)

This project's own enhanced GA. Where `StandardGA` deliberately bypasses most of the
shared framework, `GEA` is the opposite: it uses every component described there and
runs **five operator stages per generation** in fixed sequence:

| Stage | Operator | Rate | Description |
|---|---|---|---|
| 1 | **Crossover** | `crossover_rate` | `choose_crossover` — one of the 4 standard operators, per pair. |
| 2 | **Mutation** | `mutation_rate` | `choose_mutation` — one of the 9 standard operators, per individual. |
| 3 | **RC crossover** | `rc_rate` | Robust Chromosome: per gene, inherit from whichever parent has more remaining capacity slack (`cvar`). |
| 4 | **DM** | `dm_rate` | Directed Mutation: move each individual's single worst-assigned job to its cheapest feasible facility. |
| 5 | **GI** | `injection_rate` | Gene Injection: replace a handful of random genes with fresh random facility assignments. |

All five offspring pools are merged with the current population; `DiversitySelector`
picks the next generation. Additional enhancements:

- **Repair**: `RFRepair` by default.
- **Stagnation immigrants**: if best solution hasn't improved for `stagnation_limit`
  iterations, `immigrant_rate · population_size` fresh random individuals are injected.
- **Memetic local search**: enabled.

> **Design notes:** `GEAScenario1`, `GEAScenario2`, and `GEAScenario3` are
> **single-enhancement ablations** of this full variant — each adds only one of stages
> 3, 4, or 5 on top of standard crossover + mutation, without the other two. The RC
> crossover helper (`crossover_robust_chromosome`) is implemented once in
> `src/operators/crossover.py`; the RC, DM, and GI driver methods live on `BaseGA`
> (`_robust_chromosome_crossover`, `_directed_mutation`, `_gene_injection`) so all
> subclasses that need them (GEA, Scenario1/2/3) inherit rather than duplicate them.

---

## 3. `AdaptiveGEA` (`src/algos/ga_adaptive.py`)

> **Scope note:** This algorithm is included in tuning and final experiment runs
> alongside the rest of the algorithms, but its results **may not be reported** in this
> paper. It will be formally proposed in a follow-up project with a new cost function.

Identical to `GEA` except the crossover and mutation rates adapt every generation
based on how much improvement each operator produced. Each operator's rate is scaled by
a lambda multiplier (`lambda_crossover`, `lambda_mutation`, both starting at `1.0`):

1. Each offspring's improvement over its **own parent baseline** is normalized:
   `delta = (parent_cost - child_cost) / (parent_cost + epsilon)`, accumulated per
   operator. Comparing to the parent (not the global best) prevents lambda from
   collapsing toward `lambda_min` once the population converges.
2. `lambda_new = clip(lambda_old + alpha · delta_sum, lambda_min, lambda_max)`.
3. Next generation's operator counts become `base_rate · population_size · lambda`.

---

## 4. `SimulatedAnnealing` (`src/algos/ga_sa.py`)

> Bertsimas, D., & Tsitsiklis, J. (1993). Simulated annealing. *Statistical Science*,
> 8(1), 10–15.

Classic **single-solution** SA. Population size is forced to `1` — population-based SA
is not acceptable in the literature; the algorithm starts from one initial solution and
anneals it only. Each iteration, one neighbor is proposed via `choose_mutation` and:

- accepted outright if better (`cost ≤ current`), or
- accepted with Metropolis probability `exp(-Δ/T)` if worse.

Temperature is cooled geometrically after each step:
`T ← max(T_min, T · cooling_rate)`. There is no crossover, no population selection,
no immigrants.

> **Design notes:** A population-based multi-walker SA would share a temperature
> schedule across independent chains with no selection pressure between them — this
> does not correspond to the SA algorithm described in the literature. The
> single-solution version is used throughout this project.

---

## 5. `ParticleSwarm` (`src/algos/ga_pso.py`)

> Kennedy, J., & Eberhart, R. (1995, November). Particle swarm optimization.
> *Proceedings of ICNN'95* (Vol. 4, pp. 1942–1948).

Discrete PSO adapted for integer-encoded assignment problems. The original PSO operates
over a continuous real-valued space; here each gene `x[j]` is an integer facility
index. The standard PSO velocity equation is preserved exactly, but applied over the
integer domain via rounding and clamping:

**Velocity update** (real-valued, one entry per job):
```
v[j] ← w · v[j]
      + c1 · r1 · (pbest[j] − x[j])    # cognitive: pull toward personal best
      + c2 · r2 · (gbest[j] − x[j])    # social:    pull toward global best
```

**Position update** (integer, clamped to valid facility range):
```
x_new[j] = clip( round( x[j] + v[j] ), 0, I−1 )  →  repair  →  evaluate
```

Velocity is clamped to `[−I, I]` (I = number of facilities) to bound the maximum
step size. This extension preserves the original PSO velocity semantics — cognitive
and social pulls are proportional to the integer distance between positions —
while mapping updates back into the feasible integer domain via rounding and repair.

Particle identity (`self.particles` / `self.personal_best` / `self.velocities`) is
maintained separately from `self.population`'s ordering, since `BaseGA.run()` re-sorts
`self.population` by cost after every step, which would corrupt positional identity.

> **Design notes:** PSO is a continuous-space algorithm by origin. Applying it directly
> to integer-encoded GQAP requires explicit justification: the velocity equation
> produces a real-valued displacement, which is discretized to the nearest integer
> facility index and then repaired for capacity feasibility. This is the standard
> extension used in the discrete/integer PSO literature for non-permutation integer
> problems.

---

## 6. `HybridGAPSO` (`src/algos/ga_hybrid_gapso.py`)

> Juang, C. F. (2004). A hybrid of genetic algorithm and particle swarm optimization
> for recurrent network design. *IEEE Transactions on Systems, Man, and Cybernetics,
> Part B*, 34(2), 997–1006.

Each generation, the population is ranked by cost and split in two:

- the better-ranked `pso_fraction` undergo a PSO-style discrete move (exploitation,
  pulled toward personal best or global best via `choose_crossover`, or diversified via
  `choose_mutation` for the inertia term)
- the rest are regenerated by ordinary GA crossover + mutation via `choose_crossover` /
  `choose_mutation`, selected via `select_from_pool` (exploration)

An elitist replacement step guarantees the swarm's best-ever solution survives into
the next generation, as in the original paper.

> **Design notes:** The GA component uses the same `choose_crossover` / `choose_mutation`
> operator sets as all other algorithms in this project.

---

## 7. `HybridGASA` (`src/algos/ga_hybrid_gasa.py`)

> Chen, P. H., & Shahandashti, S. M. (2009). Hybrid of genetic algorithm and simulated
> annealing for multiple project scheduling with multiple resource constraints.
> *Automation in Construction*, 18(4), 434–443.

GA crossover + mutation (via `choose_crossover` / `choose_mutation`, same operator sets
as all other algorithms) supply every candidate move; what changes is the **acceptance
rule**. Each child must first pass a Metropolis test against its own parent baseline
before it is eligible to join the survivor pool:

- a child better than its baseline is always accepted,
- a worse child is accepted with probability `exp(-Δ/T)`.

`T` anneals down geometrically over the run, so early generations tolerate
quality-losing moves (escaping local optima) while late generations behave like a
plain elitist GA.

> **Design notes:** This hybrid retains a population-based structure (unlike standalone
> SA, §4), which is appropriate here because the annealing only governs the *acceptance
> criterion* — the population-level crossover and diversity selection still operate as
> in GEA.

---

## 8. `GEAScenario1` — Crossover with Robust Chromosome (`src/algos/ga_gea_scenario_1.py`)

Modernization of `GEA` — Scenario 1: **RC (Robust Chromosome) crossover**.

Runs three operators each generation at **fixed rates** — no adaptive lambda:

1. **Regular crossover** (`choose_crossover`, all 4 operators) at `crossover_rate`.
2. **Regular mutation** (`choose_mutation`, all 9 operators) at `mutation_rate`.
3. **RC crossover** (`rc_rate`): per gene, the child inherits from whichever parent's
   assigned facility has *more remaining capacity slack* (`cvar`) — the more
   capacity-robust gene, independent of its assignment cost.

All three sets of offspring are pooled with the current population; `DiversitySelector`
picks the next generation.

> **Design notes:** Adaptive lambda control (`lambda_rc`, `alpha`, `epsilon`) has been
> removed from this scenario. The adaptive version is out of scope for this paper and
> will be proposed separately. All crossover and mutation operators are identical to
> those used in the base GEA and StandardGA.

---

## 9. `GEAScenario2` — Directed Mutation (`src/algos/ga_gea_scenario_2.py`)

Modernization of `GEA` — Scenario 2: **DM (Directed Mutation)**.

Runs three operators each generation at **fixed rates** — no adaptive lambda:

1. **Regular crossover** (`choose_crossover`, all 4 operators) at `crossover_rate`.
2. **Regular mutation** (`choose_mutation`, all 9 operators) at `mutation_rate`.
3. **DM** (`mutation_greedy_reassign`, `dm_rate`): moves each selected individual's
   single worst-assigned job to its cheapest feasible facility — a directed,
   fitness-improving move rather than a blind random one.

> **Design notes:** As with Scenario 1, adaptive lambda control has been removed.
> Crossover and mutation operator sets are identical to the rest of the family.

---

## 10. `GEAScenario3` — Gene Injection (`src/algos/ga_gea_scenario_3.py`)

Modernization of `GEA` — Scenario 3: **GI (Gene Injection)**.

Runs three operators each generation at **fixed rates** — no adaptive lambda:

1. **Regular crossover** (`choose_crossover`, all 4 operators) at `crossover_rate`.
2. **Regular mutation** (`choose_mutation`, all 9 operators) at `mutation_rate`.
3. **GI** (`mutation_random`, `injection_rate`): replaces a handful of random genes
   with brand-new random facility assignments, injecting fresh genetic material rather
   than perturbing the existing assignment.

> **Design notes:** As with Scenarios 1 and 2, adaptive lambda control has been
> removed. Crossover and mutation operator sets are identical to the rest of the family.

---

## Not implemented

`GSAIS-KMeans`, `GSAIS-DBSCAN`, and `GSAIS-NN` (Gromov, Sohrabi, & Fathollahi-Fard,
*Genetic Speciation Algorithm with Interplay among Species*, under review at
*Algorithms*) are **not implemented**: the paper is unpublished and not yet available,
so there is no reliable basis for a faithful implementation of the speciation/interplay
mechanism. This is acceptable at the current stage.

---

## Hyperparameter Tuning (`scripts/tune_algorithm.py`)

Tuning is performed using the **Taguchi method**: an orthogonal array experiment that
efficiently explores a discrete grid of candidate hyperparameter combinations with far
fewer runs than a full factorial search.

### Method

For each algorithm, **three candidate values** are defined per tunable hyperparameter.
These candidate values are provided by the research team based on domain knowledge and
preliminary exploration. The candidates are arranged into a Taguchi orthogonal array
(e.g. L9 for up to 4 factors, L27 for up to 13 factors), where each row of the array
is one parameter combination to evaluate.

Each combination is run **30 times** on a single representative test problem, and the
result considered for that combination is the **average of the 30 minimum costs** found
across those runs. The winning combination is the one with the lowest average.

After selecting the best combination, **Relative Percentage Deviation (RPD)** is
computed for each candidate combination:

```
RPD = (cost_candidate − cost_best) / cost_best × 100 %
```

where `cost_best` is the minimum cost found across all combinations. RPD plots and a
parameter-combination table are reported in the paper.

### Infrastructure (`scripts/tune_algorithm.py`)

The current implementation uses **Optuna** (Bayesian / TPE search) over continuous
parameter ranges as a working approximation until the Taguchi candidate values are
finalized by the research team. Once the three candidate values per parameter are
provided, the search will be replaced with a Taguchi orthogonal-array grid.

Each algorithm has its own tune config in `scripts/conf/tune_algorithm/<algo>.yaml`,
which specifies:
- `runs`: number of independent runs per candidate (target: 30)
- `n_candidates`: number of Optuna trials (will become number of Taguchi rows)
- `param_space`: `[min, max]` bounds per parameter (will become `[v1, v2, v3]` lists)
- `output_file`: path for tuning results JSON

To run tuning for a specific algorithm:
```bash
poetry run python scripts/tune_algorithm.py --config-name="tune_algorithm/<algo>"
# e.g.:
poetry run python scripts/tune_algorithm.py --config-name="tune_algorithm/gea"
```

### Algorithms in scope for tuning

| Algorithm | Key tunable parameters |
|---|---|
| `StandardGA` | `crossover_rate`, `mutation_rate`, `elitism_count` |
| `GEA` | `crossover_rate`, `mutation_rate`, `rc_rate`, `dm_rate`, `injection_rate`, `stagnation_limit`, `immigrant_rate` |
| `SA` | `initial_temperature`, `cooling_rate`, `min_temperature` |
| `PSO` | `inertia_weight`, `cognitive_weight`, `social_weight`, `stagnation_limit` |
| `HybridGAPSO` | `pso_fraction`, `inertia_weight`, `cognitive_weight`, `social_weight`, `crossover_rate`, `mutation_rate` |
| `HybridGASA` | `crossover_rate`, `mutation_rate`, `initial_temperature`, `cooling_rate` |
| `GEAScenario1` | `crossover_rate`, `mutation_rate`, `rc_rate`, `stagnation_limit` |
| `GEAScenario2` | `crossover_rate`, `mutation_rate`, `dm_rate`, `stagnation_limit` |
| `AdaptiveGEA` | `crossover_rate`, `mutation_rate`, `alpha`, `lambda_min`, `lambda_max`, `stagnation_limit` |
| `GEAScenario3` | `crossover_rate`, `mutation_rate`, `injection_rate`, `stagnation_limit` |

---

## Final Experiment

After tuning, each algorithm is run on every test case to produce the paper's
main results table.

### Protocol

- **Runs per algorithm per test case**: 30 independent runs, each from a fresh random
  seed.
- **Time limit**: 1000 seconds of wall-clock time per run.
- **Statistics reported**: Min, Max, AVG, Std across the 30 runs per (algorithm, test
  case) cell. Additionally, the **number of Best** (algorithm achieves the best known
  Min on that instance) and **number of Unique Best** (algorithm is the only one
  achieving that Min) are reported at the bottom of the main table, computed from the
  Min column.

### Data stored per run

Each run stores the following for analysis and plotting:

1. **Best cost** — the minimum cost found at the end of the run (last element of the
   BestCosts list below).
2. **BestCosts list** (`cost_history` in `GALogger`) — the running best cost recorded
   at the end of every iteration, i.e. `best_cost[t]` for `t = 1, 2, ..., T`. Used to
   plot minimization curves and to compute Std interval plots (§3.3.14). Saved only in
   final experiment runs (`run.py`), not during tuning (memory-intensive for 30×n_candidates
   trials). Also stored: `nfe_history[t]` — the cumulative NFE at the same snapshot points,
   pairing each cost sample with its matching computational budget.
3. **NFE (Number of Function Evaluations)** (`nfe` in `GALogger`) — incremented once
   per individual cost evaluation: population initialization, each crossover/mutant/immigrant
   child, and each local-search probe. The final total is stored per run; the average
   across 30 runs per (algorithm, instance) pair is reported in the NFE table.

### Analysis outputs

| Output | Description |
|---|---|
| **Main results table** | Min / Max / AVG / Std × algorithm × test case; #Best and #Unique Best at the bottom. |
| **Std interval plots** | Standard deviation across 30 runs plotted as a function of iteration, for all algorithms on each test case — measures robustness and convergence stability. |
| **CPU time table** | Wall-clock time per run; plotted separately for small-scale and large-scale instances. |
| **Optimality Gap (OG) table** | `(algo_min − exact) / exact × 100 %` for small-scale instances where the exact solution is known (e.g. `c201535`). |
| **Hitting time** | Across the 30 runs, identifies the best run and records the iteration and wall-clock time at which the final minimum was first reached. Reported as a table or plot. |
| **NFE table** | Rounded average NFE for each algorithm × test case (2D table: rows = instances, columns = algorithms). Enables comparison of computational effort relative to solution quality. |

> **Implementation status:** `run.py` stores Min/Max/AVG/Std, hitting time, NFE
> (per-run and mean), and full BestCosts + NFE histories per run (the latter only in
> final-experiment runs, not during tuning). OG (Optimality Gap) is **pending** —
> requires known exact solutions for the small-scale instances.
