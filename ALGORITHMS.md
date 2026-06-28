# Algorithms

This project solves the Generalized Quadratic Assignment Problem (GQAP, see
[PROBLEM_DESCRIPTION.md](PROBLEM_DESCRIPTION.md)) with ten metaheuristics, all sharing
the same solution representation, problem instance format, and a common scaffolding
class (`BaseGA`). This document describes that shared scaffolding first, then each
algorithm built on top of it.

## Shared framework (`src/algos/base.py`)

### Representation

A candidate solution (`Individual`, in `src/data/models.py`) is a permutation array
`permutation` of length `J` (number of jobs), where `permutation[j]` is the facility
index assigned to job `j`. Alongside it, every `Individual` carries:

- `cost`: the GQAP objective value (`inf` if the assignment violates a facility's
  capacity)
- `cvar`: per-facility remaining capacity slack (`b_i - load_i`), used by some
  operators below as a feasibility-robustness signal, not just by repair

A problem instance (`Model`) holds the assignment-cost matrix `cij`, resource-usage
matrix `aij`, facility capacities `bi`, facility-distance matrix `DIS`, and
job-interaction matrix `F`.

### Cost evaluation (`src/costs.py`)

`cost_function_perm` computes a permutation's cost from scratch (assignment cost +
quadratic interaction cost), returning `inf` if any facility is over capacity.
`cost_function_perm_delta` computes the same cost incrementally from a known-feasible
baseline cost, touching only the positions that changed — an O(K·J) update instead of
a full O(J²) recompute when K (positions changed) is small, which is the common case
for a single crossover or mutation. Every algorithm below routes its offspring through
the batched version of this (`evaluate_permutation_delta_batch`), each paired with
whichever parent/individual it's "closest to" so the delta stays cheap.

### Operators

**Crossover** (`src/operators/crossover.py`), dispatched uniformly at random by
`choose_crossover` unless an algorithm calls a specific one directly:

| Operator | Mechanism |
|---|---|
| `crossover_one_point` | Splits both parents at one random point; each child is one parent's prefix + the other's suffix. |
| `crossover_two_point` | Splices a random middle segment from one parent into the other's copy, and vice versa. |
| `crossover_uniform` | Per-gene coin flip decides which parent each child inherits from. |
| `crossover_greedy` | Per-gene, child 1 takes whichever parent's assignment is cheaper (`cij`); child 2 takes the complementary (worse-of) gene, so the pair still spans the same diversity a random crossover would. |

**Mutation** (`src/operators/mutations.py`), dispatched uniformly at random by
`choose_mutation` unless an algorithm calls a specific one directly:

| Operator | Mechanism |
|---|---|
| `mutation_swap` | Swaps two adjacent genes. |
| `mutation_reversion` | Reverses a random contiguous segment. |
| `mutation_insertion` | Moves a random segment to a different position. |
| `mutation_big_swap` | Swaps two genes at arbitrary (non-adjacent) positions. |
| `mutation_random` | Reassigns 1–5 random genes to brand-new random facilities ("gene injection" — fresh genetic material rather than a perturbation of the existing assignment). |
| `mutation_scramble` | Shuffles the values within a random window of 3–5 genes. |
| `mutation_cyclic_shift` | Cyclically rotates 3 random genes. |
| `mutation_greedy_reassign` | Directed move: finds the single worst-assigned job and reassigns it to the cheapest facility with spare capacity. |
| `mutation_migration` | Capacity-aware: moves the priciest job out of the most-loaded facility into the facility with the most spare capacity. |

### Repair (`src/repair.py`)

Random initialization and every crossover/mutation can produce an infeasible
permutation (a facility over capacity). Repair iteratively evicts an overloaded
facility's job and reassigns it to a facility with spare capacity, until feasible or a
repair-attempt budget is exhausted:

- **`GreedyRepair`**: deterministic — always evicts the worst capacity violation and
  reassigns to the cheapest feasible facility.
- **`RFRepair`** (the default for the GEA family): evicts from and reassigns to a
  random subsample of candidates instead of the single greedy best, trading exact
  greediness for population diversity.

### Selection (`src/selection.py`, `DiversitySelector`)

- **Parent selection**: roulette wheel weighted by a softmax over each individual's
  mean Hamming distance to the rest of the population (more distinctive individuals
  are more likely to be picked as parents).
- **Survivor selection**: the cheapest `elite_fraction` of the merged pool
  (parents + offspring + mutants + immigrants, deduplicated) survives outright as
  elites; the rest is ranked by `diversity_weight · diversity_score + cost_weight ·
  cost_score`, with `diversity_weight` decaying linearly over the run so later
  generations converge instead of being held apart by the diversity term.

### Local search (memetic polish)

Each generation, the top 3 individuals get a per-job hill-climbing pass (try every
facility for each job, keep whichever lowers cost), auto-disabled above `J = 50` jobs
where the O(J·I) cost isn't worth it.

### Run loop

`run(time_limit)` initializes the population, then repeatedly calls the subclass's
`step()`, re-sorts by cost, polishes elites, and stops once `time_limit` (wall-clock
seconds) or `iterations` is reached, returning the best `Individual` found.

---

## 1. `StandardGA` (`src/algos/ga_standard.py`)

> Holland, J. H. (1992). Genetic algorithms. *Scientific American*, 267(1), 66–73.

The textbook simple genetic algorithm, kept as close to Holland's original description
as the capacity-constrained encoding allows. It deliberately **does not** use any of
this project's own enhancements below:

- **Selection**: plain fitness-proportionate (roulette wheel) selection on raw cost
  (`fitness = 1 / (cost + ε)`) — no diversity weighting.
- **Crossover**: single-point crossover, applied with probability `crossover_rate` per
  mating pair; otherwise the pair passes through unchanged.
- **Mutation**: per-gene mutation with probability `mutation_rate` — each gene
  independently has a chance to be reassigned to a random facility.
- **Replacement**: full generational replacement (children entirely replace the
  population), with a minimal single-individual elitism (`elitism_count`, default `1`)
  so the best-found solution is never lost to randomness — the one common,
  near-universal addition to Holland's description.
- No diversity selection, no randomized repair sampling (uses deterministic
  `GreedyRepair` by default), no stagnation immigrants, no memetic local search.

This is the literal baseline the rest of the algorithms in this project are compared
against.

## 2. `GEA` (`src/algos/ga_gea.py`)

This project's own enhanced GA — `BaseGA`'s full machinery with fixed crossover and
mutation rates:

- **Selection**: `DiversitySelector` (diversity-weighted roulette parents, elite +
  diversity/cost hybrid survivors — see "Shared framework" above).
- **Crossover / mutation**: each generation draws `crossover_rate · population_size`
  crossovers (rounded to an even count) and `mutation_rate · population_size`
  mutations, each via `choose_crossover` / `choose_mutation`'s random dispatch.
- **Repair**: `RFRepair` by default (randomized, diversity-preserving).
- **Stagnation immigrants**: if `best_solution` hasn't improved for
  `stagnation_limit` iterations, `immigrant_rate · population_size` fresh random
  individuals are injected to give crossover new material to recombine with.
- **Memetic local search**: enabled (see "Shared framework").

## 3. `AdaptiveGA` (`src/algos/ga_adaptive.py`)

The self-adaptive variant proposed for this project: identical to `GEA`, except the
crossover and mutation rates are no longer fixed — each is scaled by its own lambda
multiplier (`lambda_crossover`, `lambda_mutation`, both starting at `1.0`) that adapts
every generation to how much improvement that operator actually produced:

1. Each offspring/mutant's improvement over its **own parent baseline** is normalized:
   `delta = (parent_cost - child_cost) / (parent_cost + epsilon)`, and accumulated
   per operator (crossover sum, mutation sum).
   - Comparing to the parent baseline (not the global best) is a deliberate choice:
     comparing every child to the best-ever solution makes delta almost always
     negative once the population is decent, which would drive both lambdas toward
     `lambda_min` exactly as the run converges — choking off the exploration needed to
     escape a local optimum.
2. `lambda_new = clip(lambda_old + alpha * delta_sum, lambda_min, lambda_max)`.
3. Next generation's operator counts become
   `base_rate · population_size · lambda` instead of just `base_rate · population_size`.

An operator that's currently producing improvements gets used more; one that isn't
gets throttled back, without any manual per-instance rate tuning.

## 4. `SimulatedAnnealing` (`src/algos/ga_sa.py`)

> Bertsimas, D., & Tsitsiklis, J. (1993). Simulated annealing. *Statistical Science*, 8(1), 10–15.

Classic SA anneals a single solution; here the population is a bank of independent
annealing chains (one walker per population slot) sharing a single global temperature
schedule, reusing `BaseGA`'s batched repair/evaluation instead of looping one
permutation at a time in pure Python.

Each iteration, every walker proposes one neighbor via `choose_mutation` and:
- accepts it outright if it's better (`cost ≤` current), or
- accepts it with Metropolis probability `exp(-Δ/T)` if it's worse,

then the shared temperature is cooled geometrically: `T ← max(T_min, T · cooling_rate)`.
There is no crossover and no selection pressure between walkers — each chain only ever
competes against its own previous state. Stagnation immigrants are still used (replacing
random walkers, not added to a larger pool).

## 5. `ParticleSwarm` (`src/algos/ga_pso.py`)

> Kennedy, J., & Eberhart, R. (1995, November). Particle swarm optimization. *Proceedings of ICNN'95* (Vol. 4, pp. 1942–1948).

There's no continuous position/velocity in a permutation encoding. Following the common
discrete-PSO convention for permutation problems, each particle's "pull" is realized via
existing operators instead of vector arithmetic:

- with probability `inertia_weight`: a random move via `choose_mutation` (exploration)
- with probability `cognitive_weight`: crossover with the particle's own personal best
- with probability `social_weight`: crossover with the swarm's global best

(the three weights are normalized to sum to 1). Each particle tracks its own position
and personal best independently of `self.population`'s ordering — `BaseGA.run()`
re-sorts `self.population` by cost after every step, which would scramble a particle's
identity if it were tracked by position in that list instead.

## 6. `HybridGAPSO` (`src/algos/ga_hybrid_gapso.py`)

> Juang, C. F. (2004). A hybrid of genetic algorithm and particle swarm optimization for recurrent network design. *IEEE Transactions on Systems, Man, and Cybernetics, Part B*, 34(2), 997–1006.

Each generation, the population is ranked by cost and split in two:

- the better-ranked `pso_fraction` undergo the PSO-style move described above
  (exploitation, pulled toward personal/global best)
- the rest are regenerated by ordinary GA crossover + mutation, selected from
  among themselves via `select_from_pool` (exploration)

An elitist replacement step then guarantees the swarm's best-ever solution survives
into the next generation even if every particle's own move that round was a loss, as in
the original paper.

## 7. `HybridGASA` (`src/algos/ga_hybrid_gasa.py`)

> Chen, P. H., & Shahandashti, S. M. (2009). Hybrid of genetic algorithm and simulated annealing for multiple project scheduling with multiple resource constraints. *Automation in Construction*, 18(4), 434–443.

GA crossover/mutation still supplies every candidate move; what changes is the
acceptance rule. Each child must first pass a Metropolis test against its own parent
baseline before it's even eligible to join the survivor pool: a child better than its
baseline is always accepted, a worse one only gets a chance proportional to
`exp(-Δ/T)`. `T` anneals down geometrically over the run, so early generations tolerate
quality-losing moves (escaping local optima) while late generations behave like a plain
elitist GA.

## 8. `GEAScenario1` — Crossover with Robust Chromosome (`src/algos/gea_scenario_1.py`)

Modernization of `GEA`: replaces its randomly-dispatched crossover with a new operator,
**RC (Robust Chromosome) crossover** — per gene, the child inherits from whichever
parent's assigned facility has *more remaining capacity slack* (`cvar`), i.e. the more
capacity-robust gene, independent of its assignment cost. This operator's rate is
scaled by its own adaptive `lambda_rc` multiplier, using the exact same update/clip
rule `AdaptiveGA` uses for `lambda_crossover`:

(a) apply RC crossover, evaluate offspring;
(b) compute `delta_rc` against each offspring's parent baseline;
(c) `lambda_rc ← lambda_rc + alpha · delta_rc`;
(d) clip `lambda_rc` to `[lambda_min, lambda_max]`.

Mutation stays GEA's regular, non-adaptive mixed mutation operator.

## 9. `GEAScenario2` — Directed Mutation (`src/algos/gea_scenario_2.py`)

Modernization of `GEA`: replaces its randomly-dispatched mutation with
**DM (Directed Mutation)** — reuses `mutation_greedy_reassign` (move the single
worst-assigned job to its cheapest feasible facility), a directed, fitness-improving
move rather than a blind random one. Its rate is scaled by an adaptive `lambda_dm`
multiplier, updated and clipped the same way as `lambda_rc` above. Crossover stays
GEA's regular, non-adaptive mixed crossover operator.

## 10. `GEAScenario3` — Gene Injection (`src/algos/gea_scenario_3.py`)

Modernization of `GEA`: adds a **third** operator, **GI (Gene Injection)**, alongside
GEA's regular fixed-rate crossover and mutation. GI reuses `mutation_random` (replace a
handful of random genes with brand-new random facility assignments), injecting fresh
genetic material rather than perturbing the existing assignment. Its own rate is scaled
by an adaptive `lambda_gi` multiplier, updated and clipped the same way as `lambda_rc`
and `lambda_dm` above.

---

## Not implemented

`GSAIS-KMeans`, `GSAIS-DBSCAN`, and `GSAIS-NN` (Gromov, Sohrabi, & Fathollahi-Fard,
*Genetic Speciation Algorithm with Interplay among Species*, under review at
*Algorithms*) were requested but are **not implemented**: the paper is unpublished and
not available, so there's no reliable basis for a faithful implementation of its
speciation/interplay mechanism.
