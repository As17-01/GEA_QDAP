import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List

import numpy as np

from src.algos.utils import evaluate_permutation, timed
from src.data.models import Individual, Model
from src.operators.crossover import choose_crossover
from src.operators.mutations import choose_mutation

# =========================
# Base Genetic Algorithm
# =========================


class BaseGA(ABC):
    def __init__(self, model: Model, population_size: int, iterations: int):
        self.model = model
        self.population_size = population_size
        self.iterations = iterations

        self.rng = np.random.default_rng()
        self.population: List[Individual] = []
        self.best_solution: Individual | None = None
        self.worst_cost: float = float("inf")

        # Timing tracking
        self._timing = defaultdict(float)
        self._timing_calls = defaultdict(int)
        self._iteration_times: List[float] = []
        self.start_time: float = 0.0

        # Crossover & Mutation statistics
        self._crossover_attempts = 0
        self._crossover_valid = 0
        self._crossover_new_best = 0
        self._mutation_attempts = 0
        self._mutation_valid = 0
        self._mutation_new_best = 0

    # =========================
    # Initialization
    # =========================

    @timed("initialization")
    def initialize_population(self) -> None:
        self.population = []
        for _ in range(self.population_size):
            perm = self.rng.integers(0, self.model.I, size=self.model.J, dtype=int)
            perm = self.repair_permutation(perm)
            ind = evaluate_permutation(perm, self.model)
            self.population.append(ind)

        self.population.sort(key=lambda x: x.cost)
        self.best_solution = self.population[0]
        self.worst_cost = self.population[-1].cost

    # =========================
    # Fast Repair Algorithm
    # =========================

    def repair_permutation(self, perm: np.ndarray, max_repair_attempts: int = 100) -> np.ndarray:
        perm = perm.copy()
        I = self.model.I
        J = self.model.J
        aij = self.model.aij
        bi = self.model.bi

        # Current loads and slack
        loads = np.bincount(perm, weights=aij[perm, np.arange(J)], minlength=I)
        slack = bi - loads

        attempts = 0

        while attempts < max_repair_attempts:
            attempts += 1

            # Find all overloaded machines
            overloaded = np.where(slack < -1e-9)[0]
            if len(overloaded) == 0:
                break

            # Find all jobs on overloaded machines
            on_overloaded = np.isin(perm, overloaded)
            if not np.any(on_overloaded):
                break

            # Among jobs on overloaded machines, pick the one with largest aij on its current machine
            candidates = np.where(on_overloaded)[0]
            current_loads = aij[perm[candidates], candidates]
            j_remove = candidates[np.argmax(current_loads)]

            i_old = perm[j_remove]

            # Remove temporarily
            loads[i_old] -= aij[i_old, j_remove]
            slack[i_old] = bi[i_old] - loads[i_old]
            perm[j_remove] = -1

            # === Choose best target machine ===
            aij_j = aij[:, j_remove]

            # Prefer feasible machines with smallest load increase
            feasible_mask = slack + aij_j <= bi + 1e-9

            if np.any(feasible_mask):
                costs = np.where(feasible_mask, aij_j, np.inf)
                target = int(np.argmin(costs))
            else:
                # No feasible → go to machine with largest slack
                target = int(np.argmax(slack))

            # Assign
            perm[j_remove] = target
            loads[target] += aij[target, j_remove]
            slack[target] = bi[target] - loads[target]

        # Final safety pass: force assign remaining if any (very rare)
        if np.any(perm == -1):
            for j in np.where(perm == -1)[0]:
                target = int(np.argmax(slack))
                perm[j] = target
                loads[target] += aij[target, j]
                slack[target] = bi[target] - loads[target]

        return perm

    # =========================
    # Selection
    # =========================

    @timed("selection")
    def compute_selection_probabilities(self, beta: float = 10.0) -> np.ndarray:
        costs = np.array([ind.cost for ind in self.population], dtype=float)
        probs = np.exp(-beta * costs / self.worst_cost)
        return probs / probs.sum()

    def _roulette_wheel_selection(self, probabilities: np.ndarray) -> int:
        return int(np.searchsorted(np.cumsum(probabilities), self.rng.random(), side="right"))

    # =========================
    # Operators with improved tracking
    # =========================

    @timed("crossover")
    def crossover(self, probabilities: np.ndarray, n: int) -> List[Individual]:
        offspring = []
        self._crossover_attempts += n

        for _ in range(0, n, 2):
            i1 = self._roulette_wheel_selection(probabilities)
            i2 = self._roulette_wheel_selection(probabilities)

            p1, p2 = self.population[i1], self.population[i2]
            perms = choose_crossover((p1, p2), self.rng)

            for perm in perms:
                perm = self.repair_permutation(perm)
                child = evaluate_permutation(perm, self.model)

                offspring.append(child)
                if math.isfinite(child.cost):
                    self._crossover_valid += 1

                    if child.cost < self.best_solution.cost:
                        self._crossover_new_best += 1

        return offspring

    @timed("mutation")
    def mutate(self, n: int) -> List[Individual]:
        mutations = []
        self._mutation_attempts += n

        for _ in range(n):
            idx = self.rng.integers(0, len(self.population))
            perm = choose_mutation(self.population[idx].permutation, self.model, self.rng)
            perm = self.repair_permutation(perm)
            ind = evaluate_permutation(perm, self.model)

            mutations.append(ind)
            if math.isfinite(ind.cost):
                self._mutation_valid += 1

                if ind.cost < self.best_solution.cost:
                    self._mutation_new_best += 1

        return mutations

    # =========================
    # Evolution step
    # =========================

    @abstractmethod
    def step(self) -> None:
        pass

    # =========================
    # Run loop
    # =========================

    def run(self, time_limit: float | None = None):
        self.start_time = time.perf_counter()
        self.initialize_population()

        print(f"GA started → Population: {self.population_size:,} | Iterations: {self.iterations}\n")

        for it in range(1, self.iterations + 1):
            iter_start = time.perf_counter()

            self.step()

            iter_time = time.perf_counter() - iter_start
            self._iteration_times.append(iter_time)

            self.population.sort(key=lambda x: x.cost)
            self.best_solution = self.population[0]
            self.worst_cost = max(self.worst_cost, self.population[-1].cost)

            # Detailed logging every 50 iterations
            if it % 50 == 0:
                print(
                    f"Iter {it:5d} | Best: {self.best_solution.cost:12.4f} | "
                    f"Time: {iter_time*1000:6.2f}ms | "
                    f"CX: {self._crossover_new_best:3d}/{self._crossover_valid:5d}/{self._crossover_attempts:5d} | "
                    f"MUT: {self._mutation_new_best:3d}/{self._mutation_valid:5d}/{self._mutation_attempts:5d}"
                )

                # Reset counters for next block
                self._crossover_attempts = 0
                self._crossover_valid = 0
                self._crossover_new_best = 0
                self._mutation_attempts = 0
                self._mutation_valid = 0
                self._mutation_new_best = 0

            if time_limit and (time.perf_counter() - self.start_time) >= time_limit:
                print(f"Time limit reached at iteration {it}")
                break

        self._print_final_statistics()
        return self.best_solution

    # =========================
    # Final statistics
    # =========================

    def _print_final_statistics(self):
        total_time = time.perf_counter() - self.start_time
        num_iters = len(self._iteration_times)
        avg_iter = sum(self._iteration_times) / num_iters if num_iters > 0 else 0

        print("\n" + "=" * 90)
        print("GENETIC ALGORITHM - FINAL REPORT")
        print("=" * 90)
        print(f"Total runtime            : {total_time:8.4f} seconds")
        print(f"Iterations completed     : {num_iters}")
        print(f"Average time / iteration : {avg_iter*1000:8.2f} ms")
        print(f"Final best cost          : {self.best_solution.cost:.6f}")

        print("\nTime breakdown:")
        for op, t in sorted(self._timing.items(), key=lambda x: x[1], reverse=True):
            calls = self._timing_calls[op]
            avg = t / calls if calls > 0 else 0
            print(f"  {op:18s} : {t:8.4f}s ({calls:5d} calls, {avg*1000:6.2f} ms avg)")

        print("=" * 90)
