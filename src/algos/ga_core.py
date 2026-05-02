import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps
from typing import List

import numpy as np

from src.algos.heuristics import heuristic2
from src.algos.utils import evaluate_permutation
from src.data.models import Individual, Model
from src.operators.crossover import choose_crossover
from src.operators.mutations import choose_mutation

# =========================
# Timing Decorator
# =========================


def timed(operation_name: str):
    """Decorator to measure execution time of operations."""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.perf_counter()
            result = func(self, *args, **kwargs)
            elapsed = time.perf_counter() - start

            if not hasattr(self, "_timing"):
                self._timing = defaultdict(float)
                self._timing_calls = defaultdict(int)

            self._timing[operation_name] += elapsed
            self._timing_calls[operation_name] += 1
            return result

        return wrapper

    return decorator


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
        best = heuristic2(self.model)
        self.population = [best]

        while len(self.population) < self.population_size:
            perm = choose_mutation(best.permutation, self.model, self.rng)
            ind = evaluate_permutation(perm, self.model)
            if math.isfinite(ind.cost):
                self.population.append(ind)

        self.population.sort(key=lambda x: x.cost)
        self.best_solution = self.population[0]
        self.worst_cost = self.population[-1].cost

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
                child = evaluate_permutation(perm, self.model)
                if math.isfinite(child.cost):
                    offspring.append(child)
                    self._crossover_valid += 1

                    # Check if it improves global best
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
            ind = evaluate_permutation(perm, self.model)

            if math.isfinite(ind.cost):
                mutations.append(ind)
                self._mutation_valid += 1

                # Check if it improves global best
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
