import math
import time
from abc import abstractmethod
from typing import List

import numpy as np

from src.algos.costs import evaluate_permutation, get_diversity
from src.algos.ga_logging import LoggingGA
from src.algos.utils import timed
from src.data.models import Individual, Model
from src.operators.crossover import choose_crossover
from src.operators.mutations import choose_mutation
from src.operators.repair import RFRepair


class BaseGA(LoggingGA):
    def __init__(self, model: Model, population_size: int, iterations: int):
        super().__init__()
        self.model = model
        self.population_size = population_size
        self.iterations = iterations

        self.repair_class = RFRepair(self.model)

        # Core GA objects
        self.rng = np.random.default_rng()
        self.population: List[Individual] = []
        self.best_solution: Individual | None = None
        self.worst_cost: float = float("inf")

    @timed("repair")
    def repair_wrapper(self, perm):
        return self.repair_class.repair(perm)

    @timed("initialization")
    def initialize_population(self) -> None:
        self.population = []

        for _ in range(self.population_size):
            perm = self.rng.integers(0, self.model.I, size=self.model.J, dtype=int)
            perm = self.repair_wrapper(perm)
            ind = evaluate_permutation(perm, self.model)
            self.population.append(ind)

        self.population.sort(key=lambda x: x.cost)
        self.best_solution = self.population[0]
        self.worst_cost = self.population[-1].cost

    @timed("selection")
    def compute_selection_probabilities(self, beta: float = 10.0) -> np.ndarray:
        diversity_scores = get_diversity(population_base=self.population, population_to_eval=self.population)

        probs = np.exp(beta * diversity_scores)
        probs = probs / probs.sum()

        return probs

    def _roulette_wheel_selection(self, probabilities: np.ndarray) -> int:
        return int(np.searchsorted(np.cumsum(probabilities), self.rng.random(), side="right"))

    def select_from_pool(self, pool):
        unique_pool = list(dict.fromkeys(pool))
        n = self.population_size

        unique_pool.sort(key=lambda x: x.cost)

        n_elite = n // 2
        elite = unique_pool[:n_elite]
        remaining = unique_pool[n_elite:]

        diversity_array = get_diversity(population_base=elite, population_to_eval=remaining)
        scored_remaining = list(zip(remaining, diversity_array))
        scored_remaining.sort(key=lambda x: x[1])

        n_diverse = n - n_elite
        diverse_selected = [ind for ind, _ in scored_remaining[:n_diverse]]

        self.population = elite + diverse_selected

    @timed("crossover")
    def crossover(self, probabilities: np.ndarray, n: int) -> List[Individual]:
        offspring = []
        self._crossover_attempts += n

        for _ in range(0, n, 2):
            i1 = self._roulette_wheel_selection(probabilities)
            i2 = self._roulette_wheel_selection(probabilities)

            p1, p2 = self.population[i1], self.population[i2]
            perms = choose_crossover((p1, p2))

            for perm in perms:
                perm = self.repair_wrapper(perm)
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
            perm = choose_mutation(self.population[idx].permutation, self.model)
            perm = self.repair_wrapper(perm)
            ind = evaluate_permutation(perm, self.model)

            mutations.append(ind)
            if math.isfinite(ind.cost):
                self._mutation_valid += 1
                if ind.cost < self.best_solution.cost:
                    self._mutation_new_best += 1

        return mutations

    @abstractmethod
    def step(self) -> None:
        pass

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

            if it % 50 == 0:
                self.print_iteration_info(it, iter_time)
                self.reset_operator_counters()

            if time_limit and (time.perf_counter() - self.start_time) >= time_limit:
                print(f"Time limit reached at iteration {it}")
                break

        self.print_final_report()
        return self.best_solution
