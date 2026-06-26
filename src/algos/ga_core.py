import math
import time
from abc import abstractmethod
from typing import List

import numpy as np

from src.algos.ga_logging import LoggingGA, timed
from src.costs import cost_function_perm, evaluate_permutation
from src.data.models import Individual, Model
from src.operators.crossover import choose_crossover
from src.operators.mutations import choose_mutation
from src.repair import RFRepair
from src.selection import DiversitySelector

# Local search (memetic polishing) is only worth its O(J*I) cost-recompute overhead
# on small instances -- on the large T-datasets (J in the thousands) it would dominate runtime.
LOCAL_SEARCH_MAX_J = 50
LOCAL_SEARCH_TOP_K = 3
LOCAL_SEARCH_MAX_PASSES = 2


class BaseGA(LoggingGA):
    def __init__(
        self,
        model: Model,
        population_size: int,
        iterations: int,
        repair_class=None,
        selector: DiversitySelector | None = None,
    ):
        super().__init__()

        self.model = model
        self.population_size = population_size
        self.iterations = iterations

        self.repair_class = repair_class if repair_class is not None else RFRepair()
        self.selector = selector if selector is not None else DiversitySelector()

        # Core GA objects
        self.population: List[Individual] = []
        self.best_solution: Individual | None = None
        self.worst_cost: float = float("inf")
        self.progress: float = 0.0

    @property
    def avg_diversity(self) -> float:
        return self.selector.avg_diversity

    @timed("repair")
    def repair_wrapper(self, perm):
        return self.repair_class.repair(perm, self.model)

    @timed("initialization")
    def initialize_population(self) -> None:
        self.population = []

        for _ in range(self.population_size):
            perm = np.random.randint(0, self.model.I, size=self.model.J, dtype=int)
            perm = self.repair_wrapper(perm)
            ind = evaluate_permutation(perm, self.model)
            self.population.append(ind)

        self.population.sort(key=lambda x: x.cost)
        self.best_solution = self.population[0]
        self.worst_cost = self.population[-1].cost

    @timed("selection")
    def select_from_pool(self, pool):
        self.population = self.selector.select_from_pool(pool, self.population_size, self.progress)

    @timed("crossover")
    def crossover(self, probabilities: np.ndarray, n: int) -> List[Individual]:
        offspring = []
        self._crossover_attempts += n

        for _ in range(0, n, 2):
            i1 = self.selector.roulette_wheel_selection(probabilities)
            i2 = self.selector.roulette_wheel_selection(probabilities)

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
            idx = np.random.randint(0, len(self.population))
            init_perm = self.population[idx].permutation

            perm = choose_mutation(init_perm, self.model)
            perm = self.repair_wrapper(perm)

            ind = evaluate_permutation(perm, self.model)

            mutations.append(ind)
            if math.isfinite(ind.cost):
                self._mutation_valid += 1
                if ind.cost < self.best_solution.cost:
                    self._mutation_new_best += 1

        return mutations

    @timed("local_search")
    def local_search(self, perm: np.ndarray) -> np.ndarray:
        """Hill-climb by reassigning one job at a time to its best feasible facility."""
        perm = perm.copy()
        best_cost, _ = cost_function_perm(perm, self.model)

        for _ in range(LOCAL_SEARCH_MAX_PASSES):
            improved = False

            for j in range(self.model.J):
                original = perm[j]
                best_facility = original

                for i in range(self.model.I):
                    if i == original:
                        continue
                    perm[j] = i
                    cost, _ = cost_function_perm(perm, self.model)
                    if cost < best_cost:
                        best_cost = cost
                        best_facility = i
                        improved = True

                perm[j] = best_facility

            if not improved:
                break

        return perm

    def polish_elites(self) -> None:
        if self.model.J > LOCAL_SEARCH_MAX_J:
            return

        for idx in range(min(LOCAL_SEARCH_TOP_K, len(self.population))):
            ind = self.population[idx]
            polished_perm = self.local_search(ind.permutation)
            if polished_perm is not ind.permutation and not np.array_equal(polished_perm, ind.permutation):
                self.population[idx] = evaluate_permutation(polished_perm, self.model)

    @abstractmethod
    def step(self) -> None:
        pass

    def run(self, time_limit: float | None = None):
        self.start_time = time.perf_counter()
        self.initialize_population()

        print(f"GA started → Population: {self.population_size:,} | Iterations: {self.iterations}\n")

        for it in range(1, self.iterations + 1):
            self.progress = it / self.iterations

            iter_start = time.perf_counter()
            self.step()
            self.population.sort(key=lambda x: x.cost)
            self.polish_elites()
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
