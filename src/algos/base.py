import math
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from src.algos.logger import GALogger
from src.costs import (
    cost_function_perm,
    cost_function_perm_delta,
    evaluate_permutation,
    evaluate_permutation_delta_batch,
)
from src.data.models import Individual, Model
from src.operators.crossover import choose_crossover, crossover_robust_chromosome
from src.operators.mutations import choose_mutation, mutation_greedy_reassign, mutation_random
from src.repair import RFRepair
from src.selection import DiversitySelector

# Local search (memetic polishing) is only worth its O(J*I) cost-recompute overhead
# on small instances -- on the large T-datasets (J in the thousands) it would dominate runtime.
LOCAL_SEARCH_MAX_J = 50
LOCAL_SEARCH_TOP_K = 3
LOCAL_SEARCH_MAX_PASSES = 2


class BaseGA(ABC):
    def __init__(
        self,
        model: Model,
        population_size: int,
        iterations: int,
        repair_class=None,
        selector: DiversitySelector | None = None,
        stagnation_limit: int = 30,
        immigrant_rate: float = 0.1,
        verbose: bool = False,
    ):
        self.model = model
        self.population_size = population_size
        self.iterations = iterations
        self.verbose = verbose

        self.repair_class = repair_class if repair_class is not None else RFRepair()
        self.selector = selector if selector is not None else DiversitySelector()
        self.logger = GALogger()

        # Core GA objects
        self.population: List[Individual] = []
        self.best_solution: Individual | None = None
        self.worst_cost: float = float("inf")
        self.progress: float = 0.0

        # Stagnation-triggered random immigrants: a fixed mutation/crossover rate (or one
        # that only ever shrinks, as in AdaptiveGA) has no mechanism to react when the
        # population has actually converged on a local optimum -- this injects fresh random
        # individuals once `best_solution` has gone `stagnation_limit` iterations without
        # improving, giving crossover new material to recombine with.
        self.stagnation_limit = stagnation_limit
        self.immigrant_rate = immigrant_rate
        self.stagnation_counter = 0
        self._last_best_cost = float("inf")

    def repair_batch_wrapper(self, perms: np.ndarray) -> np.ndarray:
        with self.logger.timed("repair"):
            return self.repair_class.repair_batch(perms, self.model)

    def initialize_population(self) -> None:
        with self.logger.timed("initialization"):
            perms = np.random.randint(0, self.model.I, size=(self.population_size, self.model.J), dtype=int)
            perms = self.repair_batch_wrapper(perms)

            self.population = [evaluate_permutation(perms[i], self.model) for i in range(self.population_size)]
            self.logger.record_nfe(self.population_size)

            self.population.sort(key=lambda x: x.cost)
            self.best_solution = self.population[0]
            self.worst_cost = self.population[-1].cost

    def compute_selection_probabilities(self):
        with self.logger.timed("parent_selection"):
            return self.selector.compute_selection_probabilities(self.population)

    def select_from_pool(self, pool):
        with self.logger.timed("survivor_selection"):
            self.population = self.selector.select_from_pool(pool, self.population_size, self.progress)

    def crossover(self, probabilities: np.ndarray, n: int) -> List[Tuple[Individual, Individual]]:
        offspring = []
        valid = 0
        new_best = 0

        with self.logger.timed("crossover"):
            raw_perms = []
            baselines = []

            num_pairs = len(range(0, n, 2))
            if num_pairs:
                # Draw all parent indices for this generation in one shot: the cumsum
                # inside roulette_wheel_selection is the same every time within a step,
                # so computing it once here instead of once per draw avoids redoing
                # O(population_size) work ~num_pairs*2 times for nothing.
                parent_indices = self.selector.roulette_wheel_selection_batch(probabilities, 2 * num_pairs)

                for k in range(num_pairs):
                    i1, i2 = parent_indices[2 * k], parent_indices[2 * k + 1]

                    p1, p2 = self.population[i1], self.population[i2]
                    # Each child comes paired with whichever parent it shares the most
                    # positions with (tight per crossover type, not just "child k -> parent
                    # k"), keeping the delta-cost evaluation below cheap for both operators.
                    (child1, baseline1), (child2, baseline2) = choose_crossover((p1, p2), self.model)

                    raw_perms.extend((child1, child2))
                    baselines.extend((baseline1, baseline2))

            if raw_perms:
                repaired = self.repair_batch_wrapper(np.array(raw_perms))
                children = evaluate_permutation_delta_batch(baselines, repaired, self.model)
                self.logger.record_nfe(len(raw_perms))
                offspring = list(zip(children, baselines))

                for child in children:
                    if math.isfinite(child.cost):
                        valid += 1
                        if child.cost < self.best_solution.cost:
                            new_best += 1

        self.logger.record_crossover(n, valid, new_best)
        return offspring

    def mutate(self, n: int) -> List[Tuple[Individual, Individual]]:
        mutations = []
        valid = 0
        new_best = 0

        with self.logger.timed("mutation"):
            if n > 0:
                indices = np.random.randint(0, len(self.population), size=n)
                baselines = [self.population[idx] for idx in indices]

                raw_perms = np.array([choose_mutation(b.permutation, self.model) for b in baselines])
                repaired = self.repair_batch_wrapper(raw_perms)
                children = evaluate_permutation_delta_batch(baselines, repaired, self.model)
                self.logger.record_nfe(n)
                mutations = list(zip(children, baselines))

                for ind in children:
                    if math.isfinite(ind.cost):
                        valid += 1
                        if ind.cost < self.best_solution.cost:
                            new_best += 1

        self.logger.record_mutation(n, valid, new_best)
        return mutations

    def generate_immigrants(self, n: int) -> List[Individual]:
        if n <= 0:
            return []

        with self.logger.timed("immigrants"):
            perms = np.random.randint(0, self.model.I, size=(n, self.model.J), dtype=int)
            perms = self.repair_batch_wrapper(perms)
            immigrants = [evaluate_permutation(perms[i], self.model) for i in range(n)]
            self.logger.record_nfe(n)
            return immigrants

    def maybe_generate_immigrants(self) -> List[Individual]:
        """Called once per step, before this iteration's offspring/mutants are folded into
        the pool. `self.best_solution` still reflects the *previous* iteration's result here
        (run() updates it after step() returns), so this compares consecutive iterations and
        resets the counter on any improvement."""
        if self.best_solution is not None and self.best_solution.cost < self._last_best_cost - 1e-9:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        if self.best_solution is not None:
            self._last_best_cost = min(self._last_best_cost, self.best_solution.cost)

        if self.stagnation_counter < self.stagnation_limit:
            return []

        self.stagnation_counter = 0
        n = max(1, int(self.immigrant_rate * self.population_size))
        return self.generate_immigrants(n)

    def local_search(self, perm: np.ndarray) -> np.ndarray:
        """Hill-climb by reassigning one job at a time to its best feasible facility."""
        with self.logger.timed("local_search"):
            perm = perm.copy()
            best_cost, _ = cost_function_perm(perm, self.model)
            nfe = 1  # initial full eval above

            for _ in range(LOCAL_SEARCH_MAX_PASSES):
                improved = False

                for j in range(self.model.J):
                    # `perm` here is the current committed state (position j still at its
                    # original value) -- the natural delta baseline for each candidate i.
                    original = perm[j]
                    best_facility = original

                    for i in range(self.model.I):
                        if i == original:
                            continue
                        trial_perm = perm.copy()
                        trial_perm[j] = i
                        if math.isfinite(best_cost):
                            cost, _ = cost_function_perm_delta(perm, trial_perm, best_cost, self.model)
                        else:
                            cost, _ = cost_function_perm(trial_perm, self.model)
                        nfe += 1
                        if cost < best_cost:
                            best_cost = cost
                            best_facility = i
                            improved = True

                    perm[j] = best_facility

                if not improved:
                    break

            self.logger.record_nfe(nfe)
            return perm

    def _robust_chromosome_crossover(self, probabilities: np.ndarray, n: int) -> List[Individual]:
        """RC (Robust Chromosome) crossover driver: per gene, inherit from the parent with
        more remaining capacity slack. Used by GEA and GEAScenario1."""
        offspring = []
        valid = 0
        new_best = 0

        n = n - (n % 2)
        with self.logger.timed("crossover"):
            num_pairs = n // 2
            if num_pairs:
                parent_indices = self.selector.roulette_wheel_selection_batch(probabilities, 2 * num_pairs)

                raw_perms = []
                baselines = []
                for k in range(num_pairs):
                    i1, i2 = parent_indices[2 * k], parent_indices[2 * k + 1]
                    p1, p2 = self.population[i1], self.population[i2]

                    (child1, base1), (child2, base2) = crossover_robust_chromosome(p1, p2, self.model)
                    raw_perms.extend((child1, child2))
                    baselines.extend((base1, base2))

                repaired = self.repair_batch_wrapper(np.array(raw_perms))
                offspring = evaluate_permutation_delta_batch(baselines, repaired, self.model)
                self.logger.record_nfe(len(raw_perms))

                for child in offspring:
                    if math.isfinite(child.cost):
                        valid += 1
                        if child.cost < self.best_solution.cost:
                            new_best += 1

        self.logger.record_crossover(n, valid, new_best)
        return offspring

    def _directed_mutation(self, n: int) -> List[Individual]:
        """DM (Directed Mutation) driver: moves each individual's worst-assigned job to its
        cheapest feasible facility. Used by GEA and GEAScenario2."""
        mutations = []
        valid = 0
        new_best = 0

        with self.logger.timed("mutation"):
            if n > 0:
                indices = np.random.randint(0, len(self.population), size=n)
                baselines = [self.population[idx] for idx in indices]

                raw_perms = np.array([mutation_greedy_reassign(b.permutation, self.model) for b in baselines])
                repaired = self.repair_batch_wrapper(raw_perms)
                mutations = evaluate_permutation_delta_batch(baselines, repaired, self.model)
                self.logger.record_nfe(n)

                for child in mutations:
                    if math.isfinite(child.cost):
                        valid += 1
                        if child.cost < self.best_solution.cost:
                            new_best += 1

        self.logger.record_mutation(n, valid, new_best)
        return mutations

    def _gene_injection(self, n: int) -> List[Individual]:
        """GI (Gene Injection) driver: replaces a handful of random genes with new random
        facility assignments. Used by GEA and GEAScenario3."""
        injected = []
        valid = 0
        new_best = 0

        with self.logger.timed("gene_injection"):
            if n > 0:
                indices = np.random.randint(0, len(self.population), size=n)
                baselines = [self.population[idx] for idx in indices]

                raw_perms = np.array([mutation_random(b.permutation, self.model) for b in baselines])
                repaired = self.repair_batch_wrapper(raw_perms)
                injected = evaluate_permutation_delta_batch(baselines, repaired, self.model)
                self.logger.record_nfe(n)

                for child in injected:
                    if math.isfinite(child.cost):
                        valid += 1
                        if child.cost < self.best_solution.cost:
                            new_best += 1

        self.logger.record_mutation(n, valid, new_best)
        return injected

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
        self.logger.start_run()
        self.initialize_population()

        # Wall-clock time (seconds since run start) at which the best cost was last
        # improved. Set after initialization so there's always a valid value, then
        # updated whenever a new best is found during the main loop.
        self.hitting_time: float = self.logger.elapsed()

        if self.verbose:
            print(f"GA started → Population: {self.population_size:,} | Iterations: {self.iterations}\n")

        for it in range(1, self.iterations + 1):
            self.progress = it / self.iterations

            iter_start = self.logger.elapsed()
            self.step()
            self.population.sort(key=lambda x: x.cost)
            self.polish_elites()
            iter_time = self.logger.elapsed() - iter_start
            self.logger.record_iteration(iter_time)

            self.population.sort(key=lambda x: x.cost)
            prev_best = self.best_solution.cost if self.best_solution is not None else float("inf")
            self.best_solution = self.population[0]
            self.worst_cost = max(self.worst_cost, self.population[-1].cost)

            if self.best_solution.cost < prev_best - 1e-9:
                self.hitting_time = self.logger.elapsed()

            self.logger.record_best_cost(self.best_solution.cost)

            if it % 50 == 0:
                if self.verbose:
                    self.logger.print_iteration_info(it, iter_time, self.best_solution.cost, self.selector.avg_diversity)
                self.logger.reset_operator_counters()

            if time_limit and self.logger.elapsed() >= time_limit:
                if self.verbose:
                    print(f"Time limit reached at iteration {it}")
                break

        if self.verbose:
            self.logger.print_final_report(self.best_solution.cost)
        return self.best_solution
