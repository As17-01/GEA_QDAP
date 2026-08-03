import math

import numpy as np

from src.algos.core.base import AlgorithmBase
from src.costs import evaluate_permutation, evaluate_permutation_delta_batch
from src.data.models import Individual
from src.heuristics.heuristic2 import heuristic2
from src.operators.crossover import choose_crossover, crossover_robust_chromosome
from src.operators.mutations import choose_mutation, mutation_greedy_reassign, mutation_random
from src.repair import GreedyRepair


class StandardBase(AlgorithmBase):
    """Thesis-style standard scaffold: heuristic2 initialization, exponential parent
    selection, probabilistic crossover/mutation, and (mu+lambda) pool survivor selection."""

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        repair_class=None,
        selection_beta: float = 10.0,
        verbose: bool = False,
    ):
        super().__init__(
            model,
            population_size,
            iterations,
            repair_class=repair_class if repair_class is not None else GreedyRepair(),
            verbose=verbose,
        )
        self.selection_beta = selection_beta

    def initialize_population(self) -> None:
        with self.logger.timed("initialization"):
            population: list[Individual] = [heuristic2(self.model)]
            self.logger.record_nfe(1)

            seed_perm = population[0].permutation
            max_attempts = self.population_size * 100
            attempts = 0
            while len(population) < self.population_size and attempts < max_attempts:
                attempts += 1
                mutated_perm = choose_mutation(seed_perm, self.model)
                ind = evaluate_permutation(mutated_perm, self.model)
                self.logger.record_nfe(1)
                if math.isfinite(ind.cost):
                    population.append(ind)

            if len(population) < self.population_size:
                raise RuntimeError(
                    f"heuristic2 init could not fill population "
                    f"({len(population)}/{self.population_size}) after {max_attempts} mutations"
                )

            self.population = sorted(population, key=lambda x: x.cost)
            self.best_solution = self.population[0]
            self.worst_cost = self.population[-1].cost

    def _pool_replace(self, newcomers: list[Individual]) -> None:
        """(mu+lambda) survivor selection: merge parents with offspring, keep best N."""
        pool = self.population + newcomers
        pool.sort(key=lambda x: x.cost)
        self.population = pool[: self.population_size]

    def _parent_selection_indices(self, n: int) -> np.ndarray:
        """Roulette wheel with exp(-beta * cost / worst_cost), matching upstream run_ga."""
        costs = np.array([ind.cost for ind in self.population], dtype=float)
        worst = self.worst_cost
        if not math.isfinite(worst) or worst <= 0:
            worst = float(np.max(costs[np.isfinite(costs)])) if np.any(np.isfinite(costs)) else 1.0

        weights = np.zeros_like(costs)
        finite = np.isfinite(costs)
        weights[finite] = np.exp(-self.selection_beta * costs[finite] / worst)

        total = weights.sum()
        probs = weights / total if total > 0 else np.full(len(costs), 1.0 / len(costs))

        cumsum = np.cumsum(probs)
        draws = np.random.random(size=n)
        return np.minimum(np.searchsorted(cumsum, draws, side="right"), len(probs) - 1)

    def _standard_crossover_batch(self, n: int) -> list[tuple[Individual, Individual]]:
        n = n - (n % 2)
        num_pairs = n // 2
        if not num_pairs:
            return []

        parent_idx = self._parent_selection_indices(2 * num_pairs)
        raw_perms = []
        baselines = []
        for k in range(num_pairs):
            i1, i2 = parent_idx[2 * k], parent_idx[2 * k + 1]
            p1, p2 = self.population[i1], self.population[i2]
            (child1, base1), (child2, base2) = choose_crossover((p1, p2), self.model)
            raw_perms.extend((child1, child2))
            baselines.extend((base1, base2))

        repaired = self.repair_batch_wrapper(np.array(raw_perms))
        children = evaluate_permutation_delta_batch(baselines, repaired, self.model)
        self.logger.record_nfe(len(raw_perms))
        return list(zip(children, baselines))

    def _standard_mutate_batch(self, n: int) -> list[tuple[Individual, Individual]]:
        if n <= 0:
            return []

        indices = np.random.randint(0, len(self.population), size=n)
        baselines = [self.population[idx] for idx in indices]
        raw_perms = np.array([choose_mutation(b.permutation, self.model) for b in baselines])
        repaired = self.repair_batch_wrapper(raw_perms)
        children = evaluate_permutation_delta_batch(baselines, repaired, self.model)
        self.logger.record_nfe(n)
        return list(zip(children, baselines))

    def _standard_rc_crossover_batch(self, n: int) -> list[tuple[Individual, Individual]]:
        n = n - (n % 2)
        num_pairs = n // 2
        if not num_pairs:
            return []

        parent_idx = self._parent_selection_indices(2 * num_pairs)
        raw_perms = []
        baselines = []
        for k in range(num_pairs):
            i1, i2 = parent_idx[2 * k], parent_idx[2 * k + 1]
            p1, p2 = self.population[i1], self.population[i2]
            (child1, base1), (child2, base2) = crossover_robust_chromosome(p1, p2, self.model)
            raw_perms.extend((child1, child2))
            baselines.extend((base1, base2))

        repaired = self.repair_batch_wrapper(np.array(raw_perms))
        children = evaluate_permutation_delta_batch(baselines, repaired, self.model)
        self.logger.record_nfe(len(raw_perms))
        return list(zip(children, baselines))

    def _standard_directed_mutation_batch(self, n: int) -> list[tuple[Individual, Individual]]:
        if n <= 0:
            return []

        indices = np.random.randint(0, len(self.population), size=n)
        baselines = [self.population[idx] for idx in indices]
        raw_perms = np.array([mutation_greedy_reassign(b.permutation, self.model) for b in baselines])
        repaired = self.repair_batch_wrapper(raw_perms)
        children = evaluate_permutation_delta_batch(baselines, repaired, self.model)
        self.logger.record_nfe(n)
        return list(zip(children, baselines))

    def _standard_gene_injection_batch(self, n: int) -> list[tuple[Individual, Individual]]:
        if n <= 0:
            return []

        indices = np.random.randint(0, len(self.population), size=n)
        baselines = [self.population[idx] for idx in indices]
        raw_perms = np.array([mutation_random(b.permutation, self.model) for b in baselines])
        repaired = self.repair_batch_wrapper(raw_perms)
        children = evaluate_permutation_delta_batch(baselines, repaired, self.model)
        self.logger.record_nfe(n)
        return list(zip(children, baselines))

    def _finalize_adaptive_offspring(self, candidates: list[Individual], elitism_count: int) -> None:
        del elitism_count  # kept for config/API compatibility; pool selection ignores elitism
        self._pool_replace(candidates)

    def run_adaptive_generational(
        self,
        elitism_count: int,
        *,
        rc_rate: float = 0.0,
        dm_rate: float = 0.0,
        injection_rate: float = 0.0,
    ) -> None:
        """One standard-family adaptive generation with lambda-scaled operator counts."""
        offspring = self._adaptive_apply("base_crossover", "lambda_crossover", self._standard_crossover_batch)
        mutations = self._adaptive_apply("base_mutation", "lambda_mutation", self._standard_mutate_batch)
        candidates = offspring + mutations

        if rc_rate > 0:
            candidates += self._adaptive_apply("base_rc", "lambda_rc", self._standard_rc_crossover_batch)
        if dm_rate > 0:
            candidates += self._adaptive_apply("base_dm", "lambda_dm", self._standard_directed_mutation_batch)
        if injection_rate > 0:
            candidates += self._adaptive_apply("base_gi", "lambda_gi", self._standard_gene_injection_batch)

        self._finalize_adaptive_offspring(candidates, elitism_count)

    def run_gea_generational(
        self,
        crossover_rate: float,
        mutation_rate: float,
        elitism_count: int,
        *,
        rc_rate: float = 0.0,
        dm_rate: float = 0.0,
        injection_rate: float = 0.0,
    ) -> None:
        """One standard-family GEA generation: probabilistic crossover/mutation/scenario ops."""
        n = self.population_size
        num_pairs = n // 2 + (n % 2)
        parent_idx = self._parent_selection_indices(2 * num_pairs)

        raw_perms = []
        baselines = []
        for k in range(num_pairs):
            i1, i2 = parent_idx[2 * k], parent_idx[2 * k + 1]
            p1, p2 = self.population[i1], self.population[i2]

            if np.random.random() < crossover_rate:
                (child1, base1), (child2, base2) = choose_crossover((p1, p2), self.model)
            else:
                child1, base1 = p1.permutation, p1
                child2, base2 = p2.permutation, p2

            if np.random.random() < mutation_rate:
                child1 = choose_mutation(child1, self.model)
            if np.random.random() < mutation_rate:
                child2 = choose_mutation(child2, self.model)

            child1, base1 = self._apply_scenario_operators(child1, base1, p1, p2, rc_rate, dm_rate, injection_rate)
            child2, base2 = self._apply_scenario_operators(child2, base2, p1, p2, rc_rate, dm_rate, injection_rate)

            raw_perms.append(child1)
            baselines.append(base1)
            raw_perms.append(child2)
            baselines.append(base2)

        raw_perms = raw_perms[:n]
        baselines = baselines[:n]

        repaired = self.repair_batch_wrapper(np.array(raw_perms))
        offspring = evaluate_permutation_delta_batch(baselines, repaired, self.model)
        self.logger.record_nfe(len(raw_perms))

        self._pool_replace(list(offspring))

    def _apply_scenario_operators(self, child, base, p1, p2, rc_rate, dm_rate, injection_rate):
        if rc_rate > 0 and np.random.random() < rc_rate:
            (child, base), _ = crossover_robust_chromosome(p1, p2, self.model)
        if dm_rate > 0 and np.random.random() < dm_rate:
            child = mutation_greedy_reassign(child, self.model)
        if injection_rate > 0 and np.random.random() < injection_rate:
            child = mutation_random(child, self.model)
        return child, base

    def run_crossover_mutation_generational(
        self,
        crossover_rate: float,
        mutation_rate: float,
        elitism_count: int,
    ) -> None:
        """One standard-family generation of crossover + mutation with pool survivor selection."""
        self.run_gea_generational(crossover_rate, mutation_rate, elitism_count)

    def run_annealing_generational(
        self,
        crossover_rate: float,
        mutation_rate: float,
        elitism_count: int,
    ) -> None:
        """Standard GA offspring filtered by Metropolis acceptance, then pool survivor selection."""
        ncrossover = int(2 * round((crossover_rate * self.population_size) / 2))
        nmutation = int(math.floor(mutation_rate * self.population_size))

        offspring = self._anneal_accept(self._standard_crossover_batch(ncrossover))
        mutations = self._anneal_accept(self._standard_mutate_batch(nmutation))
        self._finalize_adaptive_offspring(offspring + mutations, elitism_count)
        self._cool_temperature()
