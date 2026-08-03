import numpy as np

from src.algos.base import AlgorithmBase
from src.costs import evaluate_permutation_delta_batch
from src.data.models import Individual
from src.operators.crossover import choose_crossover, crossover_robust_chromosome
from src.operators.mutations import choose_mutation, mutation_greedy_reassign, mutation_random
from src.repair import GreedyRepair


class StandardBase(AlgorithmBase):
    """Holland-style scaffolding: fitness-proportionate parent selection, probabilistic
    crossover/mutation, and generational replacement with elitism."""

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        repair_class=None,
        verbose: bool = False,
    ):
        super().__init__(
            model,
            population_size,
            iterations,
            repair_class=repair_class if repair_class is not None else GreedyRepair(),
            verbose=verbose,
        )

    def _fitness_parent_indices(self, n: int) -> np.ndarray:
        costs = np.array([ind.cost for ind in self.population], dtype=float)
        finite = np.isfinite(costs)

        fitness = np.zeros_like(costs)
        fitness[finite] = 1.0 / (costs[finite] + 1e-9)

        total = fitness.sum()
        probs = fitness / total if total > 0 else np.full(len(costs), 1.0 / len(costs))

        cumsum = np.cumsum(probs)
        draws = np.random.random(size=n)
        return np.minimum(np.searchsorted(cumsum, draws, side="right"), len(probs) - 1)

    def _replace_with_elitism(self, offspring: list[Individual], elitism_count: int) -> None:
        if elitism_count > 0:
            elites = sorted(self.population, key=lambda x: x.cost)[:elitism_count]
            worst_order = sorted(range(len(offspring)), key=lambda i: offspring[i].cost, reverse=True)
            for slot, elite in zip(worst_order, elites):
                offspring[slot] = elite
        self.population = offspring

    def _standard_crossover_batch(self, n: int) -> list[tuple[Individual, Individual]]:
        n = n - (n % 2)
        num_pairs = n // 2
        if not num_pairs:
            return []

        parent_idx = self._fitness_parent_indices(2 * num_pairs)
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

        parent_idx = self._fitness_parent_indices(2 * num_pairs)
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

    def _standard_offspring_fill(self, count: int) -> list[Individual]:
        if count <= 0:
            return []

        num_pairs = count // 2 + count % 2
        parent_idx = self._fitness_parent_indices(2 * num_pairs)
        raw_perms = []
        baselines = []
        for k in range(num_pairs):
            i1, i2 = parent_idx[2 * k], parent_idx[2 * k + 1]
            p1, p2 = self.population[i1], self.population[i2]
            (child1, base1), (child2, base2) = choose_crossover((p1, p2), self.model)
            child1 = choose_mutation(child1, self.model)
            child2 = choose_mutation(child2, self.model)
            raw_perms.extend((child1, child2))
            baselines.extend((base1, base2))

        raw_perms = raw_perms[:count]
        baselines = baselines[:count]
        repaired = self.repair_batch_wrapper(np.array(raw_perms))
        children = evaluate_permutation_delta_batch(baselines, repaired, self.model)
        self.logger.record_nfe(len(raw_perms))
        return list(children)

    def _finalize_adaptive_offspring(self, candidates: list[Individual], elitism_count: int) -> None:
        n = self.population_size
        if len(candidates) < n:
            candidates = candidates + self._standard_offspring_fill(n - len(candidates))
        offspring = sorted(candidates, key=lambda x: x.cost)[:n]
        self._replace_with_elitism(offspring, elitism_count)

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
        parent_idx = self._fitness_parent_indices(2 * num_pairs)

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

        self._replace_with_elitism(list(offspring), elitism_count)

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
        """One standard-family generation of crossover + mutation with generational replacement."""
        self.run_gea_generational(crossover_rate, mutation_rate, elitism_count)
