import numpy as np

from src.algos.base import AlgorithmBase
from src.costs import evaluate_permutation_delta_batch
from src.data.models import Individual
from src.operators.crossover import choose_crossover
from src.operators.mutations import choose_mutation
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

    def run_crossover_mutation_generational(
        self,
        crossover_rate: float,
        mutation_rate: float,
        elitism_count: int,
    ) -> None:
        """One standard-family generation of crossover + mutation with generational replacement."""
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
