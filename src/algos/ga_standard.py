import numpy as np

from src.algos.base import BaseGA
from src.costs import evaluate_permutation_delta_batch
from src.operators.crossover import crossover_one_point
from src.repair import GreedyRepair


class StandardGA(BaseGA):
    """Holland (1992). The textbook simple genetic algorithm: fitness-proportionate
    (roulette wheel) selection on raw cost, single-point crossover applied with
    probability `crossover_rate` per mating pair, per-gene mutation applied with
    probability `mutation_rate`, and full generational replacement.

    No diversity-aware selection, no randomized repair sampling, no stagnation
    immigrants, no memetic local search -- those are this project's own enhancements
    on top of the textbook algorithm (see ga_gea.py's GEA). This class is the literal
    baseline they get compared against, kept as close to the original description as
    the capacity-constrained encoding allows (repair is still needed for feasibility;
    Holland's original had no constraints to repair).
    """

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.01,
        elitism_count=1,
        repair_class=None,
        verbose=False,
    ):
        super().__init__(
            model,
            population_size,
            iterations,
            repair_class=repair_class if repair_class is not None else GreedyRepair(),
            verbose=verbose,
        )
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count

    def polish_elites(self) -> None:
        # The canonical algorithm has no memetic local search step.
        pass

    def _select_parents(self, n: int) -> np.ndarray:
        costs = np.array([ind.cost for ind in self.population], dtype=float)
        finite = np.isfinite(costs)

        fitness = np.zeros_like(costs)
        fitness[finite] = 1.0 / (costs[finite] + 1e-9)

        total = fitness.sum()
        probs = fitness / total if total > 0 else np.full(len(costs), 1.0 / len(costs))

        cumsum = np.cumsum(probs)
        draws = np.random.random(size=n)
        return np.minimum(np.searchsorted(cumsum, draws, side="right"), len(probs) - 1)

    def _mutate(self, perm: np.ndarray) -> np.ndarray:
        mask = np.random.random(len(perm)) < self.mutation_rate
        if mask.any():
            perm = perm.copy()
            perm[mask] = np.random.randint(0, self.model.I, size=int(mask.sum()))
        return perm

    def step(self) -> None:
        n = self.population_size
        num_pairs = n // 2 + (n % 2)
        parent_idx = self._select_parents(2 * num_pairs)

        raw_perms = []
        baselines = []
        for k in range(num_pairs):
            i1, i2 = parent_idx[2 * k], parent_idx[2 * k + 1]
            p1, p2 = self.population[i1], self.population[i2]

            if np.random.random() < self.crossover_rate:
                (child1, base1), (child2, base2) = crossover_one_point((p1, p2), self.model)
            else:
                child1, base1 = p1.permutation, p1
                child2, base2 = p2.permutation, p2

            raw_perms.append(self._mutate(child1))
            baselines.append(base1)
            raw_perms.append(self._mutate(child2))
            baselines.append(base2)

        raw_perms = raw_perms[:n]
        baselines = baselines[:n]

        repaired = self.repair_batch_wrapper(np.array(raw_perms))
        offspring = evaluate_permutation_delta_batch(baselines, repaired, self.model)

        if self.elitism_count > 0:
            elites = sorted(self.population, key=lambda x: x.cost)[: self.elitism_count]
            worst_order = sorted(range(len(offspring)), key=lambda i: offspring[i].cost, reverse=True)
            for slot, elite in zip(worst_order, elites):
                offspring[slot] = elite

        self.population = offspring
