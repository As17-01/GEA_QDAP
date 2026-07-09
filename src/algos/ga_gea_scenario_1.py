import math

import numpy as np
from numba import njit

from src.algos.base import BaseGA
from src.costs import evaluate_permutation_delta_batch
from src.data.models import Individual, Model


@njit(fastmath=True, cache=True)
def _crossover_robust_chromosome_nb(p1, p2, cvar1, cvar2):
    n = len(p1)
    child1 = np.empty(n, dtype=p1.dtype)
    child2 = np.empty(n, dtype=p1.dtype)
    n_match_p1 = 0

    for j in range(n):
        # Higher remaining capacity slack at the gene's assigned facility means that gene
        # tolerates more future capacity pressure before becoming infeasible -- the more
        # "robust" choice for this position, independent of its assignment cost.
        if cvar1[p1[j]] >= cvar2[p2[j]]:
            child1[j] = p1[j]
            child2[j] = p2[j]
            n_match_p1 += 1
        else:
            child1[j] = p2[j]
            child2[j] = p1[j]

    return child1, child2, n_match_p1


def crossover_robust_chromosome(p1: Individual, p2: Individual, model: Model):
    n = len(p1.permutation)
    child1, child2, n_match_p1 = _crossover_robust_chromosome_nb(p1.permutation, p2.permutation, p1.cvar, p2.cvar)

    # Pair each child with whichever parent it shares the most positions with -- the
    # tightest baseline for delta-cost evaluation, same convention as
    # src/operators/crossover.py's _pair_by_match_count.
    if n_match_p1 >= n - n_match_p1:
        return (child1, p1), (child2, p2)
    return (child1, p2), (child2, p1)


class GEAScenario1(BaseGA):
    """Modernization of GEA -- Scenario 1: Crossover with Robust Chromosome (RC).

    Runs three operators each generation at fixed rates (no adaptive lambda):
    - Regular crossover (randomly chosen from the 4 standard operators) at crossover_rate.
    - Regular mutation (randomly chosen from the 9 standard operators) at mutation_rate.
    - RC crossover: per gene, the child inherits from whichever parent's assigned facility
      has more remaining capacity slack (cvar) -- the more capacity-robust gene, independent
      of its assignment cost. Applied at rc_rate.

    All three sets of offspring are pooled together with the current population and the
    DiversitySelector picks the next generation.
    """

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.3,
        rc_rate=0.3,
        repair_class=None,
        selector=None,
        stagnation_limit=30,
        immigrant_rate=0.1,
        verbose=False,
    ):
        super().__init__(
            model,
            population_size,
            iterations,
            repair_class=repair_class,
            selector=selector,
            stagnation_limit=stagnation_limit,
            immigrant_rate=immigrant_rate,
            verbose=verbose,
        )

        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.rc_rate = rc_rate

    def _robust_chromosome_crossover(self, probabilities, n):
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

                for child in offspring:
                    if math.isfinite(child.cost):
                        valid += 1
                        if child.cost < self.best_solution.cost:
                            new_best += 1

        self.logger.record_crossover(n, valid, new_best)
        return offspring

    def step(self) -> None:
        probs = self.compute_selection_probabilities()

        ncrossover = int(2 * round((self.crossover_rate * self.population_size) / 2))
        nmutation = int(math.floor(self.mutation_rate * self.population_size))
        n_rc = int(math.floor(self.rc_rate * self.population_size))

        offspring = [child for child, _ in self.crossover(probs, ncrossover)]
        mutations = [child for child, _ in self.mutate(nmutation)]
        rc_offspring = self._robust_chromosome_crossover(probs, n_rc)
        immigrants = self.maybe_generate_immigrants()

        pool = self.population + offspring + mutations + rc_offspring + immigrants
        self.select_from_pool(pool)
