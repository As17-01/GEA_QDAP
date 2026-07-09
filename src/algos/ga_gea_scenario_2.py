import math

import numpy as np

from src.algos.base import BaseGA
from src.costs import evaluate_permutation_delta_batch
from src.operators.mutations import mutation_greedy_reassign


class GEAScenario2(BaseGA):
    """Modernization of GEA -- Scenario 2: Directed Mutation (DM).

    Runs three operators each generation at fixed rates (no adaptive lambda):
    - Regular crossover (randomly chosen from the 4 standard operators) at crossover_rate.
    - Regular mutation (randomly chosen from the 9 standard operators) at mutation_rate.
    - DM (mutation_greedy_reassign): moves each selected individual's single worst-assigned
      job to its cheapest feasible facility -- a directed, fitness-improving move rather
      than a blind random one. Applied at dm_rate.

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
        dm_rate=0.3,
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
        self.dm_rate = dm_rate

    def _directed_mutation(self, n):
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

                for child in mutations:
                    if math.isfinite(child.cost):
                        valid += 1
                        if child.cost < self.best_solution.cost:
                            new_best += 1

        self.logger.record_mutation(n, valid, new_best)
        return mutations

    def step(self) -> None:
        probs = self.compute_selection_probabilities()

        ncrossover = int(2 * round((self.crossover_rate * self.population_size) / 2))
        nmutation = int(math.floor(self.mutation_rate * self.population_size))
        n_dm = int(math.floor(self.dm_rate * self.population_size))

        offspring = [child for child, _ in self.crossover(probs, ncrossover)]
        mutations = [child for child, _ in self.mutate(nmutation)]
        dm_mutations = self._directed_mutation(n_dm)
        immigrants = self.maybe_generate_immigrants()

        pool = self.population + offspring + mutations + dm_mutations + immigrants
        self.select_from_pool(pool)
