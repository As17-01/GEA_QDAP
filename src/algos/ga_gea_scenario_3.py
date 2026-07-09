import math

import numpy as np

from src.algos.base import BaseGA
from src.costs import evaluate_permutation_delta_batch
from src.operators.mutations import mutation_random


class GEAScenario3(BaseGA):
    """Modernization of GEA -- Scenario 3: Gene Injection (GI).

    Runs three operators each generation at fixed rates (no adaptive lambda):
    - Regular crossover (randomly chosen from the 4 standard operators) at crossover_rate.
    - Regular mutation (randomly chosen from the 9 standard operators) at mutation_rate.
    - GI (mutation_random): replaces a handful of random genes in each selected individual
      with brand-new random facility assignments, injecting fresh genetic material rather
      than perturbing the existing assignment. Applied at injection_rate.

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
        injection_rate=0.1,
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
        self.injection_rate = injection_rate

    def _gene_injection(self, n):
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

                for child in injected:
                    if math.isfinite(child.cost):
                        valid += 1
                        if child.cost < self.best_solution.cost:
                            new_best += 1

        self.logger.record_mutation(n, valid, new_best)
        return injected

    def step(self) -> None:
        probs = self.compute_selection_probabilities()

        ncrossover = int(2 * round((self.crossover_rate * self.population_size) / 2))
        nmutation = int(math.floor(self.mutation_rate * self.population_size))
        n_gi = int(math.floor(self.injection_rate * self.population_size))

        offspring = [child for child, _ in self.crossover(probs, ncrossover)]
        mutations = [child for child, _ in self.mutate(nmutation)]
        injected = self._gene_injection(n_gi)
        immigrants = self.maybe_generate_immigrants()

        pool = self.population + offspring + mutations + injected + immigrants
        self.select_from_pool(pool)
