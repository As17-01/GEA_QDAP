import math

import numpy as np

from src.algos.base import BaseGA


class HybridGASA(BaseGA):
    """Chen & Shahandashti (2009). GA crossover/mutation still supplies every candidate
    move; what changes is the acceptance rule between a child and the parent baseline it
    was measured against (see BaseGA.crossover/mutate's (child, baseline) pairing): instead
    of every child unconditionally joining the survivor pool, it must first pass a
    Metropolis test against its own baseline, exactly as in simulated annealing. A child
    that's better than its baseline is always accepted; a worse one only gets a chance
    proportional to exp(-delta/T). T is annealed down over the run (geometric cooling), so
    early generations tolerate quality-losing moves to escape local optima while late
    generations behave like a plain elitist GA.
    """

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.3,
        initial_temperature=50.0,
        cooling_rate=0.97,
        min_temperature=1e-3,
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

        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.temperature = initial_temperature

    def _anneal_accept(self, pairs) -> list:
        accepted = []
        for child, baseline in pairs:
            if child.cost <= baseline.cost:
                accepted.append(child)
                continue

            if not (math.isfinite(child.cost) and math.isfinite(baseline.cost)):
                continue

            delta = child.cost - baseline.cost
            if np.random.random() < math.exp(-delta / self.temperature):
                accepted.append(child)

        return accepted

    def step(self) -> None:
        probs = self.compute_selection_probabilities()

        ncrossover = int(2 * round((self.crossover_rate * self.population_size) / 2))
        nmutation = int(math.floor(self.mutation_rate * self.population_size))

        offspring = self._anneal_accept(self.crossover(probs, ncrossover))
        mutations = self._anneal_accept(self.mutate(nmutation))
        immigrants = self.maybe_generate_immigrants()

        pool = self.population + offspring + mutations + immigrants
        self.select_from_pool(pool)

        self.temperature = max(self.min_temperature, self.temperature * self.cooling_rate)
