import math

import numpy as np

from src.algos.base import BaseGA
from src.costs import evaluate_permutation_delta_batch
from src.operators.mutations import mutation_random


class GEAScenario3(BaseGA):
    """Modernization of GEA -- Scenario 3: Gene Injection (GI).

    Each generation, GI (mutation_random: replace a handful of random genes in each
    selected individual with brand-new random facility assignments, injecting fresh
    genetic material rather than perturbing the existing assignment) runs as a third
    operator alongside GEA's regular crossover and mutation, both of which stay at
    fixed, non-adaptive rates. GI's own rate is scaled by an adaptive lambda_gi
    multiplier exactly as AdaptiveGA scales its lambdas (see ga_adaptive.py):

    (a) Apply GI to obtain offspring and evaluate them.
    (b) Compute delta_gi between offspring and their parent baselines.
    (c) Update lambda_gi with Eq. 2 (lambda_new = lambda_old + alpha * delta_gi).
    (d) Clip lambda_gi to [lambda_min, lambda_max] (Eq. 3).
    """

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.3,
        injection_rate=0.1,
        alpha=0.01,
        lambda_min=0.4,
        lambda_max=1.5,
        epsilon=1e-5,
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
        self.base_injection = injection_rate

        self.lambda_gi = 1.0
        self.alpha = alpha
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.epsilon = epsilon

    def _update_lambda(self, lam, delta):
        new_val = lam + self.alpha * delta
        return max(self.lambda_min, min(self.lambda_max, new_val))

    def _gene_injection(self, n):
        injected = []
        delta = 0.0

        with self.logger.timed("gene_injection"):
            if n > 0:
                indices = np.random.randint(0, len(self.population), size=n)
                baselines = [self.population[idx] for idx in indices]

                raw_perms = np.array([mutation_random(b.permutation, self.model) for b in baselines])
                repaired = self.repair_batch_wrapper(raw_perms)
                injected = evaluate_permutation_delta_batch(baselines, repaired, self.model)

                for child, baseline in zip(injected, baselines):
                    parent_cost = baseline.cost
                    if math.isfinite(parent_cost) and math.isfinite(child.cost):
                        delta += (parent_cost - child.cost) / (parent_cost + self.epsilon)

        return injected, delta

    def step(self) -> None:
        probs = self.compute_selection_probabilities()

        ncrossover = int(2 * round((self.crossover_rate * self.population_size) / 2))
        nmutation = int(math.floor(self.mutation_rate * self.population_size))
        n_gi = int(self.base_injection * self.population_size * self.lambda_gi)

        offspring = [child for child, _ in self.crossover(probs, ncrossover)]
        mutations = [child for child, _ in self.mutate(nmutation)]
        injected, delta_gi = self._gene_injection(n_gi)
        immigrants = self.maybe_generate_immigrants()

        self.lambda_gi = self._update_lambda(self.lambda_gi, delta_gi)

        pool = self.population + offspring + mutations + injected + immigrants
        self.select_from_pool(pool)
