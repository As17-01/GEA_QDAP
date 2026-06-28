import math

import numpy as np

from src.algos.base import BaseGA
from src.costs import evaluate_permutation_delta_batch
from src.operators.mutations import mutation_greedy_reassign


class GEAScenario2(BaseGA):
    """Modernization of GEA -- Scenario 2: Directed Mutation (DM).

    Each generation, DM (mutation_greedy_reassign: move each selected individual's
    single worst-assigned job to its cheapest feasible facility -- a directed,
    fitness-improving move rather than GEA's usual blind random mutation) replaces
    GEA's regular mutation operator. Its rate is scaled by an adaptive lambda_dm
    multiplier exactly as AdaptiveGA scales lambda_mutation (see ga_adaptive.py):

    (a) Apply DM to obtain offspring and evaluate them.
    (b) Compute delta_dm between offspring and their parent baselines.
    (c) Update lambda_dm with Eq. 2 (lambda_new = lambda_old + alpha * delta_dm).
    (d) Clip lambda_dm to [lambda_min, lambda_max] (Eq. 3).

    Crossover stays GEA's regular, non-adaptive mixed crossover operator.
    """

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.3,
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
        self.base_mutation = mutation_rate

        self.lambda_dm = 1.0
        self.alpha = alpha
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.epsilon = epsilon

    def _update_lambda(self, lam, delta):
        new_val = lam + self.alpha * delta
        return max(self.lambda_min, min(self.lambda_max, new_val))

    def _directed_mutation(self, n):
        mutations = []
        delta = 0.0
        valid = 0
        new_best = 0

        with self.logger.timed("mutation"):
            if n > 0:
                indices = np.random.randint(0, len(self.population), size=n)
                baselines = [self.population[idx] for idx in indices]

                raw_perms = np.array([mutation_greedy_reassign(b.permutation, self.model) for b in baselines])
                repaired = self.repair_batch_wrapper(raw_perms)
                mutations = evaluate_permutation_delta_batch(baselines, repaired, self.model)

                for child, baseline in zip(mutations, baselines):
                    parent_cost = baseline.cost
                    if math.isfinite(parent_cost) and math.isfinite(child.cost):
                        delta += (parent_cost - child.cost) / (parent_cost + self.epsilon)
                    if math.isfinite(child.cost):
                        valid += 1
                        if child.cost < self.best_solution.cost:
                            new_best += 1

        self.logger.record_mutation(n, valid, new_best)
        return mutations, delta

    def step(self) -> None:
        probs = self.compute_selection_probabilities()

        ncrossover = int(2 * round((self.crossover_rate * self.population_size) / 2))
        n_dm = int(self.base_mutation * self.population_size * self.lambda_dm)

        offspring = [child for child, _ in self.crossover(probs, ncrossover)]
        mutations, delta_dm = self._directed_mutation(n_dm)
        immigrants = self.maybe_generate_immigrants()

        self.lambda_dm = self._update_lambda(self.lambda_dm, delta_dm)

        pool = self.population + offspring + mutations + immigrants
        self.select_from_pool(pool)
