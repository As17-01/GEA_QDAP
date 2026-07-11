import math

import numpy as np

from src.algos.base import BaseGA
from src.costs import evaluate_permutation_delta_batch
from src.operators.mutations import choose_mutation
from src.repair import GreedyRepair


class SimulatedAnnealing(BaseGA):
    """Bertsimas & Tsitsiklis (1993). Classic single-solution SA: starts from one
    initial solution and anneals it via repeated Metropolis acceptance tests.

    Each iteration, one neighbor is proposed via a randomly-chosen mutation operator
    and:
    - accepted outright if it's better (cost <= current), or
    - accepted with Metropolis probability exp(-delta/T) if it's worse.

    Temperature is cooled geometrically after each step:
        T <- max(T_min, T * cooling_rate)
    """

    def __init__(
        self,
        model,
        population_size=1,
        iterations=1000,
        initial_temperature=50.0,
        cooling_rate=0.97,
        min_temperature=1e-3,
        repair_class=None,
        verbose=False,
    ):
        # Single-solution SA: population size is always 1 regardless of config.
        super().__init__(
            model,
            1,
            iterations,
            repair_class=repair_class if repair_class is not None else GreedyRepair(),
            verbose=verbose,
        )
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.temperature = initial_temperature

    def polish_elites(self) -> None:
        pass

    def maybe_generate_immigrants(self):
        return []

    def step(self) -> None:
        current = self.population[0]

        candidate_perm = choose_mutation(current.permutation, self.model)
        repaired = self.repair_batch_wrapper(np.array([candidate_perm]))
        candidates = evaluate_permutation_delta_batch([current], repaired, self.model)
        self.logger.record_nfe(1)
        candidate = candidates[0]

        if candidate.cost <= current.cost:
            self.population[0] = candidate
        elif math.isfinite(candidate.cost) and math.isfinite(current.cost):
            delta = candidate.cost - current.cost
            if np.random.random() < math.exp(-delta / self.temperature):
                self.population[0] = candidate

        self.temperature = max(self.min_temperature, self.temperature * self.cooling_rate)
