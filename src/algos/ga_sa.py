import math

import numpy as np

from src.algos.base import BaseGA
from src.costs import evaluate_permutation_delta_batch
from src.operators.mutations import choose_mutation


class SimulatedAnnealing(BaseGA):
    """Bertsimas & Tsitsiklis (1993). Classic SA anneals a single solution; here the
    population is a bank of independent annealing chains (one walker per population slot)
    sharing a single global temperature schedule, so it reuses BaseGA's batched repair/eval
    machinery and run loop instead of looping one permutation at a time in pure Python.

    Each walker proposes one neighbor per iteration via the existing mutation operators and
    accepts it outright if it's better, or with Metropolis probability exp(-delta/T) if it's
    worse. No crossover, no selection pressure between walkers -- each chain only ever
    competes against its own previous state.
    """

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
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

        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.temperature = initial_temperature

    def step(self) -> None:
        perms = np.array([choose_mutation(ind.permutation, self.model) for ind in self.population])
        repaired = self.repair_batch_wrapper(perms)
        candidates = evaluate_permutation_delta_batch(self.population, repaired, self.model)

        for i, (current, candidate) in enumerate(zip(self.population, candidates)):
            if candidate.cost <= current.cost:
                self.population[i] = candidate
                continue

            if not (math.isfinite(candidate.cost) and math.isfinite(current.cost)):
                continue

            delta = candidate.cost - current.cost
            if np.random.random() < math.exp(-delta / self.temperature):
                self.population[i] = candidate

        immigrants = self.maybe_generate_immigrants()
        if immigrants:
            replace_idx = np.random.choice(self.population_size, size=len(immigrants), replace=False)
            for idx, immigrant in zip(replace_idx, immigrants):
                self.population[idx] = immigrant

        self.temperature = max(self.min_temperature, self.temperature * self.cooling_rate)
