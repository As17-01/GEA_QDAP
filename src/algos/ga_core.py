from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from src.data.models import Individual, Model
from src.heuristics import heuristic2
from src.operators.crossover import choose_crossover
from src.operators.mutations import choose_mutation
from src.utils import evaluate_permutation


class BaseGA(ABC):
    def __init__(self, model: Model, population_size: int, iterations: int):
        self.model = model
        self.population_size = population_size
        self.iterations = iterations

        self.rng = np.random.default_rng()
        self.population: List[Individual] = []
        self.best_solution: Individual | None = None
        self.worst_cost: float = float("inf")

    # =========================
    # Initialization
    # =========================

    def initialize_population(self) -> None:
        best = heuristic2(self.model)
        self.population = [best]

        while len(self.population) < self.population_size:
            perm = choose_mutation(best.permutation, self.model, self.rng)
            ind = evaluate_permutation(perm, self.model)
            if math.isfinite(ind.cost):
                self.population.append(ind)

        self.population.sort(key=lambda x: x.cost)
        self.best_solution = self.population[0]
        self.worst_cost = self.population[-1].cost

    # =========================
    # Selection
    # =========================

    def compute_selection_probabilities(self, beta: float = 10.0) -> np.ndarray:
        costs = np.array([ind.cost for ind in self.population], dtype=float)
        probs = np.exp(-beta * costs / self.worst_cost)
        return probs / probs.sum()

    # =========================
    # Operators
    # =========================
    def _roulette_wheel_selection(self, probabilities: np.ndarray) -> int:
        return int(np.searchsorted(np.cumsum(probabilities), self.rng.random(), side="right"))

    def crossover(self, probabilities: np.ndarray, n: int) -> List[Individual]:
        offspring = []

        for _ in range(0, n, 2):
            i1 = self._roulette_wheel_selection(probabilities)
            i2 = self._roulette_wheel_selection(probabilities)

            p1, p2 = self.population[i1], self.population[i2]
            perms = choose_crossover((p1, p2), self.rng)

            for perm in perms:
                child = evaluate_permutation(perm, self.model)
                if math.isfinite(child.cost):
                    offspring.append(child)

        return offspring

    def mutate(self, n: int) -> List[Individual]:
        mutations = []

        for _ in range(n):
            idx = self.rng.integers(0, len(self.population))
            perm = choose_mutation(self.population[idx].permutation, self.model, self.rng)
            ind = evaluate_permutation(perm, self.model)

            if math.isfinite(ind.cost):
                mutations.append(ind)

        return mutations

    # =========================
    # Evolution step (customizable)
    # =========================

    @abstractmethod
    def step(self) -> None:
        pass

    # =========================
    # Run loop
    # =========================

    def run(self, time_limit: float | None = None):
        self.initialize_population()

        start = time.perf_counter()

        for _ in range(self.iterations):
            self.step()

            self.population.sort(key=lambda x: x.cost)
            self.best_solution = self.population[0]
            self.worst_cost = max(self.worst_cost, self.population[-1].cost)

            if time_limit and (time.perf_counter() - start) >= time_limit:
                break

        return self.best_solution
