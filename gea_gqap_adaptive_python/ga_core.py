import math
from typing import List, Tuple

import numpy as np

from gea_gqap_adaptive_python.heuristics import heuristic2
from gea_gqap_adaptive_python.models import Individual, Model
from gea_gqap_adaptive_python.operators import mutation
from gea_gqap_adaptive_python.utils import evaluate_permutation


def initialize_population(model: Model, population_size: int, rng: np.random.Generator) -> List[Individual]:
    population: List[Individual] = []
    best_solution = heuristic2(model)
    population.append(best_solution)

    while len(population) < population_size:
        mutated = mutation(population[0].permutation, model, rng)
        individual = evaluate_permutation(mutated, model)
        if math.isfinite(individual.cost):
            population.append(individual)

    population.sort(key=lambda ind: ind.cost)
    return population


def compute_selection_probabilities(population: List[Individual], beta: float, worst_cost: float) -> np.ndarray:
    costs = np.array([ind.cost for ind in population], dtype=float)
    probabilities = np.exp(-beta * costs / worst_cost)
    probabilities /= probabilities.sum()
    return probabilities


def update_best(population: List[Individual], best_solution: Individual) -> Individual:
    if population[0].cost < best_solution.cost:
        return population[0]
    return best_solution


def build_pool(
    population: List[Individual],
    offspring: List[Individual],
    mutations: List[Individual],
    scenario_candidates: List[Individual],
    crossover_origins: List[str],
    mutation_origins: List[str],
    scenario_origins: List[str],
) -> List[Tuple[Individual, str]]:
    pool = list(zip(population, ["previous"] * len(population)))
    pool.extend(zip(offspring, crossover_origins))
    pool.extend(zip(mutations, mutation_origins))
    pool.extend(zip(scenario_candidates, scenario_origins))
    pool.sort(key=lambda item: item[0].cost)
    return pool


def compute_contribution(top_origins: List[str]) -> Tuple[float, float, float, float]:
    total = len(top_origins)
    if total == 0:
        return (0.0, 0.0, 0.0, 0.0)

    return (
        top_origins.count("previous") / total,
        top_origins.count("crossover") / total,
        top_origins.count("mutation") / total,
        top_origins.count("scenario") / total,
    )
