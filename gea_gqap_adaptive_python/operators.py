from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .models import AlgorithmConfig, Individual, Model
from .utils import create_xij, evaluate_permutation


def roulette_wheel_selection(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    cumulative = np.cumsum(probabilities)
    r = rng.random()
    return int(np.searchsorted(cumulative, r, side="right"))


def mutation(permutation: np.ndarray, model: Model, rng: np.random.Generator) -> np.ndarray:
    operator = rng.integers(1, 6)
    if operator == 1:
        return mutation_swap(permutation, rng)
    if operator == 2:
        return mutation_reversion(permutation, rng)
    if operator == 3:
        return mutation_insertion(permutation, rng)
    if operator == 4:
        return mutation_random(permutation, model, rng)
    return mutation_big_swap(permutation, rng)


def mutation_swap(permutation: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = permutation.size
    if n < 2:
        return permutation.copy()
    point = int(rng.integers(0, n - 1))
    result = permutation.copy()
    result[point], result[point + 1] = result[point + 1], result[point]
    return result


def mutation_reversion(permutation: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = permutation.size
    if n < 2:
        return permutation.copy()
    i, j = np.sort(rng.choice(n, size=2, replace=False))
    result = permutation.copy()
    result[i : j + 1] = result[i : j + 1][::-1]
    return result


def mutation_insertion(permutation: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = permutation.size
    if n < 3:
        return permutation.copy()
    indices = np.sort(rng.choice(np.arange(1, n), size=2, replace=False))
    i, j = indices
    result = np.concatenate((permutation[i : j + 1], permutation[:i], permutation[j + 1 :]))
    return result


def mutation_random(permutation: np.ndarray, model: Model, rng: np.random.Generator) -> np.ndarray:
    n = permutation.size
    if n < 2:
        return permutation.copy()
    num_rand = int(rng.integers(1, min(6, n)))
    points = rng.choice(np.arange(0, n - 1), size=num_rand, replace=False)
    result = permutation.copy()
    for idx in points:
        result[idx] = int(rng.integers(0, model.I))
    return result


def mutation_big_swap(permutation: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = permutation.size
    if n < 2:
        return permutation.copy()
    i, j = rng.choice(n, size=2, replace=False)
    result = permutation.copy()
    result[i], result[j] = result[j], result[i]
    return result


def crossover(parents: Sequence[Individual], rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    operator = rng.integers(1, 3)
    if operator == 1:
        return crossover_one_point(parents, rng)
    return crossover_two_point(parents, rng)


def crossover_one_point(parents: Sequence[Individual], rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    parent1 = parents[0].permutation
    parent2 = parents[1].permutation
    n = parent1.size
    if n < 2:
        return parent1.copy(), parent2.copy()
    point = int(rng.integers(1, n))
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2


def crossover_two_point(parents: Sequence[Individual], rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    parent1 = parents[0].permutation
    parent2 = parents[1].permutation
    n = parent1.size
    if n < 3:
        return parent1.copy(), parent2.copy()
    points = np.sort(rng.choice(np.arange(1, n), size=2, replace=False))
    p1, p2 = (int(points[0]), int(points[1]))
    child1 = np.concatenate((parent2[:p1], parent1[p1:p2], parent2[p2:]))
    child2 = np.concatenate((parent1[:p1], parent2[p1:p2], parent1[p2:]))
    return child1, child2


def mask_mutation(index: int, permutation: np.ndarray, mask: np.ndarray, model: Model, rng: np.random.Generator) -> np.ndarray:
    if index == 1:
        return mask_mutation_swap(permutation, mask, rng)
    if index == 2:
        return mask_mutation_big_swap(permutation, mask, rng)
    if index == 3:
        return mask_mutation_inversion(permutation, mask, rng)
    if index == 4:
        return mask_mutation_displacement(permutation, mask, rng)
    return mask_mutation_perturbation(permutation, mask, model, rng)


def mask_mutation_swap(permutation: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indices = np.where(~mask)[0]
    if indices.size <= 1:
        return permutation.copy()
    point = int(rng.integers(0, indices.size - 1))
    result = permutation.copy()
    i = indices[point]
    j = indices[point + 1]
    result[[i, j]] = result[[j, i]]
    return result


def mask_mutation_big_swap(permutation: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indices = np.where(~mask)[0]
    if indices.size <= 1:
        return permutation.copy()
    i, j = rng.choice(indices, size=2, replace=False)
    result = permutation.copy()
    result[[i, j]] = result[[j, i]]
    return result


def mask_mutation_inversion(permutation: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indices = np.where(~mask)[0]
    if indices.size <= 1:
        return permutation.copy()
    i, j = np.sort(rng.choice(indices, size=2, replace=False))
    result = permutation.copy()
    result[i : j + 1] = result[i : j + 1][::-1]
    return result


def mask_mutation_displacement(permutation: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indices = np.where(~mask)[0]
    if indices.size <= 2:
        return permutation.copy()
    subset = permutation[indices]
    choices = np.sort(rng.choice(np.arange(1, subset.size), size=2, replace=False))
    i, j = choices
    temp = subset[i : j + 1]
    q1 = subset[:i]
    q2 = subset[j + 1 :]
    new_subset = np.concatenate((temp, q1, q2))
    result = permutation.copy()
    result[indices] = new_subset
    return result


def mask_mutation_perturbation(permutation: np.ndarray, mask: np.ndarray, model: Model, rng: np.random.Generator) -> np.ndarray:
    indices = np.where(~mask)[0]
    result = permutation.copy()
    if indices.size == 0:
        return result
    idx = rng.choice(indices)
    result[idx] = (result[idx] + 1) % model.I
    return result


def analyze_perm(
    population: Sequence[Individual],
    config: AlgorithmConfig,
    model: Model,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Individual, np.ndarray]:
    n_pop = len(population)
    n_genes = population[0].permutation.size
    n_fixed = int(np.floor(config.p_fixed_x * n_pop))

    mask = np.zeros((n_pop, n_genes), dtype=bool)
    perms = np.stack([ind.permutation for ind in population])

    left = perms[:, :-1]
    right = perms[:, 1:]
    pair_match = (
        (left[:, None, :] == left[None, :, :])
        & (right[:, None, :] == right[None, :, :])
    )
    pair_count = pair_match.sum(axis=1) - 1
    eligible = pair_count >= n_fixed

    for row in range(n_pop):
        col = 0
        while col < n_genes - 1:
            if eligible[row, col]:
                mask[row, col : col + 2] = True
                col += 2
            else:
                col += 1

    dominant_idx = 0
    dominant_score = -1
    for idx in range(n_pop):
        score = int(mask[idx].sum())
        if score > dominant_score:
            dominant_idx = idx
            dominant_score = score
        elif score == dominant_score and rng.random() > 0.5:
            dominant_idx = idx
            dominant_score = score

    dominant_individual = evaluate_permutation(population[dominant_idx].permutation, model)
    return (
        dominant_individual.permutation,
        mask,
        dominant_individual,
        mask[dominant_idx],
    )


def combine_q(position1: np.ndarray, position2: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    pattern_bool = pattern.astype(bool)
    return np.where(pattern_bool, position1, position2)

