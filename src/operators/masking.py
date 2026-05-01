from typing import Sequence, Tuple

import numpy as np

from src.algos.configs import AlgorithmConfig
from src.data.models import Individual, Model
from src.utils import evaluate_permutation


def roulette_wheel_selection(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    return int(np.searchsorted(np.cumsum(probabilities), rng.random(), side="right"))


# =========================
# Mask mutation
# =========================


def mask_mutation(
    index: int,
    permutation: np.ndarray,
    mask: np.ndarray,
    model: Model,
    rng: np.random.Generator,
) -> np.ndarray:
    if index == 1:
        return mask_mutation_swap(permutation, mask, rng)
    if index == 2:
        return mask_mutation_big_swap(permutation, mask, rng)
    if index == 3:
        return mask_mutation_inversion(permutation, mask, rng)
    if index == 4:
        return mask_mutation_displacement(permutation, mask, rng)
    return mask_mutation_perturbation(permutation, mask, model, rng)


def _free_indices(mask: np.ndarray) -> np.ndarray:
    return np.where(~mask)[0]


def mask_mutation_swap(permutation: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = _free_indices(mask)
    if idx.size <= 1:
        return permutation.copy()

    k = int(rng.integers(0, idx.size - 1))
    i, j = idx[k], idx[k + 1]

    result = permutation.copy()
    result[i], result[j] = result[j], result[i]
    return result


def mask_mutation_big_swap(permutation: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = _free_indices(mask)
    if idx.size <= 1:
        return permutation.copy()

    i, j = rng.choice(idx, size=2, replace=False)
    result = permutation.copy()
    result[i], result[j] = result[j], result[i]
    return result


def mask_mutation_inversion(permutation: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = _free_indices(mask)
    if idx.size <= 1:
        return permutation.copy()

    i, j = np.sort(rng.choice(idx, size=2, replace=False))
    result = permutation.copy()
    result[i : j + 1] = result[i : j + 1][::-1]
    return result


def mask_mutation_displacement(permutation: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = _free_indices(mask)
    if idx.size <= 2:
        return permutation.copy()

    subset = permutation[idx]
    i, j = np.sort(rng.choice(np.arange(1, subset.size), size=2, replace=False))

    new_subset = np.concatenate((subset[i : j + 1], subset[:i], subset[j + 1 :]))

    result = permutation.copy()
    result[idx] = new_subset
    return result


def mask_mutation_perturbation(
    permutation: np.ndarray, mask: np.ndarray, model: Model, rng: np.random.Generator
) -> np.ndarray:
    idx = _free_indices(mask)
    result = permutation.copy()

    if idx.size == 0:
        return result

    i = rng.choice(idx)
    result[i] = (result[i] + 1) % model.I
    return result


# =========================
# Analysis
# =========================


def analyze_perm(
    population: Sequence[Individual],
    config: AlgorithmConfig,
    model: Model,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Individual, np.ndarray]:
    n_pop = len(population)
    n_genes = population[0].permutation.size
    n_fixed = int(np.floor(config.p_fixed_x * n_pop))

    perms = np.stack([ind.permutation for ind in population])
    mask = np.zeros((n_pop, n_genes), dtype=bool)

    left, right = perms[:, :-1], perms[:, 1:]

    pair_match = (left[:, None, :] == left[None, :, :]) & (right[:, None, :] == right[None, :, :])
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

    scores = mask.sum(axis=1)
    best_indices = np.flatnonzero(scores == scores.max())
    dominant_idx = int(rng.choice(best_indices))

    dominant = evaluate_permutation(population[dominant_idx].permutation, model)

    return (
        dominant.permutation,
        mask,
        dominant,
        mask[dominant_idx],
    )


# =========================
# Combine
# =========================


def combine_q(position1: np.ndarray, position2: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    return np.where(pattern.astype(bool), position1, position2)
