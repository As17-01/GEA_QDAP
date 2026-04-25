from typing import Sequence, Tuple

import numpy as np

from gea_gqap_adaptive_python.models import Individual


def choose_crossover(parents: Sequence[Individual], rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if rng.integers(1, 3) == 1:
        return crossover_one_point(parents, rng)
    return crossover_two_point(parents, rng)


def crossover_one_point(parents: Sequence[Individual], rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    p1, p2 = parents[0].permutation, parents[1].permutation
    n = p1.size

    if n < 2:
        return p1.copy(), p2.copy()

    point = int(rng.integers(1, n))
    return (
        np.concatenate((p1[:point], p2[point:])),
        np.concatenate((p2[:point], p1[point:])),
    )


def crossover_two_point(parents: Sequence[Individual], rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    p1, p2 = parents[0].permutation, parents[1].permutation
    n = p1.size

    if n < 3:
        return p1.copy(), p2.copy()

    a, b = np.sort(rng.choice(np.arange(1, n), size=2, replace=False))
    return (
        np.concatenate((p2[:a], p1[a:b], p2[b:])),
        np.concatenate((p1[:a], p2[a:b], p1[b:])),
    )
