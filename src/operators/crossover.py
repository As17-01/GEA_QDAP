from typing import Sequence, Tuple

import numpy as np
from numba import njit

from src.data.models import Individual


@njit(fastmath=True, cache=True)
def crossover_one_point_nb(p1: np.ndarray, p2: np.ndarray, point: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(p1)

    child1 = np.empty(n, dtype=p1.dtype)
    child2 = np.empty(n, dtype=p1.dtype)

    child1[:point] = p1[:point]
    child1[point:] = p2[point:]

    child2[:point] = p2[:point]
    child2[point:] = p1[point:]

    return child1, child2


@njit(fastmath=True, cache=True)
def crossover_two_point_nb(p1: np.ndarray, p2: np.ndarray, a: int, b: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(p1)

    child1 = np.empty(n, dtype=p1.dtype)
    child2 = np.empty(n, dtype=p1.dtype)

    child1[:a] = p2[:a]
    child1[a:b] = p1[a:b]
    child1[b:] = p2[b:]

    child2[:a] = p1[:a]
    child2[a:b] = p2[a:b]
    child2[b:] = p1[b:]

    return child1, child2


def choose_crossover(parents: Sequence[Individual]) -> Tuple[np.ndarray, np.ndarray]:
    if np.random.randint(2) == 0:
        return crossover_one_point(parents)
    else:
        return crossover_two_point(parents)


def crossover_one_point(parents: Sequence[Individual]) -> Tuple[np.ndarray, np.ndarray]:
    p1 = parents[0].permutation
    p2 = parents[1].permutation
    n = len(p1)

    point = np.random.randint(1, n)
    return crossover_one_point_nb(p1, p2, point)


def crossover_two_point(parents: Sequence[Individual]) -> Tuple[np.ndarray, np.ndarray]:
    p1 = parents[0].permutation
    p2 = parents[1].permutation
    n = len(p1)

    idx = np.random.randint(1, n, size=2)
    a, b = np.sort(idx)

    return crossover_two_point_nb(p1, p2, a, b)
