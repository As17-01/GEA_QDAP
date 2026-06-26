from typing import Sequence, Tuple

import numpy as np
from numba import njit

from src.data.models import Individual

# Each child paired with whichever parent it shares the most positions with -- the
# tightest baseline for delta-cost evaluation (see evaluate_permutation_delta).
CrossoverResult = Tuple[Tuple[np.ndarray, Individual], Tuple[np.ndarray, Individual]]


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


def crossover_one_point(parents: Sequence[Individual]) -> CrossoverResult:
    p1, p2 = parents[0], parents[1]
    n = len(p1.permutation)

    point = np.random.randint(1, n)
    child1, child2 = crossover_one_point_nb(p1.permutation, p2.permutation, point)

    # child1 = p1's prefix + p2's tail -> matches p1 outside [point:]
    # child2 = p2's prefix + p1's tail -> matches p2 outside [point:]
    return (child1, p1), (child2, p2)


def crossover_two_point(parents: Sequence[Individual]) -> CrossoverResult:
    p1, p2 = parents[0], parents[1]
    n = len(p1.permutation)

    idx = np.random.randint(1, n, size=2)
    a, b = np.sort(idx)

    child1, child2 = crossover_two_point_nb(p1.permutation, p2.permutation, a, b)

    # child1 = p2 with a p1-segment spliced into [a:b) -> matches p2 outside [a:b)
    # child2 = p1 with a p2-segment spliced into [a:b) -> matches p1 outside [a:b)
    return (child1, p2), (child2, p1)


# Adding a new crossover operator is just adding it here -- choose_crossover picks
# uniformly among whatever is listed, with no separate count to remember to update.
CROSSOVER_OPERATORS = (
    crossover_one_point,
    crossover_two_point,
)


def choose_crossover(parents: Sequence[Individual]) -> CrossoverResult:
    op = CROSSOVER_OPERATORS[np.random.randint(len(CROSSOVER_OPERATORS))]
    return op(parents)
