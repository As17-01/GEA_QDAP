import numpy as np
from numba import njit

from src.data.models import Model


@njit(fastmath=True, cache=True)
def mutation_swap_nb(perm: np.ndarray) -> np.ndarray:
    n = len(perm)

    i = np.random.randint(0, n - 1)
    result = perm.copy()
    result[i], result[i + 1] = result[i + 1], result[i]
    return result


@njit(fastmath=True, cache=True)
def mutation_reversion_nb(perm: np.ndarray) -> np.ndarray:
    n = len(perm)

    i, j = np.sort(np.random.choice(n, 2, replace=False))
    result = perm.copy()
    result[i : j + 1] = result[i : j + 1][::-1]
    return result


@njit(fastmath=True, cache=True)
def mutation_insertion_nb(perm: np.ndarray) -> np.ndarray:
    n = len(perm)

    idx = np.sort(np.random.choice(np.arange(1, n), 2, replace=False))
    i, j = idx[0], idx[1]

    result = np.empty(n, dtype=np.int64)
    pos = 0
    result[pos : pos + (j - i + 1)] = perm[i : j + 1]
    pos += j - i + 1
    result[pos : pos + i] = perm[:i]
    pos += i
    result[pos:] = perm[j + 1 :]

    return result


@njit(fastmath=True, cache=True)
def mutation_big_swap_nb(perm: np.ndarray) -> np.ndarray:
    n = len(perm)

    i, j = np.random.choice(n, 2, replace=False)
    result = perm.copy()
    result[i], result[j] = result[j], result[i]
    return result


@njit(fastmath=True, cache=True)
def mutation_random_nb(perm: np.ndarray, max_value: int) -> np.ndarray:
    n = len(perm)

    result = perm.copy()
    num = np.random.randint(1, min(6, n) + 1)
    indices = np.random.choice(n, num, replace=False)

    for k in range(num):
        result[indices[k]] = np.random.randint(0, max_value)

    return result


# Every operator below takes (permutation, model) and returns a new permutation, even
# though most ignore `model` -- a uniform signature so MUTATION_OPERATORS can dispatch
# to any of them without special-casing.


def mutation_swap(permutation: np.ndarray, model: Model) -> np.ndarray:
    return mutation_swap_nb(permutation)


def mutation_reversion(permutation: np.ndarray, model: Model) -> np.ndarray:
    return mutation_reversion_nb(permutation)


def mutation_insertion(permutation: np.ndarray, model: Model) -> np.ndarray:
    return mutation_insertion_nb(permutation)


def mutation_big_swap(permutation: np.ndarray, model: Model) -> np.ndarray:
    return mutation_big_swap_nb(permutation)


def mutation_random(permutation: np.ndarray, model: Model) -> np.ndarray:
    return mutation_random_nb(permutation, int(model.I))


# Adding a new operator is just adding it here -- choose_mutation picks uniformly among
# whatever is listed, with no separate count (and no np.random.randint(N) literal) to
# remember to update. (Previously this dispatch used np.random.randint(4) with 5 possible
# branches, silently making mutation_big_swap unreachable.)
MUTATION_OPERATORS = (
    mutation_swap,
    mutation_reversion,
    mutation_insertion,
    mutation_random,
    mutation_big_swap,
)


def choose_mutation(permutation: np.ndarray, model: Model) -> np.ndarray:
    op = MUTATION_OPERATORS[np.random.randint(len(MUTATION_OPERATORS))]
    return op(permutation, model)
