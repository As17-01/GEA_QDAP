import numpy as np
from numba import njit, prange

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


def choose_mutation(permutation: np.ndarray, model: Model) -> np.ndarray:
    op = np.random.randint(4)

    if op == 0:
        return mutation_swap(permutation)
    if op == 1:
        return mutation_reversion(permutation)
    if op == 2:
        return mutation_insertion(permutation)
    if op == 3:
        return mutation_random(permutation, model)

    return mutation_big_swap(permutation)


# Batched dispatch for a whole generation's worth of mutants in one compiled call,
# instead of one Python<->numba dispatch (choose_mutation) per individual. Mirrors
# choose_mutation's op selection exactly, including the fact that np.random.randint(4)
# only ever yields 0-3, so mutation_big_swap is never actually selected by either path.
@njit(parallel=True, cache=True)
def _mutation_batch_nb(perms: np.ndarray, ops: np.ndarray, max_value: int) -> np.ndarray:
    n = perms.shape[0]
    for k in prange(n):
        op = ops[k]
        if op == 0:
            perms[k] = mutation_swap_nb(perms[k])
        elif op == 1:
            perms[k] = mutation_reversion_nb(perms[k])
        elif op == 2:
            perms[k] = mutation_insertion_nb(perms[k])
        else:
            perms[k] = mutation_random_nb(perms[k], max_value)
    return perms


def choose_mutation_batch(permutations: np.ndarray, model: Model) -> np.ndarray:
    n = permutations.shape[0]
    ops = np.random.randint(0, 4, size=n)
    return _mutation_batch_nb(permutations.copy(), ops, int(model.I))


def mutation_swap(permutation: np.ndarray) -> np.ndarray:
    return mutation_swap_nb(permutation)


def mutation_reversion(permutation: np.ndarray) -> np.ndarray:
    return mutation_reversion_nb(permutation)


def mutation_insertion(permutation: np.ndarray) -> np.ndarray:
    return mutation_insertion_nb(permutation)


def mutation_big_swap(permutation: np.ndarray) -> np.ndarray:
    return mutation_big_swap_nb(permutation)


def mutation_random(permutation: np.ndarray, model: Model) -> np.ndarray:
    return mutation_random_nb(permutation, int(model.I))
