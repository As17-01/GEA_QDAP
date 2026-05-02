import numpy as np
from numba import njit

from src.data.models import Model


# =========================
# Numba JIT-compiled mutations
# =========================

@njit(fastmath=True, cache=True)
def mutation_swap_nb(perm: np.ndarray) -> np.ndarray:
    n = len(perm)
    if n < 2:
        return perm.copy()
    i = np.random.randint(0, n - 1)
    result = perm.copy()
    result[i], result[i + 1] = result[i + 1], result[i]
    return result


@njit(fastmath=True, cache=True)
def mutation_reversion_nb(perm: np.ndarray) -> np.ndarray:
    n = len(perm)
    if n < 2:
        return perm.copy()
    i, j = np.sort(np.random.choice(n, 2, replace=False))
    result = perm.copy()
    result[i:j+1] = result[i:j+1][::-1]
    return result


@njit(fastmath=True, cache=True)
def mutation_insertion_nb(perm: np.ndarray) -> np.ndarray:
    n = len(perm)
    if n < 3:
        return perm.copy()
    
    idx = np.sort(np.random.choice(np.arange(1, n), 2, replace=False))
    i, j = idx[0], idx[1]
    
    result = np.empty(n, dtype=np.int64)
    pos = 0
    result[pos:pos + (j - i + 1)] = perm[i:j+1]
    pos += (j - i + 1)
    result[pos:pos + i] = perm[:i]
    pos += i
    result[pos:] = perm[j+1:]
    
    return result


@njit(fastmath=True, cache=True)
def mutation_big_swap_nb(perm: np.ndarray) -> np.ndarray:
    n = len(perm)
    if n < 2:
        return perm.copy()
    i, j = np.random.choice(n, 2, replace=False)
    result = perm.copy()
    result[i], result[j] = result[j], result[i]
    return result


@njit(fastmath=True, cache=True)
def mutation_random_nb(perm: np.ndarray, max_value: int) -> np.ndarray:
    n = len(perm)
    if n < 2:
        return perm.copy()
    
    result = perm.copy()
    num = np.random.randint(1, min(6, n) + 1)
    indices = np.random.choice(n, num, replace=False)
    
    for k in range(num):
        result[indices[k]] = np.random.randint(0, max_value)
    
    return result


# =========================
# Public Python interface
# =========================

def choose_mutation(permutation: np.ndarray, model: Model, rng: np.random.Generator) -> np.ndarray:
    op = rng.integers(1, 6)

    if op == 1:
        return mutation_swap(permutation, rng)
    if op == 2:
        return mutation_reversion(permutation, rng)
    if op == 3:
        return mutation_insertion(permutation, rng)
    if op == 4:
        return mutation_random(permutation, model, rng)
    return mutation_big_swap(permutation, rng)


def mutation_swap(permutation: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return mutation_swap_nb(permutation)


def mutation_reversion(permutation: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return mutation_reversion_nb(permutation)


def mutation_insertion(permutation: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return mutation_insertion_nb(permutation)


def mutation_big_swap(permutation: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return mutation_big_swap_nb(permutation)


def mutation_random(permutation: np.ndarray, model: Model, rng: np.random.Generator) -> np.ndarray:
    return mutation_random_nb(permutation, int(model.I))
