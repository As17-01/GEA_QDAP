import numpy as np

from gea_gqap_adaptive_python.data.models import Model


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
    n = permutation.size
    if n < 2:
        return permutation.copy()

    i = int(rng.integers(0, n - 1))
    result = permutation.copy()
    result[i], result[i + 1] = result[i + 1], result[i]
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

    i, j = np.sort(rng.choice(np.arange(1, n), size=2, replace=False))
    return np.concatenate((permutation[i : j + 1], permutation[:i], permutation[j + 1 :]))


def mutation_random(permutation: np.ndarray, model: Model, rng: np.random.Generator) -> np.ndarray:
    n = permutation.size
    if n < 2:
        return permutation.copy()

    result = permutation.copy()
    num = int(rng.integers(1, min(6, n)))
    indices = rng.choice(np.arange(0, n - 1), size=num, replace=False)

    for idx in indices:
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
