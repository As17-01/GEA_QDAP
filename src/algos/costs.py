from typing import List, Tuple

import numpy as np
from numba import njit, prange

from src.data.models import Individual, Model


def create_xij(permutation: np.ndarray, model: Model) -> np.ndarray:
    xij = np.zeros((model.I, model.J), dtype=int)
    job_indices = np.arange(model.J)
    xij[permutation, job_indices] = 1
    return xij


@njit(fastmath=True, cache=True)
def _hamming_distances_numba(perm: np.ndarray, pop_perms: np.ndarray) -> float:
    pop_size, J = pop_perms.shape
    total_dist = 0.0

    for i in prange(pop_size):
        dist = 0
        for j in range(J):
            if perm[j] != pop_perms[i, j]:
                dist += 1
        total_dist += dist

    return total_dist / pop_size


def get_diversity(
    permutation: np.ndarray,
    population_perms: List[np.ndarray],
    total_cost: float,
) -> float:
    if not population_perms:
        return float(total_cost)

    # Convert to 2D array for Numba
    sample_array = np.array(population_perms, dtype=np.int32)  # (sample_size, J)
    avg_hamming = _hamming_distances_numba(permutation.astype(np.int32), sample_array)

    diversity_score = avg_hamming / len(permutation)  # 0 = identical, 1 = completely different
    divercity_coef = 1.0 - diversity_score

    return float(divercity_coef)


def cost_function_perm(
    permutation: np.ndarray, population_perms: List[np.ndarray], model: Model
) -> Tuple[float, float, np.ndarray]:
    job_indices = np.arange(model.J)

    loads = np.bincount(
        permutation,
        weights=model.aij[permutation, job_indices],
        minlength=model.I,
    )

    capacity_slack = model.bi - loads
    assignment_cost = model.cij[permutation, job_indices].sum()

    distance_matrix = model.DIS[np.ix_(permutation, permutation)]
    interaction_cost = np.sum(distance_matrix * model.F)

    total_cost = float(assignment_cost + interaction_cost)
    diversity = get_diversity(permutation, population_perms, total_cost)
    if np.any(capacity_slack < 0):
        return float("inf"), diversity, capacity_slack
    else:
        return total_cost, diversity, capacity_slack


def evaluate_permutation(permutation: np.ndarray, population_perms: List[np.ndarray], model: Model) -> Individual:
    xij = create_xij(permutation, model)
    cost, diversity, capacity_slack = cost_function_perm(permutation, population_perms, model)

    return Individual(
        permutation=permutation.copy(),
        xij=xij,
        cost=cost,
        diversity=diversity,
        cvar=capacity_slack,
    )
