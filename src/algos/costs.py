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


def get_hidden_cost(
    permutation: np.ndarray,
    population_perms: List[np.ndarray],
    capacity_slack: np.ndarray,
    total_cost: float,
    sample_size: int = 30,
    min_sample: int = 10,
) -> float:
    # === Slack Penalty (larger slack = bigger penalty) ===
    positive_slack = np.maximum(capacity_slack, 0.0)
    mean_slack = np.mean(positive_slack)
    slack_penalty = mean_slack * total_cost

    if not population_perms:
        return float(total_cost + slack_penalty)

    n_pop = len(population_perms)
    if n_pop <= min_sample:
        sample_perms = population_perms
    else:
        # Randomly sample to speed up significantly
        indices = np.random.choice(n_pop, size=min(sample_size, n_pop), replace=False)
        sample_perms = [population_perms[i] for i in indices]

    # Convert to 2D array for Numba
    sample_array = np.array(sample_perms, dtype=np.int32)  # (sample_size, J)
    avg_hamming = _hamming_distances_numba(permutation.astype(np.int32), sample_array)

    diversity_score = avg_hamming / len(permutation)  # 0 = identical, 1 = completely different
    divercity_coef = 1.0 - diversity_score

    return float(total_cost * (1 + divercity_coef) + slack_penalty)


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

    if np.any(capacity_slack < 0):
        total_cost = float("inf")
    else:
        total_cost = float(assignment_cost + interaction_cost)
    hidden_cost = get_hidden_cost(permutation, population_perms, capacity_slack, total_cost)
    return total_cost, hidden_cost, capacity_slack


def evaluate_permutation(permutation: np.ndarray, population_perms: List[np.ndarray], model: Model) -> Individual:
    xij = create_xij(permutation, model)
    cost, hidden_cost, capacity_slack = cost_function_perm(permutation, population_perms, model)

    return Individual(
        permutation=permutation.copy(),
        xij=xij,
        cost=cost,
        hidden_cost=hidden_cost,
        cvar=capacity_slack,
    )
