from typing import List, Tuple

import numpy as np
from numba import njit, prange

from src.data.models import Individual, Model


def create_xij(permutation: np.ndarray, model: Model) -> np.ndarray:
    xij = np.zeros((model.I, model.J), dtype=int)
    job_indices = np.arange(model.J)
    xij[permutation, job_indices] = 1
    return xij


def get_diversity(population_base: List[Individual], population_to_eval: List[Individual]) -> np.ndarray:
    base_perms = np.array([ind.permutation for ind in population_base], dtype=np.int32)  # (N_base, J)
    eval_perms = np.array([ind.permutation for ind in population_to_eval], dtype=np.int32)  # (N_eval, J)

    J = base_perms.shape[1]

    diff = eval_perms[:, None, :] != base_perms[None, :, :]
    hamming = diff.sum(axis=2)
    diversities = hamming.mean(axis=1) / J

    return diversities.astype(np.float32)


def cost_function_perm(permutation: np.ndarray, model: Model) -> Tuple[float, float, np.ndarray]:
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
    if np.any(capacity_slack < 0):
        return float("inf"), capacity_slack
    else:
        return total_cost, capacity_slack


def evaluate_permutation(permutation: np.ndarray, model: Model) -> Individual:
    xij = create_xij(permutation, model)
    cost, capacity_slack = cost_function_perm(permutation, model)

    return Individual(
        permutation=permutation.copy(),
        xij=xij,
        cost=cost,
        cvar=capacity_slack,
    )
