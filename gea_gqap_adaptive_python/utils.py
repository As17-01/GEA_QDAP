from typing import Tuple

import numpy as np

from gea_gqap_adaptive_python.data.models import Individual, Model


def create_xij(permutation: np.ndarray, model: Model) -> np.ndarray:
    xij = np.zeros((model.I, model.J), dtype=int)
    job_indices = np.arange(model.J)
    xij[permutation, job_indices] = 1
    return xij


def get_unfeasibility_fine(capacity_slack):
    return float("inf")


def cost_function_perm(permutation: np.ndarray, model: Model) -> Tuple[float, np.ndarray]:
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
        unfeasibility_fine = get_unfeasibility_fine(capacity_slack)
        return unfeasibility_fine + total_cost, capacity_slack
    return total_cost, capacity_slack


def cost_function(xij: np.ndarray, model: Model) -> Tuple[float, np.ndarray]:
    x = xij.astype(float, copy=False)

    loads = (model.aij * x).sum(axis=1)
    capacity_slack = model.bi - loads
    assignment_cost = np.sum(model.cij * x)

    temp = np.einsum("ij,ik,kl->jl", x, model.DIS, x)
    interaction_cost = np.sum(temp * model.F)

    total_cost = float(assignment_cost + interaction_cost)
    if np.any(capacity_slack < 0):
        unfeasibility_fine = get_unfeasibility_fine(capacity_slack)
        return unfeasibility_fine + total_cost, capacity_slack
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


def clone_individual(individual: Individual) -> Individual:
    return Individual(
        permutation=individual.permutation.copy(),
        xij=individual.xij.copy(),
        cost=individual.cost,
        cvar=individual.cvar.copy(),
    )
