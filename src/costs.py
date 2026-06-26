import math
from typing import Tuple

import numpy as np
from numba import njit

from src.data.models import Individual, Model


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
        return float("inf"), capacity_slack
    else:
        return total_cost, capacity_slack


@njit(cache=True)
def _cost_function_perm_delta_nb(old_perm, new_perm, old_cost, aij, cij, DIS, F, bi, I, J):
    """Numba core for cost_function_perm_delta -- explicit loops, no numpy fancy-indexing,
    so the O(K*J) algorithmic win (K = positions changed) isn't eaten by per-call numpy
    dispatch overhead on small/medium J. See cost_function_perm_delta for the math.
    """
    is_changed = np.zeros(J, dtype=np.bool_)
    n_changed = 0
    for j in range(J):
        if old_perm[j] != new_perm[j]:
            is_changed[j] = True
            n_changed += 1

    delta = 0.0
    if n_changed > 0:
        for j in range(J):
            if not is_changed[j]:
                continue
            delta += cij[new_perm[j], j] - cij[old_perm[j], j]

            for l in range(J):
                if l == j:
                    continue
                # each changed-changed pair would otherwise be counted from both sides
                if is_changed[l] and l < j:
                    continue
                f_jl = F[j, l]
                if f_jl == 0.0:
                    continue
                term_new = DIS[new_perm[j], new_perm[l]] * f_jl
                term_old = DIS[old_perm[j], old_perm[l]] * f_jl
                delta += 2.0 * (term_new - term_old)

    new_cost = old_cost + delta

    loads = np.zeros(I, dtype=aij.dtype)
    for j in range(J):
        i = new_perm[j]
        loads[i] += aij[i, j]
    capacity_slack = bi - loads

    feasible = True
    for i in range(I):
        if capacity_slack[i] < 0:
            feasible = False
            break

    if feasible:
        return new_cost, capacity_slack
    return np.inf, capacity_slack


def cost_function_perm_delta(
    old_perm: np.ndarray, new_perm: np.ndarray, old_cost: float, model: Model
) -> Tuple[float, np.ndarray]:
    """Like cost_function_perm(new_perm, model), but computed incrementally from a known
    feasible old_cost for old_perm. Cheaper than a full recompute whenever new_perm differs
    from old_perm in only a handful of positions (the common case for mutation, and often
    crossover); never worse than a full recompute in the worst case (everything changed).

    old_cost must be the real (finite) numeric cost for old_perm -- callers should fall back
    to cost_function_perm when that doesn't hold (e.g. old_perm was infeasible).
    """
    cost, capacity_slack = _cost_function_perm_delta_nb(
        old_perm, new_perm, old_cost, model.aij, model.cij, model.DIS, model.F, model.bi, model.I, model.J
    )
    return float(cost), capacity_slack


def evaluate_permutation(permutation: np.ndarray, model: Model) -> Individual:
    cost, capacity_slack = cost_function_perm(permutation, model)

    return Individual(
        permutation=permutation.copy(),
        cost=cost,
        cvar=capacity_slack,
    )


def evaluate_permutation_delta(old_individual: Individual, new_permutation: np.ndarray, model: Model) -> Individual:
    """evaluate_permutation, but reusing old_individual's cost via cost_function_perm_delta
    when it's feasible (the common case); falls back to a full recompute otherwise."""
    if math.isfinite(old_individual.cost):
        cost, capacity_slack = cost_function_perm_delta(old_individual.permutation, new_permutation, old_individual.cost, model)
    else:
        cost, capacity_slack = cost_function_perm(new_permutation, model)

    return Individual(
        permutation=new_permutation.copy(),
        cost=cost,
        cvar=capacity_slack,
    )
