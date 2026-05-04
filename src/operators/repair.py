import numpy as np
from numba import njit

# ==========================================
# Selection policies (Numba compiled)
# ==========================================


@njit
def _select_target_greedy(i_job, aij, slack, bi, I, tol, subsample_size):
    best_target = -1
    best_cost = 1e18

    for i in range(I):
        if slack[i] + aij[i, i_job] <= bi[i] + tol:
            if aij[i, i_job] < best_cost:
                best_cost = aij[i, i_job]
                best_target = i

    if best_target == -1:
        max_slack = -1e18
        for i in range(I):
            if slack[i] > max_slack:
                max_slack = slack[i]
                best_target = i

    return best_target


@njit
def _select_target_rf(i_job, aij, slack, bi, I, tol, subsample_size):
    k = int(I * subsample_size)
    if k > I:
        k = I

    best_target = -1
    best_cost = 1e18

    # sample subset
    for _ in range(k):
        i = np.random.randint(0, I)

        if slack[i] + aij[i, i_job] <= bi[i] + tol:
            if aij[i, i_job] < best_cost:
                best_cost = aij[i, i_job]
                best_target = i

    if best_target == -1:
        max_slack = -1e18
        for i in range(I):
            if slack[i] > max_slack:
                max_slack = slack[i]
                best_target = i

    return best_target


# ==========================================
# Shared repair core
# ==========================================


@njit
def _repair_core(perm, aij, bi, I, J, tol, max_repair_attempts, subsample_size, select_target_fn):
    loads = np.zeros(I, dtype=aij.dtype)

    for j in range(J):
        i = perm[j]
        loads[i] += aij[i, j]

    slack = bi - loads

    attempts = 0

    while attempts < max_repair_attempts:
        attempts += 1

        # Check overload
        overloaded_exists = False
        for i in range(I):
            if slack[i] < -tol:
                overloaded_exists = True
                break
        if not overloaded_exists:
            break

        # Greedy removal
        best_j = -1
        best_val = -1.0

        for j in range(J):
            i = perm[j]
            if slack[i] < -tol:
                val = aij[i, j]
                if val > best_val:
                    best_val = val
                    best_j = j

        if best_j == -1:
            break

        i_old = perm[best_j]

        # Remove
        loads[i_old] -= aij[i_old, best_j]
        slack[i_old] = bi[i_old] - loads[i_old]
        perm[best_j] = -1

        # Insert via strategy
        target = select_target_fn(best_j, aij, slack, bi, I, tol, subsample_size)

        # Assign
        perm[best_j] = target
        loads[target] += aij[target, best_j]
        slack[target] = bi[target] - loads[target]

    # Final safety pass
    for j in range(J):
        if perm[j] == -1:
            best_target = 0
            max_slack = -1e18
            for i in range(I):
                if slack[i] > max_slack:
                    max_slack = slack[i]
                    best_target = i

            perm[j] = best_target
            loads[best_target] += aij[best_target, j]
            slack[best_target] = bi[best_target] - loads[best_target]

    return perm


# ==========================================
# Base class (pure greedy)
# ==========================================


class GreedyRepair:
    def __init__(self, model, tol: float = 1e-9):
        self.model = model
        self.tol = tol

    def _get_selector(self):
        return _select_target_greedy

    def _get_subsample_size(self):
        return 0  # unused

    def repair(self, perm: np.ndarray, max_repair_attempts: int = 100) -> np.ndarray:
        return _repair_core(
            perm.copy(),
            self.model.aij,
            self.model.bi,
            self.model.I,
            self.model.J,
            self.tol,
            max_repair_attempts,
            self._get_subsample_size(),
            self._get_selector(),
        )


class RFRepair(GreedyRepair):
    def __init__(self, model, tol: float = 1e-9, subsample_size: float = 0.2):
        super().__init__(model, tol)
        self.subsample_size = subsample_size

    def _get_selector(self):
        return _select_target_rf

    def _get_subsample_size(self):
        return self.subsample_size
