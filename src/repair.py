import numpy as np
from numba import njit, prange


# Picks the cheapest facility with spare capacity for i_job; falls back to the
# facility with the most slack if none fit.
@njit(cache=True)
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


# Same as _select_target_greedy but only samples a random subset of facilities,
# trading optimality for population diversity.
@njit(cache=True)
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


# Picks the job with the largest assignment cost among all jobs currently in overloaded
# facilities -- the exact "worst violation first" choice, deterministic.
@njit(cache=True)
def _select_evict_greedy(perm, aij, slack, tol, J, subsample_size):
    best_j = -1
    best_val = -1.0

    for j in range(J):
        i = perm[j]
        if slack[i] < -tol:
            val = aij[i, j]
            if val > best_val:
                best_val = val
                best_j = j

    return best_j


# Same as _select_evict_greedy, but only considers a random subsample (a share of all
# overloaded jobs, not an absolute count) drawn via single-pass reservoir sampling --
# trades exact "worst violation first" eviction for population diversity, same trade-off
# _select_target_rf already makes on the insertion side.
@njit(cache=True)
def _select_evict_rf(perm, aij, slack, tol, J, subsample_size):
    overloaded_count = 0
    for j in range(J):
        i = perm[j]
        if slack[i] < -tol:
            overloaded_count += 1

    if overloaded_count == 0:
        return -1

    k = int(overloaded_count * subsample_size)
    if k < 1:
        k = 1

    reservoir = np.empty(k, dtype=np.int64)
    seen = 0
    for j in range(J):
        i = perm[j]
        if slack[i] < -tol:
            if seen < k:
                reservoir[seen] = j
            else:
                r = np.random.randint(0, seen + 1)
                if r < k:
                    reservoir[r] = j
            seen += 1

    best_j = -1
    best_val = -1.0
    for idx in range(k):
        j = reservoir[idx]
        i = perm[j]
        val = aij[i, j]
        if val > best_val:
            best_val = val
            best_j = j

    return best_j


# Iteratively evicts an overloaded facility's job (chosen via select_evict_fn) and
# reassigns it via select_target_fn, until every facility is within capacity or attempts
# run out.
@njit(cache=True)
def _repair_core(perm, aij, bi, I, J, tol, max_repair_attempts, subsample_size, select_target_fn, select_evict_fn):
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

        # Removal via strategy
        best_j = select_evict_fn(perm, aij, slack, tol, J, subsample_size)

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


# Repairs a whole batch of permutations in one call -- each row is independent, so this
# runs across cores via prange instead of paying per-individual Python<->numba dispatch
# overhead for every offspring/mutant in a generation.
@njit(parallel=True, cache=True)
def _repair_core_batch(perms, aij, bi, I, J, tol, max_repair_attempts, subsample_size, select_target_fn, select_evict_fn):
    n = perms.shape[0]
    for idx in prange(n):
        perms[idx] = _repair_core(
            perms[idx], aij, bi, I, J, tol, max_repair_attempts, subsample_size, select_target_fn, select_evict_fn
        )
    return perms


# Repairs capacity violations by always evicting the worst violation and reassigning to
# the cheapest feasible facility -- fully deterministic.
class GreedyRepair:
    def __init__(self, tol: float = 1e-9):
        self.tol = tol

    def _get_target_selector(self):
        return _select_target_greedy

    def _get_evict_selector(self):
        return _select_evict_greedy

    def _get_subsample_size(self):
        return 0  # unused

    def repair(self, perm: np.ndarray, model, max_repair_attempts: int = 100) -> np.ndarray:
        return _repair_core(
            perm.copy(),
            model.aij,
            model.bi,
            model.I,
            model.J,
            self.tol,
            max_repair_attempts,
            self._get_subsample_size(),
            self._get_target_selector(),
            self._get_evict_selector(),
        )

    def repair_batch(self, perms: np.ndarray, model, max_repair_attempts: int = 100) -> np.ndarray:
        return _repair_core_batch(
            perms.copy(),
            model.aij,
            model.bi,
            model.I,
            model.J,
            self.tol,
            max_repair_attempts,
            self._get_subsample_size(),
            self._get_target_selector(),
            self._get_evict_selector(),
        )


# Repairs capacity violations by evicting from and reassigning to a random subsample of
# candidates (a share of the candidate pool, not an absolute count) -- trades exact
# greediness for population diversity.
class RFRepair(GreedyRepair):
    def __init__(self, tol: float = 1e-9, subsample_size: float = 0.2):
        super().__init__(tol)
        self.subsample_size = subsample_size

    def _get_target_selector(self):
        return _select_target_rf

    def _get_evict_selector(self):
        return _select_evict_rf

    def _get_subsample_size(self):
        return self.subsample_size
