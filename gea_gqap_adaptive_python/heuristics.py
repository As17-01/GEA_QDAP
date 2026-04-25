import numpy as np

from gea_gqap_adaptive_python.models import Individual, Model
from gea_gqap_adaptive_python.utils import evaluate_permutation


def heuristic2(model: Model) -> Individual:
    I, J = model.I, model.J

    X = np.zeros((I, J), dtype=int)
    load = np.zeros(I, dtype=float)

    # Precompute CT
    dis_sum = model.DIS.sum(axis=1)
    f_sum = model.F.sum(axis=1)

    CT = model.cij + dis_sum[:, None] + f_sum[None, :]

    # Initial assignment
    for j in range(J):
        col = CT[:, j]

        for attempt in range(3):
            i = int(np.argmin(col))

            if load[i] <= model.bi[i]:
                X[i, j] = 1
                load[i] += model.aij[i, j]
                break

            col[i] = col.max()

        else:
            i = int(np.argmin(col))
            X[i, j] = 1
            load[i] += model.aij[i, j]

    # Repair phase
    slack = model.bi - load
    Wij = X * model.aij

    max_passes = I * J
    passes = 0

    while np.any(slack < -1e-9) and passes < max_passes:
        passes += 1

        for i in range(I):
            while slack[i] < -1e-9:
                jobs = np.where(X[i] == 1)[0]
                if jobs.size == 0:
                    break

                j = jobs[np.argmax(Wij[i, jobs])]

                load[i] -= model.aij[i, j]
                slack[i] = model.bi[i] - load[i]

                X[i, j] = 0
                Wij[i, j] = 0

                target = int(np.argmin(model.aij[:, j]))
                if target == i:
                    target = int(np.argmax(slack))

                load[target] += model.aij[target, j]
                slack[target] = model.bi[target] - load[target]

                X[target, j] = 1
                Wij[target, j] = model.aij[target, j]

    permutation = np.argmax(X, axis=0)
    return evaluate_permutation(permutation, model)
