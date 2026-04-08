from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def rpd_matlab(Input: np.ndarray, flag: int) -> np.ndarray:
    """
    Python equivalent of `RPD.m`.

    MATLAB code:
      if flag==1:
        for each column i:
          Min = min(Input(:,i))
          Ans(j,i) = (Input(j,i)-Min)/Min
      else:
        for each row i:
          Min = min(Input(i,:))
          Ans(i,j) = (Input(i,j)-Min)/Min
    """
    X = np.asarray(Input, dtype=float)
    if X.ndim != 2:
        raise ValueError("Input must be a 2D array")

    Ans = X.copy()
    if flag == 1:
        for i in range(X.shape[1]):
            col = X[:, i]
            m = float(np.min(col))
            Ans[:, i] = (col - m) / m
    else:
        for i in range(X.shape[0]):
            row = X[i, :]
            m = float(np.min(row))
            Ans[i, :] = (row - m) / m
    return Ans


@dataclass(frozen=True)
class MEPFMResult:
    param_names: tuple[str, ...]
    levels: tuple[tuple[int, ...], ...]
    means_by_level: tuple[tuple[float, ...], ...]
    best_levels: tuple[int, ...]


def mepfm(
    input_rpd_mean: Sequence[float],
    table: np.ndarray,
    param_names: Sequence[str],
) -> MEPFMResult:
    """
    Main Effects Plot for Means (numeric core).

    - input_rpd_mean: shape (n_runs,)
    - table: shape (n_runs, n_params), entries are levels 1..3
    - param_names: length n_params
    """
    y = np.asarray(input_rpd_mean, dtype=float).reshape(-1)
    T = np.asarray(table, dtype=int)
    if T.ndim != 2:
        raise ValueError("table must be 2D")
    if y.shape[0] != T.shape[0]:
        raise ValueError("input_rpd_mean and table must have same n_runs")
    if len(param_names) != T.shape[1]:
        raise ValueError("param_names length must match table n_params")

    means_by_level: list[tuple[float, ...]] = []
    levels_by_param: list[tuple[int, ...]] = []
    best_levels: list[int] = []

    for j in range(T.shape[1]):
        lvls = tuple(sorted(set(int(x) for x in T[:, j].tolist())))
        lv_means: list[float] = []
        for lvl in lvls:
            idx = np.where(T[:, j] == lvl)[0]
            lv_means.append(float(np.mean(y[idx])) if idx.size else float("nan"))
        levels_by_param.append(lvls)
        means_by_level.append(tuple(lv_means))
        best_levels.append(int(lvls[int(np.nanargmin(np.array(lv_means)))]))

    return MEPFMResult(
        param_names=tuple(param_names),
        levels=tuple(levels_by_param),
        means_by_level=tuple(means_by_level),
        best_levels=tuple(best_levels),
    )


def pick_uniform(items: Sequence[str], k: int) -> list[str]:
    """Pick k elements uniformly from items (top-to-bottom), inclusive of ends."""
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)
    if k == 1:
        return [items[0]]
    n = len(items)
    idxs = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    # de-dup while preserving order
    out: list[str] = []
    seen: set[int] = set()
    for ix in idxs:
        if ix not in seen:
            out.append(items[ix])
            seen.add(ix)
    # If rounding caused fewer than k, fill by scanning.
    if len(out) < k:
        for it in items:
            if it not in out:
                out.append(it)
            if len(out) == k:
                break
    return out

