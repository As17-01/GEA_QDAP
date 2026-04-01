#!/usr/bin/env python3
"""
Debug runner for parameter tuning using RPD + Taguchi tables.

What it does (small-scale, for debugging):
- Picks 5 `c*` datasets uniformly (top-to-bottom) + 2 simplest `T*` datasets.
- For a chosen Excel sheet + block ("adaptive" by default), iterates over Taguchi runs.
- For each Taguchi row, runs the algorithm 5 times per dataset, collects best_cost.
- Builds a response matrix: rows=Taguchi runs, cols=datasets (mean best_cost over 5 runs).
- Applies RPD normalization (per-dataset column), then averages RPD across datasets.
- Runs MEPFM to pick best levels per parameter.

Outputs:
- JSON with raw costs + RPD + chosen best levels in `gea_gqap_adaptive_python/tests/RPD/results/`
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# keep single-threaded BLAS for repeatability / laptop safety
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent.parent.parent
sys_path_added = False


def _ensure_sys_path() -> None:
    global sys_path_added
    if sys_path_added:
        return
    import sys

    sys.path.insert(0, str(REPO_ROOT / "gea_gqap_adaptive_python"))
    sys_path_added = True


def _worker_process(task: Tuple[str, int, Dict, int, float, str]) -> Tuple[str, int, float]:
    """
    Multiprocessing worker (must be top-level to be picklable).
    """
    ds, run_number, cfg_kwargs, iterations, time_limit, repo_root_str = task
    from pathlib import Path
    import sys

    repo_root = Path(repo_root_str)
    sys.path.insert(0, str(repo_root / "gea_gqap_adaptive_python"))

    from gea_gqap_adaptive_python import AdaptiveAlgorithmConfig, load_model, run_adaptive_ga

    model = load_model(ds)
    cfg = AdaptiveAlgorithmConfig(
        iterations=int(iterations),
        time_limit=float(time_limit),
        **cfg_kwargs,
    )
    res = run_adaptive_ga(model, config=cfg)
    return ds, int(run_number), float(res.best_cost)


def _pick_datasets() -> List[str]:
    _ensure_sys_path()
    from gea_gqap_adaptive_python import list_available_models, load_model
    try:
        from .rpd_utils import pick_uniform  # type: ignore
    except Exception:  # pragma: no cover
        import sys

        sys.path.insert(0, str(TEST_DIR.parent))
        from rpd_utils import pick_uniform  # type: ignore

    names = list_available_models()
    c_names = sorted([n for n in names if n.startswith("c")])
    t_names = sorted([n for n in names if n.startswith("T")])

    # 5 c datasets uniformly from top to bottom
    c_sel = pick_uniform(c_names, 5)

    # 2 simplest T datasets: smallest I*J
    t_info: List[Tuple[str, int]] = []
    for n in t_names:
        m = load_model(n)
        t_info.append((n, int(m.I) * int(m.J)))
    t_info.sort(key=lambda x: (x[1], x[0]))
    t_sel = [t_info[0][0], t_info[1][0]]
    return c_sel + t_sel


def _map_param(name: str) -> str | None:
    """
    Map Excel parameter names to config fields.
    Unknown parameters are ignored (returned as None).
    """
    return {
        "Npop": "population_size",
        "PCrossover": "crossover_rate",
        "PMutation": "mutation_rate",
        "PScenario1": "p_scenario1",
        "PScenario2": "p_scenario2",
        "PScenario3": "p_scenario3",
        "PFixedX": "p_fixed_x",
        "PCrossoverScenario1": "scenario_crossover_rate",
        "PMutationScenario2": "scenario_mutation_rate",
        # adaptive params
        "alpha": "adaptive_alpha",
        "lambda_min": "adaptive_lambda_min",
        "lambda_max": "adaptive_lambda_max",
    }.get(name)


def _build_config_kwargs(param_levels: Dict[str, Tuple[float, float, float]], levels_row: List[int]) -> Dict:
    # Keep stable param order as appears in Excel (dict preserves insertion in py3.7+)
    names = list(param_levels.keys())
    if len(names) != len(levels_row):
        raise ValueError(f"Table columns != params: {len(levels_row)} vs {len(names)}")
    kwargs: Dict = {}
    for p_name, lvl in zip(names, levels_row):
        field = _map_param(p_name)
        if field is None:
            continue
        if lvl not in (1, 2, 3):
            raise ValueError(f"Unexpected level {lvl} for {p_name}")
        val = float(param_levels[p_name][lvl - 1])
        if field in ("population_size",):
            kwargs[field] = int(round(val))
        elif field in ("mask_mutation_index",):
            kwargs[field] = int(round(val))
        else:
            kwargs[field] = float(val)
    return kwargs


def main() -> None:
    _ensure_sys_path()
    from gea_gqap_adaptive_python import AdaptiveAlgorithmConfig, load_model, run_adaptive_ga
    try:
        from .excel_taguchi_parser import parse_sheet_blocks  # type: ignore
        from .rpd_utils import mepfm, rpd_matlab  # type: ignore
    except Exception:  # pragma: no cover
        import sys

        sys.path.insert(0, str(TEST_DIR.parent))
        from excel_taguchi_parser import parse_sheet_blocks  # type: ignore
        from rpd_utils import mepfm, rpd_matlab  # type: ignore

    from concurrent.futures import ProcessPoolExecutor, as_completed

    excel_path = (REPO_ROOT / "Nikita_AL-GEA.xlsx").resolve()
    sheet_name = os.environ.get("RPD_SHEET", "GEA_2")
    block_name = os.environ.get("RPD_BLOCK", "adaptive")  # "base" or "adaptive"

    # Debug-time defaults (override via env):
    iterations = int(os.environ.get("RPD_ITERATIONS", "60"))
    time_limit = float(os.environ.get("RPD_TIME_LIMIT", "3.0"))
    num_runs = int(os.environ.get("RPD_NUM_RUNS", "5"))
    max_rows = int(os.environ.get("RPD_MAX_TAGUCHI_ROWS", "0"))  # 0 => all
    num_workers = int(os.environ.get("RPD_NUM_WORKERS", os.environ.get("NUM_WORKERS", "16")))

    datasets = _pick_datasets()
    parsed = parse_sheet_blocks(excel_path, sheet_name)
    block = parsed.blocks[block_name]

    param_names = list(block.param_levels.keys())
    table = np.array(block.table, dtype=int)
    if max_rows and max_rows < table.shape[0]:
        table = table[:max_rows, :]

    response = np.zeros((table.shape[0], len(datasets)), dtype=float)
    raw_costs: List[List[List[float]]] = [
        [[0.0 for _ in range(num_runs)] for _ in datasets] for _ in range(table.shape[0])
    ]

    for run_idx in range(table.shape[0]):
        row_levels = table[run_idx, :].tolist()
        cfg_kwargs = _build_config_kwargs(block.param_levels, row_levels)

        tasks: List[Tuple[str, int, Dict, int, float, str]] = []
        for ds in datasets:
            for r in range(num_runs):
                tasks.append((ds, r, cfg_kwargs, iterations, time_limit, str(REPO_ROOT)))

        ds_to_idx = {ds: i for i, ds in enumerate(datasets)}
        costs_acc: List[List[float]] = [[float("nan")] * num_runs for _ in datasets]

        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_worker_process, t) for t in tasks]
            for fut in as_completed(futures):
                ds, r, cost = fut.result()
                di = ds_to_idx[ds]
                costs_acc[di][r] = cost

        for ds_i, costs in enumerate(costs_acc):
            if any(not np.isfinite(c) for c in costs):
                raise RuntimeError(f"Missing/invalid costs for dataset {datasets[ds_i]} at taguchi row {run_idx}")
            raw_costs[run_idx][ds_i] = costs
            response[run_idx, ds_i] = float(np.mean(costs))

    # RPD normalization per dataset (column-wise, like MATLAB flag=1)
    rpd = rpd_matlab(response, flag=1)
    mean_rpd = rpd.mean(axis=1)

    mep = mepfm(mean_rpd, table, param_names=param_names)

    out_dir = TEST_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rpd_tuning_{sheet_name}_{block_name}.json"

    payload = {
        "excel": {"path": str(excel_path), "sheet": sheet_name, "block": block_name},
        "datasets": datasets,
        "debug_overrides": {"iterations": iterations, "time_limit": time_limit, "num_runs": num_runs},
        "param_names": param_names,
        "param_levels": {k: list(v) for k, v in block.param_levels.items()},
        "taguchi_table": block.table,
        "response_mean_best_cost": response.tolist(),
        "raw_best_costs": raw_costs,
        "rpd": rpd.tolist(),
        "mean_rpd": mean_rpd.tolist(),
        "mepfm": {
            "best_levels": list(mep.best_levels),
            "levels": [list(x) for x in mep.levels],
            "means_by_level": [list(map(float, x)) for x in mep.means_by_level],
        },
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path}")
    print("Best levels:", mep.best_levels)


if __name__ == "__main__":
    main()

