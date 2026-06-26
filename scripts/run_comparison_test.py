#!/usr/bin/env python3

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import hydra
from omegaconf import DictConfig

from src.algos.utils import seed_all
from src.data.model_loader import load_model

SCRIPT_DIR = Path(__file__).resolve().parent

# =========================
# Utils
# =========================


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0, "median": 0, "min": 0, "max": 0, "std": 0}
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


# =========================
# Single run functions
# =========================


def run_single_experiment(
    dataset_name: str,
    algo_label: str,
    run_num: int,
    ga_cfg: DictConfig,
    time_limit: float,
):
    seed = hash((dataset_name, algo_label, run_num)) & 0x7FFFFFFF
    seed_all(seed)

    try:
        model = load_model(dataset_name)
        ga = hydra.utils.instantiate(ga_cfg, model=model)
        best = ga.run(time_limit=time_limit)

        return (algo_label, run_num, best.cost, None)

    except Exception as e:
        return (algo_label, run_num, None, str(e))


# =========================
# Dataset runner (sequential)
# =========================


def run_dataset_tests(dataset_name: str, algo_label: str, ga_cfg: DictConfig, time_limit: float, runs: int):
    results = {algo_label: []}
    errors = 0

    print(f"   Running {runs} runs for {algo_label}...")

    for r in range(1, runs + 1):
        print(f"     → {algo_label} run {r}/{runs}", end=" ")

        _, _, cost, err = run_single_experiment(dataset_name, algo_label, r, ga_cfg, time_limit)

        if err:
            errors += 1
            print(f"[ERROR] {err}")
        else:
            results[algo_label].append(cost)
            print(f"→ cost = {cost:.2f}")

    stats = {k: calculate_statistics(v) for k, v in results.items()}

    return {
        "dataset": dataset_name,
        "results": stats,
        "errors": errors,
    }


# =========================
# Main
# =========================


@hydra.main(version_base=None, config_path="conf", config_name="standard")
def main(cfg: DictConfig) -> None:
    datasets = list(cfg.datasets.names)
    algo_label = cfg.ga._target_.rsplit(".", 1)[-1]
    runs = cfg.run.runs
    time_limit = cfg.run.time_limit

    all_results = []

    print(f"[{_ts()}] Starting sequential testing")
    print(f"   Algorithm    : {algo_label}")
    print(f"   Datasets     : {len(datasets)}")
    print(f"   Runs per algo: {runs}")
    print(f"   GA params    : {cfg.ga}\n")

    for i, ds in enumerate(datasets, 1):
        print(f"[{_ts()}] Dataset {i}/{len(datasets)}: {ds}")

        res = run_dataset_tests(ds, algo_label, cfg.ga, time_limit, runs)
        all_results.append(res)

        print(f"   Results:")
        for algo, stats in res["results"].items():
            print(
                f"     {algo:10s} | mean={stats['mean']:8.2f} | " f"min={stats['min']:8.2f} | std={stats['std']:6.2f}"
            )

    # Save results (resolved relative to this script, independent of the invocation cwd)
    out = SCRIPT_DIR / cfg.output_file
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[{_ts()}] Finished! Results saved to {out}")


if __name__ == "__main__":
    main()
