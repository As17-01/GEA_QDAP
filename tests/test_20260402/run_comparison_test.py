#!/usr/bin/env python3

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent.parent))

from src.algos.ga_adaptive import AdaptiveGA
from src.algos.ga_standard import StandardGA
from src.data.model_loader import list_available_models, load_model

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 16))


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
# Single runs
# =========================


def _run_standard(model, iterations, pop_size):
    ga = StandardGA(model, population_size=pop_size, iterations=iterations)
    best = ga.run()
    return best.cost


def _run_adaptive(model, iterations, pop_size):
    ga = AdaptiveGA(model, population_size=pop_size, iterations=iterations)
    best = ga.run()
    return best.cost


def _worker(task):
    dataset_name, algo_type, run_num, iterations, pop_size = task

    seed = hash((dataset_name, algo_type, run_num)) & 0x7FFFFFFF
    np.random.seed(seed)

    try:
        model = load_model(dataset_name)

        if algo_type == "adaptive":
            cost = _run_adaptive(model, iterations, pop_size)
        else:
            cost = _run_standard(model, iterations, pop_size)

        return (algo_type, run_num, cost, None)

    except Exception as e:
        return (algo_type, run_num, None, str(e))


# =========================
# Dataset runner
# =========================


def run_dataset_tests(dataset_name: str, iterations: int, pop_size: int, runs: int):
    tasks = []
    for algo_type in ["standard", "adaptive"]:
        for r in range(1, runs + 1):
            tasks.append((dataset_name, algo_type, r, iterations, pop_size))

    results = {"standard": [], "adaptive": []}
    errors = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(_worker, t) for t in tasks]

        for f in as_completed(futures):
            algo_type, run_num, cost, err = f.result()

            if err:
                errors += 1
                print(f"[ERR] {dataset_name} {algo_type} run {run_num}: {err}")
            else:
                results[algo_type].append(cost)

    stats = {}
    for k, v in results.items():
        stats[k] = calculate_statistics(v)

    return {
        "dataset": dataset_name,
        "results": stats,
        "errors": errors,
    }


# =========================
# Main
# =========================


def main():
    datasets = list_available_models()

    ITERATIONS = 1000
    POP_SIZE = 350
    RUNS = 30

    all_results = []

    print(f"[{_ts()}] Start testing")

    for i, ds in enumerate(datasets, 1):
        print(f"\n[{_ts()}] Dataset {i}/{len(datasets)}: {ds}")

        res = run_dataset_tests(ds, ITERATIONS, POP_SIZE, RUNS)
        all_results.append(res)

        for algo, stats in res["results"].items():
            print(f"{algo:10s} | mean={stats['mean']:.1f} | min={stats['min']:.1f} | std={stats['std']:.1f}")

    out = Path("results.json")
    with out.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[{_ts()}] Done → {out}")


if __name__ == "__main__":
    main()
