#!/usr/bin/env python3

import json
import os
import statistics
import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent.parent))

from src.algos.ga_adaptive import AdaptiveGA
from src.algos.ga_standard import StandardGA
from src.data.model_loader import list_available_models, load_model

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
# Single run
# =========================


def run_single_experiment(
    dataset_name: str,
    algo_type: str,
    run_num: int,
    iterations: int,
    pop_size: int,
) -> Tuple[str, int, float, str]:

    # Deterministic seed per run
    seed = hash((dataset_name, algo_type, run_num)) & 0x7FFFFFFF
    np.random.seed(seed)

    try:
        model = load_model(dataset_name)

        if algo_type == "adaptive":
            ga = AdaptiveGA(model, population_size=pop_size, iterations=iterations)
        else:
            ga = StandardGA(model, population_size=pop_size, iterations=iterations)

        best = ga.run()

        return (algo_type, run_num, best.cost, None)

    except Exception as e:
        return (algo_type, run_num, None, str(e))


def run_single_experiment_wrapper(args):
    return run_single_experiment(*args)


# =========================
# Dataset runner (parallel)
# =========================


def run_dataset_tests(dataset_name: str, iterations: int, pop_size: int, runs: int):
    results = {"standard": [], "adaptive": []}
    errors = 0

    print(f"   Running {runs} runs for standard + adaptive (parallel)...")

    # Build all tasks
    tasks = []
    for algo_type in ["standard", "adaptive"]:
        for r in range(1, runs + 1):
            tasks.append((dataset_name, algo_type, r, iterations, pop_size))

    # Parallel execution
    n_proc = max(1, cpu_count() - 1)
    print(f"   Using {n_proc} processes")

    with Pool(processes=n_proc) as pool:
        outputs = pool.map(run_single_experiment_wrapper, tasks)

    # Collect results
    for algo_type, run_num, cost, err in outputs:
        if err:
            errors += 1
            print(f"     → {algo_type} run {run_num} [ERROR] {err}")
        else:
            results[algo_type].append(cost)
            print(f"     → {algo_type} run {run_num} → cost = {cost:.2f}")

    stats = {k: calculate_statistics(v) for k, v in results.items()}

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

    print(f"[{_ts()}] Starting PARALLEL testing")
    print(f"   Datasets     : {len(datasets)}")
    print(f"   Runs per algo: {RUNS}")
    print(f"   Iterations   : {ITERATIONS}")
    print(f"   Population   : {POP_SIZE}\n")

    for i, ds in enumerate(datasets, 1):
        print(f"[{_ts()}] Dataset {i}/{len(datasets)}: {ds}")

        res = run_dataset_tests(ds, ITERATIONS, POP_SIZE, RUNS)
        all_results.append(res)

        print(f"   Results:")
        for algo, stats in res["results"].items():
            print(
                f"     {algo:10s} | mean={stats['mean']:8.2f} | " f"min={stats['min']:8.2f} | std={stats['std']:6.2f}"
            )

    # Save results
    out = Path("results.json")
    with out.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[{_ts()}] Finished! Results saved to {out}")


if __name__ == "__main__":
    main()
