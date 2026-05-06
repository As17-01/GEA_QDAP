#!/usr/bin/env python3

import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
# Single run functions
# =========================


def run_single_experiment(
    dataset_name: str,
    algo_type: str,
    run_num: int,
    iterations: int,
    pop_size: int,
    crossover_rate: float,
    mutation_rate: float,
):

    seed = hash((dataset_name, algo_type, run_num)) & 0x7FFFFFFF
    np.random.seed(seed)

    try:
        model = load_model(dataset_name)

        if algo_type == "adaptive":
            ga = AdaptiveGA(
                model,
                population_size=pop_size,
                iterations=iterations,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
            )
            best = ga.run()
        else:
            ga = StandardGA(
                model,
                population_size=pop_size,
                iterations=iterations,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
            )
            best = ga.run()

        return (algo_type, run_num, best.cost, None)

    except Exception as e:
        return (algo_type, run_num, None, str(e))


# =========================
# Dataset runner (sequential)
# =========================


def run_dataset_tests(
    dataset_name: str, iterations: int, pop_size: int, crossover_rate: float, mutation_rate: float, runs: int
):
    results = {"standard": [], "adaptive": []}
    errors = 0

    print(f"   Running {runs} runs for standard...")

    for algo_type in ["standard"]:
        for r in range(1, runs + 1):
            print(f"     → {algo_type} run {r}/{runs}", end=" ")

            algo_type, run_num, cost, err = run_single_experiment(
                dataset_name, algo_type, r, iterations, pop_size, crossover_rate, mutation_rate
            )

            if err:
                errors += 1
                print(f"[ERROR] {err}")
            else:
                results[algo_type].append(cost)
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


def main():
    datasets = list_available_models()

    ITERATIONS = 1000
    POP_SIZE = 350
    CROSSOVER_RATE = 0.7
    MUTATION_RATE = 0.3
    RUNS = 1

    all_results = []

    print(f"[{_ts()}] Starting sequential testing")
    print(f"   Datasets     : {len(datasets)}")
    print(f"   Runs per algo: {RUNS}")
    print(f"   Iterations   : {ITERATIONS}")
    print(f"   Population   : {POP_SIZE}")
    print(f"   Crossover rt : {CROSSOVER_RATE}")
    print(f"   Mutation rt  : {MUTATION_RATE}\n")

    for i, ds in enumerate(datasets, 1):
        print(f"[{_ts()}] Dataset {i}/{len(datasets)}: {ds}")

        res = run_dataset_tests(ds, ITERATIONS, POP_SIZE, CROSSOVER_RATE, MUTATION_RATE, RUNS)
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
