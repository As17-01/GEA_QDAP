import statistics
from datetime import datetime
from typing import Dict, List

import hydra
from omegaconf import DictConfig

from src.data.model_loader import load_model
from src.seeding import seed_all


def timestamp() -> str:
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


def run_dataset_tests(dataset_name: str, algo_label: str, ga_cfg: DictConfig, time_limit: float, runs: int) -> dict:
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


def print_dataset_results(res: dict) -> None:
    print("   Results:")
    for algo, stats in res["results"].items():
        print(f"     {algo:10s} | mean={stats['mean']:8.2f} | min={stats['min']:8.2f} | std={stats['std']:6.2f}")
