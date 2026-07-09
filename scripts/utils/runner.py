import os

# Must be set before numba/numpy are imported anywhere in this process -- numba reads
# NUMBA_NUM_THREADS once at its own first import and refuses to change it afterward, and
# BLAS backends (used by numpy under the hood) similarly lock in their thread count at
# first use. Worker processes (forked or spawned) inherit these from the parent's
# environment at process-creation time, so setting them here -- before the `from src...`
# imports below pull numba/numpy in -- is what makes them take effect in every worker.
# Each worker does its own numeric work single-threaded; the parallelism comes entirely
# from the process pool, so N workers don't also each spawn a thread pool across every
# core (oversubscription). run_test.sbatch sets the same four BLAS variables at the shell
# level already; setdefault here just makes ad-hoc local runs (outside sbatch) safe too,
# without overriding whatever the caller already set.
for _env_var in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List

import hydra
from omegaconf import OmegaConf

from src.data.model_loader import load_model
from src.seeding import seed_all

# Lets conf/run/common.yaml derive output_file from ga._target_ (e.g.
# "...ga_adaptive.AdaptiveGEA" -> "adaptive") instead of each config repeating its own output
# filename. Registered here (imported by every entry-point script before its @hydra.main
# composes a config) rather than in each script, so it's done exactly once.
OmegaConf.register_new_resolver("algo_label", lambda target: target.rsplit(".", 1)[-1].removesuffix("GA").lower(), replace=True)


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
    ga_cfg: dict,
    time_limit: float,
):
    seed = hash((dataset_name, algo_label, run_num)) & 0x7FFFFFFF
    seed_all(seed)

    start = time.perf_counter()
    try:
        model = load_model(dataset_name)
        ga = hydra.utils.instantiate(ga_cfg, model=model)
        best = ga.run(time_limit=time_limit)
        hitting_time = getattr(ga, "hitting_time", None)

        return (dataset_name, run_num, best.cost, time.perf_counter() - start, hitting_time, None)

    except Exception as e:
        return (dataset_name, run_num, None, time.perf_counter() - start, None, str(e))


def run_all_experiments(
    datasets: List[str],
    algo_label: str,
    ga_cfg: dict,
    time_limit: float,
    runs: int,
    workers: int,
) -> List[dict]:
    """Runs every (dataset, run) combination across a process pool and returns one stats
    dict per dataset, in the same shape run_dataset_tests used to produce. Each run is an
    independent GA call (fresh model load, own seed), so this parallelizes flatly across
    every (dataset, run) pair rather than nesting a pool per dataset.
    """
    tasks = [(ds, r) for ds in datasets for r in range(1, runs + 1)]
    results_by_dataset: Dict[str, List[float]] = {ds: [] for ds in datasets}
    runtimes_by_dataset: Dict[str, List[float]] = {ds: [] for ds in datasets}
    hitting_times_by_dataset: Dict[str, List[float]] = {ds: [] for ds in datasets}
    errors_by_dataset: Dict[str, int] = {ds: 0 for ds in datasets}
    completed_by_dataset: Dict[str, int] = {ds: 0 for ds in datasets}

    print(f"   Running {len(tasks)} experiments ({len(datasets)} datasets x {runs} runs) across {workers} workers...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_single_experiment, ds, algo_label, r, ga_cfg, time_limit): (ds, r) for ds, r in tasks
        }

        for future in as_completed(futures):
            ds, _ = futures[future]
            _, _, cost, runtime, hitting_time, err = future.result()

            runtimes_by_dataset[ds].append(runtime)
            if err:
                errors_by_dataset[ds] += 1
            else:
                results_by_dataset[ds].append(cost)
                if hitting_time is not None:
                    hitting_times_by_dataset[ds].append(hitting_time)
            completed_by_dataset[ds] += 1

            # Print one aggregated line per dataset, once all its runs are in, instead of
            # one line per individual run -- with many datasets/runs/workers running at
            # once, per-run lines from concurrent futures interleave into unreadable noise.
            if completed_by_dataset[ds] == runs:
                stats = calculate_statistics(results_by_dataset[ds])
                total_runtime = sum(runtimes_by_dataset[ds])
                avg_hit = statistics.mean(hitting_times_by_dataset[ds]) if hitting_times_by_dataset[ds] else None
                hit_note = f" ; avg hitting time = {avg_hit:.1f}s" if avg_hit is not None else ""
                error_note = f" ({errors_by_dataset[ds]} errors)" if errors_by_dataset[ds] else ""
                print(
                    f"   {ds} run → avg cost = {stats['mean']:.2f} ; std = {stats['std']:.2f} ; "
                    f"total runtime = {total_runtime:.2f}s{hit_note}{error_note}"
                )

    return [
        {
            "dataset": ds,
            "results": {algo_label: calculate_statistics(results_by_dataset[ds])},
            "runtime": {algo_label: {**calculate_statistics(runtimes_by_dataset[ds]), "total": sum(runtimes_by_dataset[ds])}},
            "hitting_time": {
                algo_label: {
                    "mean": float(statistics.mean(hitting_times_by_dataset[ds])) if hitting_times_by_dataset[ds] else None,
                    "min": float(min(hitting_times_by_dataset[ds])) if hitting_times_by_dataset[ds] else None,
                    "max": float(max(hitting_times_by_dataset[ds])) if hitting_times_by_dataset[ds] else None,
                    "std": float(statistics.stdev(hitting_times_by_dataset[ds])) if len(hitting_times_by_dataset[ds]) > 1 else 0.0,
                }
            },
            "errors": errors_by_dataset[ds],
        }
        for ds in datasets
    ]
