#!/usr/bin/env python3
"""Tunes scripts/conf/components/common.yaml's repair/selection knobs (algorithm
parameters -- population_size, iterations, crossover_rate, mutation_rate -- are left as
configured in conf/tune.yaml's `ga` block, inherited from standard.yaml). Only StandardGA is
used as the evaluation algorithm.

Search is delegated to Optuna's TPE sampler instead of blind random sampling: each trial's
result feeds back into the sampler's model of which regions of conf/tune.yaml's param_space
tend to score well, so later trials concentrate there instead of sampling uniformly forever.

The baseline (current common.yaml values) is run once upfront; every trial's score is its
mean cost on each dataset relative to the baseline's mean cost on that dataset, averaged
across datasets -- a fixed reference avoids datasets with much larger absolute costs
dominating the ranking, and keeps the objective stable for Optuna (unlike scoring relative to
"best seen so far among trials", which would retroactively change past trials' scores).
Writes the winning candidate's values back into conf/components/common.yaml if it beats the
baseline, and dumps a JSON artifact with every trial's per-dataset stats, the winner, and the
(fixed) algorithm parameters used for the run.
"""

import json
import math
import re
import statistics
from pathlib import Path

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf

from comparison_utils import run_all_experiments, timestamp

SCRIPT_DIR = Path(__file__).resolve().parent

optuna.logging.set_verbosity(optuna.logging.WARNING)

OmegaConf.register_new_resolver("algo_label", lambda target: target.rsplit(".", 1)[-1].removesuffix("GA").lower(), replace=True)


def apply_components(base_ga_cfg: dict, components: dict) -> dict:
    ga_cfg = {**base_ga_cfg, "repair_class": dict(base_ga_cfg["repair_class"]), "selector": dict(base_ga_cfg["selector"])}
    for dotted_key, value in components.items():
        section, field = dotted_key.split(".", 1)
        ga_cfg[section][field] = value
    return ga_cfg


def effective_mean(dataset_entry: dict, runs: int) -> float:
    """calculate_statistics reports mean=0 when a dataset had zero successful runs (e.g.
    every run errored), which would look like a perfect score -- treat that as worst-case
    instead so a candidate can't "win" a dataset by failing on it."""
    if dataset_entry["errors"] >= runs:
        return math.inf
    return dataset_entry["mean"]


def evaluate(components: dict, base_ga_cfg: dict, datasets: list[str], runs: int, time_limit: float, workers: int) -> dict:
    ga_cfg = apply_components(base_ga_cfg, components)
    per_dataset_results = run_all_experiments(datasets, "Standard", ga_cfg, time_limit, runs, workers)
    return {
        res["dataset"]: {**res["results"]["Standard"], "runtime": res["runtime"]["Standard"], "errors": res["errors"]}
        for res in per_dataset_results
    }


def relative_score(per_dataset: dict, baseline_per_dataset: dict, datasets: list[str], runs: int) -> float:
    ratios = []
    for ds in datasets:
        baseline_mean = effective_mean(baseline_per_dataset[ds], runs)
        mean = effective_mean(per_dataset[ds], runs)
        if not math.isfinite(baseline_mean):
            continue
        ratios.append(mean / (baseline_mean + 1e-9) if math.isfinite(mean) else math.inf)
    return float(statistics.mean(ratios)) if ratios else math.inf


def update_common_yaml(path: Path, components: dict) -> None:
    text = path.read_text()
    for dotted_key, value in components.items():
        field = dotted_key.split(".", 1)[1]
        pattern = re.compile(rf"^(\s*{re.escape(field)}:\s*).*$", re.MULTILINE)
        text, n = pattern.subn(lambda m: f"{m.group(1)}{value}", text, count=1)
        if n == 0:
            raise ValueError(f"field {field!r} not found in {path}")
    path.write_text(text)


@hydra.main(version_base=None, config_path="conf", config_name="tune")
def main(cfg: DictConfig) -> None:
    datasets = list(cfg.datasets.names)
    time_limit = cfg.run.time_limit
    workers = int(cfg.run.workers)

    base_ga_cfg = OmegaConf.to_container(cfg.ga, resolve=True)
    algo_class = base_ga_cfg["_target_"].rsplit(".", 1)[-1]
    if algo_class != "StandardGA":
        raise SystemExit(f"tune_components.py only tunes StandardGA, got {algo_class} -- run with ga=standard's defaults intact")

    algorithm_params = {
        k: base_ga_cfg[k] for k in ("population_size", "iterations", "crossover_rate", "mutation_rate")
    }

    runs = cfg.tune.runs
    param_space = OmegaConf.to_container(cfg.tune.param_space, resolve=True)

    baseline_components = {key: base_ga_cfg[key.split(".", 1)[0]][key.split(".", 1)[1]] for key in param_space}

    print(f"[{timestamp()}] Algorithm params (fixed): {algorithm_params}")
    print(f"[{timestamp()}] Baseline: {baseline_components}")

    baseline_per_dataset = evaluate(baseline_components, base_ga_cfg, datasets, runs, time_limit, workers)
    candidate_records = [{"components": baseline_components, "per_dataset": baseline_per_dataset, "overall_score": 1.0}]

    print(f"[{timestamp()}] Tuning components with Optuna TPE: {cfg.tune.n_candidates - 1} trials x {len(datasets)} datasets x {runs} runs")

    def objective(trial: optuna.Trial) -> float:
        components = {key: trial.suggest_float(key, low, high) for key, (low, high) in param_space.items()}
        print(f"\n[{timestamp()}] Trial {trial.number + 1}/{cfg.tune.n_candidates - 1}: {components}")

        per_dataset = evaluate(components, base_ga_cfg, datasets, runs, time_limit, workers)
        score = relative_score(per_dataset, baseline_per_dataset, datasets, runs)
        candidate_records.append({"components": components, "per_dataset": per_dataset, "overall_score": score})
        return score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.tune.seed))
    study.optimize(objective, n_trials=max(0, cfg.tune.n_candidates - 1))

    best = min(candidate_records, key=lambda c: c["overall_score"])

    print(f"\n[{timestamp()}] Best candidate: {best['components']} (score={best['overall_score']:.4f})")

    if best["components"] is baseline_components:
        print(f"[{timestamp()}] Baseline remains best -- leaving common.yaml unchanged")
    else:
        components_path = SCRIPT_DIR / cfg.tune.components_path
        update_common_yaml(components_path, best["components"])
        print(f"[{timestamp()}] Wrote best candidate to {components_path}")

    artifact = {
        "timestamp": timestamp(),
        "algorithm": "Standard",
        "algorithm_params": algorithm_params,
        "datasets": datasets,
        "runs_per_dataset": runs,
        "candidates": candidate_records,
        "best": best,
    }

    out = SCRIPT_DIR / cfg.tune.output_file
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(artifact, f, indent=2)

    print(f"[{timestamp()}] Saved tuning artifact to {out}")


if __name__ == "__main__":
    main()
