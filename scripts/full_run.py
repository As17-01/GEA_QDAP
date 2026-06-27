#!/usr/bin/env python3

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from comparison_utils import run_all_experiments, timestamp

SCRIPT_DIR = Path(__file__).resolve().parent

# Lets conf/run/common.yaml derive output_file from ga._target_ (e.g.
# "...ga_adaptive.AdaptiveGA" -> "adaptive") instead of each config repeating its own
# output filename.
OmegaConf.register_new_resolver("algo_label", lambda target: target.rsplit(".", 1)[-1].removesuffix("GA").lower())


@hydra.main(version_base=None, config_path="conf", config_name="standard")
def main(cfg: DictConfig) -> None:
    datasets = list(cfg.datasets.names)
    algo_label = cfg.ga._target_.rsplit(".", 1)[-1]
    runs = cfg.run.runs
    time_limit = cfg.run.time_limit
    workers = int(cfg.run.workers)
    ga_cfg = OmegaConf.to_container(cfg.ga, resolve=True)

    print(f"[{timestamp()}] Starting parallel testing")
    print(f"   Algorithm    : {algo_label}")
    print(f"   Datasets     : {len(datasets)}")
    print(f"   Runs per algo: {runs}")
    print(f"   Workers      : {workers}")
    print(f"   GA params    : {cfg.ga}\n")

    all_results = run_all_experiments(datasets, algo_label, ga_cfg, time_limit, runs, workers)

    out = SCRIPT_DIR / cfg.run.output_file
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[{timestamp()}] Finished! Results saved to {out}")


if __name__ == "__main__":
    main()
