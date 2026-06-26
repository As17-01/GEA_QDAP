#!/usr/bin/env python3

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from comparison_utils import print_dataset_results, run_dataset_tests, timestamp

SCRIPT_DIR = Path(__file__).resolve().parent


@hydra.main(version_base=None, config_path="conf", config_name="standard")
def main(cfg: DictConfig) -> None:
    datasets = list(cfg.datasets.names)
    algo_label = cfg.ga._target_.rsplit(".", 1)[-1]
    runs = cfg.run.runs
    time_limit = cfg.run.time_limit

    print(f"[{timestamp()}] Starting sequential testing")
    print(f"   Algorithm    : {algo_label}")
    print(f"   Datasets     : {len(datasets)}")
    print(f"   Runs per algo: {runs}")
    print(f"   GA params    : {cfg.ga}\n")

    all_results = []
    for i, ds in enumerate(datasets, 1):
        print(f"[{timestamp()}] Dataset {i}/{len(datasets)}: {ds}")

        res = run_dataset_tests(ds, algo_label, cfg.ga, time_limit, runs)
        all_results.append(res)
        print_dataset_results(res)

    out = SCRIPT_DIR / cfg.output_file
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[{timestamp()}] Finished! Results saved to {out}")


if __name__ == "__main__":
    main()
