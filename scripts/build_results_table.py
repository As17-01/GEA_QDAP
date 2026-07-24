#!/usr/bin/env python3
"""
Build a self-contained HTML results table from per-algorithm JSON files.

Each algorithm run.py produces  results/<algo>.json  with this structure:
  [{"dataset": "T1", "results": {"AlgoClass": {mean, median, min, max, std}},
    "runtime": {...}, "hitting_time": {...}, "errors": 0}, ...]

Usage:
    python scripts/build_results_table.py
    python scripts/build_results_table.py --results-dir path/to/results --output out.html
"""

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils.labels import algo_label

# (ga._target_, display name) — stems match results/*.json via algo_label in conf/run/common.yaml
ALGOS = [
    ("src.algos.ga_standard.StandardGA",       "Standard GA"),
    ("src.algos.ga_sa.SimulatedAnnealing",    "SA"),
    ("src.algos.ga_pso.ParticleSwarm",        "PSO"),
    ("src.algos.ga_gea.GEA",                  "GEA"),
    ("src.algos.ga_gea_scenario_1.GEAScenario1", "GEA-S1"),
    ("src.algos.ga_gea_scenario_2.GEAScenario2", "GEA-S2"),
    ("src.algos.ga_gea_scenario_3.GEAScenario3", "GEA-S3"),
    ("src.algos.ga_hybrid_gapso.HybridGAPSO", "GA+PSO"),
    ("src.algos.ga_hybrid_gasa.HybridGASA",   "GA+SA"),
    ("src.algos.ga_adaptive.AdaptiveGEA",     "AdaptiveGEA"),
]
ALGO_ORDER = [algo_label(t) for t, _ in ALGOS]
ALGO_DISPLAY = {algo_label(t): name for t, name in ALGOS}

DATASET_ORDER = [
    "T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14",
    "c201535","c201555","c201575","c300695","c300775",
    "c302055","c302075","c302095","c351535","c351595",
]

# Datasets that have a known optimal (for OG computation)
KNOWN_OPTIMAL: dict[str, float] = {
    # "c201535": 12345,  # fill in once exact solutions are known
}


def _fmt(v: float | None, decimals: int = 0) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if decimals == 0:
        return f"{v:,.0f}"
    return f"{v:,.{decimals}f}"


def load_results(results_dir: Path) -> dict[str, dict[str, dict]]:
    """Return {algo_stem: {dataset: record}} for every non-tune JSON in results_dir."""
    data: dict[str, dict[str, dict]] = {}
    for p in sorted(results_dir.glob("*.json")):
        if "tune" in p.stem:
            continue
        try:
            records = json.loads(p.read_text())
        except Exception as exc:
            print(f"  Warning: could not parse {p.name}: {exc}")
            continue
        # Each record has one key inside "results" — the algo class name.
        # We index by file stem so all lookups are consistent.
        data[p.stem] = {r["dataset"]: r for r in records}
    return data


def _get_stats(record: dict, key: str) -> dict | None:
    """Extract the stats sub-dict from results/runtime/hitting_time blocks."""
    block = record.get(key, {})
    if not block:
        return None
    # The block is keyed by the algo class label — we just want the values.
    for v in block.values():
        return v
    return None


def build_html(data: dict[str, dict[str, dict]], output: Path) -> None:
    # Determine which algos and datasets are actually present
    present_algos = [a for a in ALGO_ORDER if a in data]
    present_datasets = [d for d in DATASET_ORDER
                        if any(d in data[a] for a in present_algos)]

    # ---------- pre-compute best-mean and best-min per dataset ----------
    best_mean: dict[str, float] = {}
    best_min:  dict[str, float] = {}
    for ds in present_datasets:
        means = [
            (_get_stats(data[a][ds], "results") or {}).get("mean")
            for a in present_algos if ds in data[a]
        ]
        mins = [
            (_get_stats(data[a][ds], "results") or {}).get("min")
            for a in present_algos if ds in data[a]
        ]
        means = [m for m in means if m is not None]
        mins  = [m for m in mins  if m is not None]
        best_mean[ds] = min(means) if means else float("inf")
        best_min[ds]  = min(mins)  if mins  else float("inf")

    # ---------- HTML generation ----------
    algo_headers = "".join(
        f'<th colspan="3">{ALGO_DISPLAY.get(a, a)}</th>'
        for a in present_algos
    )
    sub_headers = "".join(
        "<th>Mean ± Std</th><th>Best</th><th>Hit(s)</th>"
        for _ in present_algos
    )

    rows_html = ""
    for ds in present_datasets:
        cells = f'<td class="ds-name">{ds}</td>'
        for algo in present_algos:
            if ds not in data[algo]:
                cells += '<td colspan="3" class="missing">—</td>'
                continue
            rec = data[algo][ds]
            r   = _get_stats(rec, "results") or {}
            ht  = _get_stats(rec, "hitting_time") or {}
            errs = rec.get("errors", 0)

            mean_ = r.get("mean")
            std_  = r.get("std")
            min_  = r.get("min")
            hit_  = ht.get("mean")

            is_best_mean = mean_ is not None and abs(mean_ - best_mean[ds]) < 1e-3
            is_best_min  = min_  is not None and abs(min_  - best_min[ds])  < 1e-3

            mean_str = (
                f"{_fmt(mean_)} ± {_fmt(std_)}"
                if mean_ is not None and std_ is not None else "—"
            )
            min_str  = _fmt(min_) if min_ is not None else "—"
            hit_str  = _fmt(hit_, 1) if hit_ is not None else "—"

            err_badge = f' <span class="err">({errs}✗)</span>' if errs else ""

            cells += (
                f'<td class="{"best" if is_best_mean else ""}">{mean_str}{err_badge}</td>'
                f'<td class="{"best" if is_best_min else ""}">{min_str}</td>'
                f'<td class="hit">{hit_str}</td>'
            )
        rows_html += f"<tr>{cells}</tr>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GQAP Results Table</title>
<style>
  :root {{
    --bg: #ffffff; --fg: #1a1a2e; --border: #c8d0e0;
    --head-bg: #f0f2f7; --best-bg: #d4edda; --best-fg: #155724;
    --hit-fg: #6c757d; --miss-fg: #adb5bd; --err-fg: #c0392b;
    --ds-fg: #2c3e50; --stripe: #f8f9fc;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #12131a; --fg: #e2e8f0; --border: #2d3348;
             --head-bg: #1e2136; --best-bg: #1a3a2a; --best-fg: #6ee7a0;
             --hit-fg: #8899b0; --miss-fg: #4a5568; --err-fg: #fc8181;
             --ds-fg: #94a3b8; --stripe: #161824; }}
  }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg);
          color: var(--fg); padding: 24px; font-size: 13px; }}
  h1   {{ font-size: 1.2em; margin-bottom: 4px; }}
  p.sub {{ color: var(--hit-fg); margin: 0 0 16px; font-size: 0.85em; }}
  .wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; min-width: 100%; white-space: nowrap; }}
  th, td {{ padding: 5px 10px; border: 1px solid var(--border); text-align: right; }}
  th {{ background: var(--head-bg); font-weight: 600; text-align: center; }}
  td.ds-name {{ text-align: left; font-weight: 600; color: var(--ds-fg); position: sticky; left: 0; background: var(--bg); }}
  tr:nth-child(even) td {{ background: var(--stripe); }}
  tr:nth-child(even) td.ds-name {{ background: var(--stripe); }}
  td.best {{ background: var(--best-bg); color: var(--best-fg); font-weight: 700; }}
  tr:nth-child(even) td.best {{ background: var(--best-bg); }}
  td.hit  {{ color: var(--hit-fg); font-size: 0.9em; }}
  td.missing {{ color: var(--miss-fg); text-align: center; }}
  .err  {{ color: var(--err-fg); font-size: 0.8em; }}
  .legend {{ margin-top: 12px; font-size: 0.82em; color: var(--hit-fg); }}
  .legend span.b {{ display: inline-block; width: 12px; height: 12px;
                    background: var(--best-bg); border: 1px solid var(--border);
                    vertical-align: middle; margin-right: 4px; }}
</style>
</head>
<body>
<h1>GQAP Algorithm Comparison</h1>
<p class="sub">Mean ± Std and Best (Min) cost over all runs. Green = best value in row. Hit = avg wall-clock time (s) to first reach the best cost.</p>
<div class="wrap">
<table>
<thead>
  <tr>
    <th rowspan="2">Dataset</th>
    {algo_headers}
  </tr>
  <tr>{sub_headers}</tr>
</thead>
<tbody>
{rows_html}</tbody>
</table>
</div>
<div class="legend">
  <span class="b"></span> Best value in row &nbsp;|&nbsp; Hit(s) = avg hitting time &nbsp;|&nbsp; ✗ = failed runs
</div>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(f"Table written to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=SCRIPT_DIR / "results",
        help="Directory containing per-algorithm JSON files (default: scripts/results/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "results" / "summary_table.html",
        help="Output HTML file path",
    )
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir} ...")
    data = load_results(args.results_dir)
    if not data:
        print("No result JSON files found. Run the experiments first with run.py.")
        return

    print(f"Found {len(data)} algorithm(s): {', '.join(data)}")
    build_html(data, args.output)


if __name__ == "__main__":
    main()
