#!/usr/bin/env python3
"""
Запуск run_rpd_tuning_debug.py последовательно для каждого листа Taguchi в taguchi_config_RPD.json.

По умолчанию: листы GEA_1, GEA_2, GEA_3, блок adaptive (как в Excel). Для каждого листа
пишется отдельный JSON: results/rpd_tuning_<sheet>_<block>.json

Переопределение окружения (iterations, time_limit, …) — те же переменные, что у
run_rpd_tuning_debug.py (RPD_ITERATIONS, RPD_TIME_LIMIT, RPD_NUM_RUNS, …).

Пример (полноценный тюнинг на кластере):
  export RPD_ITERATIONS=1000 RPD_TIME_LIMIT=1000 RPD_NUM_RUNS=5
  python3 run_rpd_tune_all_sheets.py

Дальше перенос коэффициентов в тест: `export_recommended_to_test_config.py` и `README_RPD.md`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent


def _sheet_names_from_config() -> list[str]:
    cfg_path = Path(os.environ.get("RPD_TAGUCHI_CONFIG", str(TEST_DIR / "taguchi_config_RPD.json"))).resolve()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return sorted(data.get("sheets", {}).keys())


def main() -> None:
    raw = os.environ.get("RPD_SHEETS", "").strip()
    if raw:
        sheets = [s.strip() for s in raw.split(",") if s.strip()]
    else:
        sheets = _sheet_names_from_config()

    block = os.environ.get("RPD_BLOCK", "adaptive")
    print(f"Sheets: {sheets}, block={block}", flush=True)

    sys.path.insert(0, str(TEST_DIR))
    os.environ["RPD_BLOCK"] = block

    import run_rpd_tuning_debug as rpd  # noqa: E402

    for sheet in sheets:
        os.environ["RPD_SHEET"] = sheet
        print(f"\n{'='*60}\n>>> RPD sheet={sheet} block={block}\n{'='*60}", flush=True)
        rpd.main()


if __name__ == "__main__":
    main()
