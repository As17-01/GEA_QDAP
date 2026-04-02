"""Merge base `algorithm` dict with optional per-model overrides from test_config.json."""
from __future__ import annotations

from typing import Any, Dict


def merged_algorithm_for_model(
    base: Dict[str, Any],
    by_model: Dict[str, Dict[str, Any]] | None,
    model_key: str,
) -> Dict[str, Any]:
    out = dict(base or {})
    if by_model and model_key in by_model:
        out.update(by_model[model_key])
    return out
