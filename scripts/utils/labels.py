def algo_label(target: str) -> str:
    """Derive results filename stem from ga._target_ (e.g. StandardGA -> standard)."""
    return target.rsplit(".", 1)[-1].removesuffix("GA").lower()
