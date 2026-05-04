import time
from collections import defaultdict
from functools import wraps

from src.data.models import Individual


def clone_individual(individual: Individual) -> Individual:
    return Individual(
        permutation=individual.permutation.copy(),
        xij=individual.xij.copy(),
        cost=individual.cost,
        diversity=individual.diversity,
        cvar=individual.cvar.copy(),
    )


def timed(operation_name: str):
    """Decorator to measure execution time of operations."""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.perf_counter()
            result = func(self, *args, **kwargs)
            elapsed = time.perf_counter() - start

            if not hasattr(self, "_timing"):
                self._timing = defaultdict(float)
                self._timing_calls = defaultdict(int)

            self._timing[operation_name] += elapsed
            self._timing_calls[operation_name] += 1
            return result

        return wrapper

    return decorator
