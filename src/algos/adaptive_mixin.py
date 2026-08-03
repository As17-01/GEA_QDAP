import math
from typing import Callable, List, Tuple

from src.data.models import Individual


class AdaptiveRatesMixin:
    """Lambda-scaled operator rate control for crossover, mutation, and GEA scenario ops."""

    def _init_adaptive_rates(
        self,
        crossover_rate: float,
        mutation_rate: float,
        alpha: float,
        lambda_min: float,
        lambda_max: float,
        epsilon: float,
    ) -> None:
        self.base_crossover = crossover_rate
        self.base_mutation = mutation_rate
        self.lambda_crossover = 1.0
        self.lambda_mutation = 1.0
        self.alpha = alpha
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.epsilon = epsilon

    def _init_adaptive_scenario_rate(self, base_attr: str, lambda_attr: str, rate: float) -> None:
        setattr(self, base_attr, rate)
        setattr(self, lambda_attr, 1.0)

    def _update_lambda(self, lam: float, delta: float) -> float:
        new_val = lam + self.alpha * delta
        return max(self.lambda_min, min(self.lambda_max, new_val))

    def _accumulate_delta(self, pairs: List[Tuple[Individual, Individual]]) -> float:
        delta = 0.0
        for child, baseline in pairs:
            parent_cost = baseline.cost
            if math.isfinite(parent_cost) and math.isfinite(child.cost):
                delta += (parent_cost - child.cost) / (parent_cost + self.epsilon)
        return delta

    def _adaptive_apply(
        self,
        base_attr: str,
        lambda_attr: str,
        apply_fn: Callable[[int], List[Tuple[Individual, Individual]]],
    ) -> List[Individual]:
        base = getattr(self, base_attr)
        lam = getattr(self, lambda_attr)
        n = int(base * self.population_size * lam)
        pairs = apply_fn(n)
        setattr(self, lambda_attr, self._update_lambda(lam, self._accumulate_delta(pairs)))
        return [child for child, _ in pairs]

    def _adaptive_crossover_and_mutation(
        self,
        crossover_fn,
        mutate_fn,
    ) -> Tuple[List[Individual], List[Individual]]:
        offspring = self._adaptive_apply("base_crossover", "lambda_crossover", crossover_fn)
        mutations = self._adaptive_apply("base_mutation", "lambda_mutation", mutate_fn)
        return offspring, mutations

    def _adaptive_robust_chromosome_crossover(self, crossover_fn) -> List[Individual]:
        return self._adaptive_apply("base_rc", "lambda_rc", crossover_fn)

    def _adaptive_directed_mutation(self, mutate_fn) -> List[Individual]:
        return self._adaptive_apply("base_dm", "lambda_dm", mutate_fn)

    def _adaptive_gene_injection(self, inject_fn) -> List[Individual]:
        return self._adaptive_apply("base_gi", "lambda_gi", inject_fn)
