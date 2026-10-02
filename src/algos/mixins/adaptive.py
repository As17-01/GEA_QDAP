import math
from typing import Callable, List, Tuple

import numpy as np

from src.costs import evaluate_permutation_delta_batch
from src.data.models import Individual
from src.operators.mutations import mutation_random

_LAMBDA_ATTRS = (
    "lambda_crossover",
    "lambda_mutation",
    "lambda_rc",
    "lambda_dm",
    "lambda_gi",
)


class AdaptiveRatesMixin:
    """ALGEA reward/punishment updates and survivor selection (paper, pp. 1–12)."""

    lambda_history: list[dict[str, float]]

    def _init_adaptive_rates(
        self,
        crossover_rate: float,
        mutation_rate: float,
        alpha: float,
        lambda_min: float,
        lambda_max: float,
        epsilon: float,
        *,
        gamma: float = 0.2,
        tournament_size: int = 3,
        attenuate_reward: bool = True,
    ) -> None:
        if not all(math.isfinite(value) for value in (alpha, lambda_min, lambda_max, epsilon, gamma)):
            raise ValueError("adaptive parameters must be finite")
        if alpha < 0 or not 0 <= lambda_min <= lambda_max <= 1 or epsilon <= 0:
            raise ValueError("require alpha >= 0, 0 <= lambda_min <= lambda_max <= 1, and epsilon > 0")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be a fraction between 0 and 1")
        if not isinstance(tournament_size, int) or isinstance(tournament_size, bool) or tournament_size < 1:
            raise ValueError("tournament_size must be a positive integer")
        if any(not math.isfinite(rate) or rate < 0 for rate in (crossover_rate, mutation_rate)):
            raise ValueError("operator rates must be finite and nonnegative")
        self.base_crossover = crossover_rate
        self.base_mutation = mutation_rate
        self.alpha = alpha
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.epsilon = epsilon
        self.gamma = gamma
        self.tournament_size = tournament_size
        self.attenuate_reward = attenuate_reward
        self.enforce_unique_population = True
        self.lambda_crossover = self._initial_lambda()
        self.lambda_mutation = self._initial_lambda()
        self.reset_adaptive_rates()

    def _initial_lambda(self) -> float:
        return self.lambda_min + (self.lambda_max - self.lambda_min) / 2

    def reset_adaptive_rates(self) -> None:
        """Start each run at the midpoint of the permitted range (Eq. 4)."""
        for name in _LAMBDA_ATTRS:
            if hasattr(self, name):
                setattr(self, name, self._initial_lambda())
        self.lambda_history = []

    def snapshot_lambdas(self) -> dict[str, float]:
        """Current adaptive λ values (only attributes present on this algorithm)."""
        return {name: float(getattr(self, name)) for name in _LAMBDA_ATTRS if hasattr(self, name)}

    def record_lambda_snapshot(self) -> None:
        """Append post-iteration λ values for plotting / analysis."""
        if not hasattr(self, "lambda_history"):
            self.lambda_history = []
        self.lambda_history.append(self.snapshot_lambdas())

    def _init_adaptive_scenario_rate(self, base_attr: str, lambda_attr: str, rate: float) -> None:
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("operator rates must be finite and nonnegative")
        setattr(self, base_attr, rate)
        setattr(self, lambda_attr, self._initial_lambda())

    def _update_lambda(self, lam: float, reward: float, punishment: float) -> float:
        """Apply Eq. 63; only the positive reward fades with generation progress."""
        magnitude = reward + punishment + self.epsilon
        remaining = max(0.0, min(1.0, 1.0 - self.progress)) if self.attenuate_reward else 1.0
        new_val = lam + self.alpha * (remaining * reward / magnitude - punishment / magnitude)
        return max(self.lambda_min, min(self.lambda_max, new_val))

    def _normalized_improvement(self, child: Individual, baseline: Individual) -> float:
        if not math.isfinite(baseline.cost) or not math.isfinite(child.cost):
            return 0.0
        return (baseline.cost - child.cost) / (abs(baseline.cost) + self.epsilon)

    def _operator_performance(self, pairs: List[Tuple[Individual, Individual]]) -> Tuple[float, float]:
        """Separate positive improvement and deterioration magnitudes (Eqs. 13–14)."""
        reward = 0.0
        punishment = 0.0
        for child, baseline in pairs:
            delta = self._normalized_improvement(child, baseline)
            reward += max(delta, 0.0)
            punishment += max(-delta, 0.0)
        return reward, punishment

    def _adaptive_apply(
        self,
        base_attr: str,
        lambda_attr: str,
        apply_fn: Callable[[int], List[Tuple[Individual, Individual]]],
    ) -> List[Tuple[Individual, Individual]]:
        base = getattr(self, base_attr)
        lam = getattr(self, lambda_attr)
        n = int(base * self.population_size * lam)
        pairs = apply_fn(n)
        reward, punishment = self._operator_performance(pairs)
        setattr(self, lambda_attr, self._update_lambda(lam, reward, punishment))
        return pairs

    def _adaptive_crossover_and_mutation(
        self,
        crossover_fn,
        mutate_fn,
    ) -> Tuple[List[Tuple[Individual, Individual]], List[Tuple[Individual, Individual]]]:
        offspring = self._adaptive_apply("base_crossover", "lambda_crossover", crossover_fn)
        mutations = self._adaptive_apply("base_mutation", "lambda_mutation", mutate_fn)
        return offspring, mutations

    def _adaptive_robust_chromosome_crossover(self, crossover_fn) -> List[Tuple[Individual, Individual]]:
        return self._adaptive_apply("base_rc", "lambda_rc", crossover_fn)

    def _adaptive_directed_mutation(self, mutate_fn) -> List[Tuple[Individual, Individual]]:
        return self._adaptive_apply("base_dm", "lambda_dm", mutate_fn)

    def _adaptive_gene_injection(self, inject_fn) -> List[Tuple[Individual, Individual]]:
        return self._adaptive_apply("base_gi", "lambda_gi", inject_fn)

    def _fill_adaptive_pool(self, candidates: dict[Individual, float]) -> None:
        """Supply unique feasible candidates if deduplication leaves fewer than Npop."""
        attempts = 0
        max_attempts = self.population_size * 100
        while len(candidates) < self.population_size and attempts < max_attempts:
            if not candidates:
                break
            n = min(self.population_size - len(candidates), max_attempts - attempts)
            sources = list(candidates)
            indices = np.random.randint(0, len(sources), size=n)
            baselines = [sources[idx] for idx in indices]
            raw_perms = np.array([mutation_random(ind.permutation, self.model) for ind in baselines])
            repaired = self.repair_batch_wrapper(raw_perms)
            children = evaluate_permutation_delta_batch(baselines, repaired, self.model)
            self.logger.record_nfe(n)
            attempts += n
            for child, baseline in zip(children, baselines):
                if math.isfinite(child.cost):
                    delta = self._normalized_improvement(child, baseline)
                    candidates[child] = max(candidates.get(child, -math.inf), delta)
        if len(candidates) < self.population_size:
            raise RuntimeError(
                f"adaptive selection could not fill a unique feasible population "
                f"({len(candidates)}/{self.population_size}) after {attempts} attempts"
            )

    def _select_adaptive_survivors(
        self,
        subpopulations: List[List[Tuple[Individual, Individual]]],
        extra_individuals: List[Individual] | None = None,
    ) -> None:
        """Paper p. 10: group elites, improvement-ranked fraction, then tournaments."""
        with self.logger.timed("survivor_selection"):
            candidates: dict[Individual, float] = {}
            elites: dict[Individual, None] = {}
            for pairs in subpopulations:
                feasible = [child for child, _ in pairs if math.isfinite(child.cost)]
                if feasible:
                    elites[min(feasible, key=lambda ind: ind.cost)] = None
                for child, baseline in pairs:
                    if math.isfinite(child.cost):
                        delta = self._normalized_improvement(child, baseline)
                        candidates[child] = max(candidates.get(child, -math.inf), delta)
            for individual in extra_individuals or []:
                if math.isfinite(individual.cost):
                    candidates.setdefault(individual, 0.0)

            self._fill_adaptive_pool(candidates)
            selected = sorted(elites, key=lambda ind: ind.cost)[: self.population_size]
            for individual in selected:
                candidates.pop(individual)

            n_improvement = min(int(self.gamma * len(candidates)), self.population_size - len(selected))
            ranked = sorted(candidates, key=lambda ind: (-candidates[ind], ind.cost))
            selected.extend(ranked[:n_improvement])
            remaining = ranked[n_improvement:]

            while len(selected) < self.population_size:
                indices = np.random.choice(
                    len(remaining), size=min(self.tournament_size, len(remaining)), replace=False
                )
                winner = min(indices, key=lambda idx: remaining[idx].cost)
                selected.append(remaining.pop(winner))
            self.population = sorted(selected, key=lambda ind: ind.cost)
