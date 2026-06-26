from typing import List

import numpy as np

from src.data.models import Individual


def get_diversity(population_base: List[Individual], population_to_eval: List[Individual]) -> np.ndarray:
    base_perms = np.array([ind.permutation for ind in population_base], dtype=np.int32)  # (N_base, J)
    eval_perms = np.array([ind.permutation for ind in population_to_eval], dtype=np.int32)  # (N_eval, J)

    J = base_perms.shape[1]

    diff = eval_perms[:, None, :] != base_perms[None, :, :]
    hamming = diff.sum(axis=2)
    diversities = hamming.mean(axis=1) / J

    return diversities.astype(np.float32)


# Parent selection (diversity-weighted roulette wheel) and survivor selection
# (elite + diversity/cost hybrid, decaying toward pure cost over the run).
class DiversitySelector:
    def __init__(
        self,
        beta: float = 10.0,
        diversity_weight_start: float = 0.7,
        diversity_weight_end: float = 0.2,
    ):
        self.beta = beta
        self.diversity_weight_start = diversity_weight_start
        self.diversity_weight_end = diversity_weight_end
        self.avg_diversity: float = 0.0

    def compute_selection_probabilities(self, population: List[Individual]) -> np.ndarray:
        diversity_scores = get_diversity(population_base=population, population_to_eval=population)

        probs = np.exp(self.beta * diversity_scores)
        probs = probs / probs.sum()

        self.avg_diversity = np.mean(diversity_scores)
        return probs

    def roulette_wheel_selection(self, probabilities: np.ndarray) -> int:
        return int(np.searchsorted(np.cumsum(probabilities), np.random.random(), side="right"))

    def select_from_pool(self, pool: List[Individual], population_size: int, progress: float) -> List[Individual]:
        unique_pool = list(dict.fromkeys(pool))
        n = population_size

        # Sort by cost (lower is better)
        unique_pool.sort(key=lambda x: x.cost)

        n_elite = n // 2
        elite = unique_pool[:n_elite]
        remaining = unique_pool[n_elite:]

        diversity_array = get_diversity(population_base=elite, population_to_eval=remaining)

        # Normalize costs so lower cost -> higher score
        costs = np.array([ind.cost for ind in remaining], dtype=float)

        if len(costs) > 1 and costs.max() != costs.min():
            cost_scores = 1.0 - (costs - costs.min()) / (costs.max() - costs.min())
        else:
            cost_scores = np.ones_like(costs)

        # Decay diversity pressure over the run so the population can settle
        # onto the best cost found instead of being pinned apart by diversity.
        diversity_weight = (
            self.diversity_weight_start - (self.diversity_weight_start - self.diversity_weight_end) * progress
        )
        cost_weight = 1.0 - diversity_weight

        combined_scores = diversity_weight * diversity_array + cost_weight * cost_scores

        scored_remaining = list(zip(remaining, combined_scores))
        scored_remaining.sort(key=lambda x: x[1], reverse=True)

        n_diverse = n - n_elite
        diverse_selected = [ind for ind, _ in scored_remaining[:n_diverse]]

        return elite + diverse_selected
