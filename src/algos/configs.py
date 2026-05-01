from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.data.models import Individual

# =========================
# Algorithm configuration
# =========================


@dataclass
class AlgorithmConfig:
    iterations: int = 1000
    population_size: int = 350

    crossover_rate: float = 0.7
    mutation_rate: float = 0.3

    scenario_crossover_rate: float = 0.5
    scenario_mutation_rate: float = 0.2

    p_fixed_x: float = 0.9
    p_scenario1: float = 0.3
    p_scenario2: float = 0.3
    p_scenario3: float = 0.5

    mask_mutation_index: int = 2
    enable_scenario: Tuple[bool, bool, bool] = (True, True, True)

    time_limit: Optional[float] = 1000.0
    deduplicate: bool = False


@dataclass
class AdaptiveAlgorithmConfig(AlgorithmConfig):
    adaptive_epsilon: float = 1e-5
    adaptive_alpha: float = 0.01
    adaptive_lambda_min: float = 0.4
    adaptive_lambda_max: float = 1.5


# =========================
# Statistics
# =========================


@dataclass
class AlgorithmStats:
    contribution_rate: List[Tuple[float, float, float, float]] = field(default_factory=list)
    best_cost_trace: List[float] = field(default_factory=list)


@dataclass
class AdaptiveAlgorithmStats(AlgorithmStats):
    lambda_history: List[Tuple[float, float, float, float, float]] = field(default_factory=list)
    delta_history: List[Tuple[float, float, float, float, float]] = field(default_factory=list)


# =========================
# Results
# =========================


@dataclass
class AlgorithmResult:
    best_cost: float
    best_individual: Individual
    population: List[Individual]
    stats: AlgorithmStats
    elapsed_time: float


@dataclass
class AdaptiveAlgorithmResult(AlgorithmResult):
    adaptive_stats: AdaptiveAlgorithmStats
