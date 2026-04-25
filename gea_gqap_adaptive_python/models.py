from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# =========================
# Core problem definition
# =========================

@dataclass(frozen=True)
class Model:
    I: int
    J: int

    cij: np.ndarray  # (I, J)
    aij: np.ndarray  # (I, J)
    bi: np.ndarray   # (I,)
    DIS: np.ndarray  # (I, I)
    F: np.ndarray    # (J, J)

    def __post_init__(self) -> None:
        expected_shapes = {
            "cij": (self.I, self.J),
            "aij": (self.I, self.J),
            "bi": (self.I,),
            "DIS": (self.I, self.I),
            "F": (self.J, self.J),
        }

        actual_arrays = {
            "cij": self.cij,
            "aij": self.aij,
            "bi": self.bi,
            "DIS": self.DIS,
            "F": self.F,
        }

        for name, expected_shape in expected_shapes.items():
            if actual_arrays[name].shape != expected_shape:
                raise ValueError(f"{name} shape mismatch: expected {expected_shape}, got {actual_arrays[name].shape}")


# =========================
# Genetic representation
# =========================

@dataclass
class Individual:
    permutation: np.ndarray      # (J,)
    xij: np.ndarray              # (I, J)
    cost: float
    cvar: np.ndarray             # (I,)


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
