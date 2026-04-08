from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Model:
    """Problem definition for the generalized quadratic assignment problem."""

    I: int
    J: int
    cij: np.ndarray  # shape: (I, J)
    aij: np.ndarray  # shape: (I, J)
    bi: np.ndarray  # shape: (I,)
    DIS: np.ndarray  # shape: (I, I)
    F: np.ndarray  # shape: (J, J)

    def __post_init__(self) -> None:
        if self.cij.shape != (self.I, self.J):
            raise ValueError("cij matrix shape mismatch")
        if self.aij.shape != (self.I, self.J):
            raise ValueError("aij matrix shape mismatch")
        if self.bi.shape != (self.I,):
            raise ValueError("bi vector length mismatch")
        if self.DIS.shape != (self.I, self.I):
            raise ValueError("DIS matrix shape mismatch")
        if self.F.shape != (self.J, self.J):
            raise ValueError("F matrix shape mismatch")


@dataclass
class Individual:
    """Representation of a single chromosome."""

    permutation: np.ndarray  # length J, values in [0, I)
    xij: np.ndarray  # shape (I, J), binary assignment
    cost: float
    cvar: np.ndarray  # shape (I,), capacity slack


@dataclass
class AlgorithmConfig:
    """
    Параметры GEA/GA по статье (Table 1): 1000 поколений, популяция 350,
    ограничение по времени 1000 с. Остановка — по достижении лимита итераций или времени.
    """
    iterations: int = 1000  # Maximum number of generations (paper: 1000)
    population_size: int = 350  # Paper: 350
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
    time_limit: Optional[float] = 1000.0  # Max runtime in seconds (paper: 1000 s)
    deduplicate: bool = False  # если True, в новую популяцию не попадают дубликаты; нехватка заполняется случайными


@dataclass
class AdaptiveAlgorithmConfig(AlgorithmConfig):
    """Extended config with adaptive parameters."""
    adaptive_epsilon: float = 1e-5
    adaptive_alpha: float = 0.01
    adaptive_lambda_min: float = 0.4
    adaptive_lambda_max: float = 1.5


@dataclass
class AlgorithmStats:
    contribution_rate: List[Tuple[float, float, float, float]] = field(default_factory=list)
    best_cost_trace: List[float] = field(default_factory=list)


@dataclass
class AdaptiveAlgorithmStats(AlgorithmStats):
    """Extended stats with adaptive lambda history."""
    lambda_history: List[Tuple[float, float, float, float, float]] = field(default_factory=list)
    delta_history: List[Tuple[float, float, float, float, float]] = field(default_factory=list)


@dataclass
class AlgorithmResult:
    best_cost: float
    best_individual: Individual
    population: List[Individual]
    stats: AlgorithmStats
    elapsed_time: float


@dataclass
class AdaptiveAlgorithmResult(AlgorithmResult):
    """Extended result with adaptive metrics."""
    adaptive_stats: AdaptiveAlgorithmStats

