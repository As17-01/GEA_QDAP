from dataclasses import dataclass

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
    bi: np.ndarray  # (I,)
    DIS: np.ndarray  # (I, I)
    F: np.ndarray  # (J, J)

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
    permutation: np.ndarray  # (J,)
    xij: np.ndarray  # (I, J)
    cost: float
    hidden_cost: float
    cvar: np.ndarray  # (I,)
