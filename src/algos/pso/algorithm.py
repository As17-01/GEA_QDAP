from src.algos.core.base import AlgorithmBase
from src.algos.core.improved_base import ImprovedBase
from src.algos.mixins.pso import PSOMixin
from src.repair import GreedyRepair


class StandardParticleSwarm(AlgorithmBase, PSOMixin):
    """Discrete PSO with greedy repair and no stagnation immigrants."""

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        inertia_weight=0.4,
        cognitive_weight=0.3,
        social_weight=0.3,
        v_max=None,
        repair_class=None,
        verbose=False,
    ):
        super().__init__(
            model,
            population_size,
            iterations,
            repair_class=repair_class if repair_class is not None else GreedyRepair(),
            verbose=verbose,
        )
        self.w = inertia_weight
        self.c1 = cognitive_weight
        self.c2 = social_weight
        self.v_max = float(v_max) if v_max is not None else float(model.I)
        self._init_pso_tracking(use_velocities=True)

    def initialize_population(self) -> None:
        super().initialize_population()
        self._sync_pso_from_population()

    def step(self) -> None:
        self._velocity_pso_step(
            inertia_weight=self.w,
            cognitive_weight=self.c1,
            social_weight=self.c2,
            v_max=self.v_max,
        )
        self._commit_particles_to_population()


class ImprovedParticleSwarm(ImprovedBase, PSOMixin):
    """Discrete PSO with RF repair, diversity tracking, and stagnation immigrants."""

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        inertia_weight=0.4,
        cognitive_weight=0.3,
        social_weight=0.3,
        v_max=None,
        repair_class=None,
        selector=None,
        stagnation_limit=30,
        immigrant_rate=0.1,
        verbose=False,
    ):
        super().__init__(
            model,
            population_size,
            iterations,
            repair_class=repair_class,
            selector=selector,
            stagnation_limit=stagnation_limit,
            immigrant_rate=immigrant_rate,
            verbose=verbose,
        )
        self.w = inertia_weight
        self.c1 = cognitive_weight
        self.c2 = social_weight
        self.v_max = float(v_max) if v_max is not None else float(model.I)
        self._init_pso_tracking(use_velocities=True)

    def initialize_population(self) -> None:
        super().initialize_population()
        self._sync_pso_from_population()

    def step(self) -> None:
        self._velocity_pso_step(
            inertia_weight=self.w,
            cognitive_weight=self.c1,
            social_weight=self.c2,
            v_max=self.v_max,
        )
        self._inject_pso_immigrants(self.maybe_generate_immigrants())
        self._commit_particles_to_population()


# Backward-compatible alias for unmigrated imports.
ParticleSwarm = ImprovedParticleSwarm
