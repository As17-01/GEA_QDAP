import numpy as np

from src.algos.base import BaseGA
from src.costs import evaluate_permutation_delta_batch


class ParticleSwarm(BaseGA):
    """Kennedy & Eberhart (1995). Discrete PSO for integer-encoded assignment problems.

    The standard continuous PSO velocity equation is extended to integer positions:

        v[j] = w * v[j]
              + c1 * r1 * (pbest[j] - x[j])   (cognitive: pull toward personal best)
              + c2 * r2 * (gbest[j] - x[j])   (social: pull toward global best)

    where x[j] and pbest/gbest[j] are integer facility indices. Velocity is
    real-valued and clamped to [-v_max, v_max]. Position is updated by rounding
    x + v to the nearest integer and clamping to [0, I-1], then repaired for
    capacity feasibility.

    Particle identity (self.particles / self.personal_best / self.velocities) is
    tracked independently of self.population's ordering -- BaseGA.run() re-sorts
    self.population by cost after every step(), which would scramble a particle's
    identity if it were tracked by position in that list instead.
    """

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
        # Default v_max = number of facilities: the largest meaningful step is to
        # reassign every job to the opposite end of the facility index range.
        self.v_max = float(v_max) if v_max is not None else float(model.I)

        self.particles = []
        self.personal_best = []
        self.velocities = []

    def initialize_population(self) -> None:
        super().initialize_population()
        self.particles = list(self.population)
        self.personal_best = list(self.population)
        self.velocities = [np.zeros(self.model.J, dtype=float) for _ in range(self.population_size)]

    def step(self) -> None:
        gbest = self.best_solution
        I = self.model.I
        J = self.model.J

        new_perms = []
        for k in range(len(self.particles)):
            particle = self.particles[k]
            pbest = self.personal_best[k]
            vel = self.velocities[k]

            r1 = np.random.random(J)
            r2 = np.random.random(J)

            new_vel = (
                self.w * vel
                + self.c1 * r1 * (pbest.permutation.astype(float) - particle.permutation.astype(float))
                + self.c2 * r2 * (gbest.permutation.astype(float) - particle.permutation.astype(float))
            )
            new_vel = np.clip(new_vel, -self.v_max, self.v_max)
            self.velocities[k] = new_vel

            new_pos = np.round(particle.permutation.astype(float) + new_vel).astype(np.int64)
            new_pos = np.clip(new_pos, 0, I - 1)
            new_perms.append(new_pos)

        repaired = self.repair_batch_wrapper(np.array(new_perms))
        updated = evaluate_permutation_delta_batch(self.particles, repaired, self.model)

        for i, candidate in enumerate(updated):
            self.particles[i] = candidate
            if candidate.cost < self.personal_best[i].cost:
                self.personal_best[i] = candidate

        immigrants = self.maybe_generate_immigrants()
        if immigrants:
            replace_idx = np.random.choice(self.population_size, size=len(immigrants), replace=False)
            for idx, immigrant in zip(replace_idx, immigrants):
                self.particles[idx] = immigrant
                self.personal_best[idx] = immigrant
                self.velocities[idx] = np.zeros(self.model.J, dtype=float)

        self.population = list(self.particles)
