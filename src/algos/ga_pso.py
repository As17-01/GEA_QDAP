import numpy as np

from src.algos.base import BaseGA
from src.costs import evaluate_permutation_delta_batch
from src.operators.crossover import choose_crossover
from src.operators.mutations import choose_mutation


class ParticleSwarm(BaseGA):
    """Kennedy & Eberhart (1995). There's no continuous position/velocity here -- following
    the common discrete-PSO convention for permutation-encoded problems, each particle's
    pull toward its personal best and the swarm's global best is realized via crossover with
    those references, and the inertia term via a random mutation move. inertia/cognitive/
    social weights act as the probabilities of each move type (normalized to sum to 1)
    rather than continuous velocity coefficients.

    Particle identity (each particle's own position and personal best) is tracked in
    self.particles/self.personal_best, independent of self.population's ordering --
    BaseGA.run() re-sorts self.population by cost after every step(), which would scramble
    a particle's identity if it were tracked by position in that list instead.
    """

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        inertia_weight=0.4,
        cognitive_weight=0.3,
        social_weight=0.3,
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

        total_weight = inertia_weight + cognitive_weight + social_weight
        self.inertia_weight = inertia_weight / total_weight
        self.cognitive_weight = cognitive_weight / total_weight
        self.social_weight = social_weight / total_weight

        self.particles = []
        self.personal_best = []

    def initialize_population(self) -> None:
        super().initialize_population()
        self.particles = list(self.population)
        self.personal_best = list(self.population)

    def step(self) -> None:
        gbest = self.best_solution

        new_perms = []
        for position, pbest in zip(self.particles, self.personal_best):
            r = np.random.random()
            if r < self.inertia_weight:
                candidate = choose_mutation(position.permutation, self.model)
            elif r < self.inertia_weight + self.cognitive_weight:
                (child1, _), _ = choose_crossover((position, pbest), self.model)
                candidate = child1
            else:
                (child1, _), _ = choose_crossover((position, gbest), self.model)
                candidate = child1
            new_perms.append(candidate)

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

        self.population = list(self.particles)
