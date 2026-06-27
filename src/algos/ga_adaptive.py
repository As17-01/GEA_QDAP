import math

from src.algos.ga_core import BaseGA


class AdaptiveGA(BaseGA):
    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.3,
        alpha=0.01,
        lambda_min=0.4,
        lambda_max=1.5,
        epsilon=1e-5,
        repair_class=None,
        selector=None,
        stagnation_limit=30,
        immigrant_rate=0.1,
    ):
        super().__init__(
            model,
            population_size,
            iterations,
            repair_class=repair_class,
            selector=selector,
            stagnation_limit=stagnation_limit,
            immigrant_rate=immigrant_rate,
        )

        self.base_crossover = crossover_rate
        self.base_mutation = mutation_rate

        self.lambda_crossover = 1.0
        self.lambda_mutation = 1.0

        self.alpha = alpha
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.epsilon = epsilon

    def _update_lambda(self, lam, delta):
        new_val = lam + self.alpha * delta
        return max(self.lambda_min, min(self.lambda_max, new_val))

    def step(self) -> None:
        probs = self.compute_selection_probabilities()

        ncrossover = int(self.base_crossover * self.population_size * self.lambda_crossover)
        nmutation = int(self.base_mutation * self.population_size * self.lambda_mutation)

        crossover_delta = 0.0
        mutation_delta = 0.0

        # Generate offspring from crossover. Delta is measured against each child's own
        # parent baseline (not the global best) -- comparing every child to the best-ever
        # solution makes delta almost always negative once the population is decent, which
        # drove both lambdas toward lambda_min exactly as the run converged, choking off the
        # exploration needed to escape a local optimum.
        offspring = []
        for child, baseline in self.crossover(probs, ncrossover):
            parent_cost = baseline.cost
            if math.isfinite(parent_cost) and math.isfinite(child.cost):
                crossover_delta += (parent_cost - child.cost) / (parent_cost + self.epsilon)
            offspring.append(child)

        # Generate mutations
        mutations = []
        for child, baseline in self.mutate(nmutation):
            parent_cost = baseline.cost
            if math.isfinite(parent_cost) and math.isfinite(child.cost):
                mutation_delta += (parent_cost - child.cost) / (parent_cost + self.epsilon)
            mutations.append(child)

        # Update adaptive parameters
        self.lambda_crossover = self._update_lambda(self.lambda_crossover, crossover_delta)
        self.lambda_mutation = self._update_lambda(self.lambda_mutation, mutation_delta)

        immigrants = self.maybe_generate_immigrants()

        pool = self.population + offspring + mutations + immigrants
        self.select_from_pool(pool)
