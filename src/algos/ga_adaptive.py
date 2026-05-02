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
    ):
        super().__init__(model, population_size, iterations)

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

        # Generate offspring from crossover
        offspring = []
        for child in self.crossover(probs, ncrossover):
            parent_cost = self.best_solution.cost
            delta = (parent_cost - child.cost) / (parent_cost + self.epsilon)
            crossover_delta += delta
            offspring.append(child)

        # Generate mutations
        mutations = []
        for child in self.mutate(nmutation):
            parent_cost = self.best_solution.cost
            delta = (parent_cost - child.cost) / (parent_cost + self.epsilon)
            mutation_delta += delta
            mutations.append(child)

        # Update adaptive parameters
        self.lambda_crossover = self._update_lambda(self.lambda_crossover, crossover_delta)
        self.lambda_mutation = self._update_lambda(self.lambda_mutation, mutation_delta)

        # Combine pool and remove duplicates
        pool = self.population + offspring + mutations

        # Remove duplicate individuals (using __eq__ and __hash__ from Individual)
        unique_pool = list(dict.fromkeys(pool))  # Preserves first occurrence order

        # Sort by cost and keep top individuals
        unique_pool.sort(key=lambda x: x.cost)
        self.population = unique_pool[: self.population_size]
