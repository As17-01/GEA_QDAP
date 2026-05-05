import math

from src.algos.ga_core import BaseGA


class StandardGA(BaseGA):
    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.3,
    ):
        super().__init__(model, population_size, iterations)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def step(self) -> None:
        probs = self.compute_selection_probabilities()

        ncrossover = int(2 * round((self.crossover_rate * self.population_size) / 2))
        nmutation = int(math.floor(self.mutation_rate * self.population_size))

        offspring = self.crossover(probs, ncrossover)
        mutations = self.mutate(nmutation)

        pool = self.population + offspring + mutations
        self.select_from_pool(pool)
