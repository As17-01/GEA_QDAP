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
        repair_class=None,
        selector=None,
    ):
        super().__init__(model, population_size, iterations, repair_class=repair_class, selector=selector)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def step(self) -> None:
        probs = self.selector.compute_selection_probabilities(self.population)

        ncrossover = int(2 * round((self.crossover_rate * self.population_size) / 2))
        nmutation = int(math.floor(self.mutation_rate * self.population_size))

        offspring = self.crossover(probs, ncrossover)
        mutations = self.mutate(nmutation)

        pool = self.population + offspring + mutations
        self.select_from_pool(pool)
