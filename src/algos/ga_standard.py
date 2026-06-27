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
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def step(self) -> None:
        probs = self.compute_selection_probabilities()

        ncrossover = int(2 * round((self.crossover_rate * self.population_size) / 2))
        nmutation = int(math.floor(self.mutation_rate * self.population_size))

        offspring = [child for child, _ in self.crossover(probs, ncrossover)]
        mutations = [child for child, _ in self.mutate(nmutation)]
        immigrants = self.maybe_generate_immigrants()

        pool = self.population + offspring + mutations + immigrants
        self.select_from_pool(pool)
