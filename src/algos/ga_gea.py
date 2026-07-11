import math

from src.algos.base import BaseGA


class GEA(BaseGA):
    """This project's own enhanced GA: diversity-aware selection, randomized capacity
    repair, stagnation-triggered immigrants, and memetic local search (all via BaseGA),
    running **five operator stages per generation** in fixed sequence:

    1. Crossover       — standard crossover pool via choose_crossover (4 operators).
    2. Mutation        — standard mutation pool via choose_mutation (9 operators).
    3. RC crossover    — Robust Chromosome: per gene inherits from the parent with more
                         remaining capacity slack (GEAScenario1's enhancement).
    4. DM              — Directed Mutation: moves each individual's single worst-assigned
                         job to its cheapest feasible facility (GEAScenario2's enhancement).
    5. GI              — Gene Injection: replaces a handful of random genes with fresh
                         random facility assignments (GEAScenario3's enhancement).

    All five offspring pools are merged with the current population; DiversitySelector
    picks the next generation. GEAScenario1/2/3 each add only one of stages 3/4/5,
    making them single-enhancement ablations of this full variant.
    """

    def __init__(
        self,
        model,
        population_size=350,
        iterations=1000,
        crossover_rate=0.7,
        mutation_rate=0.3,
        rc_rate=0.3,
        dm_rate=0.3,
        injection_rate=0.1,
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
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.rc_rate = rc_rate
        self.dm_rate = dm_rate
        self.injection_rate = injection_rate

    def step(self) -> None:
        probs = self.compute_selection_probabilities()

        ncrossover = int(2 * round((self.crossover_rate * self.population_size) / 2))
        nmutation = int(math.floor(self.mutation_rate * self.population_size))
        n_rc = int(math.floor(self.rc_rate * self.population_size))
        n_dm = int(math.floor(self.dm_rate * self.population_size))
        n_gi = int(math.floor(self.injection_rate * self.population_size))

        offspring = [child for child, _ in self.crossover(probs, ncrossover)]    # stage 1
        mutations = [child for child, _ in self.mutate(nmutation)]               # stage 2
        rc_offspring = self._robust_chromosome_crossover(probs, n_rc)            # stage 3
        dm_mutations = self._directed_mutation(n_dm)                             # stage 4
        gi_injections = self._gene_injection(n_gi)                               # stage 5
        immigrants = self.maybe_generate_immigrants()

        pool = self.population + offspring + mutations + rc_offspring + dm_mutations + gi_injections + immigrants
        self.select_from_pool(pool)
