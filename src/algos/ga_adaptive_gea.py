from src.algos.adaptive_mixin import AdaptiveRatesMixin
from src.algos.base import BaseGA


class AdaptiveGEA(BaseGA, AdaptiveRatesMixin):
    """Full GEA (crossover, mutation, RC, DM, GI) with lambda-scaled operator rates."""

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
        alpha=0.01,
        lambda_min=0.4,
        lambda_max=1.5,
        epsilon=1e-5,
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
        self._init_adaptive_rates(crossover_rate, mutation_rate, alpha, lambda_min, lambda_max, epsilon)
        self._init_adaptive_scenario_rate("base_rc", "lambda_rc", rc_rate)
        self._init_adaptive_scenario_rate("base_dm", "lambda_dm", dm_rate)
        self._init_adaptive_scenario_rate("base_gi", "lambda_gi", injection_rate)

    def step(self) -> None:
        probs = self.compute_selection_probabilities()
        offspring, mutations = self._adaptive_crossover_and_mutation(probs)
        rc_offspring = self._adaptive_robust_chromosome_crossover(probs)
        dm_mutations = self._adaptive_directed_mutation()
        gi_injections = self._adaptive_gene_injection()
        immigrants = self.maybe_generate_immigrants()

        pool = (
            self.population + offspring + mutations + rc_offspring
            + dm_mutations + gi_injections + immigrants
        )
        self.select_from_pool(pool)
