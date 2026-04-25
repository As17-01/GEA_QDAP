import math
import time
from typing import List, Sequence, Tuple

import numpy as np

from gea_gqap_adaptive_python.ga_core import (
    build_pool,
    compute_contribution,
    compute_selection_probabilities,
    initialize_population,
    update_best,
)
from gea_gqap_adaptive_python.masking import analyze_perm, combine_q, mask_mutation, roulette_wheel_selection
from gea_gqap_adaptive_python.models import AlgorithmConfig, AlgorithmResult, AlgorithmStats, Individual, Model
from gea_gqap_adaptive_python.operators.crossover import choose_crossover
from gea_gqap_adaptive_python.operators.mutations import choose_mutation
from gea_gqap_adaptive_python.utils import evaluate_permutation

DEFAULT_INSTRUCTION = (True, True, True)


def _ensure_instruction_tuple(instruction: Sequence[bool] | None) -> Tuple[bool, bool, bool]:
    if instruction is None:
        return DEFAULT_INSTRUCTION
    if len(instruction) != 3:
        raise ValueError("instruction must contain exactly three boolean flags")
    return tuple(bool(x) for x in instruction)


def run_ga(
    model: Model,
    config: AlgorithmConfig | None = None,
    instruction: Sequence[bool] | None = None,
) -> AlgorithmResult:
    cfg = config or AlgorithmConfig()
    instruction_tuple = _ensure_instruction_tuple(instruction or cfg.enable_scenario)
    rng = np.random.default_rng()

    start_time = time.perf_counter()

    ncrossover = int(2 * round((cfg.crossover_rate * cfg.population_size) / 2))
    nmutation = int(math.floor(cfg.mutation_rate * cfg.population_size))

    ncrossover_scenario = int(math.floor(cfg.scenario_crossover_rate * (cfg.p_scenario3 * cfg.population_size)))
    nmutate_scenario = int(math.floor(cfg.scenario_mutation_rate * (cfg.p_scenario3 * cfg.population_size)))

    population = initialize_population(model, cfg.population_size, rng)
    best_solution = population[0]
    worst_cost = population[-1].cost
    beta = 10.0

    stats = AlgorithmStats()

    for _ in range(cfg.iterations):
        probabilities = compute_selection_probabilities(population, beta, worst_cost)

        offspring: List[Individual] = []
        crossover_origins: List[str] = []

        for _ in range(0, ncrossover, 2):
            i1 = roulette_wheel_selection(probabilities, rng)
            i2 = roulette_wheel_selection(probabilities, rng)
            parents = (population[i1], population[i2])

            child_perm1, child_perm2 = choose_crossover(parents, rng)

            for perm in (child_perm1, child_perm2):
                child = evaluate_permutation(perm, model)
                if math.isfinite(child.cost):
                    offspring.append(child)
                    crossover_origins.append("crossover")

        mutations: List[Individual] = []
        mutation_origins: List[str] = []

        for _ in range(nmutation):
            idx = rng.integers(0, len(population))
            perm = choose_mutation(population[idx].permutation, model, rng)
            ind = evaluate_permutation(perm, model)
            if math.isfinite(ind.cost):
                mutations.append(ind)
                mutation_origins.append("mutation")

        scenario_candidates: List[Individual] = []
        scenario_origins: List[str] = []

        if any(instruction_tuple):
            n_pop = len(population)

            p1 = min(max(1, int(cfg.p_scenario1 * cfg.population_size)), n_pop)
            p2 = min(max(1, int(cfg.p_scenario2 * cfg.population_size)), n_pop)
            p3 = min(max(1, int(cfg.p_scenario3 * cfg.population_size)), n_pop)

            if instruction_tuple[0] and p1 >= 2 and ncrossover_scenario > 0:
                _, _, dominant, _ = analyze_perm(population[:p1], cfg, model, rng)
                for _ in range(ncrossover_scenario):
                    idx = roulette_wheel_selection(probabilities, rng)
                    child_perm1, child_perm2 = choose_crossover((dominant, population[idx]), rng)
                    for perm in (child_perm1, child_perm2):
                        child = evaluate_permutation(perm, model)
                        if math.isfinite(child.cost):
                            scenario_candidates.append(child)
                            scenario_origins.append("scenario")

            if instruction_tuple[1] and p2 >= 1 and nmutate_scenario > 0:
                _, mask_matrix, _, _ = analyze_perm(population[:p2], cfg, model, rng)
                mask_slice = mask_matrix[:p2]
                for _ in range(nmutate_scenario):
                    ii = int(rng.integers(0, p2))
                    perm = mask_mutation(
                        cfg.mask_mutation_index,
                        population[ii].permutation,
                        mask_slice[ii],
                        model,
                        rng,
                    )
                    child = evaluate_permutation(perm, model)
                    if math.isfinite(child.cost):
                        scenario_candidates.append(child)
                        scenario_origins.append("scenario")

            if instruction_tuple[2] and p3 >= 1 and nmutate_scenario > 0:
                _, _, dominant, dominant_mask = analyze_perm(population[:p3], cfg, model, rng)
                tail_indices = np.arange(max(0, n_pop - p3), n_pop)
                for _ in range(nmutate_scenario):
                    jj = int(rng.choice(tail_indices))
                    perm = combine_q(
                        dominant.permutation,
                        population[jj].permutation,
                        dominant_mask,
                    )
                    child = evaluate_permutation(perm, model)
                    if math.isfinite(child.cost):
                        scenario_candidates.append(child)
                        scenario_origins.append("scenario")

        pool = build_pool(
            population,
            offspring,
            mutations,
            scenario_candidates,
            crossover_origins,
            mutation_origins,
            scenario_origins,
        )

        population = [ind for ind, _ in pool[: cfg.population_size]]
        top_origins = [o for _, o in pool[: cfg.population_size]]

        stats.contribution_rate.append(compute_contribution(top_origins))

        population.sort(key=lambda ind: ind.cost)
        worst_cost = max(worst_cost, population[-1].cost)
        best_solution = update_best(population, best_solution)

        stats.best_cost_trace.append(best_solution.cost)

        if cfg.time_limit and (time.perf_counter() - start_time) >= cfg.time_limit:
            break

    return AlgorithmResult(
        best_cost=best_solution.cost,
        best_individual=best_solution,
        population=population,
        stats=stats,
        elapsed_time=time.perf_counter() - start_time,
    )
