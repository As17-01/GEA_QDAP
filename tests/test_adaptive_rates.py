import math

import hydra
import numpy as np
import pytest

from src.algos.adaptive.ga import ImprovedAdaptiveGA, StandardAdaptiveGA
from src.algos.adaptive.gea import ImprovedAdaptiveGEA, StandardAdaptiveGEA
from src.algos.core import improved_base, standard_base
from src.algos.mixins.adaptive import AdaptiveRatesMixin
from src.costs import evaluate_permutation
from src.data.model_loader import load_model
from src.data.models import Individual, Model
from src.repair import GreedyRepair, IdentityRepair
from src.seeding import seed_all
from tests.conftest import ALL_ALGORITHM_CONFIGS, load_ga_config

ADAPTIVE_CONFIGS = [name for name in ALL_ALGORITHM_CONFIGS if "adaptive" in name]


def individual(cost, chromosome=0):
    return Individual(np.array([chromosome]), cost, np.array([0.0]))


def adaptive_controller(**overrides):
    controller = AdaptiveRatesMixin()
    params = dict(crossover_rate=0.7, mutation_rate=0.3, alpha=0.05, lambda_min=0.1, lambda_max=0.9, epsilon=1e-12)
    params.update(overrides)
    controller._init_adaptive_rates(**params)
    controller.progress = 0.0
    return controller


def test_reward_and_punishment_examples_from_paper_pages_4_and_5():
    controller = adaptive_controller()
    baseline = individual(100.0)
    for costs, expected_reward, expected_punishment, expected_signal in [
        ([98, 96, 99, 101, 100.5], 0.07, 0.015, 0.6470588235294118),
        ([99, 102, 101, 99.5], 0.015, 0.03, -1 / 3),
    ]:
        pairs = [(individual(cost), baseline) for cost in costs]
        reward, punishment = controller._operator_performance(pairs)
        assert reward == pytest.approx(expected_reward)
        assert punishment == pytest.approx(expected_punishment)
        assert controller._update_lambda(0.5, reward, punishment) == pytest.approx(0.5 + 0.05 * expected_signal)


def test_early_and_late_update_examples_from_paper_pages_8_and_9():
    controller = adaptive_controller()
    rewarded = controller._update_lambda(0.5, 0.75, 0.25)
    assert rewarded == pytest.approx(0.525)
    assert controller._update_lambda(rewarded, 0.30, 0.70) == pytest.approx(0.505)

    controller.progress = 0.9
    assert controller._update_lambda(0.5, 0.8, 0.2) == pytest.approx(0.494)
    controller.progress = 1.0
    assert controller._update_lambda(0.5, 1.0, 0.0) == 0.5
    assert controller._update_lambda(0.5, 0.0, 1.0) == pytest.approx(0.45)


def test_optional_nonattenuated_variant_still_rewards_at_final_generation():
    controller = adaptive_controller(attenuate_reward=False)
    controller.progress = 1.0
    assert controller._update_lambda(0.5, 0.75, 0.25) == pytest.approx(0.525)


def test_abs_parent_cost_preserves_improvement_sign_for_negative_objectives():
    controller = adaptive_controller()
    assert controller._normalized_improvement(individual(-110), individual(-100)) == pytest.approx(0.1)
    assert controller._normalized_improvement(individual(-90), individual(-100)) == pytest.approx(-0.1)
    assert math.isfinite(controller._normalized_improvement(individual(-1), individual(0)))


def test_empty_unchanged_and_nonfinite_batches_are_neutral():
    controller = adaptive_controller()
    same = individual(100)
    pairs = [(same, same), (individual(math.inf), same), (same, individual(math.nan))]
    assert controller._operator_performance([]) == (0.0, 0.0)
    assert controller._operator_performance(pairs) == (0.0, 0.0)
    assert controller._update_lambda(0.5, 0.0, 0.0) == 0.5


def test_bounds_and_normalization_limit_updates_regardless_of_batch_size():
    controller = adaptive_controller()
    assert controller._update_lambda(0.89, 1.0, 0.0) == 0.9
    assert controller._update_lambda(0.11, 0.0, 1.0) == 0.1
    first = controller._update_lambda(0.5, 0.07, 0.015)
    larger = controller._update_lambda(0.5, 7.0, 1.5)
    assert larger == pytest.approx(first)
    assert abs(larger - 0.5) <= controller.alpha


def test_full_probability_range_clips_rewards_and_punishments_to_endpoints():
    controller = adaptive_controller(lambda_min=0.0, lambda_max=1.0, alpha=1.0)
    assert controller.lambda_crossover == controller.lambda_mutation == 0.5
    assert controller._update_lambda(0.9, 1.0, 0.0) == 1.0
    assert controller._update_lambda(0.1, 0.0, 1.0) == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"epsilon": 0},
        {"alpha": -0.1},
        {"lambda_min": 0.95},
        {"lambda_min": -0.01},
        {"lambda_max": 1.01},
        {"lambda_min": 1.1, "lambda_max": 1.2},
        {"lambda_max": math.inf},
        {"gamma": 20},
        {"tournament_size": 0},
        {"tournament_size": 1.5},
        {"mutation_rate": -0.1},
    ],
)
def test_invalid_adaptive_parameters_are_rejected(overrides):
    with pytest.raises(ValueError):
        adaptive_controller(**overrides)


@pytest.fixture
def small_model():
    return Model(
        I=2,
        J=4,
        cij=np.array([[1.0] * 4, [2.0] * 4]),
        aij=np.ones((2, 4)),
        bi=np.full(2, 10.0),
        DIS=np.zeros((2, 2)),
        F=np.zeros((4, 4)),
    )


@pytest.mark.parametrize("robust", [False, True])
def test_improved_crossovers_use_best_parent_for_learning_and_original_parent_for_costs(
    small_model, monkeypatch, robust
):
    ga = ImprovedAdaptiveGEA(small_model, population_size=2, repair_class=IdentityRepair())
    cheap = evaluate_permutation(np.array([0, 0, 0, 0]), small_model)
    expensive = evaluate_permutation(np.array([1, 1, 1, 1]), small_model)
    ga.population = [cheap, expensive]
    ga.best_solution = cheap
    monkeypatch.setattr(ga.selector, "roulette_wheel_selection_batch", lambda probabilities, size: np.array([0, 1]))
    child_perm = np.array([0, 1, 1, 1])
    children = ((child_perm, expensive), (child_perm, cheap))
    if robust:
        monkeypatch.setattr(improved_base, "crossover_robust_chromosome", lambda p1, p2, model: children)
        pairs = ga._robust_chromosome_crossover(np.array([0.5, 0.5]), 2, best_parent_reference=True)
    else:
        monkeypatch.setattr(improved_base, "choose_crossover", lambda parents, model: children)
        pairs = ga.crossover(np.array([0.5, 0.5]), 2, best_parent_reference=True)
    expected_cost = evaluate_permutation(child_perm, small_model).cost
    assert all(child.cost == expected_cost for child, _ in pairs)
    assert all(reference is cheap for _, reference in pairs)
    reward, punishment = ga._operator_performance(pairs)
    assert reward == 0.0
    assert punishment > 0.0


@pytest.mark.parametrize("robust", [False, True])
def test_standard_crossovers_use_best_participating_parent(small_model, monkeypatch, robust):
    ga = StandardAdaptiveGEA(small_model, population_size=2)
    cheap = evaluate_permutation(np.array([0, 0, 0, 0]), small_model)
    expensive = evaluate_permutation(np.array([1, 1, 1, 1]), small_model)
    ga.population = [cheap, expensive]
    child_perm = np.array([0, 1, 1, 1])
    monkeypatch.setattr(
        standard_base, "choose_crossover", lambda parents, model: ((child_perm, expensive), (child_perm, cheap))
    )
    if robust:
        ga.p_scenario1 = 1.0
        monkeypatch.setattr(standard_base, "analyze_perm", lambda *args, **kwargs: (None, None, cheap, None))
        monkeypatch.setattr(ga, "_selection_probabilities", lambda: np.array([0.0, 1.0]))
        pairs = ga._thesis_scenario1_pairs(1, best_parent_reference=True)
    else:
        monkeypatch.setattr(ga, "_parent_selection_indices", lambda size: np.array([0, 1]))
        pairs = ga._standard_crossover_pairs(2, best_parent_reference=True)
    assert all(reference is cheap for _, reference in pairs)
    assert all(child.cost == 7.0 for child, _ in pairs)


def test_selection_preserves_group_elites_and_prefers_improvement_over_absolute_cost(small_model):
    ga = StandardAdaptiveGA(small_model, population_size=3, gamma=1.0)
    parent_elite = individual(1, 0)
    child_elite = individual(2, 1)
    low_cost = individual(3, 2)
    high_improvement = individual(10, 3)
    same_chromosome = individual(10, 3)
    ga._select_adaptive_survivors(
        [
            [(parent_elite, parent_elite), (low_cost, low_cost)],
            [(child_elite, individual(3)), (high_improvement, individual(100)), (same_chromosome, individual(100))],
        ]
    )
    assert set(ga.population) == {parent_elite, child_elite, high_improvement}
    assert len(ga.population) == len(set(ga.population)) == 3


def test_tournaments_fill_survivors_without_duplicates(small_model, monkeypatch):
    ga = StandardAdaptiveGA(small_model, population_size=3, gamma=0.0, tournament_size=2)
    candidates = [individual(cost, cost) for cost in [1, 4, 3, 2]]
    draws = []

    def draw_indices(n, size, replace):
        draws.append((size, replace))
        return np.arange(n - size, n)

    monkeypatch.setattr(np.random, "choice", draw_indices)
    ga._select_adaptive_survivors([[(ind, ind) for ind in candidates]])
    assert [ind.cost for ind in ga.population] == [1, 2, 3]
    assert draws == [(2, False), (2, False)]


def test_duplicate_pool_is_replenished_and_evaluations_are_counted(small_model):
    ga = StandardAdaptiveGA(small_model, population_size=4)
    seed = evaluate_permutation(np.zeros(4, dtype=int), small_model)
    seed_all(42)
    ga._select_adaptive_survivors([[(seed, seed)] * 4])
    assert len(ga.population) == len(set(ga.population)) == 4
    assert seed in ga.population
    assert all(math.isfinite(ind.cost) for ind in ga.population)
    assert ga.logger.nfe >= 3


def test_impossible_unique_population_fails_with_bounded_attempts(small_model, monkeypatch):
    ga = StandardAdaptiveGA(small_model, population_size=2)
    seed = evaluate_permutation(np.zeros(4, dtype=int), small_model)
    monkeypatch.setattr("src.algos.mixins.adaptive.mutation_random", lambda perm, model: perm.copy())
    with pytest.raises(RuntimeError, match="unique feasible population"):
        ga._select_adaptive_survivors([[(seed, seed)]])
    assert ga.logger.nfe == 200


def test_memetic_polishing_preserves_adaptive_population_uniqueness(small_model, monkeypatch):
    ga = ImprovedAdaptiveGA(small_model, population_size=2)
    cheap = evaluate_permutation(np.zeros(4, dtype=int), small_model)
    expensive = evaluate_permutation(np.ones(4, dtype=int), small_model)
    ga.population = [cheap, expensive]
    monkeypatch.setattr(ga, "local_search", lambda perm: cheap.permutation.copy())
    ga.polish_elites()
    assert len(ga.population) == len(set(ga.population)) == 2


@pytest.mark.parametrize("config_name", ADAPTIVE_CONFIGS)
@pytest.mark.parametrize("dataset", ["T1", "T2"])
def test_every_adaptive_variant_has_bounded_lambdas_and_unique_full_population(config_name, dataset):
    ga = hydra.utils.instantiate(load_ga_config(config_name), model=load_model(dataset))
    midpoint = (ga.lambda_min + ga.lambda_max) / 2
    assert 0 <= ga.lambda_min <= ga.lambda_max <= 1
    initial = ga.snapshot_lambdas()
    assert all(value == pytest.approx(midpoint) for value in initial.values())
    seed_all(42)
    ga.run(time_limit=None)
    assert len(ga.lambda_history) == 5
    assert len(ga.population) == len(set(ga.population)) == ga.population_size
    assert all(math.isfinite(ind.cost) for ind in ga.population)
    for snapshot in ga.lambda_history:
        assert snapshot.keys() == initial.keys()
        assert all(ga.lambda_min <= value <= ga.lambda_max for value in snapshot.values())


@pytest.mark.parametrize("algorithm_class", [StandardAdaptiveGA, ImprovedAdaptiveGA])
def test_repeated_runs_reset_midpoint_and_history(algorithm_class, small_model):
    ga = algorithm_class(small_model, population_size=4, iterations=2, repair_class=GreedyRepair())
    for _ in range(2):
        ga.lambda_crossover = ga.lambda_max
        ga.lambda_mutation = ga.lambda_min
        seed_all(42)
        ga.run(time_limit=None)
        assert len(ga.lambda_history) == 2
        assert abs(ga.lambda_history[0]["lambda_crossover"] - 0.5) <= ga.alpha
        assert abs(ga.lambda_history[0]["lambda_mutation"] - 0.5) <= ga.alpha
