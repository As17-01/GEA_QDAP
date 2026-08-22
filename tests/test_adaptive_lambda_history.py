import hydra

from src.algos.adaptive.ga import ImprovedAdaptiveGA
from src.data.model_loader import load_model
from src.seeding import seed_all


def test_lambda_history_recorded_during_run():
    model = load_model("T1")
    ga = ImprovedAdaptiveGA(model=model, population_size=20, iterations=5, verbose=False)
    seed_all(42)
    ga.run(time_limit=None)
    assert hasattr(ga, "lambda_history")
    assert len(ga.lambda_history) == 5
    assert "lambda_crossover" in ga.lambda_history[0]
    assert "lambda_mutation" in ga.lambda_history[0]
    assert "lambda_rc" not in ga.lambda_history[0]
