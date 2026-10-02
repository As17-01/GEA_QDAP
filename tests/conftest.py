import os

# Pin BLAS/numba to one thread before any numpy/numba import so prange-based repair and
# cost evaluation stay deterministic under pytest (parallel numba RNG order varies).
for _env_var in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
CONF_DIR = SCRIPTS / "conf"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from utils.labels import algo_label

OmegaConf.register_new_resolver("algo_label", algo_label, replace=True)

ALL_ALGORITHM_CONFIGS = [
    "standard",
    "gea",
    "standard_gea",
    "ga",
    "adaptive",
    "standard_adaptive",
    "sa",
    "standard_sa",
    "pso",
    "standard_pso",
    "hybrid_ga_pso",
    "standard_hybrid_ga_pso",
    "hybrid_ga_sa",
    "standard_hybrid_ga_sa",
    "gea_scenario_1",
    "standard_gea_scenario_1",
    "gea_scenario_2",
    "standard_gea_scenario_2",
    "gea_scenario_3",
    "standard_gea_scenario_3",
    "adaptive_gea",
    "standard_adaptive_gea",
    "adaptive_gea_scenario_1",
    "standard_adaptive_gea_scenario_1",
    "adaptive_gea_scenario_2",
    "standard_adaptive_gea_scenario_2",
    "adaptive_gea_scenario_3",
    "standard_adaptive_gea_scenario_3",
]

TEST_DATASETS = ["T1", "T2"]

SMOKE_ITERATIONS = 5
SMOKE_POPULATION_SIZE = 20
SMOKE_SEED = 42

# Golden best costs for seed=42, 5 iterations, population_size=20, single-threaded numba.
# When updating after standard-scaffold changes, regenerate standard_* entries only;
# improved config values (ga, gea, adaptive, …) must stay frozen unless those algorithms change.
EXPECTED_COSTS = {
    ("standard", "T1"): 2997987.839673718,
    ("standard", "T2"): 10900438.301304156,
    ("gea", "T1"): 2347569.9124305504,
    ("gea", "T2"): 8255699.864957599,
    ("standard_gea", "T1"): 2980776.543499155,
    ("standard_gea", "T2"): 10871861.534145605,
    ("ga", "T1"): 2732629.9151111124,
    ("ga", "T2"): 10450864.87404784,
    ("adaptive", "T1"): 2732629.9151111124,
    ("adaptive", "T2"): 10450864.87404784,
    ("standard_adaptive", "T1"): 2997987.839673718,
    ("standard_adaptive", "T2"): 10890326.98163593,
    ("sa", "T1"): 2895981.3465877837,
    ("sa", "T2"): 10761332.947231574,
    ("standard_sa", "T1"): 3016604.5998349353,
    ("standard_sa", "T2"): 10907543.375230681,
    ("pso", "T1"): 2789377.937690473,
    ("pso", "T2"): 10577601.313234571,
    ("standard_pso", "T1"): 3010853.7585912915,
    ("standard_pso", "T2"): 10890570.624395194,
    ("hybrid_ga_pso", "T1"): 2794315.4194674296,
    ("hybrid_ga_pso", "T2"): 10404461.293815063,
    ("standard_hybrid_ga_pso", "T1"): 2993473.601733859,
    ("standard_hybrid_ga_pso", "T2"): 10876828.244051142,
    ("hybrid_ga_sa", "T1"): 2718518.493335408,
    ("hybrid_ga_sa", "T2"): 10472435.664510876,
    ("standard_hybrid_ga_sa", "T1"): 2990790.2838919656,
    ("standard_hybrid_ga_sa", "T2"): 10877062.543743756,
    ("gea_scenario_1", "T1"): 2370783.1805182355,
    ("gea_scenario_1", "T2"): 8480483.839949256,
    ("standard_gea_scenario_1", "T1"): 2993473.6017338596,
    ("standard_gea_scenario_1", "T2"): 10871861.534145605,
    ("gea_scenario_2", "T1"): 2710676.020270306,
    ("gea_scenario_2", "T2"): 10448479.303009778,
    ("standard_gea_scenario_2", "T1"): 2996621.896163943,
    ("standard_gea_scenario_2", "T2"): 10854702.306285208,
    ("gea_scenario_3", "T1"): 2767143.880134292,
    ("gea_scenario_3", "T2"): 10274580.865259748,
    ("standard_gea_scenario_3", "T1"): 2991646.1147859795,
    ("standard_gea_scenario_3", "T2"): 10885451.341142755,
    ("adaptive_gea", "T1"): 2347569.9124305504,
    ("adaptive_gea", "T2"): 8255699.864957599,
    ("standard_adaptive_gea", "T1"): 2993086.4256607853,
    ("standard_adaptive_gea", "T2"): 10884681.624545049,
    ("adaptive_gea_scenario_1", "T1"): 2269421.6162845492,
    ("adaptive_gea_scenario_1", "T2"): 8454591.59733361,
    ("standard_adaptive_gea_scenario_1", "T1"): 3004457.3093706504,
    ("standard_adaptive_gea_scenario_1", "T2"): 10855149.992332228,
    ("adaptive_gea_scenario_2", "T1"): 2710676.020270306,
    ("adaptive_gea_scenario_2", "T2"): 10448479.303009778,
    ("standard_adaptive_gea_scenario_2", "T1"): 3008688.152893045,
    ("standard_adaptive_gea_scenario_2", "T2"): 10896844.312788308,
    ("adaptive_gea_scenario_3", "T1"): 2746056.568474839,
    ("adaptive_gea_scenario_3", "T2"): 10388735.219151733,
    ("standard_adaptive_gea_scenario_3", "T1"): 3003620.3450601893,
    ("standard_adaptive_gea_scenario_3", "T2"): 10895536.79993075,
}


def load_ga_config(config_name: str, *, iterations: int = SMOKE_ITERATIONS, population_size: int = SMOKE_POPULATION_SIZE) -> dict:
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(
            config_name=config_name,
            overrides=[
                f"ga.iterations={iterations}",
                f"ga.population_size={population_size}",
            ],
        )
    return OmegaConf.to_container(cfg.ga, resolve=True)
