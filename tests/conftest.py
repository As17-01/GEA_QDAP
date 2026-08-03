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
    "improved_ga",
    "adaptive",
    "sa",
    "pso",
    "hybrid_ga_pso",
    "hybrid_ga_sa",
    "gea_scenario_1",
    "gea_scenario_2",
    "gea_scenario_3",
    "adaptive_gea",
    "adaptive_gea_scenario_1",
    "adaptive_gea_scenario_2",
    "adaptive_gea_scenario_3",
]

TEST_DATASETS = ["T1", "T2"]

SMOKE_ITERATIONS = 5
SMOKE_POPULATION_SIZE = 20
SMOKE_SEED = 42

# Golden best costs for seed=42, 5 iterations, population_size=20, single-threaded numba.
EXPECTED_COSTS = {
    ("standard", "T1"): 2775437.583027137,
    ("gea", "T1"): 2347569.9124305504,
    ("improved_ga", "T1"): 2732629.9151111124,
    ("adaptive", "T1"): 2732629.9151111124,
    ("sa", "T1"): 2915146.7320602443,
    ("pso", "T1"): 2789377.937690473,
    ("hybrid_ga_pso", "T1"): 2794315.4194674296,
    ("hybrid_ga_sa", "T1"): 2718518.493335408,
    ("gea_scenario_1", "T1"): 2370783.1805182355,
    ("gea_scenario_2", "T1"): 2710676.020270306,
    ("gea_scenario_3", "T1"): 2767143.880134292,
    ("adaptive_gea", "T1"): 2347569.9124305504,
    ("adaptive_gea_scenario_1", "T1"): 2269421.6162845492,
    ("adaptive_gea_scenario_2", "T1"): 2710676.020270306,
    ("adaptive_gea_scenario_3", "T1"): 2746056.568474839,
    ("standard", "T2"): 10397195.68806942,
    ("gea", "T2"): 8255699.864957599,
    ("improved_ga", "T2"): 10450864.87404784,
    ("adaptive", "T2"): 10450864.87404784,
    ("sa", "T2"): 10858202.723611688,
    ("pso", "T2"): 10577601.313234571,
    ("hybrid_ga_pso", "T2"): 10404461.293815063,
    ("hybrid_ga_sa", "T2"): 10472435.664510876,
    ("gea_scenario_1", "T2"): 8480483.839949256,
    ("gea_scenario_2", "T2"): 10448479.303009778,
    ("gea_scenario_3", "T2"): 10274580.865259748,
    ("adaptive_gea", "T2"): 8255699.864957599,
    ("adaptive_gea_scenario_1", "T2"): 8454591.59733361,
    ("adaptive_gea_scenario_2", "T2"): 10448479.303009778,
    ("adaptive_gea_scenario_3", "T2"): 10388735.219151733,
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
