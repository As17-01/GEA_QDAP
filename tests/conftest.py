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
EXPECTED_COSTS = {
    ("standard", "T1"): 3000128.073460968,
    ("gea", "T1"): 2347569.9124305504,
    ("standard_gea", "T1"): 2919758.9789992007,
    ("ga", "T1"): 2732629.9151111124,
    ("adaptive", "T1"): 2732629.9151111124,
    ("standard_adaptive", "T1"): 2902323.982293173,
    ("sa", "T1"): 2895981.3465877837,
    ("standard_sa", "T1"): 2915146.7320602443,
    ("pso", "T1"): 2789377.937690473,
    ("standard_pso", "T1"): 2875071.2576413266,
    ("hybrid_ga_pso", "T1"): 2794315.4194674296,
    ("standard_hybrid_ga_pso", "T1"): 2899280.004329535,
    ("hybrid_ga_sa", "T1"): 2718518.493335408,
    ("standard_hybrid_ga_sa", "T1"): 2974258.974010832,
    ("gea_scenario_1", "T1"): 2370783.1805182355,
    ("standard_gea_scenario_1", "T1"): 2863778.922987876,
    ("gea_scenario_2", "T1"): 2710676.020270306,
    ("standard_gea_scenario_2", "T1"): 2945611.6491405866,
    ("gea_scenario_3", "T1"): 2767143.880134292,
    ("standard_gea_scenario_3", "T1"): 2936471.125612959,
    ("adaptive_gea", "T1"): 2347569.9124305504,
    ("standard_adaptive_gea", "T1"): 2813323.659038525,
    ("adaptive_gea_scenario_1", "T1"): 2269421.6162845492,
    ("standard_adaptive_gea_scenario_1", "T1"): 2819847.6170662385,
    ("adaptive_gea_scenario_2", "T1"): 2710676.020270306,
    ("standard_adaptive_gea_scenario_2", "T1"): 2904765.0705134897,
    ("adaptive_gea_scenario_3", "T1"): 2746056.568474839,
    ("standard_adaptive_gea_scenario_3", "T1"): 2911733.3629647214,
    ("standard", "T2"): 10881696.878478287,
    ("gea", "T2"): 8255699.864957599,
    ("standard_gea", "T2"): 10300122.045170257,
    ("ga", "T2"): 10450864.87404784,
    ("adaptive", "T2"): 10450864.87404784,
    ("standard_adaptive", "T2"): 10779828.745348178,
    ("sa", "T2"): 10761332.947231574,
    ("standard_sa", "T2"): 10858202.723611688,
    ("pso", "T2"): 10577601.313234571,
    ("standard_pso", "T2"): 10637528.135064742,
    ("hybrid_ga_pso", "T2"): 10404461.293815063,
    ("standard_hybrid_ga_pso", "T2"): 10802560.434713153,
    ("hybrid_ga_sa", "T2"): 10472435.664510876,
    ("standard_hybrid_ga_sa", "T2"): 10728259.027471127,
    ("gea_scenario_1", "T2"): 8480483.839949256,
    ("standard_gea_scenario_1", "T2"): 10871861.534145605,
    ("gea_scenario_2", "T2"): 10448479.303009778,
    ("standard_gea_scenario_2", "T2"): 10699015.788875153,
    ("gea_scenario_3", "T2"): 10274580.865259748,
    ("standard_gea_scenario_3", "T2"): 10740354.553883309,
    ("adaptive_gea", "T2"): 8255699.864957599,
    ("standard_adaptive_gea", "T2"): 10407479.795839554,
    ("adaptive_gea_scenario_1", "T2"): 8454591.59733361,
    ("standard_adaptive_gea_scenario_1", "T2"): 10062549.399854176,
    ("adaptive_gea_scenario_2", "T2"): 10448479.303009778,
    ("standard_adaptive_gea_scenario_2", "T2"): 10713418.340782963,
    ("adaptive_gea_scenario_3", "T2"): 10388735.219151733,
    ("standard_adaptive_gea_scenario_3", "T2"): 10717729.370651912,
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
