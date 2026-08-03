#!/bin/bash
#SBATCH --job-name=gea_gqap_full
#SBATCH --error=gea_gqap_full-%j.err
#SBATCH --output=gea_gqap_full-%j.log
#SBATCH --partition=rocky
#SBATCH --time=336:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gpus=0

# Runs all standard and improved configs in one job.
# To run separately, use run_full_standard.sbatch and run_full_improved.sbatch.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUM_WORKERS=16

module load Python
source activate gea_gqap_v2_env

ALGORITHMS=(
    standard
    ga
    gea
    standard_gea
    adaptive
    standard_adaptive
    sa
    standard_sa
    pso
    standard_pso
    hybrid_ga_pso
    standard_hybrid_ga_pso
    hybrid_ga_sa
    standard_hybrid_ga_sa
    gea_scenario_1
    standard_gea_scenario_1
    gea_scenario_2
    standard_gea_scenario_2
    gea_scenario_3
    standard_gea_scenario_3
    adaptive_gea
    standard_adaptive_gea
    adaptive_gea_scenario_1
    standard_adaptive_gea_scenario_1
    adaptive_gea_scenario_2
    standard_adaptive_gea_scenario_2
    adaptive_gea_scenario_3
    standard_adaptive_gea_scenario_3
)

for algo in "${ALGORITHMS[@]}"; do
    echo "=== Running ${algo} ==="
    poetry run python scripts/run.py --config-name="${algo}"
done
