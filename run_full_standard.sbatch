#!/bin/bash
#SBATCH --job-name=gea_gqap_full_std
#SBATCH --error=gea_gqap_full_std-%j.err
#SBATCH --output=gea_gqap_full_std-%j.log
#SBATCH --partition=rocky
#SBATCH --time=336:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gpus=0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUM_WORKERS=16

module load Python
source activate gea_gqap_v2_env

ALGORITHMS=(
    standard
    standard_gea
    standard_adaptive
    standard_sa
    standard_pso
    standard_hybrid_ga_pso
    standard_hybrid_ga_sa
    standard_gea_scenario_1
    standard_gea_scenario_2
    standard_gea_scenario_3
    standard_adaptive_gea
    standard_adaptive_gea_scenario_1
    standard_adaptive_gea_scenario_2
    standard_adaptive_gea_scenario_3
)

for algo in "${ALGORITHMS[@]}"; do
    echo "=== Running ${algo} ==="
    poetry run python scripts/run.py --config-name="${algo}"
done
