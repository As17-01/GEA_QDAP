#!/bin/bash
#SBATCH --job-name=gea_gqap_tune_algo
#SBATCH --error=gea_gqap_tune_algo-%j.err
#SBATCH --output=gea_gqap_tune_algo-%j.log
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
    gea
    adaptive
    sa
    pso
    hybrid_ga_pso
    hybrid_ga_sa
    gea_scenario_1
    gea_scenario_2
    gea_scenario_3
)

for algo in "${ALGORITHMS[@]}"; do
    echo "=== Tuning ${algo} ==="
    poetry run python scripts/tune_algorithm.py --config-name="tune_algorithm/${algo}"
done
