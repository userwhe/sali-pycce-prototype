#!/bin/bash
#SBATCH --job-name=sali-shard-array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-359%20
#SBATCH --chdir=/gscratch/scrubbed/whe3/sali/repo
#SBATCH --output=/gscratch/scrubbed/whe3/sali/logs/%x-%A_%a.out
#SBATCH --error=/gscratch/scrubbed/whe3/sali/logs/%x-%A_%a.err

set -euo pipefail

WORK_ROOT=/gscratch/scrubbed/whe3/sali
CHECKOUT_DIR="${WORK_ROOT}/repo"
PROJECT_DIR="${CHECKOUT_DIR}/sali"
DATASET_DIR="${WORK_ROOT}/datasets/paper-low-full"
VENV="${WORK_ROOT}/venv"

if [[ ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
  PROJECT_DIR="${CHECKOUT_DIR}"
fi

mkdir -p "${WORK_ROOT}/logs"
cd "${PROJECT_DIR}"
source "${VENV}/bin/activate"

python scripts/shard_jobs.py \
  --preset paper \
  --field low \
  --dataset-dir "${DATASET_DIR}" \
  --shard-size 10000 \
  --normalization-samples 20000 \
  --shard-dtype float32 \
  --raw-signal-dtype float32 \
  write \
  --shard-id "${SLURM_ARRAY_TASK_ID}"
