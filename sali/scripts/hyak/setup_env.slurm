#!/bin/bash
#SBATCH --job-name=sali-setup-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/gscratch/scrubbed/whe3/sali/repo
#SBATCH --output=/gscratch/scrubbed/whe3/sali/logs/%x-%j.out
#SBATCH --error=/gscratch/scrubbed/whe3/sali/logs/%x-%j.err

set -euo pipefail

WORK_ROOT=/gscratch/scrubbed/whe3/sali
CHECKOUT_DIR="${WORK_ROOT}/repo"
PROJECT_DIR="${CHECKOUT_DIR}/sali"
CONDA_ENV="${WORK_ROOT}/conda-env"

if [[ ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
  PROJECT_DIR="${CHECKOUT_DIR}"
fi

mkdir -p "${WORK_ROOT}/logs" "${WORK_ROOT}/datasets"
cd "${PROJECT_DIR}"

module load conda

if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
  conda create --yes --prefix "${CONDA_ENV}" python=3.12 pip
fi

conda run --no-capture-output --prefix "${CONDA_ENV}" python -m pip install --upgrade pip
conda run --no-capture-output --prefix "${CONDA_ENV}" python -m pip install -e "${PROJECT_DIR}[dev]"
conda run --no-capture-output --prefix "${CONDA_ENV}" python scripts/shard_jobs.py --help >/dev/null
