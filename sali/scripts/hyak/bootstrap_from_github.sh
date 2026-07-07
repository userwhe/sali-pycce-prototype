#!/bin/bash

set -euo pipefail

WORK_ROOT="${SALI_WORK_ROOT:-/gscratch/scrubbed/whe3/sali}"
CHECKOUT_DIR="${WORK_ROOT}/repo"
PROJECT_SUBDIR="${SALI_PROJECT_SUBDIR:-sali}"
PROJECT_DIR="${CHECKOUT_DIR}/${PROJECT_SUBDIR}"
REPO_URL="${SALI_REPO_URL:-https://github.com/userwhe/sali-pycce-prototype.git}"
REPO_BRANCH="${SALI_REPO_BRANCH:-codex/sali-reproduction}"
SUBMIT_MODE="${SALI_SUBMIT_MODE:-setup}"

echo "host: $(hostname)"
echo "user: $(whoami)"
echo "work root: ${WORK_ROOT}"
echo "repo branch: ${REPO_BRANCH}"
echo "submit mode: ${SUBMIT_MODE}"

mkdir -p "${WORK_ROOT}/logs" "${WORK_ROOT}/datasets"

if [[ -d "${CHECKOUT_DIR}/.git" ]]; then
  git -C "${CHECKOUT_DIR}" fetch origin "${REPO_BRANCH}"
  git -C "${CHECKOUT_DIR}" checkout "${REPO_BRANCH}"
  git -C "${CHECKOUT_DIR}" reset --hard "origin/${REPO_BRANCH}"
elif [[ -e "${CHECKOUT_DIR}" ]]; then
  echo "ERROR: ${CHECKOUT_DIR} exists but is not a git checkout." >&2
  echo "Move it aside or set SALI_WORK_ROOT to a new directory." >&2
  exit 1
else
  git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${CHECKOUT_DIR}"
fi

if [[ ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
  if [[ -f "${CHECKOUT_DIR}/pyproject.toml" ]]; then
    PROJECT_DIR="${CHECKOUT_DIR}"
  else
    echo "ERROR: could not find SALI pyproject.toml in ${PROJECT_DIR} or ${CHECKOUT_DIR}" >&2
    exit 1
  fi
fi
echo "project dir: ${PROJECT_DIR}"

hyakstorage --home || true
hyakstorage /gscratch/scrubbed/whe3 || true
df -h /gscratch/scrubbed/whe3 || true

if [[ "${SUBMIT_MODE}" == "setup" || "${SUBMIT_MODE}" == "none" ]]; then
  echo "Setup complete. No Slurm jobs submitted."
  echo "Run hyakalloc and choose SALI_HYAK_ACCOUNT and SALI_HYAK_PARTITION before submitting."
  hyakalloc || true
  exit 0
fi

: "${SALI_HYAK_ACCOUNT:?Set SALI_HYAK_ACCOUNT to the exact account shown by hyakalloc}"
: "${SALI_HYAK_PARTITION:?Set SALI_HYAK_PARTITION to the exact partition shown by hyakalloc}"

cd "${PROJECT_DIR}"

submit_env() {
  sbatch --parsable \
    -A "${SALI_HYAK_ACCOUNT}" \
    -p "${SALI_HYAK_PARTITION}" \
    scripts/hyak/setup_env.slurm
}

submit_prepare() {
  local dependency="$1"
  sbatch --parsable \
    -A "${SALI_HYAK_ACCOUNT}" \
    -p "${SALI_HYAK_PARTITION}" \
    --dependency="afterok:${dependency}" \
    scripts/hyak/prepare_shards.slurm
}

submit_array() {
  local dependency="$1"
  shift
  sbatch --parsable \
    -A "${SALI_HYAK_ACCOUNT}" \
    -p "${SALI_HYAK_PARTITION}" \
    --dependency="afterok:${dependency}" \
    "$@" \
    scripts/hyak/generate_shards_array.slurm
}

submit_finalize() {
  local dependency="$1"
  sbatch --parsable \
    -A "${SALI_HYAK_ACCOUNT}" \
    -p "${SALI_HYAK_PARTITION}" \
    --dependency="afterok:${dependency}" \
    scripts/hyak/finalize_shards.slurm
}

env_job="$(submit_env)"
env_job="${env_job%%;*}"
echo "env setup job: ${env_job}"

if [[ "${SUBMIT_MODE}" == "env" ]]; then
  squeue -u "$(whoami)" || true
  exit 0
fi

prepare_job="$(submit_prepare "${env_job}")"
prepare_job="${prepare_job%%;*}"
echo "prepare job: ${prepare_job}"

case "${SUBMIT_MODE}" in
  prepare)
    ;;
  pilot)
    pilot_job="$(submit_array "${prepare_job}" --array=0-3%2)"
    pilot_job="${pilot_job%%;*}"
    echo "pilot array job: ${pilot_job}"
    ;;
  full)
    array_job="$(submit_array "${prepare_job}")"
    array_job="${array_job%%;*}"
    finalize_job="$(submit_finalize "${array_job}")"
    finalize_job="${finalize_job%%;*}"
    echo "full array job: ${array_job}"
    echo "finalize job: ${finalize_job}"
    ;;
  *)
    echo "ERROR: unsupported SALI_SUBMIT_MODE=${SUBMIT_MODE}" >&2
    echo "Use setup, env, prepare, pilot, or full." >&2
    exit 2
    ;;
esac

squeue -u "$(whoami)" || true
