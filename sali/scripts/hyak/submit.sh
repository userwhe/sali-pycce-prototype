#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGIN="${SALI_HYAK_LOGIN:-whe3@klone.hyak.uw.edu}"
ACCOUNT="${SALI_HYAK_ACCOUNT:-stf}"
PARTITION="${SALI_HYAK_PARTITION:-compute-hugemem}"
WORK_ROOT="${SALI_WORK_ROOT:-/gscratch/scrubbed/whe3/sali}"
MODE="${1:-setup}"
REMOTE_USER="${LOGIN%@*}"

if [[ "${REMOTE_USER}" == "${LOGIN}" ]]; then
  REMOTE_USER="${USER:-whe3}"
fi

validate_token() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[A-Za-z0-9._/-]+$ ]]; then
    echo "ERROR: ${name} contains unsupported characters: ${value}" >&2
    exit 2
  fi
}

run_bootstrap() {
  local mode="$1"
  validate_token "SALI_SUBMIT_MODE" "${mode}"
  validate_token "SALI_WORK_ROOT" "${WORK_ROOT}"

  if [[ "${mode}" == "setup" || "${mode}" == "none" ]]; then
    ssh "${LOGIN}" "SALI_WORK_ROOT=${WORK_ROOT} SALI_SUBMIT_MODE=${mode} bash -s" \
      < "${SCRIPT_DIR}/bootstrap_from_github.sh"
    return
  fi

  validate_token "SALI_HYAK_ACCOUNT" "${ACCOUNT}"
  validate_token "SALI_HYAK_PARTITION" "${PARTITION}"
  ssh "${LOGIN}" \
    "SALI_WORK_ROOT=${WORK_ROOT} SALI_HYAK_ACCOUNT=${ACCOUNT} SALI_HYAK_PARTITION=${PARTITION} SALI_SUBMIT_MODE=${mode} bash -s" \
    < "${SCRIPT_DIR}/bootstrap_from_github.sh"
}

case "${MODE}" in
  setup|none|env|prepare|pilot|full)
    run_bootstrap "${MODE}"
    ;;
  queue)
    ssh "${LOGIN}" "squeue -u ${REMOTE_USER}"
    ;;
  storage)
    ssh "${LOGIN}" "hyakstorage --home; hyakstorage ${WORK_ROOT} || true; df -h /gscratch/scrubbed/whe3"
    ;;
  count)
    ssh "${LOGIN}" "find ${WORK_ROOT}/datasets/paper-low-full -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l; du -sh ${WORK_ROOT}/datasets/paper-low-full 2>/dev/null || true"
    ;;
  logs)
    ssh "${LOGIN}" "ls -lt ${WORK_ROOT}/logs 2>/dev/null | head -20; for f in ${WORK_ROOT}/logs/sali-setup-env-*.err ${WORK_ROOT}/logs/sali-setup-env-*.out ${WORK_ROOT}/logs/sali-shard-prepare-*.err ${WORK_ROOT}/logs/sali-shard-prepare-*.out ${WORK_ROOT}/logs/sali-shard-array-*.err ${WORK_ROOT}/logs/sali-shard-array-*.out; do [ -f \"\$f\" ] || continue; echo; echo ===== \"\$f\" =====; tail -120 \"\$f\"; done"
    ;;
  *)
    cat >&2 <<EOF
Usage: $0 [setup|env|prepare|pilot|full|queue|storage|count|logs]

Environment overrides:
  SALI_HYAK_LOGIN      default: whe3@klone.hyak.uw.edu
  SALI_HYAK_ACCOUNT    default: stf
  SALI_HYAK_PARTITION  default: compute-hugemem
  SALI_WORK_ROOT       default: /gscratch/scrubbed/whe3/sali
EOF
    exit 2
    ;;
esac
