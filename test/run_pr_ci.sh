#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
MATRIX_RUNNER="${REPO_ROOT}/test/run_ci_matrix.sh"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/ci_common.sh"

DEFAULT_BUILD_ROOT="/tmp/adept-self-hosted-ci"
DEFAULT_RESULTS_FILE=""
DEFAULT_MASTER_REF="auto"

BUILD_ROOT="${DEFAULT_BUILD_ROOT}"
RESULTS_FILE="${DEFAULT_RESULTS_FILE}"
MASTER_REF="${DEFAULT_MASTER_REF}"
LCG_SETUP=""
CUDA_ARCH="auto"
CTEST_TIMEOUT_SEC=0
JOBS="auto"
FETCH_MASTER=1
FORCE_REBUILD=0
REFRESH_MASTER=0

log() {
  printf '[pr-ci] %s\n' "$*"
}

die() {
  printf '[pr-ci] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs the AdePT PR CI flow on a self-hosted runner:
  1. build PR and upstream master reference for async/split/mixed
  2. run physics drift comparisons
  3. always run unit tests on async
  4. run validation on async + split only when drift differs from master

Options:
  --build-root <path>        Build/cache root (default: ${DEFAULT_BUILD_ROOT})
  --results-file <path>      Write KEY=VALUE results for workflow consumption
  --lcg-setup <path>         Explicit LCG setup script
  --master-ref <git-ref>     Upstream reference for drift (default: ${DEFAULT_MASTER_REF})
  --no-fetch-master          Do not let the matrix runner fetch the master ref
  --cuda-arch <arch|auto>    CUDA arch passed to the matrix runner (default: auto)
  --jobs <N|auto>            Parallel build jobs passed through to the matrix runner (default: auto)
  --ctest-timeout-sec <sec>  Optional per-test timeout passed to ctest --timeout
  --force-rebuild            Force clean rebuilds in the matrix runner
  --refresh-master           Refresh cached master worktree/builds
  -h, --help                 Show this help
EOF
}

select_lcg_setup() {
  LCG_SETUP=$(select_devadept_lcg_setup "${LCG_SETUP}") || die "Failed to resolve devAdePT LCG setup"
}

run_ctest() {
  local -a cmd=(ctest "$@")
  if [[ "${CTEST_TIMEOUT_SEC}" -gt 0 ]]; then
    cmd+=(--timeout "${CTEST_TIMEOUT_SEC}")
    "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

run_and_capture() {
  local __var_name=$1
  shift
  if "$@"; then
    printf -v "${__var_name}" '%s' 0
  else
    printf -v "${__var_name}" '%s' 1
  fi
}

build_all_targets() {
  local build_dir=$1
  if [[ ! -d "${build_dir}" ]]; then
    log "Build directory not found: ${build_dir}"
    return 1
  fi
  if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
    log "Build directory is not configured: ${build_dir}"
    return 1
  fi

  local jobs=${JOBS}
  if [[ "${jobs}" == "auto" ]]; then
    jobs=$(nproc)
  fi

  log "Building all targets in ${build_dir} (-j${jobs})"
  cmake --build "${build_dir}" --target all -j"${jobs}"
}

write_results() {
  [[ -n "${RESULTS_FILE}" ]] || return

  {
    printf 'LCG_SETUP=%q\n' "${LCG_SETUP}"
    printf 'BUILD_ROOT=%q\n' "${BUILD_ROOT}"
    printf 'MASTER_REF=%q\n' "${MASTER_REF}"
    printf 'DRIFT_STATUS=%s\n' "${DRIFT_STATUS}"
    printf 'UNIT_STATUS=%s\n' "${UNIT_STATUS}"
    printf 'VALIDATION_RAN=%s\n' "${VALIDATION_RAN}"
    printf 'VALIDATION_STATUS=%s\n' "${VALIDATION_STATUS}"
    printf 'FULL_CI_STATUS=%s\n' "${FULL_CI_STATUS}"
  } > "${RESULTS_FILE}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-root)
      [[ $# -ge 2 ]] || die "Missing value for --build-root"
      BUILD_ROOT=$2
      shift 2
      ;;
    --results-file)
      [[ $# -ge 2 ]] || die "Missing value for --results-file"
      RESULTS_FILE=$2
      shift 2
      ;;
    --lcg-setup)
      [[ $# -ge 2 ]] || die "Missing value for --lcg-setup"
      LCG_SETUP=$2
      shift 2
      ;;
    --master-ref)
      [[ $# -ge 2 ]] || die "Missing value for --master-ref"
      MASTER_REF=$2
      shift 2
      ;;
    --no-fetch-master)
      FETCH_MASTER=0
      shift
      ;;
    --cuda-arch)
      [[ $# -ge 2 ]] || die "Missing value for --cuda-arch"
      CUDA_ARCH=$2
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || die "Missing value for --jobs"
      JOBS=$2
      shift 2
      ;;
    --ctest-timeout-sec)
      [[ $# -ge 2 ]] || die "Missing value for --ctest-timeout-sec"
      CTEST_TIMEOUT_SEC=$2
      shift 2
      ;;
    --force-rebuild)
      FORCE_REBUILD=1
      shift
      ;;
    --refresh-master)
      REFRESH_MASTER=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

[[ -x "${MATRIX_RUNNER}" ]] || die "Matrix runner not found: ${MATRIX_RUNNER}"
[[ "${CTEST_TIMEOUT_SEC}" =~ ^[0-9]+$ ]] || die "Invalid --ctest-timeout-sec '${CTEST_TIMEOUT_SEC}'"
if [[ "${JOBS}" != "auto" && ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  die "Invalid --jobs '${JOBS}'. Expected 'auto' or a positive integer."
fi

select_lcg_setup

# shellcheck disable=SC1090
set +u
source "${LCG_SETUP}"
set -u
normalize_lcg_cuda_env || die "Failed to normalize CUDA environment from LCG setup"

mkdir -p "${BUILD_ROOT}"

DRIFT_STATUS=0
UNIT_STATUS=1
VALIDATION_RAN=0
VALIDATION_STATUS=0
FULL_CI_STATUS=1

matrix_common_args=(
  --suite drift
  --build-root "${BUILD_ROOT}"
  --master-ref "${MASTER_REF}"
  --cuda-arch "${CUDA_ARCH}"
  --jobs "${JOBS}"
  --ctest-timeout-sec "${CTEST_TIMEOUT_SEC}"
)

if [[ "${FETCH_MASTER}" -eq 0 ]]; then
  matrix_common_args+=(--no-fetch-master)
fi
if [[ "${FORCE_REBUILD}" -eq 1 ]]; then
  matrix_common_args+=(--force-rebuild)
fi
if [[ "${REFRESH_MASTER}" -eq 1 ]]; then
  matrix_common_args+=(--refresh-master)
fi

log "Using LCG setup: ${LCG_SETUP}"
log "Build root: ${BUILD_ROOT}"
log "Master reference: ${MASTER_REF}"
log "CUDA arch: ${CUDA_ARCH}"
log "Build jobs: ${JOBS}"
log "Running drift matrix for async/split/mixed"

for cfg in async split mixed; do
  log "Drift phase for config: ${cfg}"
  if ! "${MATRIX_RUNNER}" "${matrix_common_args[@]}" --configs "${cfg}"; then
    DRIFT_STATUS=1
  fi
done

MONOL_BUILD_DIR="${BUILD_ROOT}/BUILD_MONOL"
SPLIT_BUILD_DIR="${BUILD_ROOT}/BUILD_SPLIT_ON"

log "Ensuring async build has all test targets"
if build_all_targets "${MONOL_BUILD_DIR}"; then
  log "Running async unit tests"
  run_and_capture UNIT_STATUS run_ctest --test-dir "${MONOL_BUILD_DIR}" --output-on-failure -L unit -j1
else
  log "Skipping async unit tests because the async build is unavailable"
  UNIT_STATUS=1
fi

if [[ "${DRIFT_STATUS}" -ne 0 ]]; then
  VALIDATION_RAN=1

  validation_monol_status=1
  validation_split_status=1

  log "Drift differs from master; running validation on async and split"

  if [[ "${UNIT_STATUS}" -eq 0 ]]; then
    run_and_capture validation_monol_status run_ctest --test-dir "${MONOL_BUILD_DIR}" --output-on-failure -L validation -j1
  else
    log "Skipping async validation because the async build or unit stage failed"
  fi

  if build_all_targets "${SPLIT_BUILD_DIR}"; then
    run_and_capture validation_split_status run_ctest --test-dir "${SPLIT_BUILD_DIR}" --output-on-failure -L validation -j1
  else
    log "Skipping split validation because the split build is unavailable"
  fi

  if [[ "${validation_monol_status}" -eq 0 && "${validation_split_status}" -eq 0 ]]; then
    VALIDATION_STATUS=0
  else
    VALIDATION_STATUS=1
  fi
else
  log "Drift matched master; validation is skipped"
  VALIDATION_RAN=0
  VALIDATION_STATUS=0
fi

if [[ "${UNIT_STATUS}" -eq 0 && ( "${DRIFT_STATUS}" -eq 0 || "${VALIDATION_STATUS}" -eq 0 ) ]]; then
  FULL_CI_STATUS=0
else
  FULL_CI_STATUS=1
fi

write_results

log "Drift status: ${DRIFT_STATUS}"
log "Unit status: ${UNIT_STATUS}"
log "Validation ran: ${VALIDATION_RAN}"
log "Validation status: ${VALIDATION_STATUS}"
log "Full CI status: ${FULL_CI_STATUS}"

exit "${FULL_CI_STATUS}"
