#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
MATRIX_RUNNER="${REPO_ROOT}/test/run_ci_matrix.sh"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/ci_common.sh"

DEFAULT_BUILD_ROOT="/tmp/adept-nightly-ci"
DEFAULT_RESULTS_FILE=""

BUILD_ROOT="${DEFAULT_BUILD_ROOT}"
RESULTS_FILE="${DEFAULT_RESULTS_FILE}"
LCG_SETUP=""
BUILD_TYPE="Release"
CUDA_ARCH="auto"
CTEST_TIMEOUT_SEC=0
JOBS="auto"
declare -a CMAKE_EXTRA_ARGS=()

log() {
  printf '[nightly-ci] %s\n' "$*"
}

die() {
  printf '[nightly-ci] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs the nightly CI selection on the self-hosted runner:
  1. build Release matrix for async/split/mixed
  2. run unit tests on async
  3. run validation tests on async and split

Options:
  --build-root <path>        Build root (default: ${DEFAULT_BUILD_ROOT})
  --results-file <path>      Write KEY=VALUE results for workflow consumption
  --lcg-setup <path>         Explicit LCG setup script
  --build-type <type>        CMake build type (default: ${BUILD_TYPE})
  --cuda-arch <arch|auto>    CUDA arch passed to the matrix runner (default: ${CUDA_ARCH})
  --jobs <N|auto>            Parallel build jobs passed through to the matrix runner (default: ${JOBS})
  --ctest-timeout-sec <sec>  Optional per-test timeout passed to ctest --timeout
  --cmake-extra <arg>        Additional CMake argument passed through to the matrix runner
  -h, --help                 Show this help
EOF
}

write_results() {
  [[ -n "${RESULTS_FILE}" ]] || return

  {
    printf 'LCG_SETUP=%q\n' "${LCG_SETUP}"
    printf 'BUILD_ROOT=%q\n' "${BUILD_ROOT}"
    printf 'BUILD_TYPE=%q\n' "${BUILD_TYPE}"
    printf 'NIGHTLY_STATUS=%s\n' "${NIGHTLY_STATUS}"
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
    --build-type)
      [[ $# -ge 2 ]] || die "Missing value for --build-type"
      BUILD_TYPE=$2
      shift 2
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
    --cmake-extra)
      [[ $# -ge 2 ]] || die "Missing value for --cmake-extra"
      CMAKE_EXTRA_ARGS+=("$2")
      shift 2
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

LCG_SETUP=$(select_devadept_lcg_setup "${LCG_SETUP}") || die "Failed to resolve devAdePT LCG setup"

# shellcheck disable=SC1090
set +u
source "${LCG_SETUP}"
set -u
normalize_lcg_cuda_env || die "Failed to normalize CUDA environment from LCG setup"

mkdir -p "${BUILD_ROOT}"

matrix_args=(
  --suite ci
  --configs async,split,mixed
  --build-root "${BUILD_ROOT}"
  --build-type "${BUILD_TYPE}"
  --cuda-arch "${CUDA_ARCH}"
  --jobs "${JOBS}"
  --ctest-timeout-sec "${CTEST_TIMEOUT_SEC}"
)

for arg in "${CMAKE_EXTRA_ARGS[@]}"; do
  matrix_args+=(--cmake-extra "${arg}")
done

log "Using LCG setup: ${LCG_SETUP}"
log "Build type: ${BUILD_TYPE}"
log "Build root: ${BUILD_ROOT}"
log "CUDA arch: ${CUDA_ARCH}"
log "Build jobs: ${JOBS}"
log "Running Jenkins-like nightly CI selection"

# Increase self-hosted nightly validation concurrency and buffer capacity unless explicitly overridden.
: "${ADEPT_VALIDATION_DEFAULT_NUM_THREADS:=32}"
: "${ADEPT_VALIDATION_DEFAULT_NUM_TRACKSLOTS:=6}"
: "${ADEPT_VALIDATION_DEFAULT_NUM_HITSLOTS:=30}"
: "${ADEPT_VALIDATION_DEFAULT_CPU_CAPACITY_FACTOR:=5.0}"
: "${ADEPT_VALIDATION_REGIONS_NUM_THREADS:=32}"
: "${ADEPT_VALIDATION_REGIONS_NUM_TRACKSLOTS:=6}"
: "${ADEPT_VALIDATION_REGIONS_NUM_HITSLOTS:=30}"
: "${ADEPT_VALIDATION_REGIONS_CPU_CAPACITY_FACTOR:=5.0}"
: "${ADEPT_VALIDATION_WDT_NUM_THREADS:=32}"
: "${ADEPT_VALIDATION_WDT_NUM_TRACKSLOTS:=8}"
: "${ADEPT_VALIDATION_WDT_NUM_HITSLOTS:=28}"
: "${ADEPT_VALIDATION_WDT_CPU_CAPACITY_FACTOR:=5.0}"
export ADEPT_VALIDATION_DEFAULT_NUM_THREADS
export ADEPT_VALIDATION_DEFAULT_NUM_TRACKSLOTS
export ADEPT_VALIDATION_DEFAULT_NUM_HITSLOTS
export ADEPT_VALIDATION_DEFAULT_CPU_CAPACITY_FACTOR
export ADEPT_VALIDATION_REGIONS_NUM_THREADS
export ADEPT_VALIDATION_REGIONS_NUM_TRACKSLOTS
export ADEPT_VALIDATION_REGIONS_NUM_HITSLOTS
export ADEPT_VALIDATION_REGIONS_CPU_CAPACITY_FACTOR
export ADEPT_VALIDATION_WDT_NUM_THREADS
export ADEPT_VALIDATION_WDT_NUM_TRACKSLOTS
export ADEPT_VALIDATION_WDT_NUM_HITSLOTS
export ADEPT_VALIDATION_WDT_CPU_CAPACITY_FACTOR

NIGHTLY_STATUS=1
if "${MATRIX_RUNNER}" "${matrix_args[@]}"; then
  NIGHTLY_STATUS=0
fi

write_results

log "Nightly status: ${NIGHTLY_STATUS}"

exit "${NIGHTLY_STATUS}"
