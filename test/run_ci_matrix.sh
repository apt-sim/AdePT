#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

# Tracked local matrix runner for AdePT CI-like checks.
# Default mode runs fast physics_drift tests for async/split/mixed.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/ci_common.sh"

DEFAULT_MASTER_REF="auto"

BUILD_TYPE="Release"
SUITE="drift"
CONFIG_LIST="async,split,mixed"
BUILD_ROOT="${SCRIPT_DIR}/build"
MASTER_REF="${DEFAULT_MASTER_REF}"
FETCH_MASTER=1
KEEP_WORKTREE=1
FORCE_REBUILD=0
FORCE_MASTER_UPDATE=0
LCG_SETUP=""
CUDA_ARCH="auto"
JOBS="auto"
G4VG_DIR="${G4VG_DIR:-}"
MASTER_COMMIT=""
CTEST_TIMEOUT_SEC=0
declare -a CMAKE_EXTRA_ARGS=()

log() {
  printf '[matrix-ci] %s\n' "$*"
}

die() {
  printf '[matrix-ci] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Runs AdePT in a local matrix similar to Jenkins:
  - async (default flags)
  - split (-DADEPT_USE_SPLIT_KERNELS=ON)
  - mixed (-DADEPT_MIXED_PRECISION=ON)

Options:
  --suite <drift|ci>         Test suite to run (default: drift)
  --configs <list>           Comma list: async,split,mixed (default: async,split,mixed)
  --build-type <type>        CMake build type (default: Release)
  --cuda-arch <arch|auto>    CUDA arch (default: auto)
  --jobs <N|auto>            Parallel build jobs (default: auto = nproc)
  --lcg-setup <path>         Explicit LCG setup script (default: devAdePT EL9 view)
  --build-root <path>        Build root directory (default: ${BUILD_ROOT})
  --master-ref <git-ref>     Master reference for drift comparisons (default: ${DEFAULT_MASTER_REF})
  --no-fetch-master          Do not fetch remote master refs before running drift
  --keep-worktree            Keep master worktree cache (default: ON)
  --no-keep-worktree         Remove master worktree after run
  --force-rebuild            Remove build directories before reconfiguring
  --refresh-master           Recreate the cached master worktree
  --g4vg-dir <path>          Path containing G4VGConfig.cmake (disables FetchContent G4VG)
  --cmake-extra <arg>        Additional CMake arg (repeatable)
  --ctest-timeout-sec <sec>  Optional per-test timeout passed to ctest --timeout (default: 0 = disabled)
  -h, --help                 Show help

Suite behavior:
  drift (default): builds PR + master for each config and runs physics_drift_* tests
  ci              : runs Jenkins-like selection for each config (can take much longer)
USAGE
}

select_lcg_setup() {
  LCG_SETUP=$(select_devadept_lcg_setup "${LCG_SETUP}") || die "Failed to resolve devAdePT LCG setup"
}

fetch_remote_master_ref() {
  local remote_name=$1

  [[ "${FETCH_MASTER}" -eq 1 ]] || return
  git -C "${REPO_ROOT}" remote get-url "${remote_name}" >/dev/null 2>&1 || return

  log "Fetching ${remote_name}/master"
  if ! git -C "${REPO_ROOT}" fetch --no-tags "${remote_name}" \
      +refs/heads/master:refs/remotes/"${remote_name}"/master >/dev/null 2>&1; then
    log "Fetch failed for ${remote_name}/master; falling back to local refs"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      [[ $# -ge 2 ]] || die "Missing value for --suite"
      SUITE="$2"
      shift 2
      ;;
    --configs)
      [[ $# -ge 2 ]] || die "Missing value for --configs"
      CONFIG_LIST="$2"
      shift 2
      ;;
    --build-type)
      [[ $# -ge 2 ]] || die "Missing value for --build-type"
      BUILD_TYPE="$2"
      shift 2
      ;;
    --cuda-arch)
      [[ $# -ge 2 ]] || die "Missing value for --cuda-arch"
      CUDA_ARCH="$2"
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || die "Missing value for --jobs"
      JOBS="$2"
      shift 2
      ;;
    --lcg-setup)
      [[ $# -ge 2 ]] || die "Missing value for --lcg-setup"
      LCG_SETUP="$2"
      shift 2
      ;;
    --build-root)
      [[ $# -ge 2 ]] || die "Missing value for --build-root"
      BUILD_ROOT="$2"
      shift 2
      ;;
    --master-ref)
      [[ $# -ge 2 ]] || die "Missing value for --master-ref"
      MASTER_REF="$2"
      shift 2
      ;;
    --no-fetch-master)
      FETCH_MASTER=0
      shift
      ;;
    --keep-worktree)
      KEEP_WORKTREE=1
      shift
      ;;
    --no-keep-worktree)
      KEEP_WORKTREE=0
      shift
      ;;
    --force-rebuild)
      FORCE_REBUILD=1
      shift
      ;;
    --refresh-master)
      FORCE_MASTER_UPDATE=1
      shift
      ;;
    --g4vg-dir)
      [[ $# -ge 2 ]] || die "Missing value for --g4vg-dir"
      G4VG_DIR="$2"
      shift 2
      ;;
    --cmake-extra)
      [[ $# -ge 2 ]] || die "Missing value for --cmake-extra"
      CMAKE_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --ctest-timeout-sec)
      [[ $# -ge 2 ]] || die "Missing value for --ctest-timeout-sec"
      CTEST_TIMEOUT_SEC="$2"
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

[[ "${CTEST_TIMEOUT_SEC}" =~ ^[0-9]+$ ]] || die "Invalid --ctest-timeout-sec '${CTEST_TIMEOUT_SEC}'. Expected a non-negative integer."
if [[ "${JOBS}" != "auto" && ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  die "Invalid --jobs '${JOBS}'. Expected 'auto' or a positive integer."
fi

case "${SUITE}" in
  drift|ci)
    ;;
  *)
    die "Invalid --suite '${SUITE}'. Allowed: drift, ci"
    ;;
esac

BUILD_TARGET="integrationTest"
if [[ "${SUITE}" == "ci" ]]; then
  BUILD_TARGET="all"
fi

IFS=',' read -r -a CONFIGS <<< "${CONFIG_LIST}"
[[ ${#CONFIGS[@]} -gt 0 ]] || die "No configs provided"

for cfg in "${CONFIGS[@]}"; do
  case "${cfg}" in
    async|split|mixed)
      ;;
    *)
      die "Invalid config '${cfg}'. Allowed: async, split, mixed"
      ;;
  esac
done

select_lcg_setup

# shellcheck disable=SC1090
set +u
source "${LCG_SETUP}"
set -u
normalize_lcg_cuda_env || die "Failed to normalize CUDA environment from LCG setup"

if [[ "${CUDA_ARCH}" == "auto" ]]; then
  detected=""
  if [[ -x /usr/local/cuda/extras/demo_suite/deviceQuery ]]; then
    detected=$(/usr/local/cuda/extras/demo_suite/deviceQuery 2>/dev/null | awk '/CUDA Capability/ {print $NF; exit}' | tr -d '.')
  fi
  if [[ -z "${detected}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    detected=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .' || true)
  fi
  if [[ ! "${detected}" =~ ^[0-9]+$ ]]; then
    detected=""
  fi
  if [[ -z "${detected}" ]]; then
    detected="75"
    log "Could not detect CUDA capability; using fallback ${detected}"
  fi
  CUDA_ARCH="${detected}"
fi

if [[ "${JOBS}" == "auto" ]]; then
  JOBS=$(nproc)
fi

# Intentionally do not auto-detect repository-local G4VG builds.
# They may be linked against a different Geant4 toolchain and can silently
# contaminate runtime library resolution.
# Use --g4vg-dir explicitly when a non-builtin G4VG is desired.
if [[ -n "${G4VG_DIR}" ]]; then
  [[ -f "${G4VG_DIR}/G4VGConfig.cmake" ]] || die "G4VGConfig.cmake not found under ${G4VG_DIR}"
fi

mkdir -p "${BUILD_ROOT}"

declare -a COMMON_CMAKE_ARGS
COMMON_CMAKE_ARGS=(
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
  "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
  "-DADEPT_BUILD_TESTING=ON"
)
if [[ -n "${G4VG_DIR}" ]]; then
  COMMON_CMAKE_ARGS+=("-DADEPT_USE_BUILTIN_G4VG=OFF" "-DG4VG_DIR=${G4VG_DIR}")
fi
COMMON_CMAKE_ARGS+=("${CMAKE_EXTRA_ARGS[@]}")

config_suffix() {
  case "$1" in
    async) echo "MONOL" ;;
    split) echo "SPLIT_ON" ;;
    mixed) echo "MIXED_PRECISION" ;;
  esac
}

build_config() {
  local role=$1
  local source_dir=$2
  local build_dir=$3
  local cfg=$4

  local -a cfg_args=()
  case "${cfg}" in
    async)
      ;;
    split)
      cfg_args+=("-DADEPT_USE_SPLIT_KERNELS=ON")
      ;;
    mixed)
      cfg_args+=("-DADEPT_MIXED_PRECISION=ON")
      ;;
    *)
      die "Internal error: unsupported config '${cfg}'"
      ;;
  esac

  if [[ "${FORCE_REBUILD}" -eq 1 && -d "${build_dir}" ]]; then
    log "Removing existing build directory for ${role}/${cfg}: ${build_dir}"
    rm -rf "${build_dir}"
  fi

  log "Configuring ${cfg} (${role}) in ${build_dir}"
  cmake -S "${source_dir}" -B "${build_dir}" "${COMMON_CMAKE_ARGS[@]}" "${cfg_args[@]}"

  log "Building target '${BUILD_TARGET}' (${cfg}, ${role}, -j${JOBS})"
  cmake --build "${build_dir}" --target "${BUILD_TARGET}" -j"${JOBS}"

  local exe="${build_dir}/BuildProducts/bin/integrationTest"
  [[ -x "${exe}" ]] || die "Build completed but executable missing: ${exe}"
}

resolve_master_ref() {
  if [[ "${MASTER_REF}" != "auto" ]]; then
    case "${MASTER_REF}" in
      upstream/master)
        fetch_remote_master_ref upstream
        ;;
      origin/master)
        fetch_remote_master_ref origin
        ;;
    esac

    if git -C "${REPO_ROOT}" rev-parse --verify --quiet "${MASTER_REF}^{commit}" >/dev/null; then
      echo "${MASTER_REF}"
      return
    fi

    die "Could not resolve requested master reference '${MASTER_REF}'"
  fi

  fetch_remote_master_ref upstream
  fetch_remote_master_ref origin

  local ref
  for ref in upstream/master origin/master master main HEAD; do
    if git -C "${REPO_ROOT}" rev-parse --verify --quiet "${ref}^{commit}" >/dev/null; then
      echo "${ref}"
      return
    fi
  done

  die "Could not resolve a master reference"
}

MASTER_WORKTREE_DIR="${BUILD_ROOT}/AdePT_master_reference"
cleanup() {
  if [[ "${KEEP_WORKTREE}" -eq 0 && -d "${MASTER_WORKTREE_DIR}" ]]; then
    git -C "${REPO_ROOT}" worktree remove --force "${MASTER_WORKTREE_DIR}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

prepare_master_worktree() {
  local ref=$1
  local target_commit current_commit
  target_commit=$(git -C "${REPO_ROOT}" rev-parse "${ref}^{commit}")
  MASTER_COMMIT="${target_commit}"

  if [[ -d "${MASTER_WORKTREE_DIR}" ]]; then
    current_commit=$(git -C "${MASTER_WORKTREE_DIR}" rev-parse HEAD 2>/dev/null || true)
    if [[ "${FORCE_MASTER_UPDATE}" -eq 0 && "${current_commit}" == "${target_commit}" ]]; then
      log "Reusing master worktree at ${MASTER_WORKTREE_DIR} (${target_commit:0:12})"
      git -C "${MASTER_WORKTREE_DIR}" submodule update --init >/dev/null
      return
    fi
    git -C "${REPO_ROOT}" worktree remove --force "${MASTER_WORKTREE_DIR}" >/dev/null 2>&1 || rm -rf "${MASTER_WORKTREE_DIR}"
  fi

  log "Creating master worktree at ${MASTER_WORKTREE_DIR} from ${ref} (${target_commit:0:12})"
  git -C "${REPO_ROOT}" worktree add --force "${MASTER_WORKTREE_DIR}" "${target_commit}" >/dev/null
  git -C "${MASTER_WORKTREE_DIR}" submodule update --init >/dev/null
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

run_drift_tests() {
  local pr_build_dir=$1
  local master_build_dir=$2

  local pr_exec="${pr_build_dir}/BuildProducts/bin/integrationTest"
  local master_exec="${master_build_dir}/BuildProducts/bin/integrationTest"

  [[ -x "${pr_exec}" ]] || die "PR executable not found: ${pr_exec}"
  [[ -x "${master_exec}" ]] || die "Master executable not found: ${master_exec}"

  local drift_count
  drift_count=$(run_ctest --test-dir "${pr_build_dir}" -N -R '^physics_drift_' | awk '/Total Tests:/ {print $3}')
  if [[ -z "${drift_count}" || "${drift_count}" == "0" ]]; then
    die "No physics_drift tests were found in ${pr_build_dir}"
  fi

  log "Running physics_drift tests in ${pr_build_dir}"
  ADEPT_MASTER_EXECUTABLE="${master_exec}" \
  run_ctest --test-dir "${pr_build_dir}" --output-on-failure -R '^physics_drift_' -j1
}

run_ci_subset_tests() {
  local cfg=$1
  local build_dir=$2

  case "${cfg}" in
    async)
      log "Running monolithic CI subset: unit + validation"
      run_ctest --test-dir "${build_dir}" --output-on-failure -L unit -j1
      run_ctest --test-dir "${build_dir}" --output-on-failure -L validation -j1
      ;;
    split)
      log "Running split CI subset: validation"
      run_ctest --test-dir "${build_dir}" --output-on-failure -L validation -j1
      ;;
    mixed)
      log "Skipping mixed CI subset tests (no mixed non-drift stage in current Jenkins pipeline)"
      ;;
    *)
      die "Internal error: unsupported config '${cfg}'"
      ;;
  esac
}

log "Repository root: ${REPO_ROOT}"
log "Suite: ${SUITE}"
log "Configs: ${CONFIG_LIST}"
log "Build root: ${BUILD_ROOT}"
log "Build type: ${BUILD_TYPE}"
log "CUDA arch: ${CUDA_ARCH}"
log "Build jobs: ${JOBS}"
log "LCG setup: ${LCG_SETUP}"
log "Master ref request: ${MASTER_REF}"
log "CTest timeout (sec): ${CTEST_TIMEOUT_SEC}"

if [[ "${SUITE}" == "drift" ]]; then
  resolved_ref=$(resolve_master_ref)
  master_commit=$(git -C "${REPO_ROOT}" rev-parse "${resolved_ref}^{commit}")
  head_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)
  if [[ "${master_commit}" == "${head_commit}" ]]; then
    log "Warning: master reference (${resolved_ref}) matches current HEAD"
  fi
  prepare_master_worktree "${resolved_ref}"
fi

for cfg in "${CONFIGS[@]}"; do
  suffix=$(config_suffix "${cfg}")
  pr_build_dir="${BUILD_ROOT}/BUILD_${suffix}"

  build_config "pr" "${REPO_ROOT}" "${pr_build_dir}" "${cfg}"

  if [[ "${SUITE}" == "drift" ]]; then
    master_build_dir="${BUILD_ROOT}/BUILD_MASTER_REFERENCE_${suffix}"
    build_config "master" "${MASTER_WORKTREE_DIR}" "${master_build_dir}" "${cfg}"
    run_drift_tests "${pr_build_dir}" "${master_build_dir}"
  else
    run_ci_subset_tests "${cfg}" "${pr_build_dir}"
  fi
done

log "Matrix run completed successfully"
