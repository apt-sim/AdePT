#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

DEFAULT_BUILD_ROOT="/tmp/adept-athena-performance"
DEFAULT_ARTIFACTS_DIR="${REPO_ROOT}/athena-performance-results"

MODE=""
BUILD_ROOT="${DEFAULT_BUILD_ROOT}"
ARTIFACTS_DIR="${DEFAULT_ARTIFACTS_DIR}"
RESULTS_FILE=""

ADEPT_REPOSITORY=""
ADEPT_REF=""

ATHENA_REPOSITORY="https://gitlab.cern.ch/atlas/athena.git"
ATHENA_BASE_REF="main"
ATHENA_GPU_REPOSITORY=""
ATHENA_GPU_REF=""

ATLAS_EXTERNALS_REPOSITORY="https://gitlab.cern.ch/atlas/atlasexternals.git"
ATLAS_EXTERNALS_BASE_REF="21.1.X-simGPU"

THREADS="96"
EVENTS="1000"
REPETITIONS="5"

ADEPT_SHORT_SHA=""
RUN_LABEL=""
MODE_LABEL=""
RUN_DIR_LABEL=""

ATHENA_WORKTREE=""
ATHENA_HEAD_SHA=""
GPU_BUILD_DIR=""
GPU_BUILD_RELATIVE=""
BUILD_LOG=""
RUN_DIR=""
SUMMARY_MARKDOWN_FILE=""
ATLAS_EXTERNALS_LOCAL_REPO=""
ATLAS_EXTERNALS_LOCAL_REF=""
ATLAS_EXTERNALS_LOCAL_SHA=""
OVERALL_REAL_MEAN_SEC=""
OVERALL_REAL_STDDEV_SEC=""
PER_RUN_REAL_TIMES_CSV=""
PER_RUN_MEASUREMENT_COUNTS_CSV=""

PERF_STATUS=1

log() {
  printf '[athena-perf] %s\n' "$*"
}

die() {
  printf '[athena-perf] ERROR: %s\n' "$*" >&2
  exit 1
}

retry_command() {
  local attempts=${1:-3}
  shift

  local try
  for try in $(seq 1 "${attempts}"); do
    if "$@"; then
      return 0
    fi
    if [[ "${try}" -lt "${attempts}" ]]; then
      log "Command failed (attempt ${try}/${attempts}): $*"
      sleep 10
    fi
  done

  return 1
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs the Athena+AtlasExternals performance benchmark for the checked-out AdePT
revision and writes a markdown summary suitable for GitHub Actions.

Required options:
  --mode <mono|split>            Benchmark mode to run
  --adept-repository <owner/repo>
                                  AdePT repository to build from
  --adept-ref <git-ref>          AdePT ref/SHA to build from
  --athena-gpu-repository <url>  Athena GPU branch repository/clone URL
  --athena-gpu-ref <git-ref>     Athena GPU branch to merge on top of main

Optional options:
  --build-root <path>            Working root (default: ${DEFAULT_BUILD_ROOT})
  --artifacts-dir <path>         Directory for logs and summary output
  --results-file <path>          Write KEY=VALUE results for workflow consumption
  --athena-repository <url>      Athena base repository
  --athena-base-ref <ref>        Athena base branch/ref (default: ${ATHENA_BASE_REF})
  --atlasexternals-repository <url>
                                  AtlasExternals base repository
  --atlasexternals-base-ref <ref>
                                  AtlasExternals base branch/ref (default: ${ATLAS_EXTERNALS_BASE_REF})
  --threads <N>                  ATHENA_CORE_NUMBER and T<N> log tag (default: ${THREADS})
  --events <N>                   Number of events per run (default: ${EVENTS})
  --repetitions <N>              Number of repeated runs (default: ${REPETITIONS})
  -h, --help                     Show this help
EOF
}

verify_remote_ref() {
  local repo_url=$1
  local ref=$2

  retry_command 3 git ls-remote --exit-code "${repo_url}" \
    "${ref}" "refs/heads/${ref}" "refs/tags/${ref}" >/dev/null 2>&1
}

preflight_checks() {
  command -v git >/dev/null 2>&1 || die "git is required"
  command -v python3 >/dev/null 2>&1 || die "python3 is required"

  [[ -r /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh ]] || \
    die "ATLASLocalRootBase setup is not visible inside the container"
  [[ -r /cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/CampaignInputs/mc23/EVNT/mc23_13p6TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.evgen.EVNT.e8514/EVNT.32288062._002040.pool.root.1 ]] || \
    die "Benchmark EVNT input is not visible inside the container"

  if [[ -d /afs ]]; then
    if ls /afs >/dev/null 2>&1; then
      log "/afs is visible and readable inside the container"
    else
      log "/afs is mounted inside the container but is not readable with the current credentials"
    fi
  else
    log "/afs is not visible inside the container"
  fi

  log "Checking remote access to Athena base repository"
  verify_remote_ref "${ATHENA_REPOSITORY}" "${ATHENA_BASE_REF}" || \
    die "Could not resolve ${ATHENA_REPOSITORY}@${ATHENA_BASE_REF}"

  log "Checking remote access to Athena GPU merge repository"
  verify_remote_ref "${ATHENA_GPU_REPOSITORY}" "${ATHENA_GPU_REF}" || \
    die "Could not resolve ${ATHENA_GPU_REPOSITORY}@${ATHENA_GPU_REF}"

  log "Checking remote access to AtlasExternals repository"
  verify_remote_ref "${ATLAS_EXTERNALS_REPOSITORY}" "${ATLAS_EXTERNALS_BASE_REF}" || \
    die "Could not resolve ${ATLAS_EXTERNALS_REPOSITORY}@${ATLAS_EXTERNALS_BASE_REF}"
}

write_results() {
  [[ -n "${RESULTS_FILE}" ]] || return

  {
    printf 'PERF_STATUS=%s\n' "${PERF_STATUS}"
    printf 'MODE=%q\n' "${MODE}"
    printf 'BUILD_ROOT=%q\n' "${BUILD_ROOT}"
    printf 'ARTIFACTS_DIR=%q\n' "${ARTIFACTS_DIR}"
    printf 'ADEPT_REPOSITORY=%q\n' "${ADEPT_REPOSITORY}"
    printf 'ADEPT_REF=%q\n' "${ADEPT_REF}"
    printf 'ATHENA_REPOSITORY=%q\n' "${ATHENA_REPOSITORY}"
    printf 'ATHENA_BASE_REF=%q\n' "${ATHENA_BASE_REF}"
    printf 'ATHENA_GPU_REPOSITORY=%q\n' "${ATHENA_GPU_REPOSITORY}"
    printf 'ATHENA_GPU_REF=%q\n' "${ATHENA_GPU_REF}"
    printf 'ATHENA_HEAD_SHA=%q\n' "${ATHENA_HEAD_SHA}"
    printf 'ATLAS_EXTERNALS_REPOSITORY=%q\n' "${ATLAS_EXTERNALS_REPOSITORY}"
    printf 'ATLAS_EXTERNALS_BASE_REF=%q\n' "${ATLAS_EXTERNALS_BASE_REF}"
    printf 'ATLAS_EXTERNALS_LOCAL_SHA=%q\n' "${ATLAS_EXTERNALS_LOCAL_SHA}"
    printf 'THREADS=%q\n' "${THREADS}"
    printf 'EVENTS=%q\n' "${EVENTS}"
    printf 'REPETITIONS=%q\n' "${REPETITIONS}"
    printf 'BUILD_LOG=%q\n' "${BUILD_LOG}"
    printf 'RUN_DIR=%q\n' "${RUN_DIR}"
    printf 'SUMMARY_MARKDOWN_FILE=%q\n' "${SUMMARY_MARKDOWN_FILE}"
    printf 'OVERALL_REAL_MEAN_SEC=%q\n' "${OVERALL_REAL_MEAN_SEC}"
    printf 'OVERALL_REAL_STDDEV_SEC=%q\n' "${OVERALL_REAL_STDDEV_SEC}"
    printf 'PER_RUN_REAL_TIMES_CSV=%q\n' "${PER_RUN_REAL_TIMES_CSV}"
    printf 'PER_RUN_MEASUREMENT_COUNTS_CSV=%q\n' "${PER_RUN_MEASUREMENT_COUNTS_CSV}"
  } > "${RESULTS_FILE}"
}

copy_runtime_logs() {
  [[ -d "${RUN_DIR}" ]] || return
  mkdir -p "${ARTIFACTS_DIR}/run-logs"

  find "${RUN_DIR}" -maxdepth 1 -type f \
    \( -name "adept_T*.log" -o -name "log.AtlasG4Tf_AdePT_T*" -o -name "run_adept.sh" -o -name "run_all_5.sh" \) \
    -exec cp {} "${ARTIFACTS_DIR}/run-logs/" \;
}

write_failure_summary() {
  local exit_code=$1

  mkdir -p "${ARTIFACTS_DIR}"
  SUMMARY_MARKDOWN_FILE="${ARTIFACTS_DIR}/summary.md"

  cat > "${SUMMARY_MARKDOWN_FILE}" <<EOF
### Athena Performance Benchmark (${MODE})

The benchmark failed before a complete timing summary could be produced.

- Mode: \`${MODE}\`
- AdePT: \`${ADEPT_REPOSITORY}@${ADEPT_REF}\`
- Athena base: \`${ATHENA_REPOSITORY}@${ATHENA_BASE_REF}\`
- Athena GPU merge: \`${ATHENA_GPU_REPOSITORY}@${ATHENA_GPU_REF}\`
- AtlasExternals base: \`${ATLAS_EXTERNALS_REPOSITORY}@${ATLAS_EXTERNALS_BASE_REF}\`
- Exit code: \`${exit_code}\`
- Build log: \`${BUILD_LOG}\`
EOF
}

on_exit() {
  local exit_code=$1
  trap - EXIT

  if [[ "${exit_code}" -ne 0 ]]; then
    PERF_STATUS="${exit_code}"
  fi

  mkdir -p "${ARTIFACTS_DIR}"
  copy_runtime_logs || true

  if [[ -z "${SUMMARY_MARKDOWN_FILE}" || ! -f "${SUMMARY_MARKDOWN_FILE}" ]]; then
    write_failure_summary "${exit_code}" || true
  fi

  write_results || true
  exit "${exit_code}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || die "Missing value for --mode"
      MODE=$2
      shift 2
      ;;
    --build-root)
      [[ $# -ge 2 ]] || die "Missing value for --build-root"
      BUILD_ROOT=$2
      shift 2
      ;;
    --artifacts-dir)
      [[ $# -ge 2 ]] || die "Missing value for --artifacts-dir"
      ARTIFACTS_DIR=$2
      shift 2
      ;;
    --results-file)
      [[ $# -ge 2 ]] || die "Missing value for --results-file"
      RESULTS_FILE=$2
      shift 2
      ;;
    --adept-repository)
      [[ $# -ge 2 ]] || die "Missing value for --adept-repository"
      ADEPT_REPOSITORY=$2
      shift 2
      ;;
    --adept-ref)
      [[ $# -ge 2 ]] || die "Missing value for --adept-ref"
      ADEPT_REF=$2
      shift 2
      ;;
    --athena-repository)
      [[ $# -ge 2 ]] || die "Missing value for --athena-repository"
      ATHENA_REPOSITORY=$2
      shift 2
      ;;
    --athena-base-ref)
      [[ $# -ge 2 ]] || die "Missing value for --athena-base-ref"
      ATHENA_BASE_REF=$2
      shift 2
      ;;
    --athena-gpu-repository)
      [[ $# -ge 2 ]] || die "Missing value for --athena-gpu-repository"
      ATHENA_GPU_REPOSITORY=$2
      shift 2
      ;;
    --athena-gpu-ref)
      [[ $# -ge 2 ]] || die "Missing value for --athena-gpu-ref"
      ATHENA_GPU_REF=$2
      shift 2
      ;;
    --atlasexternals-repository)
      [[ $# -ge 2 ]] || die "Missing value for --atlasexternals-repository"
      ATLAS_EXTERNALS_REPOSITORY=$2
      shift 2
      ;;
    --atlasexternals-base-ref)
      [[ $# -ge 2 ]] || die "Missing value for --atlasexternals-base-ref"
      ATLAS_EXTERNALS_BASE_REF=$2
      shift 2
      ;;
    --threads)
      [[ $# -ge 2 ]] || die "Missing value for --threads"
      THREADS=$2
      shift 2
      ;;
    --events)
      [[ $# -ge 2 ]] || die "Missing value for --events"
      EVENTS=$2
      shift 2
      ;;
    --repetitions)
      [[ $# -ge 2 ]] || die "Missing value for --repetitions"
      REPETITIONS=$2
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

[[ "${MODE}" == "mono" || "${MODE}" == "split" ]] || die "Expected --mode mono or split"
[[ -n "${ADEPT_REPOSITORY}" ]] || die "--adept-repository is required"
[[ -n "${ADEPT_REF}" ]] || die "--adept-ref is required"
[[ -n "${ATHENA_GPU_REPOSITORY}" ]] || die "--athena-gpu-repository is required"
[[ -n "${ATHENA_GPU_REF}" ]] || die "--athena-gpu-ref is required"
[[ "${THREADS}" =~ ^[1-9][0-9]*$ ]] || die "Invalid --threads '${THREADS}'"
[[ "${EVENTS}" =~ ^[1-9][0-9]*$ ]] || die "Invalid --events '${EVENTS}'"
[[ "${REPETITIONS}" =~ ^[1-9][0-9]*$ ]] || die "Invalid --repetitions '${REPETITIONS}'"

case "${MODE}" in
  mono)
    MODE_LABEL="Mono_NoLeaks"
    RUN_DIR_LABEL="Mono_NoLeaks"
    ;;
  split)
    MODE_LABEL="Split_NoLeaks"
    RUN_DIR_LABEL="Split_NoLeaks"
    ;;
esac

ADEPT_SHORT_SHA=${ADEPT_REF:0:7}
RUN_LABEL="${MODE}-${ADEPT_SHORT_SHA}"

mkdir -p "${BUILD_ROOT}" "${ARTIFACTS_DIR}"
BUILD_ROOT=$(cd "${BUILD_ROOT}" && pwd)
ARTIFACTS_DIR=$(cd "${ARTIFACTS_DIR}" && pwd)

ATHENA_WORKTREE="${BUILD_ROOT}/athena"
GPU_BUILD_DIR="${BUILD_ROOT}/gpu_build_${MODE_LABEL}_${ADEPT_SHORT_SHA}"
GPU_BUILD_RELATIVE="../$(basename "${GPU_BUILD_DIR}")"
BUILD_LOG="${ARTIFACTS_DIR}/output_build.txt"
RUN_DIR="${BUILD_ROOT}/runs_validation_GammaRR/${RUN_DIR_LABEL}"
SUMMARY_MARKDOWN_FILE="${ARTIFACTS_DIR}/summary.md"

trap 'on_exit $?' EXIT

log "Mode: ${MODE}"
log "Build root: ${BUILD_ROOT}"
log "Artifacts dir: ${ARTIFACTS_DIR}"
log "AdePT under test: ${ADEPT_REPOSITORY}@${ADEPT_REF}"
log "Athena base: ${ATHENA_REPOSITORY}@${ATHENA_BASE_REF}"
log "Athena GPU merge: ${ATHENA_GPU_REPOSITORY}@${ATHENA_GPU_REF}"

git_checkout_ref() {
  local repo_dir=$1
  local branch_name=$2
  local ref=$3

  if git -C "${repo_dir}" rev-parse --verify --quiet "origin/${ref}^{commit}" >/dev/null; then
    git -C "${repo_dir}" checkout -B "${branch_name}" "origin/${ref}" >/dev/null
  else
    git -C "${repo_dir}" checkout -B "${branch_name}" "${ref}" >/dev/null
  fi
}

clone_athena() {
  retry_command 3 git clone "${ATHENA_REPOSITORY}" "${ATHENA_WORKTREE}" >/dev/null
  git_checkout_ref "${ATHENA_WORKTREE}" ci-base "${ATHENA_BASE_REF}"
  git -C "${ATHENA_WORKTREE}" remote add adept-gpu "${ATHENA_GPU_REPOSITORY}"
  retry_command 3 git -C "${ATHENA_WORKTREE}" fetch --no-tags adept-gpu "${ATHENA_GPU_REF}" >/dev/null
  git -C "${ATHENA_WORKTREE}" merge --no-edit FETCH_HEAD >/dev/null
  ATHENA_HEAD_SHA=$(git -C "${ATHENA_WORKTREE}" rev-parse HEAD)
}

prepare_local_atlasexternals() {
  local repo_dir="${BUILD_ROOT}/atlasexternals-under-test"
  local cmake_file branch_name

  retry_command 3 git clone "${ATLAS_EXTERNALS_REPOSITORY}" "${repo_dir}" >/dev/null
  git_checkout_ref "${repo_dir}" ci-base "${ATLAS_EXTERNALS_BASE_REF}"

  cmake_file="${repo_dir}/External/AdePT/CMakeLists.txt"
  python3 - "${cmake_file}" "https://github.com/${ADEPT_REPOSITORY}.git" "${ADEPT_REF}" "${MODE}" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
adept_url = sys.argv[2]
adept_ref = sys.argv[3]
mode = sys.argv[4]
split_value = "TRUE" if mode == "split" else "FALSE"

text = path.read_text()

source_pattern = re.compile(
    r'set\(\s*ATLAS_ADEPT_SOURCE\s*.*?CACHE STRING "The source for AdePT" \)',
    re.S,
)
replacement = (
    'set( ATLAS_ADEPT_SOURCE\n'
    f'   "GIT_REPOSITORY;{adept_url};GIT_TAG;{adept_ref}"\n'
    '   CACHE STRING "The source for AdePT" )'
)
text, source_count = source_pattern.subn(replacement, text, count=1)
if source_count != 1:
    raise SystemExit("Could not rewrite ATLAS_ADEPT_SOURCE")

split_pattern = re.compile(r'-DADEPT_USE_SPLIT_KERNELS:BOOL=(TRUE|FALSE)')
text, split_count = split_pattern.subn(
    f'-DADEPT_USE_SPLIT_KERNELS:BOOL={split_value}',
    text,
    count=1,
)
if split_count != 1:
    raise SystemExit("Could not rewrite ADEPT_USE_SPLIT_KERNELS")

path.write_text(text)
PY

  branch_name="ci-athena-performance-${RUN_LABEL}"
  git -C "${repo_dir}" config user.name "AdePT Performance CI"
  git -C "${repo_dir}" config user.email "actions@users.noreply.github.com"
  git -C "${repo_dir}" checkout -B "${branch_name}" >/dev/null
  git -C "${repo_dir}" add External/AdePT/CMakeLists.txt
  git -C "${repo_dir}" commit -m "CI performance benchmark for ${ADEPT_REF}" >/dev/null

  ATLAS_EXTERNALS_LOCAL_REPO="${repo_dir}"
  ATLAS_EXTERNALS_LOCAL_REF="${branch_name}"
  ATLAS_EXTERNALS_LOCAL_SHA=$(git -C "${repo_dir}" rev-parse HEAD)
}

prepare_athena_gpu_env() {
  export ATLAS_LOCAL_ROOT_BASE="/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase"
  # shellcheck disable=SC1091
  source "${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh" --quiet
  asetup none,gcc14.2,cmakesetup
  export ATLAS_NIGHTLY_PLATFORM="${BINARY_TAG:-${LCG_PLATFORM:-${CMTCONFIG}}}"
  export ATLAS_NIGHTLY_G4PATH="${ATLAS_NIGHTLY_G4PATH:-/cvmfs/atlas-nightlies.cern.ch/repo/sw/main--simGPU_AthSimulation_${ATLAS_NIGHTLY_PLATFORM}/Geant4}"
  export G4PATH="${ATLAS_NIGHTLY_G4PATH}"
}

build_athena() {
  (
    prepare_athena_gpu_env
    export AtlasExternals_URL="${ATLAS_EXTERNALS_LOCAL_REPO}"
    export AtlasExternals_REF="${ATLAS_EXTERNALS_LOCAL_REF}"
    cd "${ATHENA_WORKTREE}"
    ./Projects/AthSimulation/build_gpu.sh -b "${GPU_BUILD_RELATIVE}" > "${BUILD_LOG}" 2>&1
  )
}

rewrite_setup_run() {
  cat > "${GPU_BUILD_DIR}/setup_run.sh" <<EOF
export ATLAS_LOCAL_ROOT_BASE="/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase"
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
asetup none,gcc14.2,cmakesetup
export ATLAS_NIGHTLY_PLATFORM=\${BINARY_TAG:-\${LCG_PLATFORM:-\${CMTCONFIG}}}
export ATLAS_NIGHTLY_G4PATH=\${ATLAS_NIGHTLY_G4PATH:-/cvmfs/atlas-nightlies.cern.ch/repo/sw/main--simGPU_AthSimulation_\${ATLAS_NIGHTLY_PLATFORM}/Geant4}
export G4PATH=\${ATLAS_NIGHTLY_G4PATH}
source ${ATHENA_WORKTREE}/Projects/AthSimulation/build_env.sh -b ${GPU_BUILD_DIR}/externals
source ${GPU_BUILD_DIR}/build/\${LCG_PLATFORM}/setup.sh
EOF
  chmod +x "${GPU_BUILD_DIR}/setup_run.sh"
}

prepare_run_dir() {
  mkdir -p "${RUN_DIR}"
  cp "${REPO_ROOT}/test/athena-performance/run_all_5.sh" "${RUN_DIR}/run_all_5.sh"
  cp "${REPO_ROOT}/test/athena-performance/${MODE}/run_adept.sh" "${RUN_DIR}/run_adept.sh"
  chmod +x "${RUN_DIR}/run_all_5.sh" "${RUN_DIR}/run_adept.sh"
}

run_benchmark() {
  (
    set -euo pipefail
    # shellcheck disable=SC1091
    source "${GPU_BUILD_DIR}/setup_run.sh"
    cd "${RUN_DIR}"
    export ADEPT_MAX_EVENTS="${EVENTS}"
    export ADEPT_REPETITIONS="${REPETITIONS}"
    export ADEPT_OUTPUT_HITS_FILE="test.CA.HITS.pool_AdePT_E${EVENTS}.root"
    ./run_all_5.sh "${THREADS}" "${REPETITIONS}"
  )
}

summarize_results() {
  local env_file="${ARTIFACTS_DIR}/summary.env"

  python3 - "${RUN_DIR}" "${THREADS}" "${REPETITIONS}" "${MODE}" "${ADEPT_REPOSITORY}" "${ADEPT_REF}" \
    "${ATHENA_REPOSITORY}" "${ATHENA_BASE_REF}" "${ATHENA_HEAD_SHA}" \
    "${ATHENA_GPU_REPOSITORY}" "${ATHENA_GPU_REF}" \
    "${ATLAS_EXTERNALS_REPOSITORY}" "${ATLAS_EXTERNALS_BASE_REF}" "${ATLAS_EXTERNALS_LOCAL_SHA}" \
    "${SUMMARY_MARKDOWN_FILE}" "${BUILD_LOG}" "${GPU_BUILD_DIR}/setup_run.sh" <<'PY' > "${env_file}"
import math
import pathlib
import re
import statistics
import sys

(
    run_dir,
    threads,
    repetitions,
    mode,
    adept_repo,
    adept_ref,
    athena_repo,
    athena_base_ref,
    athena_head_sha,
    athena_gpu_repo,
    athena_gpu_ref,
    atlasexternals_repo,
    atlasexternals_base_ref,
    atlasexternals_sha,
    summary_path,
    build_log,
    setup_run_path,
) = sys.argv[1:]

run_dir_path = pathlib.Path(run_dir)
summary_path = pathlib.Path(summary_path)
real_re = re.compile(r"Real=([0-9.]+)s")
log_pattern = f"log.AtlasG4Tf_AdePT_T{threads}_run*"

per_run = []
for atlas_log in sorted(run_dir_path.glob(log_pattern)):
    values = []
    for line in atlas_log.read_text(errors="ignore").splitlines():
        match = real_re.search(line)
        if match:
            values.append(float(match.group(1)))
    if len(values) <= 1:
        continue
    trimmed = values[1:]
    per_run.append((atlas_log.name, sum(trimmed) / len(trimmed), len(trimmed)))

if not per_run:
    raise SystemExit("No benchmark logs with parseable Real= measurements were found")

means = [item[1] for item in per_run]
overall_mean = statistics.mean(means)
overall_stddev = statistics.stdev(means) if len(means) > 1 else 0.0

mode_label = "Mono_NoLeaks" if mode == "mono" else "Split_NoLeaks"
build_cmd = f"./Projects/AthSimulation/build_gpu.sh -b ../gpu_build_{mode_label}_{adept_ref[:7]} > output_build.txt"
run_cmd = f"bash -lc 'source {setup_run_path} && cd {run_dir} && ./run_all_5.sh {threads} {repetitions}'"

lines = [
    f"### Athena Performance Benchmark ({mode})",
    "",
    f"- AdePT: `{adept_repo}@{adept_ref}`",
    f"- Athena base: `{athena_repo}@{athena_base_ref}`",
    f"- Athena merged head: `{athena_head_sha}`",
    f"- Athena GPU merge source: `{athena_gpu_repo}@{athena_gpu_ref}`",
    f"- AtlasExternals base: `{atlasexternals_repo}@{atlasexternals_base_ref}`",
    f"- AtlasExternals benchmark commit: `{atlasexternals_sha}`",
    f"- Build command: `{build_cmd}`",
    f"- Run command: `{run_cmd}`",
    f"- Configured repetitions: `{repetitions}`",
    (
        "- Runtime extraction: per-run average of `Real=` measurements with the first "
        "measurement skipped, matching "
        "`grep 'Real=' ... | sed ... | awk 'NR>1 { sum += $1; n++ } ...'`."
    ),
    f"- Build log: `{build_log}`",
    "",
    "| Run log | Average real time (s) | Measurements |",
    "| --- | ---: | ---: |",
]

for log_name, mean_value, measurement_count in per_run:
    lines.append(f"| `{log_name}` | {mean_value:.3f} | {measurement_count} |")

lines.extend(
    [
        "",
        f"- Overall average real time: `{overall_mean:.3f} s`",
        f"- Overall standard deviation: `{overall_stddev:.3f} s`",
    ]
)

summary_path.write_text("\n".join(lines) + "\n")

print(f"OVERALL_REAL_MEAN_SEC={overall_mean:.6f}")
print(f"OVERALL_REAL_STDDEV_SEC={overall_stddev:.6f}")
print("PER_RUN_REAL_TIMES_CSV=" + ",".join(f"{value:.6f}" for value in means))
print("PER_RUN_MEASUREMENT_COUNTS_CSV=" + ",".join(str(item[2]) for item in per_run))
PY

  # shellcheck disable=SC1090
  source "${env_file}"
}

preflight_checks
clone_athena
prepare_local_atlasexternals
build_athena
rewrite_setup_run
prepare_run_dir
run_benchmark
summarize_results

PERF_STATUS=0
log "Overall average real time: ${OVERALL_REAL_MEAN_SEC}s"
