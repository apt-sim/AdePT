#! /usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

# Build master in separate build directories (async, split, mixed) and compare
# deterministic validation output against the corresponding PR executables.

set -eu -o pipefail

WORKSPACE_DIR=$1
PR_SOURCE_DIR=$2
BUILD_TYPE=$3
CUDA_CAPABILITY=$4

MASTER_SOURCE_DIR="${WORKSPACE_DIR}/AdePT_master_reference"
CI_TMP_BASE_DIR="${WORKSPACE_DIR}/master_vs_pr_validation_tmp"

cleanup_master_reference() {
  git -C "${PR_SOURCE_DIR}" worktree remove --force "${MASTER_SOURCE_DIR}" >/dev/null 2>&1 || true
  rm -rf "${WORKSPACE_DIR}/BUILD_MASTER_REFERENCE_ASYNC_ON" \
         "${WORKSPACE_DIR}/BUILD_MASTER_REFERENCE_SPLIT_ON" \
         "${WORKSPACE_DIR}/BUILD_MASTER_REFERENCE_MIXED_PRECISION" \
         "${CI_TMP_BASE_DIR}"
}

run_config_comparison() {
  local config_name=$1
  local pr_executable=$2
  shift 2
  local extra_cmake_options=("$@")

  local master_binary_dir="${WORKSPACE_DIR}/BUILD_MASTER_REFERENCE_${config_name}"
  local ci_tmp_dir="${CI_TMP_BASE_DIR}/${config_name}"

  if [ ! -x "${pr_executable}" ]; then
    echo "Could not find PR integrationTest executable for ${config_name}: ${pr_executable}"
    exit 1
  fi

  cmake -S "${MASTER_SOURCE_DIR}" -B "${master_binary_dir}" \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_CAPABILITY}" \
        -DADEPT_BUILD_TESTING=ON \
        "${extra_cmake_options[@]}"
  cmake --build "${master_binary_dir}" --target integrationTest -j"$(nproc)"

  local master_executable
  master_executable="$(find "${master_binary_dir}" -type f -path '*/BuildProducts/bin/integrationTest' | head -n 1)"
  if [ -z "${master_executable}" ] || [ ! -x "${master_executable}" ]; then
    echo "Could not find master integrationTest executable for ${config_name} in ${master_binary_dir}"
    exit 1
  fi

  bash "${PR_SOURCE_DIR}/test/regression/scripts/validation_master_vs_pr.sh" \
       "${master_executable}" \
       "${pr_executable}" \
       "${MASTER_SOURCE_DIR}" \
       "${PR_SOURCE_DIR}" \
       "${PR_SOURCE_DIR}/test/regression/scripts" \
       "${ci_tmp_dir}"
}

trap cleanup_master_reference EXIT
cleanup_master_reference

git -C "${PR_SOURCE_DIR}" fetch --no-tags origin +refs/heads/master:refs/remotes/origin/master
git -C "${PR_SOURCE_DIR}" worktree add --force "${MASTER_SOURCE_DIR}" origin/master
git -C "${MASTER_SOURCE_DIR}" submodule update --init

run_config_comparison "ASYNC_ON" \
  "${WORKSPACE_DIR}/BUILD_ASYNC_ON/BuildProducts/bin/integrationTest"
run_config_comparison "SPLIT_ON" \
  "${WORKSPACE_DIR}/BUILD_SPLIT_ON/BuildProducts/bin/integrationTest" \
  "-DADEPT_USE_SPLIT_KERNELS=ON"
run_config_comparison "MIXED_PRECISION" \
  "${WORKSPACE_DIR}/BUILD_MIXED_PRECISION/BuildProducts/bin/integrationTest" \
  "-DADEPT_MIXED_PRECISION=ON"

trap - EXIT
cleanup_master_reference
