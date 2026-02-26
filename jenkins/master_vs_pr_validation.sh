#! /usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

# Build master in separate build directories (async, split, mixed) and run the
# dedicated CTest physics_drift tests against corresponding PR executables.

set -eu -o pipefail

WORKSPACE_DIR=$1
PR_SOURCE_DIR=$2
BUILD_TYPE=$3
CUDA_CAPABILITY=$4
MODEL_NAME=$5

MASTER_SOURCE_DIR="${WORKSPACE_DIR}/AdePT_master_reference"

cleanup_master_reference() {
  git -C "${PR_SOURCE_DIR}" worktree remove --force "${MASTER_SOURCE_DIR}" >/dev/null 2>&1 || true
  rm -rf "${WORKSPACE_DIR}/BUILD_MASTER_REFERENCE_ASYNC_ON" \
         "${WORKSPACE_DIR}/BUILD_MASTER_REFERENCE_SPLIT_ON" \
         "${WORKSPACE_DIR}/BUILD_MASTER_REFERENCE_MIXED_PRECISION"
}

run_config_comparison() {
  local config_name=$1
  local pr_binary_dir=$2
  shift 2
  local extra_cmake_options=("$@")

  local master_binary_dir="${WORKSPACE_DIR}/BUILD_MASTER_REFERENCE_${config_name}"
  local pr_executable="${pr_binary_dir}/BuildProducts/bin/integrationTest"

  if [ ! -x "${pr_executable}" ]; then
    echo "Could not find PR integrationTest executable for ${config_name}: ${pr_executable}"
    return 1
  fi
  if [ ! -d "${pr_binary_dir}" ]; then
    echo "Could not find PR build directory for ${config_name}: ${pr_binary_dir}"
    return 1
  fi

  if ! cmake -S "${MASTER_SOURCE_DIR}" -B "${master_binary_dir}" \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_CAPABILITY}" \
        -DADEPT_BUILD_TESTING=ON \
        "${extra_cmake_options[@]}"; then
    echo "Failed to configure master reference build for ${config_name}"
    return 1
  fi
  if ! cmake --build "${master_binary_dir}" --target integrationTest -j"$(nproc)"; then
    echo "Failed to build master reference integrationTest for ${config_name}"
    return 1
  fi

  local master_executable
  master_executable="$(find "${master_binary_dir}" -type f -path '*/BuildProducts/bin/integrationTest' | head -n 1)"
  if [ -z "${master_executable}" ] || [ ! -x "${master_executable}" ]; then
    echo "Could not find master integrationTest executable for ${config_name} in ${master_binary_dir}"
    return 1
  fi

  if ! (
      ADEPT_MASTER_EXECUTABLE="${master_executable}" \
      ADEPT_MASTER_SOURCE_DIR="${MASTER_SOURCE_DIR}" \
      CMAKE_SOURCE_DIR="${PR_SOURCE_DIR}" \
      CMAKE_BINARY_DIR="${pr_binary_dir}" \
      PHYSICS_DRIFT_CONFIG="${config_name}" \
      ctest -V --output-on-failure -S "${PR_SOURCE_DIR}/jenkins/adept-ctest-physics-drift.cmake,${MODEL_NAME}"
    ); then
    echo "physics_drift mismatches detected for ${config_name}"
    return 1
  fi

  return 0
}

trap cleanup_master_reference EXIT
cleanup_master_reference

git -C "${PR_SOURCE_DIR}" fetch --no-tags origin +refs/heads/master:refs/remotes/origin/master
git -C "${PR_SOURCE_DIR}" worktree add --force "${MASTER_SOURCE_DIR}" origin/master
git -C "${MASTER_SOURCE_DIR}" submodule update --init

overall_status=0

run_config_comparison "ASYNC_ON" "${WORKSPACE_DIR}/BUILD_ASYNC_ON" || overall_status=1
run_config_comparison "SPLIT_ON" "${WORKSPACE_DIR}/BUILD_SPLIT_ON" \
  "-DADEPT_USE_SPLIT_KERNELS=ON" || overall_status=1
run_config_comparison "MIXED_PRECISION" "${WORKSPACE_DIR}/BUILD_MIXED_PRECISION" \
  "-DADEPT_MIXED_PRECISION=ON" || overall_status=1

if [ "${overall_status}" -ne 0 ]; then
  echo "physics_drift detected differences in at least one configuration."
else
  echo "All physics_drift tests passed in all configurations."
fi

trap - EXIT
cleanup_master_reference
exit "${overall_status}"
