#! /usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

# Run physics_drift comparisons using already-built PR/master integrationTest
# executables in matrix build directories.
# Arguments:
#   1: workspace directory (contains AdePT checkout and BUILD_* folders)
#   2: CDash model name

set -eu -o pipefail

WORKSPACE_DIR=$1
MODEL_NAME=$2

SOURCE_DIR="${WORKSPACE_DIR}/AdePT"

if [ ! -d "${SOURCE_DIR}" ]; then
  echo "Source directory not found: ${SOURCE_DIR}"
  exit 2
fi

run_config_comparison() {
  local config_name=$1
  local ctest_binary_dir=$2
  local master_binary_dir=$3

  local pr_executable="${ctest_binary_dir}/BuildProducts/bin/integrationTest"
  local master_executable="${master_binary_dir}/BuildProducts/bin/integrationTest"

  if [ ! -x "${pr_executable}" ]; then
    echo "Could not find PR integrationTest executable for ${config_name}: ${pr_executable}"
    return 1
  fi
  if [ ! -x "${master_executable}" ]; then
    echo "Could not find master integrationTest executable for ${config_name}: ${master_executable}"
    return 1
  fi

  # Run drift tests registered in the PR build tree (CTest discovery).
  # The test script receives the master executable via environment so it can
  # compare PR output against master output for the same scenario/macro.
  if ! (
      ADEPT_MASTER_EXECUTABLE="${master_executable}" \
      CMAKE_SOURCE_DIR="${SOURCE_DIR}" \
      CMAKE_BINARY_DIR="${ctest_binary_dir}" \
      PHYSICS_DRIFT_CONFIG="${config_name}" \
      ctest -V --output-on-failure -S "${SOURCE_DIR}/jenkins/adept-ctest-physics-drift.cmake,${MODEL_NAME}"
    ); then
    echo "physics_drift mismatches detected for ${config_name}"
    return 1
  fi

  return 0
}

overall_status=0

# PR and master binaries must be built beforehand in these matrix directories.
run_config_comparison "MONOL" "${WORKSPACE_DIR}/BUILD_MONOL" "${WORKSPACE_DIR}/BUILD_MASTER_REFERENCE_MONOL" || overall_status=1
run_config_comparison "SPLIT_ON" "${WORKSPACE_DIR}/BUILD_SPLIT_ON" "${WORKSPACE_DIR}/BUILD_MASTER_REFERENCE_SPLIT_ON" || overall_status=1
run_config_comparison "MIXED_PRECISION" "${WORKSPACE_DIR}/BUILD_MIXED_PRECISION" "${WORKSPACE_DIR}/BUILD_MASTER_REFERENCE_MIXED_PRECISION" || overall_status=1

if [ "${overall_status}" -ne 0 ]; then
  echo "physics_drift detected differences in at least one configuration."
else
  echo "All physics_drift tests passed in all configurations."
fi

exit "${overall_status}"
