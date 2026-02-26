#! /usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

# Build master in a separate worktree.
# This is used for regression testing to compare deterministic validation
# output against the already built PR executable.

set -eu -o pipefail

WORKSPACE_DIR=$1
PR_SOURCE_DIR=$2
PR_EXECUTABLE=$3
BUILD_TYPE=$4
CUDA_CAPABILITY=$5

MASTER_SOURCE_DIR="${WORKSPACE_DIR}/AdePT_master_reference"
MASTER_BINARY_DIR="${WORKSPACE_DIR}/BUILD_MASTER_REFERENCE"
CI_TMP_DIR="${WORKSPACE_DIR}/master_vs_pr_validation_tmp"

cleanup_master_reference() {
  git -C "${PR_SOURCE_DIR}" worktree remove --force "${MASTER_SOURCE_DIR}" >/dev/null 2>&1 || true
  rm -rf "${MASTER_BINARY_DIR}" "${CI_TMP_DIR}"
}

trap cleanup_master_reference EXIT
cleanup_master_reference

if [ ! -x "${PR_EXECUTABLE}" ]; then
  echo "Could not find PR integrationTest executable at ${PR_EXECUTABLE}"
  exit 1
fi

git -C "${PR_SOURCE_DIR}" fetch --no-tags origin +refs/heads/master:refs/remotes/origin/master
git -C "${PR_SOURCE_DIR}" worktree add --force "${MASTER_SOURCE_DIR}" origin/master
git -C "${MASTER_SOURCE_DIR}" submodule update --init

cmake -S "${MASTER_SOURCE_DIR}" -B "${MASTER_BINARY_DIR}" \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_CAPABILITY}" \
      -DADEPT_BUILD_TESTING=ON
cmake --build "${MASTER_BINARY_DIR}" --target integrationTest -j"$(nproc)"

MASTER_EXECUTABLE="$(find "${MASTER_BINARY_DIR}" -type f -path '*/BuildProducts/bin/integrationTest' | head -n 1)"
if [ -z "${MASTER_EXECUTABLE}" ] || [ ! -x "${MASTER_EXECUTABLE}" ]; then
  echo "Could not find master integrationTest executable in ${MASTER_BINARY_DIR}"
  exit 1
fi

bash "${PR_SOURCE_DIR}/test/regression/scripts/validation_master_vs_pr.sh" \
     "${MASTER_EXECUTABLE}" \
     "${PR_EXECUTABLE}" \
     "${MASTER_SOURCE_DIR}" \
     "${PR_SOURCE_DIR}" \
     "${PR_SOURCE_DIR}/test/regression/scripts" \
     "${CI_TMP_DIR}"

trap - EXIT
cleanup_master_reference
