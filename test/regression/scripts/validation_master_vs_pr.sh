#! /usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

# PR-vs-master physics regression check:
# - generate deterministic low-stat macros for master and PR sources
# - run both builds with the same settings
# - require an exact match of the accumulated validation output

set -eu -o pipefail

# Read input parameters
MASTER_EXECUTABLE=$1
PR_EXECUTABLE=$2
MASTER_SOURCE_DIR=$3
PR_SOURCE_DIR=$4
CI_TEST_DIR=$5
CI_TMP_DIR=$6

MASTER_TMP_DIR="${CI_TMP_DIR}/master"
PR_TMP_DIR="${CI_TMP_DIR}/pr"
MASTER_MACRO="${MASTER_TMP_DIR}/master_validation_testem3.mac"
PR_MACRO="${PR_TMP_DIR}/pr_validation_testem3.mac"
MASTER_OUTPUT="master_em3_1evt_1particle"
PR_OUTPUT="pr_em3_1evt_1particle"

cleanup() {
  echo "Cleaning up temporary files..."
  rm -rf "${CI_TMP_DIR}"
}

trap cleanup EXIT
cleanup

mkdir -p "${MASTER_TMP_DIR}" "${PR_TMP_DIR}"

generate_validation_macro() {
  local source_dir=$1
  local output_macro=$2

  "${CI_TEST_DIR}/python_scripts/macro_generator.py" \
      --template "${CI_TEST_DIR}/example_template.mac" \
      --output "${output_macro}" \
      --gdml_name "${source_dir}/examples/data/testEm3.gdml" \
      --num_threads 1 \
      --num_events 1 \
      --num_trackslots 1 \
      --num_hitslots 1 \
      --num_leakslots 1 \
      --track_in_all_regions True \
      --gun_type setDefault \
      --gun_number 20 \
      --adept_seed 1234567
}

generate_validation_macro "${MASTER_SOURCE_DIR}" "${MASTER_MACRO}"
generate_validation_macro "${PR_SOURCE_DIR}" "${PR_MACRO}"

"${MASTER_EXECUTABLE}" --do_validation --allsensitive --accumulated_events \
                       -m "${MASTER_MACRO}" \
                       --output_dir "${MASTER_TMP_DIR}" \
                       --output_file "${MASTER_OUTPUT}"

"${PR_EXECUTABLE}" --do_validation --allsensitive --accumulated_events \
                   -m "${PR_MACRO}" \
                   --output_dir "${PR_TMP_DIR}" \
                   --output_file "${PR_OUTPUT}"

"${CI_TEST_DIR}/python_scripts/check_reproducibility.py" \
    --file1 "${MASTER_TMP_DIR}/${MASTER_OUTPUT}.csv" \
    --file2 "${PR_TMP_DIR}/${PR_OUTPUT}.csv" \
    --tol 0.0 --n1 20 --n2 20
