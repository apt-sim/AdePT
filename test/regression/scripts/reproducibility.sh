#! /usr/bin/env bash

# SPDX-FileCopyrightText: 2025 CERN
# SPDX-License-Identifier: Apache-2.0

# This is a CI test for reproducbility. The same 8 ttbar events are executed twice
# and then it is checked that the total energy deposition is exactly the same.

# abort on first encounted error
set -eu -o pipefail


# Read input parameters
ADEPT_EXECUTABLE=$1
PROJECT_BINARY_DIR=$2
PROJECT_SOURCE_DIR=$3
CI_TEST_DIR=$4
CI_TMP_DIR=$5

# Define cleanup function of temporary files
cleanup() {
  echo "Cleaning up temporary files..."
  rm -rf ${CI_TMP_DIR}
}

# register cleanup to be called on exit
trap cleanup EXIT
# called it directly ensure clean environment
cleanup

# Create temporary directory
mkdir -p ${CI_TMP_DIR}

# generate macro
$CI_TEST_DIR/python_scripts/macro_generator.py \
    --template ${CI_TEST_DIR}/example_template.mac \
    --output ${CI_TMP_DIR}/reproducibility.mac \
    --gdml_name ${PROJECT_SOURCE_DIR}/examples/data/cms2018_sd.gdml \
    --num_threads 4 \
    --num_events 8 \
    --num_trackslots 5 \
    --num_hitslots 4 \
    --num_leakslots 4 \
    --gun_type hepmc \
    --track_in_all_regions True\
    --event_file ${PROJECT_BINARY_DIR}/ppttbar.hepmc3


# run test
$ADEPT_EXECUTABLE --do_validation --accumulated_events -m ${CI_TMP_DIR}/reproducibility.mac --output_dir ${CI_TMP_DIR} --output_file cms_ttbar_run1
$ADEPT_EXECUTABLE --do_validation --accumulated_events -m ${CI_TMP_DIR}/reproducibility.mac --output_dir ${CI_TMP_DIR} --output_file cms_ttbar_run2 

# allow for small rounding error of 1e-6 due to summation per thread
$CI_TEST_DIR/python_scripts/check_reproducibility.py --file1 ${CI_TMP_DIR}/cms_ttbar_run1.csv \
                                                     --file2 ${CI_TMP_DIR}/cms_ttbar_run2.csv \
                                                     --tol 1e-5