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
# trap cleanup EXIT
# called it directly ensure clean environment
cleanup

# Create temporary directory
mkdir -p ${CI_TMP_DIR}

# generate macro
$CI_TEST_DIR/python_scripts/macro_generator.py \
    --template ${CI_TEST_DIR}/example_template.mac \
    --output ${CI_TMP_DIR}/reproducibility_regions.mac \
    --gdml_name ${PROJECT_SOURCE_DIR}/examples/data/testEm3_regions.gdml \
    --num_threads 4 \
    --num_events 50 \
    --num_trackslots 3 \
    --num_leakslots 0.3 \
    --num_hitslots 15 \
    --track_in_all_regions False\
    --gun_type setDefault\
    --regions "caloregion, Layer1, Layer2, Layer3, Layer4, Layer5, Layer6, Layer7, Layer8, Layer9, Layer10,\
            Layer11, Layer12, Layer13, Layer14, Layer15, Layer16, Layer17, Layer18, Layer19, Layer20,\
            Layer31, Layer32, Layer33, Layer34, Layer35, Layer36, Layer37, Layer38, Layer39, Layer40,\
            Layer41, Layer42, Layer43, Layer44, Layer45, Layer46, Layer47, Layer48, Layer49, Layer50"

# Choose a small num_leakslots value in order to test the throttling mechanism

# run test
$ADEPT_EXECUTABLE --do_validation --accumulated_events -m ${CI_TMP_DIR}/reproducibility_regions.mac --output_dir ${CI_TMP_DIR} --output_file testem3_run1 --allsensitive
$ADEPT_EXECUTABLE --do_validation --accumulated_events -m ${CI_TMP_DIR}/reproducibility_regions.mac --output_dir ${CI_TMP_DIR} --output_file testem3_run2 --allsensitive

# allow for small rounding error of 1e-6 due to summation per thread
$CI_TEST_DIR/python_scripts/check_reproducibility.py --file1 ${CI_TMP_DIR}/testem3_run1.csv \
                                                     --file2 ${CI_TMP_DIR}/testem3_run2.csv \
                                                     --tol 1e-5