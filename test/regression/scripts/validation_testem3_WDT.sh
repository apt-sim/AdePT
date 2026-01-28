#! /usr/bin/env bash

# SPDX-FileCopyrightText: 2025 CERN
# SPDX-License-Identifier: Apache-2.0

# This is a CI test for validation. 10 GeV electrons are shot at the testEm3 geometry and the energy deposition is validated
# against a high-statistics benchmark file

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

# use gun_type hepmc or setDefault
$CI_TEST_DIR/python_scripts/macro_generator.py \
    --template ${CI_TEST_DIR}/example_template.mac \
    --output ${CI_TMP_DIR}/validation_testem3_WDT.mac \
    --gdml_name ${PROJECT_SOURCE_DIR}/examples/data/testEm3_wdt.gdml \
    --num_threads 8 \
    --num_events 400 \
    --num_trackslots 3 \
    --num_hitslots 12 \
    --num_leakslots 3 \
    --track_in_all_regions True\
    --gun_type setDefault\
    --wdt_regions "WDT_Region_layers_10_40,Layer5,Layer44,Layer45"

# run test
$ADEPT_EXECUTABLE --do_validation --allsensitive --accumulated_events \
                  -m "${CI_TMP_DIR}/validation_testem3_WDT.mac" \
                  --output_dir "${CI_TMP_DIR}" \
                  --output_file "adept_em3_2.5e4_e-"

# Validating the relative error per layer
$CI_TEST_DIR/python_scripts/check_validation.py --file1 ${CI_TMP_DIR}/adept_em3_2.5e4_e-.csv \
                                                --file2 ${CI_TEST_DIR}/benchmark_files/g4hepem_em3_10e7_e-.csv \
                                                --n1 4e4 --n2 1e7 --tol 0.01 \
                                                # --plot_file plot.png # uncomment to plot the validation plot

