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

# use gun_type hepmc or setDefault
$CI_TEST_DIR/python_scripts/macro_generator.py \
    --template ${CI_TEST_DIR}/example_template.mac \
    --output ${CI_TMP_DIR}/validation_testem3_regions.mac \
    --gdml_name ${PROJECT_SOURCE_DIR}/examples/data/testEm3_regions.gdml \
    --num_threads 4 \
    --num_events 1000 \
    --num_trackslots 2 \
    --num_hitslots 18 \
    --track_in_all_regions False\
    --gun_type setDefault\
    --regions "caloregion, Layer1, Layer2, Layer3, Layer4, Layer5, Layer6, Layer7, Layer8, Layer9, Layer10,\
            Layer12, Layer13, Layer14, Layer15, Layer16, Layer17, Layer18, Layer19, Layer20,\
            Layer21, Layer22, Layer23, Layer24, Layer25, Layer26, Layer27, Layer28, Layer29, Layer30,\
            Layer31, Layer32, Layer33, Layer34, Layer35, Layer36, Layer37, Layer38, Layer39, Layer40,\
            Layer41, Layer42, Layer43, Layer44, Layer45, Layer46, Layer47, Layer48, Layer49, Layer50"

# run test
$ADEPT_EXECUTABLE --do_validation --allsensitive --accumulated_events \
                  -m "${CI_TMP_DIR}/validation_testem3_regions.mac" \
                  --output_dir "${CI_TMP_DIR}" \
                  --output_file "adept_em3_1e5_e-"

# Validating the relative error per layer
# The saved G4 benchmark file was run with --noadept and 1e7 primary 10 GeV electrons.
# Comparing against 1e5 primary electrons (500 per event, 200 events) with AdePT should give errors below 1%.
# This is a compromise between run time and accuracy of the test 
$CI_TEST_DIR/python_scripts/check_validation.py --file1 ${CI_TMP_DIR}/adept_em3_1e5_e-.csv \
                                                --file2 ${CI_TEST_DIR}/benchmark_files/g4_em3_10e7_e-.csv \
                                                --n1 1e5 --n2 1e7 --tol 0.01 \
                                                # --plot_file plot.png # uncomment to plot the validation plot

