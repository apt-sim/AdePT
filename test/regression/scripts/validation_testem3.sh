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

num_threads=${ADEPT_VALIDATION_DEFAULT_NUM_THREADS:-${ADEPT_VALIDATION_NUM_THREADS:-4}}
num_events=${ADEPT_VALIDATION_DEFAULT_NUM_EVENTS:-${ADEPT_VALIDATION_NUM_EVENTS:-600}}
num_trackslots=${ADEPT_VALIDATION_DEFAULT_NUM_TRACKSLOTS:-${ADEPT_VALIDATION_NUM_TRACKSLOTS:-3}}
num_hitslots=${ADEPT_VALIDATION_DEFAULT_NUM_HITSLOTS:-${ADEPT_VALIDATION_NUM_HITSLOTS:-12}}
cpu_capacity_factor=${ADEPT_VALIDATION_DEFAULT_CPU_CAPACITY_FACTOR:-${ADEPT_VALIDATION_CPU_CAPACITY_FACTOR:-2.5}}
gun_number=${ADEPT_VALIDATION_DEFAULT_GUN_NUMBER:-${ADEPT_VALIDATION_GUN_NUMBER:-100}}
nprimaries=$((num_events * gun_number))

echo "Validation settings (default): threads=${num_threads}, events=${num_events}, gun=${gun_number}, trackslots=${num_trackslots}, hitslots=${num_hitslots}, cpu_capacity_factor=${cpu_capacity_factor}"

# use gun_type hepmc or setDefault
$CI_TEST_DIR/python_scripts/macro_generator.py \
    --template ${CI_TEST_DIR}/example_template.mac \
    --output ${CI_TMP_DIR}/validation_testem3.mac \
    --gdml_name ${PROJECT_SOURCE_DIR}/examples/data/testEm3.gdml \
    --num_threads ${num_threads} \
    --num_events ${num_events} \
    --num_trackslots ${num_trackslots} \
    --num_hitslots ${num_hitslots} \
    --cpu_capacity_factor ${cpu_capacity_factor} \
    --gun_number ${gun_number} \
    --track_in_all_regions True\
    --gun_type setDefault 


# run test
$ADEPT_EXECUTABLE --allsensitive --accumulated_events \
                  -m "${CI_TMP_DIR}/validation_testem3.mac" \
                  --output_dir "${CI_TMP_DIR}" \
                  --output_file "adept_em3_2.5e4_e-"

# Validating the relative error per layer
# The saved G4 benchmark file was run with --noadept and 1e7 primary 10 GeV electrons.
# Comparing against the configured AdePT validation statistics should give errors below 1%.
# This is a compromise between run time and accuracy of the test 
$CI_TEST_DIR/python_scripts/check_validation.py --file1 ${CI_TMP_DIR}/adept_em3_2.5e4_e-.csv \
                                                --file2 ${CI_TEST_DIR}/benchmark_files/g4hepem_em3_10e7_e-.csv \
                                                --n1 ${nprimaries} --n2 1e7 --tol 0.01 \
                                                # --plot_file plot.png # uncomment to plot the validation plot
