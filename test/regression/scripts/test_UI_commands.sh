#! /usr/bin/env bash

# SPDX-FileCopyrightText: 2025 CERN
# SPDX-License-Identifier: Apache-2.0

# This is a CI test for testing that all UI commands work without failing

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
    --template ${CI_TEST_DIR}/test_ui_commands_template.mac \
    --output ${CI_TMP_DIR}/test_ui_commands.mac \
    --gdml_name ${PROJECT_SOURCE_DIR}/examples/data/testEm3_regions.gdml \


# run test
$ADEPT_EXECUTABLE -m "${CI_TMP_DIR}/test_ui_commands.mac" --do_validation

