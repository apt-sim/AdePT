#! /usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

# PR-vs-master physics drift check for one scenario.
# Master executable is provided via environment:
#   ADEPT_MASTER_EXECUTABLE

set -eu -o pipefail

# Read input parameters
PR_EXECUTABLE=$1
PR_SOURCE_DIR=$2
CI_TEST_DIR=$3
CI_TMP_DIR=$4
SCENARIO=$5

MASTER_EXECUTABLE=${ADEPT_MASTER_EXECUTABLE:-}

if [ -z "${MASTER_EXECUTABLE}" ]; then
  echo "Error: physics_drift '${SCENARIO}' requires ADEPT_MASTER_EXECUTABLE."
  echo "Set ADEPT_MASTER_EXECUTABLE to the integrationTest executable path of the master branch."
  echo "Example:"
  echo "  ADEPT_MASTER_EXECUTABLE=/path/to/build_of_master_branch/BuildProducts/bin/integrationTest ctest --output-on-failure -R drift"
  exit 2
fi

if [ ! -x "${MASTER_EXECUTABLE}" ]; then
  echo "Error: ADEPT_MASTER_EXECUTABLE is not an executable file: ${MASTER_EXECUTABLE}"
  exit 2
fi
NUM_THREADS=1
NUM_EVENTS=1
NUM_TRACKSLOTS=1
NUM_HITSLOTS=1
NUM_LEAKSLOTS=1
GUN_NUMBER=20
ADEPT_SEED=1234567

REGIONS_LIST="caloregion, Layer1, Layer2, Layer3, Layer4, Layer5, Layer6, Layer7, Layer8, Layer9, Layer10,\
            Layer11, Layer12, Layer13, Layer14, Layer15, Layer16, Layer17, Layer18, Layer19, Layer20,\
            Layer31, Layer32, Layer33, Layer34, Layer35, Layer36, Layer37, Layer38, Layer39, Layer40,\
            Layer41, Layer42, Layer43, Layer44, Layer45, Layer46, Layer47, Layer48, Layer49, Layer50"
WDT_REGIONS_LIST="WDT_Region_layers_10_40,Layer5,Layer44,Layer45"

cleanup() {
  rm -rf "${CI_TMP_DIR}"
}

trap cleanup EXIT
cleanup

SCENARIO_GDML=""
SCENARIO_TRACK_IN_ALL_REGIONS=""
SCENARIO_REGIONS=""
SCENARIO_WDT_REGIONS=""
SCENARIO_FIELD="0 0 0"

case "${SCENARIO}" in
  default)
    SCENARIO_GDML="testEm3.gdml"
    SCENARIO_TRACK_IN_ALL_REGIONS="True"
    ;;
  regions)
    SCENARIO_GDML="testEm3_regions.gdml"
    SCENARIO_TRACK_IN_ALL_REGIONS="False"
    SCENARIO_REGIONS="${REGIONS_LIST}"
    ;;
  wdt)
    SCENARIO_GDML="testEm3_wdt.gdml"
    SCENARIO_TRACK_IN_ALL_REGIONS="True"
    SCENARIO_WDT_REGIONS="${WDT_REGIONS_LIST}"
    ;;
  bfield)
    SCENARIO_GDML="testEm3.gdml"
    SCENARIO_TRACK_IN_ALL_REGIONS="True"
    SCENARIO_FIELD="0 0 1.0"
    ;;
  *)
    echo "Unknown scenario '${SCENARIO}'. Supported: default, regions, wdt, bfield"
    exit 2
    ;;
esac

generate_validation_macro() {
  local source_dir=$1
  local output_macro=$2
  local gdml_path=$3
  local track_in_all_regions=$4
  local regions_list=$5
  local wdt_regions_list=$6
  local detector_field=$7

  "${CI_TEST_DIR}/python_scripts/macro_generator.py" \
      --template "${CI_TEST_DIR}/example_template.mac" \
      --output "${output_macro}" \
      --gdml_name "${source_dir}/examples/data/${gdml_path}" \
      --num_threads "${NUM_THREADS}" \
      --num_events "${NUM_EVENTS}" \
      --num_trackslots "${NUM_TRACKSLOTS}" \
      --num_hitslots "${NUM_HITSLOTS}" \
      --num_leakslots "${NUM_LEAKSLOTS}" \
      --track_in_all_regions "${track_in_all_regions}" \
      --gun_type setDefault \
      --gun_number "${GUN_NUMBER}" \
      --adept_seed "${ADEPT_SEED}" \
      --regions "${regions_list}" \
      --wdt_regions "${wdt_regions_list}" \
	  --detector_field "${detector_field}"
}

SCENARIO_TMP_DIR="${CI_TMP_DIR}/${SCENARIO}"
MASTER_SCENARIO_DIR="${SCENARIO_TMP_DIR}/master"
PR_SCENARIO_DIR="${SCENARIO_TMP_DIR}/pr"
SCENARIO_MACRO="${SCENARIO_TMP_DIR}/${SCENARIO}.mac"
MASTER_OUTPUT="master_${SCENARIO}"
PR_OUTPUT="pr_${SCENARIO}"

mkdir -p "${MASTER_SCENARIO_DIR}" "${PR_SCENARIO_DIR}"

# Use one macro for both binaries to ensure identical runtime input.
generate_validation_macro "${PR_SOURCE_DIR}" "${SCENARIO_MACRO}" \
  "${SCENARIO_GDML}" "${SCENARIO_TRACK_IN_ALL_REGIONS}" "${SCENARIO_REGIONS}" "${SCENARIO_WDT_REGIONS}" "${SCENARIO_FIELD}"

"${MASTER_EXECUTABLE}" --do_validation --allsensitive --accumulated_events \
                       -m "${SCENARIO_MACRO}" \
                       --output_dir "${MASTER_SCENARIO_DIR}" \
                       --output_file "${MASTER_OUTPUT}"

"${PR_EXECUTABLE}" --do_validation --allsensitive --accumulated_events \
                   -m "${SCENARIO_MACRO}" \
                   --output_dir "${PR_SCENARIO_DIR}" \
                   --output_file "${PR_OUTPUT}"

"${CI_TEST_DIR}/python_scripts/check_reproducibility.py" \
  --file1 "${MASTER_SCENARIO_DIR}/${MASTER_OUTPUT}.csv" \
  --file2 "${PR_SCENARIO_DIR}/${PR_OUTPUT}.csv" \
  --tol 0.0

echo "Scenario '${SCENARIO}' matched exactly."
