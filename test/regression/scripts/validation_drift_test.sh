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
NUM_TRACKSLOTS=3
NUM_HITSLOTS=12
NUM_LEAKSLOTS=3
GUN_NUMBER=20
ADEPT_SEED=1234567
# The ROOT truth path requires the user callbacks because the reconstructed
# MC-truth state is only available when AdePT is told to invoke them.
CALL_USER_STEPPING_ACTION=False
CALL_USER_TRACKING_ACTION=False

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
  local call_user_stepping_action=$8
  local call_user_tracking_action=$9

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
      --call_user_stepping_action "${call_user_stepping_action}" \
      --call_user_tracking_action "${call_user_tracking_action}" \
      --detector_field "${detector_field}"
}

supports_truth_root() {
  local executable=$1
  local output

  # The helper flag allows the drift script to decide whether it can run the
  # richer ROOT truth comparison, or needs to fall back to the legacy CSV-only
  # exact comparison.
  if output=$("${executable}" --supports_truth_root 2>/dev/null); then
    if [ "${output}" = "1" ]; then
      echo "1"
      return 0
    fi
  fi

  echo "0"
}

SCENARIO_TMP_DIR="${CI_TMP_DIR}/${SCENARIO}"
MASTER_SCENARIO_DIR="${SCENARIO_TMP_DIR}/master"
PR_SCENARIO_DIR="${SCENARIO_TMP_DIR}/pr"
SCENARIO_MACRO="${SCENARIO_TMP_DIR}/${SCENARIO}.mac"
MASTER_OUTPUT="master_${SCENARIO}"
PR_OUTPUT="pr_${SCENARIO}"

mkdir -p "${MASTER_SCENARIO_DIR}" "${PR_SCENARIO_DIR}"

PR_SUPPORTS_ROOT="$(supports_truth_root "${PR_EXECUTABLE}")"
MASTER_SUPPORTS_ROOT="$(supports_truth_root "${MASTER_EXECUTABLE}")"

if [ "${PR_SUPPORTS_ROOT}" = "1" ] && [ "${MASTER_SUPPORTS_ROOT}" = "1" ]; then
  # The ROOT truth path remains exact in MT because it compares merged
  # histogram populations, whereas the legacy CSV path still depends on the
  # order of floating-point accumulation and is therefore kept single-threaded.
  NUM_THREADS=4
  NUM_EVENTS=8
  CALL_USER_STEPPING_ACTION=True
  CALL_USER_TRACKING_ACTION=True
fi

# Use one macro for both binaries to ensure identical runtime input.
# The macro generator expands the scenario settings into an executable Geant4
# macro with the right geometry, field, AdePT options, and callback settings.
generate_validation_macro "${PR_SOURCE_DIR}" "${SCENARIO_MACRO}" \
  "${SCENARIO_GDML}" "${SCENARIO_TRACK_IN_ALL_REGIONS}" "${SCENARIO_REGIONS}" "${SCENARIO_WDT_REGIONS}" \
  "${SCENARIO_FIELD}" "${CALL_USER_STEPPING_ACTION}" "${CALL_USER_TRACKING_ACTION}"

if [ "${CALL_USER_STEPPING_ACTION}" = "True" ]; then
  # ROOT truth mode: produce aggregated histogram files for PR and master and
  # compare them semantically, histogram by histogram.
  "${MASTER_EXECUTABLE}" --allsensitive --accumulated_events --truth_root \
                         -m "${SCENARIO_MACRO}" \
                         --output_dir "${MASTER_SCENARIO_DIR}" \
                         --output_file "${MASTER_OUTPUT}"

  "${PR_EXECUTABLE}" --allsensitive --accumulated_events --truth_root \
                     -m "${SCENARIO_MACRO}" \
                     --output_dir "${PR_SCENARIO_DIR}" \
                     --output_file "${PR_OUTPUT}"

  python3 "${CI_TEST_DIR}/python_scripts/check_root_histograms.py" \
    --file1 "${MASTER_SCENARIO_DIR}/${MASTER_OUTPUT}.root" \
    --file2 "${PR_SCENARIO_DIR}/${PR_OUTPUT}.root"
else
  # FIXME: This legacy CSV fallback should be removed once the master/reference
  # executable also supports --truth_root. The drift gate should then require
  # ROOT truth support instead of silently keeping the old CSV-only mode alive.
  #
  # Legacy fallback for builds without ROOT support: keep the original exact
  # CSV edep drift test unchanged.
  "${MASTER_EXECUTABLE}" --allsensitive --accumulated_events \
                         -m "${SCENARIO_MACRO}" \
                         --output_dir "${MASTER_SCENARIO_DIR}" \
                         --output_file "${MASTER_OUTPUT}"

  "${PR_EXECUTABLE}" --allsensitive --accumulated_events \
                     -m "${SCENARIO_MACRO}" \
                     --output_dir "${PR_SCENARIO_DIR}" \
                     --output_file "${PR_OUTPUT}"

  "${CI_TEST_DIR}/python_scripts/check_reproducibility.py" \
    --file1 "${MASTER_SCENARIO_DIR}/${MASTER_OUTPUT}.csv" \
    --file2 "${PR_SCENARIO_DIR}/${PR_OUTPUT}.csv" \
    --tol 0.0
fi

echo "Scenario '${SCENARIO}' matched exactly."
