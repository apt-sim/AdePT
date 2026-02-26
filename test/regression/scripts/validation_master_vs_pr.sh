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

run_with_output_on_failure() {
  local step_name=$1
  shift
  local step_log="${CI_TMP_DIR}/${step_name}.log"

  if ! "$@" >"${step_log}" 2>&1; then
    echo "Step '${step_name}' failed. Command output:"
    cat "${step_log}"
    return 1
  fi

  rm -f "${step_log}"
}

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

run_master_pr_comparison() {
  local scenario=$1
  local gdml_path=$2
  local track_in_all_regions=$3
  local regions_list=$4
  local wdt_regions_list=$5
  local detector_field=$6

  local master_scenario_dir="${MASTER_TMP_DIR}/${scenario}"
  local pr_scenario_dir="${PR_TMP_DIR}/${scenario}"
  local master_macro="${master_scenario_dir}/master_${scenario}.mac"
  local pr_macro="${pr_scenario_dir}/pr_${scenario}.mac"
  local master_output="master_${scenario}"
  local pr_output="pr_${scenario}"

  mkdir -p "${master_scenario_dir}" "${pr_scenario_dir}"

  run_with_output_on_failure "generate_master_macro_${scenario}" \
    generate_validation_macro "${MASTER_SOURCE_DIR}" "${master_macro}" \
      "${gdml_path}" "${track_in_all_regions}" "${regions_list}" "${wdt_regions_list}" "${detector_field}"

  run_with_output_on_failure "generate_pr_macro_${scenario}" \
    generate_validation_macro "${PR_SOURCE_DIR}" "${pr_macro}" \
      "${gdml_path}" "${track_in_all_regions}" "${regions_list}" "${wdt_regions_list}" "${detector_field}"

  run_with_output_on_failure "run_master_validation_${scenario}" \
    "${MASTER_EXECUTABLE}" --do_validation --allsensitive --accumulated_events \
                           -m "${master_macro}" \
                           --output_dir "${master_scenario_dir}" \
                           --output_file "${master_output}"

  run_with_output_on_failure "run_pr_validation_${scenario}" \
    "${PR_EXECUTABLE}" --do_validation --allsensitive --accumulated_events \
                       -m "${pr_macro}" \
                       --output_dir "${pr_scenario_dir}" \
                       --output_file "${pr_output}"

  run_with_output_on_failure "compare_master_pr_outputs_${scenario}" \
    "${CI_TEST_DIR}/python_scripts/check_reproducibility.py" \
    --file1 "${master_scenario_dir}/${master_output}.csv" \
    --file2 "${pr_scenario_dir}/${pr_output}.csv" \
    --tol 0.0

  echo "Scenario '${scenario}' matched exactly."
}

run_master_pr_comparison "default" "testEm3.gdml" "True" "" "" "0 0 0"
run_master_pr_comparison "regions" "testEm3_regions.gdml" "False" "${REGIONS_LIST}" "" "0 0 0"
run_master_pr_comparison "wdt" "testEm3_wdt.gdml" "True" "" "${WDT_REGIONS_LIST}" "0 0 0"
run_master_pr_comparison "bfield" "testEm3.gdml" "True" "" "" "0 0 1.0"

echo "All master-vs-PR mini validation scenarios matched exactly."
