#! /usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

# Validation coverage for /adept/directHandoff. The option makes results depend on the GPU timing, so this checks
# statistical physics agreement: the regions setup of testEm3_validation_regions (layers 21-30 on the CPU, so that
# every shower hands tracks back to Geant4) with direct handoff, against the G4HepEm reference.

set -eu -o pipefail

ADEPT_EXECUTABLE=$1
PROJECT_BINARY_DIR=$2
PROJECT_SOURCE_DIR=$3
CI_TEST_DIR=$4
CI_TMP_DIR=$5

cleanup() {
  echo "Cleaning up temporary files..."
  rm -rf "${CI_TMP_DIR}"
}

trap cleanup EXIT
cleanup

mkdir -p "${CI_TMP_DIR}"

num_threads=${ADEPT_VALIDATION_DIRECT_HANDOFF_NUM_THREADS:-${ADEPT_VALIDATION_NUM_THREADS:-8}}
num_events=${ADEPT_VALIDATION_DIRECT_HANDOFF_NUM_EVENTS:-${ADEPT_VALIDATION_NUM_EVENTS:-800}}
num_trackslots=${ADEPT_VALIDATION_DIRECT_HANDOFF_NUM_TRACKSLOTS:-${ADEPT_VALIDATION_NUM_TRACKSLOTS:-3}}
num_hitslots=${ADEPT_VALIDATION_DIRECT_HANDOFF_NUM_HITSLOTS:-${ADEPT_VALIDATION_NUM_HITSLOTS:-12}}
cpu_capacity_factor=${ADEPT_VALIDATION_DIRECT_HANDOFF_CPU_CAPACITY_FACTOR:-${ADEPT_VALIDATION_CPU_CAPACITY_FACTOR:-2.5}}
gun_number=${ADEPT_VALIDATION_DIRECT_HANDOFF_GUN_NUMBER:-${ADEPT_VALIDATION_GUN_NUMBER:-100}}
nprimaries=$((num_events * gun_number))

echo "Validation settings (direct_handoff): threads=${num_threads}, events=${num_events}, gun=${gun_number}, trackslots=${num_trackslots}, hitslots=${num_hitslots}, cpu_capacity_factor=${cpu_capacity_factor}"

"${CI_TEST_DIR}/python_scripts/macro_generator.py" \
    --template "${CI_TEST_DIR}/example_template.mac" \
    --output "${CI_TMP_DIR}/validation_testem3_direct_handoff.mac" \
    --gdml_name "${PROJECT_SOURCE_DIR}/examples/data/testEm3_regions.gdml" \
    --num_threads "${num_threads}" \
    --num_events "${num_events}" \
    --num_trackslots "${num_trackslots}" \
    --num_hitslots "${num_hitslots}" \
    --cpu_capacity_factor "${cpu_capacity_factor}" \
    --direct_handoff True \
    --gun_number "${gun_number}" \
    --track_in_all_regions False \
    --gun_type setDefault \
    --regions "caloregion, Layer1, Layer2, Layer3, Layer4, Layer5, Layer6, Layer7, Layer8, Layer9, Layer10,\
            Layer11, Layer12, Layer13, Layer14, Layer15, Layer16, Layer17, Layer18, Layer19, Layer20,\
            Layer31, Layer32, Layer33, Layer34, Layer35, Layer36, Layer37, Layer38, Layer39, Layer40,\
            Layer41, Layer42, Layer43, Layer44, Layer45, Layer46, Layer47, Layer48, Layer49, Layer50"

# Verbosity 2 emits the early-return message checked below.
sed -i 's|^/adept/setVerbosity 0$|/adept/setVerbosity 2|' \
  "${CI_TMP_DIR}/validation_testem3_direct_handoff.mac"
run_log="${CI_TMP_DIR}/validation_testem3_direct_handoff.log"

if ! "${ADEPT_EXECUTABLE}" --allsensitive --accumulated_events \
                      -m "${CI_TMP_DIR}/validation_testem3_direct_handoff.mac" \
                      --output_dir "${CI_TMP_DIR}" \
                      --output_file "adept_em3_direct_handoff_2.5e4_e-" >"${run_log}" 2>&1; then
  cat "${run_log}"
  exit 1
fi

if ! grep -qF 'returning to Geant4 with tracks handed back during the flush of event' "${run_log}"; then
  echo "The validation run did not exercise direct handoff."
  cat "${run_log}"
  exit 1
fi

"${CI_TEST_DIR}/python_scripts/check_validation.py" \
  --file1 "${CI_TMP_DIR}/adept_em3_direct_handoff_2.5e4_e-.csv" \
  --file2 "${CI_TEST_DIR}/benchmark_files/g4hepem_em3_10e7_e-.csv" \
  --n1 "${nprimaries}" --n2 1e7 --tol 0.01
