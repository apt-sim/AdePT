#! /usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

# Validation coverage for /adept/FinishLastNParticlesOnCPU. The option is
# population-dependent, so this checks statistical physics agreement.

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

num_threads=${ADEPT_VALIDATION_FINISH_ON_CPU_NUM_THREADS:-${ADEPT_VALIDATION_NUM_THREADS:-4}}
num_events=${ADEPT_VALIDATION_FINISH_ON_CPU_NUM_EVENTS:-300}
num_trackslots=${ADEPT_VALIDATION_FINISH_ON_CPU_NUM_TRACKSLOTS:-${ADEPT_VALIDATION_NUM_TRACKSLOTS:-3}}
num_hitslots=${ADEPT_VALIDATION_FINISH_ON_CPU_NUM_HITSLOTS:-${ADEPT_VALIDATION_NUM_HITSLOTS:-12}}
cpu_capacity_factor=${ADEPT_VALIDATION_FINISH_ON_CPU_CPU_CAPACITY_FACTOR:-${ADEPT_VALIDATION_CPU_CAPACITY_FACTOR:-2.5}}
gun_number=${ADEPT_VALIDATION_FINISH_ON_CPU_GUN_NUMBER:-200}
# Use a large threshold so lost CPU-returned tracks cannot hide in the
# low-energy shower tail.
finish_last_n_particles_on_cpu=${ADEPT_VALIDATION_FINISH_ON_CPU_N:-1000}
nprimaries=$((num_events * gun_number))

echo "Validation settings (finish_on_cpu): threads=${num_threads}, events=${num_events}, gun=${gun_number}, trackslots=${num_trackslots}, hitslots=${num_hitslots}, cpu_capacity_factor=${cpu_capacity_factor}, finish_last_n_particles_on_cpu=${finish_last_n_particles_on_cpu}"

"${CI_TEST_DIR}/python_scripts/macro_generator.py" \
    --template "${CI_TEST_DIR}/example_template.mac" \
    --output "${CI_TMP_DIR}/validation_testem3_finish_on_cpu.mac" \
    --gdml_name "${PROJECT_SOURCE_DIR}/examples/data/testEm3.gdml" \
    --num_threads "${num_threads}" \
    --num_events "${num_events}" \
    --num_trackslots "${num_trackslots}" \
    --num_hitslots "${num_hitslots}" \
    --cpu_capacity_factor "${cpu_capacity_factor}" \
    --finish_last_n_particles_on_cpu "${finish_last_n_particles_on_cpu}" \
    --gun_number "${gun_number}" \
    --track_in_all_regions True \
    --gun_type setDefault

# Verbosity 2 emits the CPU-finish message checked below.
sed -i 's|^/adept/setVerbosity 0$|/adept/setVerbosity 2|' \
  "${CI_TMP_DIR}/validation_testem3_finish_on_cpu.mac"
run_log="${CI_TMP_DIR}/validation_testem3_finish_on_cpu.log"

if ! "${ADEPT_EXECUTABLE}" --allsensitive --accumulated_events \
                      -m "${CI_TMP_DIR}/validation_testem3_finish_on_cpu.mac" \
                      --output_dir "${CI_TMP_DIR}" \
                      --output_file "adept_em3_finish_on_cpu_2.5e4_e-" >"${run_log}" 2>&1; then
  cat "${run_log}"
  exit 1
fi

if ! grep -qF 'particles are finished on CPU for event ' "${run_log}"; then
  echo "The validation run did not exercise finishing on the CPU."
  cat "${run_log}"
  exit 1
fi

"${CI_TEST_DIR}/python_scripts/check_validation.py" \
  --file1 "${CI_TMP_DIR}/adept_em3_finish_on_cpu_2.5e4_e-.csv" \
  --file2 "${CI_TEST_DIR}/benchmark_files/g4hepem_em3_10e7_e-.csv" \
  --n1 "${nprimaries}" --n2 1e7 --tol 0.01
