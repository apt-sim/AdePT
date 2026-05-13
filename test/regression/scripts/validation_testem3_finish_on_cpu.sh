#! /usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

# Validation coverage for /adept/FinishLastNParticlesOnCPU. The option is
# population-dependent, so this checks statistical physics agreement and also
# verifies from the transport printout that the finish-on-CPU path was used.

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
finish_last_n_particles_on_cpu=${ADEPT_VALIDATION_FINISH_ON_CPU_N:-100}
nprimaries=$((num_events * gun_number))

if [[ "${finish_last_n_particles_on_cpu}" -le 0 ]]; then
  echo "Error: ADEPT_VALIDATION_FINISH_ON_CPU_N must be positive."
  exit 2
fi

if [[ "${gun_number}" -le "${finish_last_n_particles_on_cpu}" ]]; then
  echo "Error: gun_number (${gun_number}) must be larger than FinishLastNParticlesOnCPU (${finish_last_n_particles_on_cpu})."
  echo "This test must not be configured so the full primary batch can be handed to CPU immediately."
  exit 2
fi

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

run_log="${CI_TMP_DIR}/validation_testem3_finish_on_cpu.log"

"${ADEPT_EXECUTABLE}" --allsensitive --accumulated_events \
                      -m "${CI_TMP_DIR}/validation_testem3_finish_on_cpu.mac" \
                      --output_dir "${CI_TMP_DIR}" \
                      --output_file "adept_em3_finish_on_cpu_2.5e4_e-" > "${run_log}" 2>&1

if ! grep -Eq 'Finishing (e-/e\+|gamma) of the [0-9]+ last particles' "${run_log}"; then
  echo "Error: FinishLastNParticlesOnCPU did not trigger; no finish-on-CPU transport printout was found."
  exit 1
fi

echo "Representative FinishLastNParticlesOnCPU printouts:"
awk '
  /Finishing (e-\/e\+|gamma) of the [0-9]+ last particles/ {
    print
    if (++count == 5) exit
  }
' "${run_log}"

max_finish_population=$(
  awk '
    match($0, /Finishing (e-\/e\+|gamma) of the ([0-9]+) last particles/, m) {
      if (m[2] > max) max = m[2]
      count++
    }
    END {
      if (count == 0) exit 1
      print max
    }
  ' "${run_log}"
)

if [[ "${max_finish_population}" -ge "${gun_number}" ]]; then
  echo "Error: finish-on-CPU triggered with ${max_finish_population} particles in flight, not below the per-event primary count ${gun_number}."
  exit 1
fi

if [[ "${max_finish_population}" -ge "${finish_last_n_particles_on_cpu}" ]]; then
  echo "Error: finish-on-CPU triggered with ${max_finish_population} particles in flight, expected less than ${finish_last_n_particles_on_cpu}."
  exit 1
fi

echo "FinishLastNParticlesOnCPU triggered below threshold: max observed in-flight population ${max_finish_population} < ${finish_last_n_particles_on_cpu}, with ${gun_number} primaries/event."

"${CI_TEST_DIR}/python_scripts/check_validation.py" \
  --file1 "${CI_TMP_DIR}/adept_em3_finish_on_cpu_2.5e4_e-.csv" \
  --file2 "${CI_TEST_DIR}/benchmark_files/g4hepem_em3_10e7_e-.csv" \
  --n1 "${nprimaries}" --n2 1e7 --tol 0.01
