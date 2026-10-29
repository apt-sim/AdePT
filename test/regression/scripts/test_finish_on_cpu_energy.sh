#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

set -eu -o pipefail

ADEPT_EXECUTABLE=${1:?integrationTest executable}
PROJECT_SOURCE_DIR=${2:?project source directory}
CI_TMP_DIR=${3:?test temporary directory}

cleanup() {
  rm -rf -- "${CI_TMP_DIR}"
}
trap cleanup EXIT
cleanup
mkdir -p "${CI_TMP_DIR}"

# Start below the finishing threshold so energetic particles return to the CPU.
# Two primaries also catch resuming only the first returned track of an event.
for particle in e- gamma; do
  MACRO_FILE="${CI_TMP_DIR}/finish_${particle}.mac"
  LOG_FILE="${CI_TMP_DIR}/finish_${particle}.log"
  cat >"${MACRO_FILE}" <<EOF
/run/numberOfThreads 1
/control/verbose 0
/run/verbose 0
/process/verbose 0
/tracking/verbose 0
/event/verbose 0

/detector/filename ${PROJECT_SOURCE_DIR}/examples/data/testEm3.gdml
/adept/setVerbosity 2
/adept/setSeed 1234567
/adept/setMillionsOfTrackSlots 1
/adept/setMillionsOfHitSlots 1
/adept/setCPUCapacityFactor 2.5
/adept/setCUDAStackLimit 8192
/adept/FinishLastNParticlesOnCPU 100
/adept/setTrackInAllRegions true
/process/em/applyCuts true
/run/setCut 0.7 mm
/run/initialize
/eventAction/verbose 1

/gun/setDefault
/gun/particle ${particle}
/gun/energy 10 GeV
/gun/number 2
/gun/position -220 0 0 mm
/gun/direction 1 0 0
/gun/print false
/run/beamOn 1
EOF

  if ! "${ADEPT_EXECUTABLE}" --allsensitive -m "${MACRO_FILE}" \
      --output_dir "${CI_TMP_DIR}" --output_file "finish_${particle}" >"${LOG_FILE}" 2>&1; then
    cat "${LOG_FILE}"
    exit 1
  fi

  if ! grep -q 'particles are finished on CPU for event 0' "${LOG_FILE}"; then
    echo "The ${particle} event did not exercise finishing on the CPU."
    cat "${LOG_FILE}"
    exit 1
  fi

  DEPOSITED_ENERGY=$(awk '/Total energy deposited:/ {print $(NF - 1); exit}' "${LOG_FILE}")
  # The calorimeter contains nearly all 20 GeV. Allow generous shower leakage,
  # while rejecting both discarded tracks and a single surviving 10 GeV track.
  if ! awk -v energy="${DEPOSITED_ENERGY:-0}" 'BEGIN { exit !(energy > 15000) }'; then
    echo "Expected both CPU-finished 10 GeV ${particle} primaries to deposit their energy."
    echo "Observed deposited energy: ${DEPOSITED_ENERGY:-missing} MeV; expected above 15000 MeV."
    cat "${LOG_FILE}"
    exit 1
  fi
  echo "CPU-finished ${particle} tracks deposited ${DEPOSITED_ENERGY} MeV."
done
