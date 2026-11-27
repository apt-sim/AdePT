#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

# Quick check of /adept/directHandoff. Layers 21-30 of testEm3 stay on the CPU, so every shower hands tracks back to
# Geant4 and offloads them again further downstream. With direct handoff, the workers go back to Geant4 while the GPU
# still transports the event, so this checks that every event still ends with all of its tracks: no step of a finished
# event arrives in the next one, and each event deposits nearly all of the primary energy.

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

num_events=8
MACRO_FILE="${CI_TMP_DIR}/direct_handoff.mac"
LOG_FILE="${CI_TMP_DIR}/direct_handoff.log"

regions=""
for layer in $(seq 1 20) $(seq 31 50); do
  regions+="/adept/addGPURegion Layer${layer}"$'\n'
done

cat >"${MACRO_FILE}" <<EOF
/run/numberOfThreads 2
/control/verbose 0
/run/verbose 0
/process/verbose 0
/tracking/verbose 0
/event/verbose 0

/detector/filename ${PROJECT_SOURCE_DIR}/examples/data/testEm3_regions.gdml
/adept/setVerbosity 2
/adept/setSeed 1234567
/adept/setMillionsOfTrackSlots 1
/adept/setMillionsOfHitSlots 1
/adept/setCPUCapacityFactor 2.5
/adept/setCUDAStackLimit 8192
/adept/directHandoff true
/adept/setTrackInAllRegions false
/adept/addGPURegion caloregion
${regions}
/process/em/applyCuts true
/run/setCut 0.7 mm
/run/initialize
/eventAction/verbose 1

/gun/setDefault
/gun/particle e-
/gun/energy 10 GeV
/gun/number 2
/gun/position -220 0 0 mm
/gun/direction 1 0 0
/gun/print false
/run/beamOn ${num_events}
EOF

if ! "${ADEPT_EXECUTABLE}" --allsensitive -m "${MACRO_FILE}" \
    --output_dir "${CI_TMP_DIR}" --output_file "direct_handoff" >"${LOG_FILE}" 2>&1; then
  cat "${LOG_FILE}"
  exit 1
fi

if ! grep -qF '/adept/directHandoff is on' "${LOG_FILE}"; then
  echo "The run did not use direct handoff."
  cat "${LOG_FILE}"
  exit 1
fi

if ! grep -qF 'returning to Geant4 with tracks handed back during the flush of event' "${LOG_FILE}"; then
  echo "No flush returned to Geant4 early: the run did not exercise direct handoff."
  cat "${LOG_FILE}"
  exit 1
fi

# Printed by AdePTTrackingManager::ProcessReturnedGPUSteps for returned steps of another event or worker, which is what
# an event ending before the GPU has finished its tracks leads to.
if grep -E "eventId doesn't match|threadId doesn't match" "${LOG_FILE}"; then
  echo "Returned steps reached the wrong event or worker."
  exit 1
fi

# The calorimeter contains nearly all of the 20 GeV per event. Allow generous shower leakage, while rejecting an event
# that ended before the GPU had finished its tracks.
if ! awk -v expected="${num_events}" '
    /Total energy deposited:/ {
      ++n
      energy = $(NF - 1)
      if (energy <= 15000) { print "Event energy " energy " MeV, expected above 15000 MeV"; bad = 1 }
    }
    END {
      if (n != expected) { print "Found " n " event energies, expected " expected; bad = 1 }
      exit bad
    }' "${LOG_FILE}"; then
  cat "${LOG_FILE}"
  exit 1
fi
echo "All ${num_events} events deposited their energy with direct handoff."
