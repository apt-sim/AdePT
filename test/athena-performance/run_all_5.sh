#!/usr/bin/env bash

set -euo pipefail

NWORK=${1:-96}
REPETITIONS=${2:-${ADEPT_REPETITIONS:-5}}

[[ "${REPETITIONS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: invalid repetition count '${REPETITIONS}'" >&2
  exit 1
}

for I in $(seq 1 "${REPETITIONS}"); do
  echo "=== Starting AdePT run ${I}, T=${NWORK} ==="
  ./run_adept.sh "${NWORK}" > "adept_T${NWORK}_run${I}.log" 2>&1
  if [ -f log.AtlasG4Tf ]; then
    mv log.AtlasG4Tf "log.AtlasG4Tf_AdePT_T${NWORK}_run${I}"
  else
    echo "WARNING: log.AtlasG4Tf missing after AdePT run ${I}" >&2
  fi
done

echo "All done after ${REPETITIONS} run(s)."
