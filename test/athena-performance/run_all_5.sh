#!/usr/bin/env bash

set -euo pipefail

NWORK=${1:-96}

for I in 1 2 3 4 5; do
  echo "=== Starting AdePT run ${I}, T=${NWORK} ==="
  ./run_adept.sh "${NWORK}" > "adept_T${NWORK}_run${I}.log" 2>&1
  if [ -f log.AtlasG4Tf ]; then
    mv log.AtlasG4Tf "log.AtlasG4Tf_AdePT_T${NWORK}_run${I}"
  else
    echo "WARNING: log.AtlasG4Tf missing after AdePT run ${I}" >&2
  fi
done

echo "All done."
