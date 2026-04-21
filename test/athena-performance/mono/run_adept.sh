#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

NWORK=${1:-32}
export ATHENA_CORE_NUMBER=${NWORK}
MAX_EVENTS="${ADEPT_MAX_EVENTS:-1000}"
OUTPUT_HITS_FILE="${ADEPT_OUTPUT_HITS_FILE:-test.CA.HITS.pool_AdePT_${MAX_EVENTS}.root}"

echo "=== Running AdePT with ATHENA_CORE_NUMBER=${ATHENA_CORE_NUMBER}, MAX_EVENTS=${MAX_EVENTS} ==="

PREEXEC_CMD=$(cat <<'EOF'
flags.Sim.G4Commands+=[
  "/adept/MaxWDTIterations 10",
  "/adept/setSeed 2312452",
  "/adept/CallUserTrackingAction true",
  "/adept/CallUserSteppingAction false",
  "/adept/setCovfieBfieldFile /cvmfs/atlas.cern.ch/repo/sw/database/GroupData/MagneticFieldMaps/bmagatlas_09_fullAsym20400_forGPU_v1.cvf",
  "/adept/setVerbosity 0",
  "/adept/addGPURegion EMB",
  "/adept/addGPURegion EMEC",
  "/adept/addGPURegion HEC",
  "/adept/addGPURegion PreSampLAr",
  "/adept/setTrackInAllRegions false",
  "/adept/setMillionsOfTrackSlots 8",
  "/adept/setMillionsOfHitSlots 20",
  "/adept/setCUDAStackLimit 32192",
  "/adept/setCUDAHeapLimit 84857600"
];
flags.GeoModel.EMECStandard=True
EOF
)

AtlasG4_tf.py \
  --multithreaded \
  --randomSeed 1212353 \
  --conditionsTag 'default:OFLCOND-MC21-SDR-RUN4-02' \
  --geometryVersion 'default:ATLAS-P2-RUN4-04-00-00' \
  --preInclude 'AtlasG4Tf:Campaigns.MC23aSimulationMultipleIoV' \
  --postInclude 'PyJobTransforms.UseFrontier' \
  --inputEVNTFile "/cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/CampaignInputs/mc23/EVNT/mc23_13p6TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.evgen.EVNT.e8514/EVNT.32288062._002040.pool.root.1" \
  --outputHITSFile "${OUTPUT_HITS_FILE}" \
  --maxEvents "${MAX_EVENTS}" \
  --jobNumber 1 \
  --postExec 'with open("ConfigSimCA.pkl", "wb") as f: cfg.store(f)' \
  --physicsList "FTFP_BERT_ATL_AdePT" \
  --imf False \
  --preExec "${PREEXEC_CMD}"
