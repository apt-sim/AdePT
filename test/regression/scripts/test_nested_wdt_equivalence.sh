#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

# Identical geometry, materials, cuts and seeds must give identical physics when
# some dense cells acquire their own production-cut region inside a WDT envelope.
set -eu -o pipefail

ADEPT_EXECUTABLE=${1:?integrationTest executable}
PROJECT_SOURCE_DIR=${2:?project source directory}
CI_TMP_DIR=$(realpath -m -- "${3:?test temporary directory}")
SCRIPTS_DIR="${PROJECT_SOURCE_DIR}/test/regression/scripts"

cleanup() {
  rm -rf -- "${CI_TMP_DIR}"
}
on_exit() {
  local status=$?
  if [[ ${status} -eq 0 ]]; then
    cleanup
  else
    printf 'Nested WDT test failed with exit status %d; artifacts retained in %s\n' "${status}" "${CI_TMP_DIR}" >&2
  fi
  exit "${status}"
}
trap on_exit EXIT
cleanup
mkdir -p "${CI_TMP_DIR}/flat" "${CI_TMP_DIR}/nested" "${CI_TMP_DIR}/fine"

python3 - "${PROJECT_SOURCE_DIR}/examples/data/testEm3.gdml" "${CI_TMP_DIR}" <<'PYTHON'
import copy
import pathlib
import sys
import xml.etree.ElementTree as ET

source, destination = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
tree = ET.parse(source)
structure = tree.getroot().find("structure")
lead = structure.find("volume[@name='G4_Pb']")
probe = copy.deepcopy(lead)
probe.set("name", "G4_Pb_nested_probe")
structure.insert(list(structure).index(lead) + 1, probe)
# Leave Pb in the parent as well: an actual/reference MC-index comparison would
# consume extra acceptance RNG in these equal-material child cells.
changed = 0
for volume in structure.findall("volume"):
    name = volume.get("name", "")
    if name.startswith("Layer") and int(name[5:]) % 2 == 0:
        for reference in volume.findall("physvol/volumeref"):
            if reference.get("ref") == "G4_Pb":
                reference.set("ref", "G4_Pb_nested_probe")
                changed += 1
assert changed == 25
# Both inputs contain exactly the same placements and LV clone. Only the new
# region auxiliary changes in the nested variant, with all four cuts identical.
tree.write(destination / "flat/geometry.gdml", encoding="UTF-8", xml_declaration=True)
flat_tree = copy.deepcopy(tree)
parent = tree.getroot().find("userinfo/auxiliary[@auxtype='Region']")
child = copy.deepcopy(parent)
child.set("auxvalue", "nested_equal_cuts")
child.find("auxiliary[@auxtype='volume']").set("auxvalue", "G4_Pb_nested_probe")
tree.getroot().find("userinfo").append(child)
tree.write(destination / "nested/geometry.gdml", encoding="UTF-8", xml_declaration=True)
# B also has a root outside the WDT envelope. The reference assigns that same
# physical volume an independent non-WDT region with exactly the same cuts.
for mode, base in (("flat", flat_tree), ("nested", tree)):
    outside = copy.deepcopy(base)
    volumes = outside.getroot().find("structure")
    lead_copy = copy.deepcopy(volumes.find("volume[@name='G4_Pb']"))
    lead_copy.set("name", "Outside_Pb")
    volumes.insert(0, lead_copy)
    placement = ET.SubElement(volumes.find("volume[@name='World']"), "physvol", {"name": "Outside_Pb"})
    ET.SubElement(placement, "volumeref", {"ref": "Outside_Pb"})
    ET.SubElement(placement, "position", {"name": "outside_position", "unit": "mm", "x": "220", "y": "0", "z": "0"})
    if mode == "nested":
        region = outside.getroot().find("userinfo/auxiliary[@auxvalue='nested_equal_cuts']")
        ET.SubElement(region, "auxiliary", {"auxtype": "volume", "auxvalue": "Outside_Pb"})
    else:
        region = copy.deepcopy(parent)
        region.set("auxvalue", "outside_normal_region")
        region.find("auxiliary[@auxtype='volume']").set("auxvalue", "Outside_Pb")
        outside.getroot().find("userinfo").append(region)
    outside.write(destination / mode / "outside_geometry.gdml", encoding="UTF-8", xml_declaration=True)
child.find("auxiliary[@auxtype='ecut']").set("auxvalue", "0.01")
tree.write(destination / "fine/geometry.gdml", encoding="UTF-8", xml_declaration=True)
PYTHON

TRUTH_ARGS=()
if [[ "$("${ADEPT_EXECUTABLE}" --supports_truth_root | tail -n 1)" == "1" ]]; then
  TRUTH_ARGS=(--truth_root)
fi

for scenario in gamma_outside gamma_inside electron gamma_external_region; do
  particle=gamma
  energy="1 MeV"
  position="-220 0 0 mm"
  gun_number=1
  geometry_file=geometry.gdml
  if [[ "${scenario}" == gamma_external_region ]]; then
    position="220 0 0 mm"
    gun_number=20
    geometry_file=outside_geometry.gdml
  elif [[ "${scenario}" == gamma_inside ]]; then
    position="-190.85 0 0 mm" # Inside the cloned Pb cell of layer 2.
  elif [[ "${scenario}" == electron ]]; then
    particle=e-
    energy="10 MeV"
  fi

  for mode in flat nested; do
    run_dir="${CI_TMP_DIR}/${mode}/${scenario}"
    mkdir -p "${run_dir}"
    track_all=true
    gpu_regions=""
    if [[ "${scenario}" == gamma_outside ]]; then
      track_all=false
      gpu_regions="/adept/addGPURegion caloregion"
      if [[ "${mode}" == nested ]]; then
        gpu_regions+=$'\n/adept/addGPURegion nested_equal_cuts'
      fi
    fi
    cat >"${run_dir}/run.mac" <<EOF
/run/numberOfThreads 1
/control/verbose 0
/run/verbose 0
/process/verbose 0
/tracking/verbose 0
/event/verbose 0
/detector/filename ${CI_TMP_DIR}/${mode}/${geometry_file}
/adept/setVerbosity 0
/adept/setSeed 1234567
/adept/setMillionsOfTrackSlots 1
/adept/setMillionsOfHitSlots 1
/adept/setCPUCapacityFactor 2.5
/adept/setCUDAStackLimit 8192
/adept/FinishLastNParticlesOnCPU 0
/adept/setTrackInAllRegions ${track_all}
${gpu_regions}
/adept/addWDTRegion caloregion
/adept/CallUserSteppingAction true
/adept/returnAllSteps true
/adept/CallUserTrackingAction true
/adept/returnFirstAndLastStep true
/process/em/applyCuts true
/run/setCut 0.7 mm
/run/initialize
/random/setSeeds 1234567 89101112
/eventAction/verbose 1
/gun/setDefault
/gun/particle ${particle}
/gun/energy ${energy}
/gun/number ${gun_number}
/gun/position ${position}
/gun/direction 1 0 0
/gun/print false
/run/beamOn 5
EOF
    "${ADEPT_EXECUTABLE}" --allsensitive "${TRUTH_ARGS[@]}" \
      -m "${run_dir}/run.mac" --output_dir "${run_dir}" --output_file physics \
      >"${run_dir}/run.log" 2>&1 || {
        status=$?
        cat "${run_dir}/run.log"
        exit "${status}"
      }
    if ! grep -Eq 'Total WDT roots found: [1-9]' "${run_dir}/run.log"; then
      cat "${run_dir}/run.log"
      echo "The ${mode} case did not enable a WDT envelope."
      exit 1
    fi
    if [[ "${mode}" == nested ]] && ! grep -Fq '(nested_equal_cuts)' "${run_dir}/run.log"; then
      cat "${run_dir}/run.log"
      echo "The child region did not inherit its WDT envelope."
      exit 1
    fi
  done

  # Compare each event, not only sums over the run. CSV contains physics
  # accumulators; wall-clock timings are written to a separate *_global.csv.
  python3 - "${CI_TMP_DIR}/flat/${scenario}/physics.csv" \
      "${CI_TMP_DIR}/nested/${scenario}/physics.csv" <<'PYTHON'
import csv
import math
import pathlib
import sys

left, right = map(pathlib.Path, sys.argv[1:])
a, b = left.read_bytes(), right.read_bytes()
if a != b:
    for index, (line_a, line_b) in enumerate(zip(a.splitlines(), b.splitlines()), 1):
        if line_a != line_b:
            print(f"First differing line {index}: flat={line_a!r}, nested={line_b!r}")
            break
    raise SystemExit("Flat and nested event-level physics CSVs differ")
rows = list(csv.reader(a.decode().splitlines()))
if len(rows) != 6:
    raise SystemExit(f"Expected five scored events, found {len(rows) - 1}")
values = [[float(value) for value in row] for row in rows[1:]]
if not all(math.isfinite(value) for row in values for value in row):
    raise SystemExit("Non-finite physics output")
if not sum(sum(row[1:]) for row in values) > 0:
    raise SystemExit("No scored energy deposition")
print("Five event-level CSV records match exactly.")
PYTHON

  echo "Equal-cut flat and nested WDT event CSVs matched for ${scenario}."
done

# Now change only B's electron cut. Pb also remains in A with the original
# reference cut, so extra secondary electrons prove that the interaction uses
# the actual child's couple rather than its WDT reference couple.
python3 - "${CI_TMP_DIR}" <<'PYTHON'
import pathlib
import sys
base = pathlib.Path(sys.argv[1])
macro = (base / "nested/gamma_inside/run.mac").read_text()
macro = macro.replace(str(base / "nested/geometry.gdml"), str(base / "fine/geometry.gdml"))
(base / "fine/run.mac").write_text(macro)
PYTHON
"${ADEPT_EXECUTABLE}" --allsensitive "${TRUTH_ARGS[@]}" \
  -m "${CI_TMP_DIR}/fine/run.mac" --output_dir "${CI_TMP_DIR}/fine" --output_file physics \
  >"${CI_TMP_DIR}/fine/run.log" 2>&1 || {
    status=$?
    cat "${CI_TMP_DIR}/fine/run.log"
    exit "${status}"
  }
PYTHONDONTWRITEBYTECODE=1 python3 - "${CI_TMP_DIR}" "${SCRIPTS_DIR}/python_scripts" "${#TRUTH_ARGS[@]}" <<'PYTHON'
import csv
import math
import pathlib
import sys
base = pathlib.Path(sys.argv[1])
if int(sys.argv[3]):
    sys.path.insert(0, sys.argv[2])
    from check_root_histograms import compare_histograms
    # Reuse one ROOT interpreter for all comparisons and the different-cut check.
    # Sidecars preserve exact floating-point states; file metadata is not compared.
    for scenario in ("gamma_outside", "gamma_inside", "electron", "gamma_external_region"):
        compare_histograms(str(base / "flat" / scenario / "physics.root"),
                           str(base / "nested" / scenario / "physics.root"), 0, 0)
        print(f"Equal-cut flat and nested WDT ROOT physics matched for {scenario}.")
    import ROOT
    ROOT.gROOT.SetBatch(True)
    def electron_count(path):
        file = ROOT.TFile.Open(str(path))
        histogram = file.Get("particle_type_counts")
        labels = str(file.Get("particle_type_counts__labels").GetString().Data()).splitlines()
        value = histogram.GetBinContent(labels.index("e-") + 1) if "e-" in labels else 0
        file.Close()
        return value
    equal = electron_count(base / "nested/gamma_inside/physics.root")
    fine = electron_count(base / "fine/physics.root")
    if fine <= equal:
        raise SystemExit(f"Lower child electron cut did not increase secondary electrons: {equal} -> {fine}")
    print(f"Lower child electron cut increased secondary electrons: {equal:g} -> {fine:g}.")
else:
    if (base / "fine/physics.csv").read_bytes() == (base / "nested/gamma_inside/physics.csv").read_bytes():
        raise SystemExit("Lower child electron cut did not change scored physics")
    print("Lower child electron cut changed the scored physics.")
rows = list(csv.reader((base / "fine/physics.csv").open()))
values = [[float(value) for value in row] for row in rows[1:]]
if len(values) != 5 or not all(math.isfinite(value) for row in values for value in row):
    raise SystemExit("Invalid different-cut physics output")
if not sum(sum(row[1:]) for row in values) > 0:
    raise SystemExit("No different-cut energy deposition")
PYTHON
