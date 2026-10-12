#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

"""Replay records of the AdePT geometry validation output (/adept/geometryValidationOutput).

Selects records of <prefix>_crossings.json and/or <prefix>_handoff.json and replays them on the CPU with the
geometryValidationReplay executable (built with example1 when ADEPT_GEOMETRY_VALIDATION=ON), loading the same GDML
file as the run.

Example:
  replay_geometry_validation.py --replay BuildProducts/bin/geometryValidationReplay --gdml cms2018_sd.gdml \\
      --prefix example1_handoff --kind inside-left --limit 5
"""

import argparse
import json
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Replay AdePT geometry validation records.")
    parser.add_argument("--replay", required=True, help="Path of the geometryValidationReplay executable")
    parser.add_argument("--gdml", required=True, help="GDML file of the run")
    parser.add_argument("--prefix", required=True, help="Output prefix given to /adept/geometryValidationOutput")
    parser.add_argument("--index", type=int, nargs="*", help="Replay only these record indices")
    parser.add_argument("--kind",
                        help="Crossings: only this kind (inside-left, outside-entered, off-surface, zero-step-loop, "
                             "state-mismatch-end, state-mismatch-start)")
    parser.add_argument("--category", help="Handoff: only this category (e.g. unexplained)")
    parser.add_argument("--volume", help="Only records whose volumes contain this label")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of records per file (default 20)")
    parser.add_argument("--boolean-threshold", type=int,
                        help="Boolean optimization threshold, if the run set one (VecGeom BooleanFactory)")
    return parser.parse_args()


def vec(v):
    return " ".join(repr(float(x)) for x in v)


def crossing_line(i, r):
    def path(p):
        return f"{len(p['daughter_indices'])} " + " ".join(
            f"{index} {label}" for index, label in zip(p["daughter_indices"], p["labels"]))
    # A state mismatch step started from the boundary flag of the step before (not kept in the record): pre_safety 0
    on_boundary = r['from_on_boundary'] or (r['kind'] == 'state-mismatch-end' and r.get('pre_safety', -1) == 0)
    return (f"C {i} {int(on_boundary)} {path(r['from'])} {path(r['to'])} {vec(r['pre_point'])} "
            f"{vec(r['pre_direction'])} {vec(r['point'])} {vec(r['direction'])} {int(r['stuck_push'])} "
            f"{int(r['magnetic_field'])}")


def handoff_line(i, r):
    def path(p):
        return f"{len(p)} " + " ".join(f"{name} {copy}" for name, copy in p)
    return (f"H {i} {path(r['exited'])} {path(r['expected'])} {vec(r['boundary_point'])} {vec(r['handed_point'])} "
            f"{vec(r['direction'])}")


def select(records, args, labels):
    out = []
    for i, r in enumerate(records):
        if args.index is not None and i not in args.index:
            continue
        if args.kind and r.get("kind") != args.kind:
            continue
        if args.category and r.get("category") != args.category:
            continue
        if args.volume and not any(args.volume in label for label in labels(r)):
            continue
        out.append((i, r))
    return out[: args.limit]


def main():
    args = parse_args()
    lines = []
    crossings = args.prefix + "_crossings.json"
    if os.path.exists(crossings):
        records = json.load(open(crossings))["records"]
        for i, r in select(records, args, lambda r: r["from"]["labels"] + r["to"]["labels"]):
            print(f"C {i}: {r['kind']} {r['depth_ktol']:.4g} kTolerance in {r['offending']['label']}, event "
                  f"{r['event']} track {r['track']} step {r['step']}, overlap {r['overlap']}")
            lines.append(crossing_line(i, r))
    handoff = args.prefix + "_handoff.json"
    if os.path.exists(handoff):
        records = json.load(open(handoff))["records"]
        for i, r in select(records, args, lambda r: [n for n, _ in r["exited"] + r["expected"] + r["located"]]):
            print(f"H {i}: {r['category']}, band error {r['band_error']}, event {r['event']} track {r['track']}")
            lines.append(handoff_line(i, r))
    if not lines:
        raise SystemExit("No record selected")
    command = [args.replay, args.gdml] + ([str(args.boolean_threshold)] if args.boolean_threshold else [])
    result = subprocess.run(command, input="\n".join(lines) + "\n", text=True, capture_output=True)
    for line in result.stdout.splitlines():
        # replay lines; Geant4 messages are indented by six spaces
        if line.startswith(("C ", "H ")) or (line.startswith("  ") and not line.startswith("      ")):
            print(line)
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
