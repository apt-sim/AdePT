<!--
SPDX-FileCopyrightText: 2026 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

# Geometry validation: region handoff and GPU boundary crossings

When only some regions are transported on the GPU (see
{ref}`runtime-parameters-specify-the-regions-where-the-gpu-is-used`), a track
leaving them is returned to Geant4. AdePT gives the returned track a touchable
built from the VecGeom state reached on the GPU, and Geant4 starts tracking it
with `G4Navigator::ResetHierarchyAndLocate(position, direction, touchable)`: a
search from that path which, for a point on a surface, enters a volume only if
the direction goes into it. The handoff is therefore correct as long as the step
end point lies within Geant4's tolerance band (+-`kCarTolerance`/2) of the
crossed surfaces and the final direction agrees with the crossing. A point
handed over deeper inside the volume just left would be relocated back into it
and bounce between CPU and GPU.

AdePT provides two optional geometry-validation checks for this, for validation
of the GPU navigation and for studies of its tolerance conventions. Findings are
counted and logged; they never stop the job. The checks only observe: with them
enabled, the simulation results are identical to those of a build without them.

## Build

The checks are compiled only with:

```console
cmake -S . -B ./adept-build \
  -DADEPT_GEOMETRY_VALIDATION=ON \
  <otherargs>
```

With the default `OFF`, the transport code is unchanged and the commands below
only print a notice.

## Handoff to Geant4

```console
/adept/validateHandoff true
```

For every track leaving the GPU regions, a separate navigator repeats the
relocation Geant4 does when tracking resumes, and the outcome is classified:

- *tolerance band*: the GPU end point must not be strictly inside a volume being
  left, nor strictly outside a volume being entered, as Geant4 classifies it, and
  it must lie on the surface of at least one crossed volume;
- *normal at the exit point*: the outward normal of the crossed surface is
  compared with the final direction (leaving through it, pointing back, tangent);
- *relocation*: Geant4 confirms the GPU path, or locates the track elsewhere for
  a geometric reason: the normal at the exit point shows the direction going back
  through the crossed surface (for example a deflection applied at the boundary;
  native Geant4 would take a zero step to the same place), or another volume
  starts on the same surface in the direction of motion. Geant4 may also confirm
  the GPU path at the end point itself while the short push that AdePT applies
  along the direction before returning the track moved it across a further
  surface (for example a tangent polygonal face a few `kTolerance` away); this is
  counted as *crossed by the handoff push*. Anything else is *unexplained*
  (overlaps, differences between the Geant4 and VecGeom solids).

```{figure} images/geometry_validation_handoff.png
:name: fig-validation-handoff
:alt: Classification of the tracks leaving the GPU regions.
:align: center
:width: 95%

(a) confirmed path; (b) final direction pointing back through the crossed
surface and (c) a further volume starting on that surface, both legitimate;
(d) errors.
```

The job prints `HandoffValidation` counters per worker thread at the end of the
job, with `HandoffValidation: PASSED` when every tolerance-band violation and
unexplained relocation is proven to come from an overlap in the geometry (see
{ref}`validation-overlap-proofs`), `FAILED` otherwise, with details of the first
cases.

## Boundary crossings on the GPU

```console
/adept/validateCrossings true
```

Every boundary crossing made by the GPU transport is checked right after the
relocation into the next volume, with the VecGeom solids of the volumes left and
entered, against a +-`kTolerance` band. The report gives a histogram of the
wrong-side depth of the landings in `kTolerance` units, the off-band landings
(more than `kTolerance` inside a volume being left, outside a volume being
entered, or away from every crossed surface), and the bounce-backs, where the
final direction points back through the crossed surface. This check is available
with the solid navigation only (not with `ADEPT_USE_SURF`).

```{figure} images/geometry_validation_crossings.png
:name: fig-validation-crossings
:alt: Landing classes of the GPU boundary crossings.
:align: center
:width: 95%

Landing classes against the +-`kTolerance` band, and bounce-backs.
```

Steps ended by the stuck-track push of the transport (a track making zero steps
is moved by 100 `kTolerance`) are counted separately, and the in-band crossings
made during a zero-step loop, before the push, are counted and recorded: they show
the states the loop alternates between. The `CrossingValidation`
report is a measurement: it does not pass or fail.

Between crossings, the position is also checked against the volume of the
navigation state, at the start of every step and at the end of every step that
crossed no boundary. A point more than `kTolerance` outside that volume, or as
deep inside one of its daughters, is a state mismatch: the track left or entered a
volume without a relocation, for instance because a solid missed a crossing, and
is relocated later by a zero-length step far from any surface. Mismatches are
counted and recorded with the physics and geometry step lengths, the pre-step
safety and the multiple-scattering displacement, which tell the navigation, the
safety and the displacement apart as causes.

(validation-overlap-proofs)=
## Overlap proofs

Both checks try to prove that an error comes from the geometry: the end point, or
a witness point up to 100 `kTolerance` before or after it along the final
direction, is
strictly inside two volumes of different branches of the hierarchy (an overlap),
or strictly inside a volume and strictly outside one of its ancestors (an
extrusion). Proven overlaps are listed per volume pair in the reports and in the
output files (see {ref}`validation-output`).

```{figure} images/geometry_validation_overlaps.png
:name: fig-validation-overlaps
:alt: Proof of overlaps and extrusions from the solids.
:align: center
:width: 95%

Overlap and extrusion proofs used by both checks.
```

(validation-output)=
## Output files, plots and replays

```console
/adept/geometryValidationOutput <prefix>
/adept/geometryValidationRecordCapacity 10000
```

At the end of the job the checks write `<prefix>_crossings.json` and
`<prefix>_handoff.json`: the counters, the histograms (the wrong-side landing
depth in logarithmic `kTolerance` bins, the normal component of the bounce-back
directions) and a replayable record for each off-band landing, each crossing of
a zero-step loop, each state mismatch and each handoff not confirmed by Geant4 at
the first attempt.
Records hold the navigation states before and after the crossing (daughter
indices from the world, with the volume labels as a check) or the Geant4 paths,
the pre-step and final points and
directions, the event and track, the stuck-track and magnetic-field flags and the
overlap proof. They are sorted by event and track, so the files do not depend on
the thread scheduling; at most the record capacity is kept per check, the
counters are always complete.

In multi-threaded jobs the GPU crossings are counted by the transport shared by
all workers and written when the transport is destroyed, after its last user; the
handoff checks run in each worker thread and are merged when the last worker
ends.

The histograms are drawn, as for the integration test truth histograms, with

```console
test/regression/scripts/python_scripts/plot_geometry_validation.py --prefix <prefix> --output-dir plots
```

which writes a ROOT file and one PNG per histogram. Records are summarized, and
replayed on the CPU loading the same GDML file as example1, with

```console
test/regression/scripts/python_scripts/replay_geometry_validation.py --prefix <prefix> --summary
test/regression/scripts/python_scripts/replay_geometry_validation.py \
  --replay BuildProducts/bin/geometryValidationReplay --gdml <file.gdml> \
  --prefix <prefix> [--kind inside-left] [--category unexplained] [--volume <label>] [--index i ...] \
  [--limit n] [--boolean-threshold n]
```

`--boolean-threshold` must repeat the VecGeom Boolean optimization threshold of
the run, if it set one: the replay must build the same Boolean solids.

For a GPU crossing the replay classifies the end point with every crossed solid
(Inside, safeties, distances along the final direction, normal), recomputes the
step from the pre-step point for straight steps and the relocation with the final
direction. A step that started on a boundary is also classified at the start
point moved by the navigator boundary push, with the solids of the volumes
entered, which shows a solid answering wrongly within its surface band. For a
handoff the replay repeats the Geant4 relocation at the boundary and handed
points. `geometryValidationReplay` is built with example1 when
`ADEPT_GEOMETRY_VALIDATION=ON`. How to use the summary and the replay to find the
cause of an error is described in {ref}`validation-debugging`.

(validation-debugging)=
## Debugging with the records

The checks find the errors; the records and the replay explain them. Debugging
takes four steps: triage the records, replay a few of each group, read the replay
to decide what answered wrongly, and turn the record into a reproducer.

### Triage

Run the job with `/adept/geometryValidationOutput <prefix>`, then summarize the
records without replaying them:

```console
replay_geometry_validation.py --prefix <prefix> --summary [--kind <kind>] [--volume <label>]
```

The first line repeats the counters of the report. Each following line is a group
of records with the same kind and the same volumes left and entered, largest
first, for example

```text
crossings 184555210: off-band 0 (...), zero-step loops 359, state mismatches 0 at step end / 0 at step start; records kept 359 of 359
     272 zero-step-loop       barrel_1 -> aluSideA_1: 262 tracks, at most 2 per track, max depth 0.000455 kTolerance, charged 4, stuck push 0, overlap 0; indices 0 1 2 3 4
       4 zero-step-loop       SM3rd_2 -> EMOD_4: 4 tracks, at most 1 per track, max depth 0.000107 kTolerance, charged 0, stuck push 0, overlap 0; indices 20 22 34 35
```

with the number of tracks, the largest number of records of one track, the
largest depth, how many records are of charged tracks, came after a stuck-track
push or carry an overlap proof, and the first record indices, to pass to
`--index`. A group is usually one cause: replaying one or two of its records is
enough. Handoff records are grouped by category and by the Geant4 volumes exited
and expected.

What a group means before any replay:

- An off-band group with an overlap proof is a geometry error: the overlap or the
  extrusion is listed in the report and in the output file.
- An off-band group whose tracks all come from a stuck-track push is a consequence
  of the push (it moves the track by 100 `kTolerance` without navigation); the
  cause is the zero-step loop that preceded the push.
- A zero-step-loop group with at most one or two records per track and no
  stuck-track push is a track crossing coincident surfaces: a volume touching
  another one, or a daughter touching its mother, entered with a zero step after
  leaving the first one. A real loop shows many records per track and ends with a
  stuck-track push.
- Any other off-band landing and every state mismatch are worth a replay.

### Replay

```console
replay_geometry_validation.py --replay BuildProducts/bin/geometryValidationReplay \
  --gdml <file.gdml> --prefix <prefix> --index 0 [--boolean-threshold n]
```

The replay runs the VecGeom solids and the BVH navigator on the CPU, with the
VecGeom build example1 was linked with. For each crossing record it prints:

- a header `C <index>: <volume left> -> <volume entered>, step <length> mm`, with
  `stuck-track push` and `magnetic field` when they apply;
- the classification of the end point with the final direction, by each solid of
  the volumes left (`left`), entered (`entered`) and of the deepest common
  ancestor (`common`): `Inside`, `SafetyToIn`, `SafetyToOut`, `DistanceToIn`,
  `DistanceToOut` in `kTolerance` units (`-1e+09` is the -1 of a wrong-side
  start, `inf` no crossing) and `n.d`, the outward normal of the solid at the
  point dotted with the direction (`(invalid)` when the point is not on its
  surface);
- the same classification of the pre-step point with the pre-step direction
  (`pre`, `pre/cmn`) and, for a step that started on a boundary, of that point
  moved by the navigator boundary push, by the volume left (`pushed`) and by the
  volumes entered (`pushed>`). This block also prints the parameters of the solid
  entered and the local point and direction in that solid with 17 digits;
- `recomputed step`: the straight step recomputed from the pre-step point, the
  distance of its end point from the recorded one in `kTolerance` units, and the
  next volume. Curved steps in a field are not recomputed;
- `exit climb / entry before relocation` and `relocation with the final
  direction`: the state the navigator reaches at the end point, and whether it is
  the recorded one.

Handoff records print the Geant4 relocation at the boundary point and at the
handed point, with `G4 Inside`, `DistanceToIn`, `DistanceToOut` and `n.d` of
every volume involved.

The depth of a record is measured with the safeties of the solids, so it carries
their precision: for very large solids (radii of kilometers) this is a fraction
of `kTolerance`.

### Reading the replay

| What the replay shows | Cause |
| --- | --- |
| The recomputed step reproduces the recorded end point, and a solid left or entered classifies it beyond its band (`Inside outside` for the volume entered, `Inside inside` deeper than `kTolerance` for the volume left) | The distance of that solid lands outside its own surface band: a solid error. Make a reproducer (below). |
| `pushed>`: the volume entered classifies the pushed pre-step point as outside, or gives a `DistanceToIn` beyond the band, although the step entered it | The solid answers wrongly from a point within its surface band. |
| The recomputed end point differs from the recorded one by more than `kTolerance` (straight steps) | The GPU and the CPU disagree: a difference of the device code, of the navigation state, or of the Boolean threshold given to the replay. |
| `relocation ... DIFFERENT from the recorded state`, with an overlap proof | A geometry overlap or extrusion. |
| `relocation ... DIFFERENT from the recorded state` after a zero step, no overlap proof | Coincident surfaces: the relocation goes one level further, into a volume touching the one entered. |
| Zero-step loop: the volume entered is `Inside surface` with a `DistanceToOut` well beyond `kTolerance`, and the relocation is the recorded state | A legitimate zero-length crossing between touching volumes, not an error. |
| State mismatch with a displacement larger than the pre-step safety | The multiple-scattering displacement left the safety sphere: the safety (of the volume or of a daughter) was too large, or the displacement was not limited by it. |
| State mismatch with a displacement within the pre-step safety | The straight step missed a boundary: replay the step (`state-mismatch-end` records keep the geometry step) to find the solid whose distance missed it. |
| State mismatch at a step start | The previous step or its relocation left the track in the wrong state: replay the records of the same track before it. |

### From a record to a reproducer

The replay prints the parameters of the solid involved (or find them in the GDML
file with the logical volume name) and the point and direction in its local frame
with 17 digits. These are enough to reproduce the error in a VecGeom unit test:
build the solid, call the distance or the classification shown wrong by the
replay, and compare with a reference computed in higher precision (`long double`)
from the same local point. The test fails before the fix of the solid; after the
fix, rebuild example1 with the fixed VecGeom, replay the same records (the
`recomputed step` line shows the new end point) and rerun the job to see the
group disappear.

### Worked example: a far mirror in LHCb

The LHCb 2024 geometry, all regions on the GPU, push-free stepping, gave in the
`CrossingValidation` report

```text
CrossingValidation   off-band landings (beyond +-kTolerance): 1 (0 inside a volume being left, 0 outside a volume being entered, 1 farther than kTolerance from every crossed surface; 0 proven to come from an overlap, 0 after a stuck-track push)
```

The summary shows one group, without overlap or stuck-track push:

```text
       1 off-surface          pvRich1Mirror2Seg12 -> pvRich1Mirror2Q2Master: 1 tracks, at most 1 per track, max depth 1.4 kTolerance, charged 0, stuck push 0, overlap 0; indices 0
```

The replay of record 0 gives

```text
C 0: pvRich1Mirror2Seg12 -> pvRich1Mirror2Q2Master, step 0.856381 mm
  end point, final direction (distances in kTolerance):
    left     level  6 pvRich1Mirror2Seg12      Inside outside SafetyToIn  1.4 SafetyToOut -1e+09 DistanceToIn inf DistanceToOut -1e+09 n.d (invalid) 0.9689
    common   level  5 pvRich1Mirror2Q2Master   Inside inside  ...
  pre-step point pushed by the navigator boundary push (10 kTolerance):
    local point (2654100.825030609, -158.23057569505181, -5.0193571025656656) direction (-0.96891571455553982, -0.16480283140934263, -0.18450573120310274)
  recomputed step 0.856380718 mm, end point 0 kTolerance from the recorded one, next pvRich1Mirror2Q2Master (boundary)
  relocation with the final direction: pvRich1Mirror2Q2Master (same as the recorded state)
```

The CPU reproduces the step exactly and the relocation is right, but the mirror
segment, a spherical shell of 2.654 km radius cut to a 0.44 m patch, classifies the
end of its own `DistanceToOut` 1.4 `kTolerance` outside its inner sphere (radius
2654100 mm), which the track was leaving through. The cause is
in the solid: in the frame of the sphere the local coordinates are of the order of
the radius, and the residual |p|²-R² used by the sphere loses about ε·R, 0.6
`kTolerance` at this radius. A unit test with the segment parameters, the local
point above and a `long double` reference reproduces the landing error (the
reference puts the recorded end point 1.16 `kTolerance` inside the inner sphere);
the segment parameters are the precision test of the sphere far-section fix in
VecGeom. With example1 rebuilt on the fixed VecGeom, the replay of the same record
recomputes a different end point, 0.46 `kTolerance` from the recorded one, and the
next job reports no off-band landing.

### Worked example: zero-length crossings that are not errors

The same campaign reported zero-step loops in other geometries, for instance one
in CMS D110:

```text
       1 zero-step-loop       HGCalEE -> HGCalEEAbsorberLead27: 1 tracks, at most 1 per track, max depth 0.000153 kTolerance, charged 1, stuck push 0, overlap 0; indices 0
```

```text
C 0: HGCalEE -> HGCalEEAbsorberLead27, step 0 mm
  end point, final direction (distances in kTolerance):
    entered  level  9 HGCalEEAbsorberLead27    Inside surface SafetyToIn 0.000153 SafetyToOut 0.000153 DistanceToIn 0.00296 DistanceToOut 9.59e+10 n.d -0.05184
  recomputed step 0 mm, end point 0 kTolerance from the recorded one, next HGCalEEAbsorberLead27 (boundary)
  relocation with the final direction: HGCalEEAbsorberLead27 (same as the recorded state)
```

The track starts on the surface of the lead absorber, still in its mother
`HGCalEE`, and enters the absorber with a zero step: the absorber
classifies the point on its surface, the direction enters it (`n.d` negative) and
it is 96 mm thick along the track. The CPU agrees with the GPU at every stage: the
volumes touch, nothing is wrong. In ALICE, the 272 records `barrel_1 ->
aluSideA_1` show the same pattern with a second zero step: the relocation with the
final direction lands in the daughter `aluSideAVac_1`, which touches the surface of
its mother there.

## Example 1 tests

With `ADEPT_GEOMETRY_VALIDATION=ON` and examples built, the `example1-handoff`
test runs the CMS 2018 geometry with only `EcalRegion` on the GPU, a 3.8 T field
and both checks, and passes on `HandoffValidation: PASSED`. Further geometries
defining their own regions, for example GDML files with `Region` auxiliary
information, are added at configuration time:

```console
-DADEPT_HANDOFF_VALIDATION_GEOMETRIES="<file.gdml>=<Region>[,<Region>...];..."
```

Each entry adds a test `example1-handoff-<file>`.
`ADEPT_HANDOFF_VALIDATION_CUDA_STACK` (default `8192`) sets the CUDA stack limit
of these tests; deep Boolean solids may need more.

## Cost

Measured on the CMS 2018 geometry, `EcalRegion` on the GPU, 3.8 T, 10 events of
200 electrons of 10 GeV (RTX 4090):

| Build and mode | Run time relative to a build without the option |
| --- | :---: |
| `ADEPT_GEOMETRY_VALIDATION=OFF` | 1.00 |
| `ON`, no check enabled | 0.99 |
| `ON`, `/adept/validateHandoff` | 1.00 |
| `ON`, `/adept/validateCrossings` | 1.02 |
| `ON`, both | 1.03 |

Run-to-run fluctuations are about 0.5%.

The figures on this page are generated with `make geometry-validation-figures` in `docs`.
