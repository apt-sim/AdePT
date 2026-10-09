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
  starts on the same surface in the direction of motion. Anything else is
  *unexplained* (overlaps, differences between the Geant4 and VecGeom solids).

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
/adept/crossingRecordsFile crossings.csv
/adept/crossingRecordCapacity 10000
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

Off-band landings are written to the records file, one line each, with the point
and direction in the frame of the offending volume (to replay the query on its
solid) and in the global frame, the depth in `kTolerance` units, the normal
component of the direction, the volumes left and reached, and the overlap proof.
The `CrossingValidation` report is a measurement: it does not pass or fail.

(validation-overlap-proofs)=
## Overlap proofs

Both checks try to prove that an error comes from the geometry: the end point, or
a witness point up to 100 `kTolerance` further along the final direction, is
strictly inside two volumes of different branches of the hierarchy (an overlap),
or strictly inside a volume and strictly outside one of its ancestors (an
extrusion). Proven overlaps are listed per volume pair in the reports and in the
records file.

```{figure} images/geometry_validation_overlaps.png
:name: fig-validation-overlaps
:alt: Proof of overlaps and extrusions from the solids.
:align: center
:width: 95%

Overlap and extrusion proofs used by both checks.
```

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
