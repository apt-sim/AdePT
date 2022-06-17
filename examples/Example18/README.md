<!--
SPDX-FileCopyrightText: 2022 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

## Example 18

Example based on Example 13, but restricting the step length in case of magnetic field for better performance.

 * arbitrary geometry via gdml file (tested with cms2018.gdml from VecGeom persistency/gdml/gdmls folder) and optionally a magnetic field with constant Bz,
 * geometry read as Geant4 geometry, reading in regions and cuts, to initialize G4HepEm data
 * geometry read then into VecGeom, and synchronized to GPU
 * G4HepEm material-cuts couple indices mapped to VecGeom logical volume id's
 * physics processes for e-/e+ (including MSC) and gammas using G4HepEm.
 * scoring per placed volume, no sensitive detector feature

Electrons, positrons, and gammas are stored in separate containers in device memory.
Free positions in the storage are handed out with monotonic slot numbers, slots are not reused.
Active tracks are passed via three queues per particle type (see `struct ParticleQueues`).
Results are reproducible using one RANLUX++ state per track.

Additionally, the kernels score energy deposit and the charged track length per volume.

### Kernels

This example uses one stream per particle type to launch kernels asynchronously.
They are synchronized via a forth stream using CUDA events.

#### `TransportElectrons<bool IsElectron>`

1. Obtain safety unless the track is currently on a boundary.
2. Determine physics step limit, including conversion to geometric step length according to MSC.
3. Query geometry (or optionally magnetic field) to get geometry step length.
4. Convert geometry to true step length according to MSC, apply net direction change and discplacement.
5. Apply continuous effects; kill track if stopped.
6. If the particle reaches a boundary, perform relocation.
7. If not, and if there is a discrete process:
 1. Sample the final state.
 2. Update the primary and produce secondaries.

#### `TransportGammas`

1. Determine the physics step limit.
2. Query VecGeom to get geometry step length (no magnetic field for neutral particles!).
3. If the particle reaches a boundary, perform relocation.
4. If not, and if there is a discrete process:
 1. Sample the final state.
 2. Update the primary and produce secondaries.

#### `FinishIteration`

Clear the queues and return the tracks in flight.
This kernel runs after all secondary particles were produced.

#### `InitPrimaries` and `InitParticleQueues`

Used to initialize multiple primary particles with separate seeds.
