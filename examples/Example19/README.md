<!--
SPDX-FileCopyrightText: 2022 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

## Example 19

Example based on Example 18, but the workflow is split in smaller kernels. This allows for more fine-granular measurements and optimisations.

 * arbitrary geometry via gdml file (tested with cms2018.gdml from VecGeom persistency/gdml/gdmls folder) and optionally a magnetic field with constant Bz,
 * geometry read as Geant4 geometry, reading in regions and cuts, to initialize G4HepEm data
 * geometry read then into VecGeom, and synchronized to GPU
 * G4HepEm material-cuts couple indices mapped to VecGeom logical volume id's
 * physics processes for e-/e+ (including MSC) and gammas using G4HepEm.
 * scoring per placed volume, no sensitive detector feature
 * configurable particle gun via command line arguments
   * E.g., use `-gunpos -220,0,0 -gundir 1,0,0` for `testEm3.gdml`. For `cms2018.gdml`, the default gun is correct.

Electrons, positrons, and gammas are stored in separate containers in device memory.
Free positions in the storage are handed out with monotonic slot numbers, slots are not reused.
Active tracks are passed via three queues per particle type (see `struct ParticleQueues`).
Results are reproducible using one RANLUX++ state per track.

Additionally, the kernels score energy deposit and the charged track length per volume.

### Kernels

This example uses one stream per particle type to launch kernels asynchronously.
They are synchronized via a fourth stream using CUDA events.

#### `TransportElectrons<bool IsElectron>`

1. Obtain safety unless the track is currently on a boundary.
2. Determine physics step limit, including conversion to geometric step length according to MSC.
3. Query geometry (or optionally magnetic field) to get geometry step length.
4. Convert geometry to true step length according to MSC, apply net direction change and displacement.
5. Apply continuous effects; kill track if stopped.
6. If the particle reaches a boundary, perform relocation.
7. If not, and if there is a discrete process, hand over to interaction kernel.

#### `TransportGammas`

1. Determine the physics step limit.
2. Query VecGeom to get geometry step length (no magnetic field for neutral particles!).
3. If the particle reaches a boundary, perform relocation.
4. If not, and if there is a discrete process, hand over to interaction kernel.

#### Interaction kernels
In electrons.cu and gammas.cu, there is dedicated kernels for all discrete processes. These were
extracted from the main transport kernels in example18.
1. Find which particles will undergo the interaction that the respective kernel will take care of.
2. Sample the final state.
3. Update the primary and produce secondaries.

#### `FinishIteration`

Clear the queues and return the tracks in flight.
This kernel runs after all secondary particles were produced.

#### `InitPrimaries` and `InitParticleQueues`

Used to initialize multiple primary particles with separate seeds.
