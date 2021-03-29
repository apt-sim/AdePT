<!--
SPDX-FileCopyrightText: 2021 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

## Example 8

Demonstrator of particle transportation on GPUs, with:

 * geometry via VecGeom and a magnetic field with constant Bz,
 * physics processes for e-/e+ using G4HepEm.

Tracks are entirely stored in device memory (not unified!) with a monotonic slot number.
Slots are not reused and active tracks are passed via queues.
Results are reproducible using one RANLUX++ state per track.

### Kernels

Note: There are no gamma processes in this example.
To avoid infinite particles, they immediately pair-produce, if allowed, or deposit their energy.
This also means the kernels only deal with electrons and positrons.

#### `PerformStep`

1. Determine physics step limit.
2. Call magnetic field to get geometry step length.
3. Apply continuous effects; kill track if stopped.
4. If the particle reaches a boundary, enqueue for relocation.
5. If not, and if there is a discrete process:
 1. Sample the final state.
 2. Update the primary and produce secondaries.
    Gammas immediately pair-produce if their energy is above the threshold, or deposit their energy.

#### `RelocateToNextVolume`

If the particle reached a boundary, update the state by pushing to the next volume.
This kernel is a parallel implementation of `LoopNavigator::RelocateToNextVolume` and uses one warp (32 threads) per particle.

1. One thread removes all volumes from the state that were left, eventually arriving at one volume that contains the pushed point.
2. All threads are used to check the daughter volumes in parallel. This step is repeated as long as one thread finds a volume to descend into.
3. Finally, one thread leaves all assembly volumes.

#### `FinishStep`

Clear the queues and return the tracks in flight.

#### `InitPrimaries` and `InitQueue`

Used to initialize multiple primary particles with separate seeds.
