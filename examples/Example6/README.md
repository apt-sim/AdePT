<!--
SPDX-FileCopyrightText: 2021 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

## Example 6

Demonstrator of particle transportation on GPUs, with:

 * one container for all tracks,
 * geometry via VecGeom and a magnetic field with constant Bz,
 * physics processes for e-/e+ using G4HepEm.

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
5. If not, and if there is a discrete process, enqueue the particle to perform the interaction.

#### `RelocateToNextVolume`

If the particle reached a boundary, update the state by pushing to the next volume.
This kernel is a parallel implementation of `LoopNavigator::RelocateToNextVolume` and uses one warp (32 threads) per particle.

1. One thread removes all volumes from the state that were left, eventually arriving at one volume that contains the pushed point.
2. All threads are used to check the daughter volumes in parallel. This step is repeated as long as one thread finds a volume to descend into.
3. Finally, one thread leaves all assembly volumes.

#### `PerformDiscreteInteractions`

Perform the discrete interaction and sample the final state, updating the primary and producing secondaries.
Gammas immediately pair-produce if their energy is above the threshold, or deposit their energy.

#### `init_track`

Init the primary track, seed the RANLUX++ state.
