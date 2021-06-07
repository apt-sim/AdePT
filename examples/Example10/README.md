<!--
SPDX-FileCopyrightText: 2021 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

## Example 10

Demonstrator of particle transportation on GPUs, with:

 * geometry via VecGeom and a magnetic field with constant Bz,
 * physics processes for e-/e+ and gammas using G4HepEm.

Electrons, positrons, and gammas are stored in separate containers in device memory.
Free positions in the storage are handed out with monotonic slot numbers, slots are not reused.
Active tracks are passed via three queues per particle type (see `struct ParticleQueues`).
Results are reproducible using one RANLUX++ state per track.

Compared to the previous Example 9, this one reduces the number of iterations with two ideas:
1. On the device, each transportation kernel can make several steps
   (up to a configurable limit, currently 4) as long as the particle
   stays alive and inside the volume. At volume boundary, relocation
   still happens in a separate kernel.
2. The host submits multiple kernels (currently 4 per particle type)
   into the appropriate streams and only synchronizes once at the end.
   Proper ordering on the device is ensured via events, extending the
   previous approach. The code uses an upper bound on the number of
   particles to determine the launch configuration.

In total, this reduces the number of iterations for trackML from 214
with Example 9 to 29 using this example with identical results (energy
deposition, number of secondaries, number of hit volume boundaries).
However, best performance for this setup is obtained with only one
step per kernel and submitting one kernel per iteration - in other
words, exactly what Example 9 does.

TL;DR: This example is currently meant as a proof-of-concept and for
playing with the parameters (number of steps and submitted kernels).

### Kernels

This example uses one stream per particle type to launch kernels asynchronously.
They are synchronized via a forth stream using CUDA events.

#### `TransportElectrons<bool IsElectron>`

1. Determine physics step limit.
2. Call magnetic field to get geometry step length.
3. Apply continuous effects; kill track if stopped.
4. If the particle reaches a boundary, enqueue for relocation.
5. If not, and if there is a discrete process:
   1. Sample the final state.
   2. Update the primary and produce secondaries.

#### `TransportGammas`

1. Determine the physics step limit.
2. Query VecGeom to get geometry step length (no magnetic field for neutral particles!).
3. If the particle reaches a boundary, enqueue for relocation.
4. If not, and if there is a discrete process:
   1. Sample the final state.
   2. Update the primary and produce secondaries.

#### `RelocateToNextVolume`

If the particle reached a boundary, update the state by pushing to the next volume.
This kernel is a parallel implementation of `LoopNavigator::RelocateToNextVolume` and uses one warp (32 threads) per particle.

1. One thread removes all volumes from the state that were left, eventually arriving at one volume that contains the pushed point.
2. All threads are used to check the daughter volumes in parallel. This step is repeated as long as one thread finds a volume to descend into.
3. Finally, one thread leaves all assembly volumes.

Note: This kernel is called three times, once for each particle type (with separate containers) in the respective stream.

#### `FinishIteration`

Clear the queues and return the tracks in flight.
This kernel runs after all secondary particles were produced.

#### `InitPrimaries` and `InitParticleQueues`

Used to initialize multiple primary particles with separate seeds.
