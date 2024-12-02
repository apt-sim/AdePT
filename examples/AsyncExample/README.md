<!--
SPDX-FileCopyrightText: 2022 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

## Async Example

Example demonstrating how to use one single asynchronous AdePT from Geant4 using the fast simulation hooks.

This example is based on Example14 with the following modifications:
- A slot manager that can recycle slots is employed. This allows to transport considerably larger
  numbers of particles without running out of memory. The slot manager works as follows:
  - If a new slot is needed, a slot number is fetched atomically from a list of free slots.
  - Once a track dies, its slot is marked as to be freed. It cannot be freed while other tracks are transported
    because this might race with allocating new slots.
  - Periodically, the slots to be freed are copied over into the list of free slots.
- Only one instance of AdePT runs asynchronously in a separate thread.
  - Each G4 worker can enqueue tracks, which are all transported in parallel.
  - A transport loop is running continuously, which transports particles, injects new particles, and retrieves leaked tracks and hits.
  - The G4 workers communicate with the AdePT thread via state machines, so it is clear when events finish or need to start transporting.
  - As long as the G4 workers have CPU work to do, they don't block while the GPU transport is running.
  - Each track knows which G4 worker it came from, and the scoring structures are replicated for each G4 worker that is active.
- AdePT not only runs in the region that is named as GPU region in the config, it also transports particles in all daughter regions of the "AdePT region". This required refactoring the geometry visitor that sets up the GPU region.