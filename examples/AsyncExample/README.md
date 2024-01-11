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

The example can be run both as full Geant4 simulation or as Geant4 + AdePT simulation where the simulation of the EM calorimeter + daughter regions is done on the GPU. The selection of the two running modes can be done at run time using a command.

To activate AdePT:
/param/ActivateModel AdePT

and to inactivate it (run full Geant4):
/adeptint/adept/activate true
/param/InActivateModel AdePT

The selection of the detector region in which AdePT is suppossed to be used, is done through:
/adeptint/detector/regionname EcalRegion

The selection of sensitive volumes can be done through:
/adeptint/detector/addsensitivevolume EAPD_01
/adeptint/detector/addsensitivevolume EAPD_02
etc.

(the selection of sensitive volumes through auxiliary information in the GDML file is not yet supported)

The interfacing to Geant4 is done through the standard 'fast simulation' hooks. The EMShowerModel class implements the DoIt method which passes particles from Geant4 tracking to AdePT tracking. The particles are added to a buffer and asynchronously enqueued for GPU transport. The output are energy depositions per sensitive volume, which are converted into Geant4 hits and any 'outgoing' particles, which are put back on the Geant4 stack. The 'Flush' method is called when there are no more Geant4 particles on the stack (but before the event is finished) to block until the GPU transport for the G4 event in question has finished.

The SensitiveDetector class implements two 'ProcessHits' methods. One is the standard one used by the Geant4 scoring, while the second one is used to convert G4FastHits (so energy deposition from AdePT scoring) into the Geant4 hits (directly comparable to full simulation). The scoring consists of basic 'calorimeter scoring' where we have one hit per active calorimeter cell and we record the total energy deposited in that cell in the event.

The EventAction class prints all the hits at the end of the event, if
`/adeptint/event/verbose > 1`

Higher verbosity levels will print considerably more information about the G4 <--> AdePT communication.