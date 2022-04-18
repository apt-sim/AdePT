<!--
SPDX-FileCopyrightText: 2022 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

## example17

Example17 is identical o Example14 (see below) but using TrackManager for reusing track slots.

Example demonstrating how to call AdePT from Geant4 using the fast simulation hooks.

NOTE: This requires the buffer 'flush' method to be implemented through the FastSimulationManager(s)
and called in the event loop of the G4EventManager (not part of the Geant4 release yet). Patch provided to be applied on the code.

This example is based on Example13, but instead of a 'standalone' AdePT application, it provides 'AdeptIntegration' class allowing it to be called from another transport code.

The example can be run both as full Geant4 simulation or as Geant4 + AdePT simulation where the simulation of the EM calorimeter is done on the GPU. The selection of the two running modes can be done at run time using a command. 

To activate AdePT:
/param/ActivateModel AdePT

and to inactivate it (run full Geant4):
/example17/adept/activate true 
/param/InActivateModel AdePT

The selection of the detector region in which AdePT is suppossed to be used, is done through:
/example17/detector/regionname EcalRegion

The selection of sensitive volumes can be done through:
/example17/detector/addsensitivevolume EAPD_01
/example17/detector/addsensitivevolume EAPD_02
etc.

(the selection of sensitive volumes through auxiliary information in the GDML file is not yet supported)

The interfacing to Geant4 is done through the standard 'fast simulation' hooks. The EMShowerModel class implements the DoIt method which passes particles from Geant4 tracking to AdePT tracking. The particles are added to the buffer and when the chosen threshold is reached, the 
shower simulation on the GPU is triggered. The output are energy depositions per sensitive volume, which are converted into Geant4 hits and any 'outgoing' particles, which are put back on the Geant4 stack. The 'Flush' method is called when there are no more Geant4 particles on the stack (but before the event is finished) to simulate any remaining particles in the AdePT buffer (below the threshold).

The EMShowerModel object is instanciated and initialized in DetectorConstruction::ConstructSDandField(), which results in an AdeptIntegration object instanciated per thread. 

The SensitiveDetector class implements two 'ProcessHits' methods. One is the standard one used by the Geant4 scoring, while the second one is used to convert G4FastHits (so energy deposition from AdePT scorring) into the Geant4 hits (directly comparable to full simulation). The scoring consists of basic 'calorimeter scoring' where we have one hit per active calorimeter cell and we record the total energy deposited in that cell in the event.

The EventAction class prints all the hits at the end of the event, if /example17/event/verbose > 1