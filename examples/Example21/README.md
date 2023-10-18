<!--
SPDX-FileCopyrightText: 2023 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

## example21

Example21 demonstrates the integration of AdePT into Geant4 using G4VTrackingManager machinery.
The AdePT code is based on Example17.

The example can be run both as full Geant4 simulation or as Geant4 + AdePT simulation where the simulation of the EM particles is done on the GPU. The selection of the two running modes can be done at run time using a command. 

To activate AdePT:



and to inactivate it (run full Geant4):





The selection of sensitive volumes can be done through:
/example17/detector/addsensitivevolume EAPD_01
/example17/detector/addsensitivevolume EAPD_02
etc.
