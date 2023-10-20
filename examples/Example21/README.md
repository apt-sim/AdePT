<!--
SPDX-FileCopyrightText: 2023 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

## example21

Example21 demonstrates the integration of AdePT into Geant4 using G4VTrackingManager machinery.
The AdePT code is based on Example17.

The example can be run both as full Geant4 simulation or as Geant4 + AdePT simulation where the simulation of the EM particles is done on the GPU.

By default, the example runs with AdePT on (FTFP_BERT_AdePT physics list).
In order to run without AdePT (Geant4 only) you need to call with the the extra argument --no_AdePT as follows:

<mybin>/example21 -m example21.mac --no_AdePT

The selection of sensitive volumes can be done through:
/example17/detector/addsensitivevolume EAPD_01
/example17/detector/addsensitivevolume EAPD_02
etc.
