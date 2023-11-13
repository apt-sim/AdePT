<!--
SPDX-FileCopyrightText: 2023 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

## example22

example22 demonstrates the integration of AdePT into Geant4 using G4VTrackingManager machinery.
The AdePT code is based on Example17.

The example can be run both as full Geant4 simulation or as Geant4 + AdePT simulation where the simulation of the EM particles is done on the GPU.

By default, the example runs with AdePT on (FTFP_BERT_AdePT physics list).
In order to run without AdePT (Geant4 only) you need to call with the the extra argument --no_AdePT as follows:

example22 -m example22.mac --no_AdePT

The selection of sensitive volumes can be done through:
/example22/detector/addsensitivevolume EAPD_01
/example22/detector/addsensitivevolume EAPD_02
etc.
