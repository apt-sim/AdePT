<!--
SPDX-FileCopyrightText: 2023 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

## example21

Example21 demonstrates the integration of AdePT into Geant4 using G4VTrackingManager machinery.
The AdePT code is based on Example17.

The example can be run both as full Geant4 simulation or as Geant4 + AdePT simulation where the simulation of the EM particles is done on the GPU.

The FTFP_BERT_AdePT physics list runs Geant4 with AdePT, while the FTFP_BERT_HepEm one runs Geant4 only (with G4HepEm EM models).

The selection of sensitive volumes can be done through:
/example17/detector/addsensitivevolume EAPD_01
/example17/detector/addsensitivevolume EAPD_02
etc.
