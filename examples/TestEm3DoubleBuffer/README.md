<!--
SPDX-FileCopyrightText: 2022 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

## TestEm3DoubleBuffer

Basically the same as TestEm3, but the electron, positron and gamma tracks are stored dense.
There are two arrays per track type.
Each transport kernel reads from one array and stores surviving tracks into the other array.
This way, no empty track slots are generated in the track arrays.
