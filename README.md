<!--
SPDX-FileCopyrightText: 2020 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

# AdePT

Accelerated demonstrator of electromagnetic Particle Transport

## Build Requirements

The following packages are a required to build and run:

- CMake >= 3.18
- C/C++ Compiler with C++14 support
- CUDA Toolkit (tested 10.1, min version TBD)

To configure, simply run:

```console
$ cmake -S. -B./adept-build <otherargs>
```
As <otherargs> one needs to provide the paths to the dependence librarties VecCore and VecGeom:
```console
   -DVecCore_DIR=<path_to_veccore_installation>/lib/cmake/VecCore \
   -DVecGeom_DIR=<path_to_vecgeom_installation>/lib/cmake/VecGeom \
   [-DVc_DIR=<path_to_vc_installation/lib/cmake/Vc] #only in case VecGeom was compiled using Vc backend
   [-DCMAKE_PREFIX_PATH=<alpakaInstallDir>] #only in case you want to build FisherPrice_Alpaka. <alpakaInstallDir> should point at the folder in which "include/alpaka/" is found.
```
When running CentOS, replace `lib/` by `lib64`.

To build, run:

```console
$ cmake --build ./adept-build
```

## Copyright

AdePT code is Copyright (C) CERN, 2020, for the benefit of the AdePT project.
Any other code in the project has (C) and license terms clearly indicated.

Contributions of all authors to AdePT and their institutes are acknowledged in
the `AUTHORS.md` file.

