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

To build, simply run:

```console
$ cmake -S. -B./adept-build <otherargs>
...
$ cmake --build ./adept-build
```

## Copyright

AdePT code is Copyright (C) CERN, 2020, for the benefit of the AdePT project.
Any other code in the project has (C) and license terms clearly indicated.

Contributions of all authors to AdePT and their institutes are acknowledged in
the `AUTHORS.md` file.
