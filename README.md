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
- VecCore [library](https://github.com/root-project/veccore) 0.7.0 (recommended, but older versions >= 0.5.0 also work)
- VecGeom [library](https://gitlab.cern.ch/VecGeom/VecGeom) >= 1.1.20

A suitable environment may be set up either from CVMFS (requires the sft.cern.ch and projects.cern.ch repos
to be available on the local system):
```console
$ source /cvmfs/sft.cern.ch/lcg/views/devAdePT/latest/x86_64-centos7-gcc11-opt/setup.sh
```

or from the supplied [spack](https://spack.io) environment file:
```console
$ spack env create adept-spack ./scripts/spack.yaml
$ spack -e adept-spack concretize -f
$ spack -e adept-spack install
...
$ spack env activate -p adept-spack
```

Note that the above assumes your spack configuration defaults to use a suitable C++ compiler and has
`cuda_arch` set appropriately for the hardware you will be running on.

You can also build the packages manually as follows. To configure and build VecCore, simply run:
```console
$ cmake -S. -B./veccore-build -DCMAKE_INSTALL_PREFIX="<path_to_veccore_installation>"
$ cmake --build ./veccore-build --target install
```

Add your CUDA installation to the PATH and LD_LIBRARY_PATH environment variables, as in:
```console
$ export PATH=${PATH}:/usr/local/cuda/bin
$ export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
```

Find the CUDA architecture for the target GPU. If you installed the CUDA demo suite, the fastest way is to use the deviceQuery executable from `extras/demo_suite`. This lists the CUDA capability for all installed GPUs, remember the value for your target:
```console
$ /usr/local/cuda/extras/demo_suite/deviceQuery
Device 0: "GeForce RTX 2080 SUPER"
  CUDA Capability Major/Minor version number:    7.5 (cuda_architecture=75)
...
Device 1: "Quadro K4200"
  CUDA Capability Major/Minor version number:    3.0 (cuda_architecture=30)
```

To configure and build VecGeom, use the configuration options below, using as <cuda_architecture> the value from the step above:
```console
$ cmake -S. -B./vecgeom-build \
  -DCMAKE_INSTALL_PREFIX="<path_to_vecgeom_installation>" \
  -DCMAKE_PREFIX_PATH="<path_to_veccore_installation>" \
  -DVECGEOM_ENABLE_CUDA=ON \
  -DVECGEOM_GDML=ON \
  -DBACKEND=Scalar \
  -DCMAKE_CUDA_ARCHITECTURES=<cuda_architecture> \
  -DVECGEOM_USE_NAVINDEX=ON \
  -DCMAKE_BUILD_TYPE=Release
$ cmake --build ./vecgeom-build --target install -- -j6 ### build using 6 threads and install
```

To configure AdePT, simply run:

```console
$ cmake -S. -B./adept-build <otherargs>
```
As <otherargs> one needs to provide the paths to the dependence libraries VecCore and VecGeom, and optionally the path to the Alpaka installation (in case you want to build FisherPrice_Alpaka)
```console
   -DCMAKE_PREFIX_PATH="<path_to_veccore_installation>;<path_to_vecgeom_installation>;[<alpakaInstallDir>]" \
   -DCMAKE_CUDA_ARCHITECTURES=<cuda_architecture> \
   -DCMAKE_BUILD_TYPE=Release
```

To build, run:

```console
$ cmake --build ./adept-build -- -j6 ### build using 6 threads
```

The provided examples and tests can be run from the build directory:
```console
$ cd adept-build
$ CUDA_VISIBLE_DEVICES=0 BuildProducts/bin/<executable>   ### use the device number matching the selected <cuda_architecture>
```

## Copyright

AdePT code is Copyright (C) CERN, 2020, for the benefit of the AdePT project.
Any other code in the project has (C) and license terms clearly indicated.

Contributions of all authors to AdePT and their institutes are acknowledged in
the `AUTHORS.md` file.
