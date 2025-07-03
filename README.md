<!--
SPDX-FileCopyrightText: 2020 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

# AdePT

Accelerated demonstrator of electromagnetic Particle Transport

## Build Requirements

The following packages are a required to build and run:

- CMake >= 3.25.2
- C/C++ Compiler with C++20 support
- CUDA Toolkit (> 12 recommended, tested 10.1, min version TBD)
- [Geant4](https://gitlab.cern.ch/geant4/geant4) > 11.0
- VecCore [library](https://github.com/root-project/veccore) 0.8.2
- VecGeom [library](https://gitlab.cern.ch/VecGeom/VecGeom) >=  2.0.0-rc.4 
- G4HepEm [library](https://github.com/mnovak42/g4hepem) >= tag 20250610
- optional: HepMC3 [library](https://gitlab.cern.ch/hepmc/HepMC3)

### Setting up the environment and installing the dependencies

#### 1. From CVMFS (recommended for users)

A suitable environment may be set up either from CVMFS (requires the sft.cern.ch and projects.cern.ch repos
to be available on the local system):
```console
$ source /cvmfs/sft.cern.ch/lcg/views/devAdePT/latest/x86_64-el9-gcc13-opt/setup.sh
```

#### 2. Via Spack (outdated, currently not recommended)

The dependencies may be installed via spack supplied [spack](https://spack.io) environment file:
Note: the spack environment has not been tested for a while and might be outdated
```console
$ spack env create adept-spack ./scripts/spack.yaml
$ spack -e adept-spack concretize -f
$ spack -e adept-spack install
...
$ spack env activate -p adept-spack
```

Note that the above assumes your spack configuration defaults to use a suitable C++ compiler and has
`cuda_arch` set appropriately for the hardware you will be running on.

#### 3. Manually (recommended for developers)

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

Find the CUDA architecture for the target GPU. The GPU can be found via `nvidia-smi`. Then, the list of compute capabilities is available on the [nvidia-website](https://developer.nvidia.com/cuda-gpus). 
Then, the required `cuda_architecture` is the compute capability without the `.` for the version, i.e., compute capability 8.9 corresponds to `<cuda_architecture> = 89`.

To configure and build VecGeom, use the configuration options below, using as <cuda_architecture> the value from the step above:
```console
$ cmake -S. -B./vecgeom-build \
  -DCMAKE_INSTALL_PREFIX="<path_to_vecgeom_installation>" \
  -DCMAKE_PREFIX_PATH="<path_to_veccore_installation>" \
  -DVECGEOM_ENABLE_CUDA=ON \
  -DVECGEOM_GDML=ON \
  -DBACKEND=Scalar \
  -DCMAKE_CUDA_ARCHITECTURES=<cuda_architecture> \
  -DCMAKE_BUILD_TYPE=Release
$ cmake --build ./vecgeom-build --target install -- -j 6 ### build using 6 threads and install
```
For faster performance with with the solid model, the option `-DVECGEOM_NAV=index` can be used over the default of `-DVECGEOM_NAV=tuple`.


To configure and build Geant4, there are many options available, see the [documentation](https://geant4-userdoc.web.cern.ch/UsersGuides/InstallationGuide/html/installguide.html#geant4-build-options). For AdePT it is important that VecGeom is switched off via ` -DGEANT4_USE_USOLIDS=OFF` (which it is by default), to avoid conflicting the GPU VecGeom build needed for AdePT. Furthermore GDML is required. For example, the following settings work:

```console
cmake -S. -B./geant4-build \
 -DCMAKE_INSTALL_PREFIX="<path_to_geant4_installation>" \
 -DGEANT4_USE_SYSTEM_EXPAT=OFF \
 -DGEANT4_USE_GDML=ON \
 -DGEANT4_INSTALL_DATA=ON \
 -DGEANT4_USE_USOLIDS=OFF
cmake --build ./geant4-build --target install -- -j 6
```

To configure and build G4HepEm, use the configuration options below:
```console
$ cmake -S. -B./g4hepem-build \
  -DCMAKE_INSTALL_PREFIX="<path_to_g4hepem_installation>" \
  -DCMAKE_PREFIX_PATH="<path_to_geant4_installation>" \
  -DG4HepEm_EARLY_TRACKING_EXIT=ON \
  -DG4HepEm_CUDA_BUILD=ON
$ cmake --build ./g4hepem-build --target install -- -j 6 ### build using 6 threads and install
```

HepMC3 is optional, but strongly recommended to be able to use the HepMC3 gun in AdePT for realistic events:
```console
cmake -S. -B./hepmc3-build \
  -DCMAKE_INSTALL_PREFIX=<path_to_hepmc3_installation>
  -DHEPMC3_ENABLE_ROOTIO=OFF \
  -DHEPMC3_ENABLE_PYTHON=OFF \
cmake --build ./hepmc3-build --target install -- -j 6 ### build using 6 threads and install
```

### Building AdePT

To configure AdePT, simply run:

```console
$ cmake -S. -B./adept-build \
   -DCMAKE_CUDA_ARCHITECTURES=<cuda_architecture> \
   -DCMAKE_BUILD_TYPE=Release \
   <otherargs>
```
where `<otherargs>` are additional options from the Build Options below to configure the build.
If one did not rely on an environment setup via CVMFS or Spack, one also must provide in `<otherargs>` the paths to the dependence libraries VecCore, VecGeom, G4HepEm, and optionally HepMC3
```console
   -DCMAKE_PREFIX_PATH="<path_to_veccore_installation>;<path_to_vecgeom_installation>;<path_to_g4hepem_installation>;<path_to_hepmc3_installation>" \
```

#### Build Options
The table below shows the available CMake options for AdePT that may be used to configure the build:

|Option|Default|Description|
|------|:-----:|-----------|
|ASYNC_MODE|OFF|Enable the asynchronous kernel scheduling mode. Recommended and significantly faster than the synchronous mode in many occasions |
|ADEPT_USE_EXT_BFIELD|OFF|Use external B field from file via the covfie library. If ON, the constant field values are ignored and only B fields from file are accepted! |
|USE_SPLIT_KERNELS|OFF| Run split version of the transport kernels. Requires ASYNC_MODE=ON |
|ADEPT_USE_SURF|OFF| Enable surface model navigation on GPU (still in development, unstable for geometries with overlaps) |
|ADEPT_USE_SURF_SINGLE|OFF|Use mixed precision in the surface model|
|DEBUG_SINGLE_THREAD|OFF| Run transport kernels in single thread mode |
|ADEPT_DEBUG_TRACK|0| Debug tracking level (0=off, >0=on with levels) |

To build, run:

```console
$ cmake --build ./adept-build -- -j6 ### build using 6 threads
```

The provided examples and tests can be run from the build directory. `example1` is a standalone G4 application with AdePT integration.
It can be run with
```console
$ cd adept-build
$ ./BuildProducts/bin/example1 -m <macro_file>   ### for more option, use -h
```
In the build folder, several example `<macro_file>` are generated, such as `example1.mac` or `example1_ttbar.mac`.

## Including AdePT in other CMake projects

In order to include AdePT in a separate project we need to run:

```
find_package(AdePT)
```

Which has the same dependencies as before (VecGeom, VecCore and G4HepEM).

Then, for the targets using AdePT:

```
cuda_rdc_target_include_directories(example_target <SCOPE> 
                                    <TARGET INCLUDE DIRECTORIES>
                                    ${AdePT_INCLUDE_DIRS})

cuda_rdc_target_link_libraries(example_target <SCOPE>
                               <TARGET LINK LIBRARIES>
                               ${AdePT_LIBRARIES})
```
Note that the cuda_rdc is required, which is inherited from VecGeom and needed to avoid multi-cuda dependency issues.


## Copyright

AdePT code is Copyright (C) CERN, 2020, for the benefit of the AdePT project.
Any other code in the project has (C) and license terms clearly indicated.

Contributions of all authors to AdePT and their institutes are acknowledged in
the `AUTHORS.md` file.
