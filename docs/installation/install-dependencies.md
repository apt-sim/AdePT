# Install Dependencies

AdePT requires the following software stack:

- CMake >= 3.25.2
- C/C++ compiler with C++20 support
- CUDA toolkit (CUDA 12+ recommended)
- Geant4 > 11.0
- VecCore
- VecGeom (with CUDA and GDML support)
- G4HepEm
- Optional: HepMC3

## Recommended: CVMFS Environment

If CVMFS is available, this is the fastest way to get a compatible stack:

```console
source /cvmfs/sft.cern.ch/lcg/views/devAdePT/latest/x86_64-el9-gcc13-opt/setup.sh
```

## Manual Dependency Builds (Developer Path)

Use this when you need full control over versions and build options.

### VecCore

```console
cmake -S . -B ./veccore-build \
  -DCMAKE_INSTALL_PREFIX=<path_to_veccore_installation>
cmake --build ./veccore-build --target install -- -j6
```

### VecGeom

```console
cmake -S . -B ./vecgeom-build \
  -DCMAKE_INSTALL_PREFIX=<path_to_vecgeom_installation> \
  -DCMAKE_PREFIX_PATH=<path_to_veccore_installation> \
  -DVECGEOM_ENABLE_CUDA=ON \
  -DVECGEOM_GDML=ON \
  -DBACKEND=Scalar \
  -DCMAKE_CUDA_ARCHITECTURES=<cuda_architecture> \
  -DCMAKE_BUILD_TYPE=Release
cmake --build ./vecgeom-build --target install -- -j6
```

Tip: you can set `-DVECGEOM_NAV=<mode>` to tune memory/speed trade-offs.

- Use `-DVECGEOM_NAV=index` when geometry is sufficiently small. It uses more
  memory to build the geometry tree, but is faster at runtime.
- Use `-DVECGEOM_NAV=tuple` for larger geometries or tighter memory budgets. It
  uses less memory, but is slightly slower.


### Geant4

```console
cmake -S . -B ./geant4-build \
  -DCMAKE_INSTALL_PREFIX=<path_to_geant4_installation> \
  -DGEANT4_USE_SYSTEM_EXPAT=OFF \
  -DGEANT4_USE_GDML=ON \
  -DGEANT4_INSTALL_DATA=ON \
  -DGEANT4_USE_USOLIDS=OFF
cmake --build ./geant4-build --target install -- -j6
```

### G4HepEm

```console
cmake -S . -B ./g4hepem-build \
  -DCMAKE_INSTALL_PREFIX=<path_to_g4hepem_installation> \
  -DCMAKE_PREFIX_PATH=<path_to_geant4_installation> \
  -DG4HepEm_EARLY_TRACKING_EXIT=ON \
  -DG4HepEm_CUDA_BUILD=ON
cmake --build ./g4hepem-build --target install -- -j6
```

### Optional: HepMC3

```console
cmake -S . -B ./hepmc3-build \
  -DCMAKE_INSTALL_PREFIX=<path_to_hepmc3_installation> \
  -DHEPMC3_ENABLE_ROOTIO=OFF \
  -DHEPMC3_ENABLE_PYTHON=OFF
cmake --build ./hepmc3-build --target install -- -j6
```
