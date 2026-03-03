<!--
SPDX-FileCopyrightText: 2026 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

# Build From Source

## Configure

```console
cmake -S . -B ./adept-build \
  -DCMAKE_CUDA_ARCHITECTURES=<cuda_architecture> \
  -DCMAKE_BUILD_TYPE=Release \
  <otherargs>
```

If not using CVMFS, pass dependency prefixes:

```console
-DCMAKE_PREFIX_PATH="<path_to_geant4_installation>;<path_to_veccore_installation>;<path_to_vecgeom_installation>;<path_to_g4hepem_installation>;<path_to_hepmc3_installation>"
```

## Build

```console
cmake --build ./adept-build -- -j6
```

## Important CMake Options

| Option | Default | Description |
| --- | :---: | --- |
| `ADEPT_USE_EXT_BFIELD` | `OFF` | Use external B field from file via the covfie library. If ON, the constant field values are ignored and only B fields from file are accepted! |
| `ADEPT_USE_SPLIT_KERNELS` | `OFF` | Run split version of the transport kernels |
| `ADEPT_USE_SURF` | `OFF` | Enable surface model navigation on GPU (still in development, unstable for geometries with overlaps) |
| `ADEPT_MIXED_PRECISION` | `OFF` | Use B-field integration and surface model in mixed precision |
| `ADEPT_DEBUG_SINGLE_THREAD` | `OFF` | Run transport kernels in single thread mode |
| `ADEPT_DEBUG_TRACK` | `0` | Debug tracking level (0=off, >0=on with levels) |
| `ADEPT_ENFORCE_STRICT_FLAGS` | `OFF` | Use strict compiler flags, as also used in CMSSW. Many warnings are promoted to errors using this flag. |
| `ADEPT_BUILD_TESTING` | `OFF` | Build unit and regression tests |
| `ADEPT_BUILD_EXAMPLES` | `OFF` | Build examples |
| `ADEPT_ENABLE_POWER_METER` | `OFF` | Compile the code for consumption measurement |
| `ADEPT_STEPPINGACTION` | `NONE` | SteppingAction mode: `NONE`, `CMS`, or `LHCb` |
| `ADEPT_USE_BUILTIN_G4VG` | `ON` | Fetch and build G4VG as part of AdePT (used when Geant4 integration is enabled) |

## Optional: Run Tests

```console
ctest --test-dir ./adept-build --output-on-failure
```
