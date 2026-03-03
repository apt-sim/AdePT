<!--
SPDX-FileCopyrightText: 2026 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

# Integration Into Geant4 Applications

AdePT is consumed as a CMake package.

## CMake Integration

```cmake
find_package(AdePT REQUIRED)
```

For targets that use AdePT:

```cmake
cuda_rdc_target_include_directories(example_target <SCOPE>
  <TARGET INCLUDE DIRECTORIES>
  ${AdePT_INCLUDE_DIRS})

cuda_rdc_target_link_libraries(example_target <SCOPE>
  <TARGET LINK LIBRARIES>
  ${AdePT_LIBRARIES})
```

`cuda_rdc_*` is required to keep CUDA device-link behavior consistent with the
VecGeom/AdePT build model.

## Runtime Configuration

At runtime, AdePT parameters are configured through Geant4 UI commands in the
`/adept/` directory. See:

- [Runtime Parameters](runtime-parameters.md)

In practice this means your Geant4 macro configures `/adept/*` before
`/run/initialize`.

## Reference Integration Examples

See repository examples and regression integration test:

- `examples/Example1`
- `examples/AsyncExample`
- `test/regression/IntegrationTest`
