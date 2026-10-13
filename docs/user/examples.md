<!--
SPDX-FileCopyrightText: 2026 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

# Examples

AdePT ships runnable examples in `examples/` and integration-focused regression
macros in `test/regression`.

## Build With Examples Enabled

```console
cmake -S . -B ./adept-build \
  -DADEPT_BUILD_EXAMPLES=ON \
  -DCMAKE_CUDA_ARCHITECTURES=<cuda_architecture>
cmake --build ./adept-build -- -j6
```

## Example 1

```console
cd adept-build
./BuildProducts/bin/example1 -h
./BuildProducts/bin/example1 -m <macro_file>
```

`example1` is the main standalone Geant4 application with AdePT integration.
Generated macros in the build tree include `example1.mac` and
`example1_ttbar.mac`. With `-DADEPT_GEOMETRY_VALIDATION=ON`, the
`example1_handoff.mac` macro (only `EcalRegion` on the GPU, with the geometry
validation of the handoff and of the boundary crossings) is generated and run by the `example1-handoff` test,
see [Geometry validation](geometry-validation.md).

## Integration Test Macros

The regression integration test also provides realistic macro setups, including
`/adept/*` command usage:

- `test/regression/macros/integrationtest.mac.in`
- `test/regression/scripts/test_ui_commands_template.mac`
