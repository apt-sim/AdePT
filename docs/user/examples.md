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
Generated macros in the build tree include `example1.mac`,
`example1_ttbar.mac` and `example1_handoff.mac`, run by the `example1-handoff`
test (see [Handoff validation](handoff-validation.md)).

### Magnetic field commands

The field commands available in `example1` and `integrationTest` depend on the
`ADEPT_USE_EXT_BFIELD` build option:

| Build option | Supported commands |
| --- | --- |
| `OFF` | `/detector/setField Bx By Bz unit` for a uniform field |
| `ON` | `/detector/setCovfieBfieldFile path` for Geant4 and `/adept/setCovfieBfieldFile path` for the GPU |

Unsupported field commands fail immediately and stop macro execution. This also
applies to a zero uniform-field vector in an external-field build and to a uniform
field supplied alongside a covfie file. For an external field, supply the same map
to both file commands. With no field commands, the default is no field.
The uniform-field `example1-handoff` test is registered only with
`ADEPT_USE_EXT_BFIELD=OFF`.

## Integration Test Macros

The regression integration test also provides realistic macro setups, including
`/adept/*` command usage:

- `test/regression/macros/integrationtest.mac.in`
- `test/regression/scripts/test_ui_commands_template.mac`
