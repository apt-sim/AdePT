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

## Geant4 Physics Integration

After linking AdePT, integrate it into your Geant4 physics setup with the
approach that matches your EM physics configuration.

### Case 1: Complete AdePT Physics List

If you use the `FTFP_BERT` physics list as is, with `G4EmStandardPhysics` option
0 and no specializations such as Woodcock tracking, use the complete AdePT
physics list `FTFP_BERT_AdePT`.

### Case 2: AdePT EM Constructor

If you use a different hadronic physics setup, but still use
`G4EmStandardPhysics` option 0 without special EM customizations, register
`G4EmStandardPhysics_AdePT` instead of `G4EmStandardPhysics`.

The public headers for this integration are under `AdePT/g4integration`. For
example, a physics-list wrapper can include and register the AdePT EM physics
constructor as:

```cpp
#include <AdePT/g4integration/G4EmStandardPhysics_AdePT.hh>

RegisterPhysics(new G4EmStandardPhysics_AdePT(ver));
```

### Case 3: Direct Tracking-Manager Integration

If you use another EM configuration, such as a different `G4EmStandardPhysics`
option, Woodcock tracking, or experiment-specific G4HepEm settings, register
`AdePTTrackingManager` directly for electrons, positrons, and gammas. This is
the pattern to use when wrapping an existing custom EM physics constructor.

The direct tracking-manager integration needs both the AdePT public integration
headers and the G4HepEm configuration header:

```cpp
#include <AdePT/g4integration/AdePTConfiguration.hh>
#include <AdePT/g4integration/AdePTTrackingManager.hh>

#include <G4HepEmConfig.hh>
#include <G4HepEmParameters.hh>

#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4Positron.hh"
#include "G4SystemOfUnits.hh"
```

In the physics constructor, keep the `AdePTConfiguration` alive for the lifetime
of the tracking manager. Configure AdePT first, construct the tracking manager,
then configure the underlying G4HepEm tracking settings and attach the manager to
the particle definitions:

```cpp
// Required: In the physics-constructor constructor or equivalent setup:
fAdePTConfiguration = new AdePTConfiguration();

// Optional AdePT-side defaults. If these are set in ConstructProcess(), they override the usual /adept/* macro commands, since those are normally processed before /run/initialize. This might be useful to avoid having to call many UI commands for a fixed setup.
fAdePTConfiguration->SetCUDAStackLimit(8192);
fAdePTConfiguration->SetTrackInAllRegions(true);
fAdePTConfiguration->SetCallUserTrackingAction(true);

// Required: In ConstructProcess(), after registering the EM processes for the custom
// EM constructor, create an AdePTTrackingManager:
fTrackingManager = new AdePTTrackingManager(fAdePTConfiguration, /*verbosity=*/0);

// Optional: change the G4HepEm configuration
auto *g4hepemConfig = fTrackingManager->GetG4HepEmConfig();

// Optional: Match the G4HepEm behavior to the EM constructor that is being replaced.
g4hepemConfig->SetMultipleStepsInMSCWithTransportation(false);
g4hepemConfig->SetEnergyLossStepLimitFunctionParameters(0.8, 1.0 * CLHEP::mm);

// Optional: enable Woodcock tracking for the relevant detector regions.
g4hepemConfig->SetWoodcockTrackingRegion("ECALRegion");

// Optional: tune low-energy tracking cuts or other G4HepEm parameters.
auto *g4hepemParameters = g4hepemConfig->GetG4HepEmParameters();
g4hepemParameters->fElectronTrackingCut = 1.0 * CLHEP::MeV;
g4hepemParameters->fGammaTrackingCut    = 1.0 * CLHEP::MeV;

// Required: Attach the tracking manager to e-/e+ and gamma.
G4Electron::Definition()->SetTrackingManager(fTrackingManager);
G4Positron::Definition()->SetTrackingManager(fTrackingManager);
G4Gamma::Definition()->SetTrackingManager(fTrackingManager);
```

Only the construction of `AdePTConfiguration`, `AdePTTrackingManager`, and the
three `SetTrackingManager` calls are mandatory. The `G4HepEmConfig` calls should
mirror the custom EM setup that is being replaced. For example, an ATLAS-like setup
may add Woodcock regions and disable multiple MSC transportation steps, while an
LHCb-like option-2 setup may set the energy-loss step-limit function and
tracking cuts.

This step enables the AdePT transport/tracking integration in the Geant4
application.

## Runtime Configuration

At runtime, AdePT parameters are configured through Geant4 UI commands in the
`/adept/` directory. See:

- [Runtime Parameters](runtime-parameters.md)

In practice this means your Geant4 macro configures `/adept/*` before
`/run/initialize`.

## Reference Integration Examples

See repository examples and regression integration test:

- `examples/Example1`
- `test/regression/IntegrationTest`
