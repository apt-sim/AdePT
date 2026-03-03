# Runtime Parameters

<!--
This file is auto-generated.
Do not edit manually.
-->

This page is generated from:

- `src/AdePTConfigurationMessenger.cc` (authoritative command definitions and guidance)
- `test/regression/scripts/test_ui_commands_template.mac` (example invocations used in CI)

Regenerate with:

```console
python3 docs/scripts/generate_runtime_parameters.py
```

Categories are sourced from `// ADEPT_DOCS_SECTION: ...` markers in
`src/AdePTConfigurationMessenger.cc`.

Examples come from the CI UI command macro. If a command shows `-` in the
Example column, add an invocation in `test_ui_commands_template.mac`.

## Misc
| Command | Value type | Description | Constraints | Example |
| --- | --- | --- | --- | --- |
| `/adept/setVerbosity` | integer | Set verbosity level for the AdePT integration layer | - | `/adept/setVerbosity 0` |
| `/adept/setSeed` | integer | Set the base seed for the rng. Default: 1234567 | - | `/adept/setSeed 1242342` |

## Setting Up the GPU
| Command | Value type | Description | Constraints | Example |
| --- | --- | --- | --- | --- |
| `/adept/setMillionsOfTrackSlots` | double | Set the total number of track slots that will be allocated on the GPU, in millions | - | `/adept/setMillionsOfTrackSlots 1` |
| `/adept/setMillionsOfLeakSlots` | double | Set the total number of leak slots that will be allocated on the GPU, in millions | - | `/adept/setMillionsOfLeakSlots 1` |
| `/adept/setMillionsOfHitSlots` | double | Set the total number of hit slots that will be allocated on the GPU, in millions | - | `/adept/setMillionsOfHitSlots 1` |
| `/adept/setHitBufferThreshold` | double | Set the usage threshold at which the GPU steps are copied from the buffer and not taken directly by the G4 workers | parameter `HitBufferThreshold`; range `HitBufferThreshold>=0.&&HitBufferThreshold<=1.` | `/adept/setHitBufferThreshold 0.8` |
| `/adept/setCPUCapacityFactor` | double | Sets the CPUCapacity factor for scoring with respect to the GPU (see: /adept/setMillionsOfHitSlots). Must at least be 2.5 | parameter `CPUCapacityFactor`; range `CPUCapacityFactor>=2.5` | `/adept/setCPUCapacityFactor 3` |
| `/adept/setHitBufferSafetyFactor` | double | Sets the HitBuffer safety factor for stalling the GPU. If nParticlesInFlight * HitBufferSafetyFactor > NumHitSlotsLeft, the GPU will stall. Default: 1.5 | - | `/adept/setHitBufferSafetyFactor 1.5` |
| `/adept/setCUDAStackLimit` | integer | Set the CUDA device stack limit | - | `/adept/setCUDAStackLimit 8192` |
| `/adept/setCUDAHeapLimit` | integer | Set the CUDA device heap limit | - | `/adept/setCUDAHeapLimit 8192` |

## Specify the regions where the GPU is used
| Command | Value type | Description | Constraints | Example |
| --- | --- | --- | --- | --- |
| `/adept/setTrackInAllRegions` | bool | If true, particles are tracked on the GPU across the whole geometry | - | `/adept/setTrackInAllRegions false` |
| `/adept/addGPURegion` | string | Add a region in which transport will be done on GPU | - | `/adept/addGPURegion Layer1` |
| `/adept/removeGPURegion` | string | Remove a region in which transport will be done on GPU (so it will be done on the CPU) | - | `/adept/removeGPURegion Layer3` |

## User Actions
| Command | Value type | Description | Constraints | Example |
| --- | --- | --- | --- | --- |
| `/adept/CallUserSteppingAction` | bool | If true, the UserSteppingAction is called for on every step. WARNING: The steps are currently not sorted, that means it is not guaranteed that the UserSteppingAction is called in order, i.e., it could get called on the secondary before the primary has finished its track. NOTE: This means that every single step is recorded on GPU and send back to CPU, which can impact performance | - | `/adept/CallUserSteppingAction true` |
| `/adept/CallUserTrackingAction` | bool | If true, the PostUserTrackingAction is called for on every track. NOTE: This means that the last step of every track is recorded on GPU and send back to CPU | - | `/adept/CallUserTrackingAction true` |

## Special Settings
| Command | Value type | Description | Constraints | Example |
| --- | --- | --- | --- | --- |
| `/adept/SpeedOfLight` | bool | If true, all electrons, positrons, gammas handed over to AdePT are immediately killed. WARNING: Only to be used for testing the speed and fraction of EM, all results are wrong! | - | `/adept/SpeedOfLight false` |
| `/adept/setVecGeomGDML` | string | Temporary method for setting the geometry to use with VecGeom | - | `/adept/setVecGeomGDML $gdml_name` |
| `/adept/setCovfieBfieldFile` | string | Set the path the the covfie file for reading in an external magnetic field | - | - |
| `/adept/FinishLastNParticlesOnCPU` | integer | Set N, the number of last N particles per event that are finished on CPU. Default: 0. This is an important parameter for handling loopers in a magnetic field | - | `/adept/FinishLastNParticlesOnCPU 100` |
| `/adept/MaxWDTIterations` | integer | Set N, the number of maximum Woodcock tracking iterations per step before giving the gamma back to the normal gamma kernel. Default: 5. This can be used to optimize the performance in highly granular geometries | - | `/adept/MaxWDTIterations 5` |

## Special settings for G4EmStandard_AdePT physics constructor
| Command | Value type | Description | Constraints | Example |
| --- | --- | --- | --- | --- |
| `/adept/addWDTRegion` | string | Add a region in which the gamma transport is done via Woodcock tracking. NOTE: This ONLY applies to the AdePTPhysics, if the PhysicsList uses ANY other physics (which is done in LHCb, CMS, ATLAS) then this will have NO effect! | - | `/adept/addWDTRegion Layer1` |
| `/adept/addWDTKineticEnergyLimit` | double | Sets a kinetic energy limit above which the gamma transport is done via Woodcock tracking in the assigned regions. NOTE: This ONLY applies to the AdePTPhysics, if the PhysicsList uses ANY other physics (which is done in LHCb, CMS, ATLAS) then this will have NO effect! | - | `/adept/addWDTKineticEnergyLimit 200 keV` |
| `/adept/SetMultipleStepsInMSCWithTransportation` | bool | If true, this configures G4HepEm to use multiple steps in MSC on CPU. This does not affect GPU transport | - | `/adept/SetMultipleStepsInMSCWithTransportation true` |
| `/adept/SetEnergyLossFluctuation` | bool | If true, this configures G4HepEm to use energy loss fluctuations. This affects both CPU and GPU transport NOTE: This is only true for the AdePTPhysics in the examples! In all other physics lists the setting is taken directly from Geant4 and this parameter does not change it. | - | `/adept/SetEnergyLossFluctuation true` |

