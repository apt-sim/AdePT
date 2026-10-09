// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0
/// \brief Implementation of the AdePTConfigurationMessenger class
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include <AdePT/g4integration/config/AdePTConfigurationMessenger.hh>
#include <AdePT/g4integration/AdePTConfiguration.hh>

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4Tokenizer.hh"
#include "G4Exception.hh"

namespace {
void ReportLockedTransportInitializationOption(G4UIcommand *command)
{
  G4ExceptionDescription description;
  description << command->GetCommandPath()
              << " cannot be changed after AdePT initialization has started. Set it before the first run.";
  command->CommandFailed(description);
}
} // namespace

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

AdePTConfigurationMessenger::AdePTConfigurationMessenger(AdePTConfiguration *adeptConfiguration)
    : G4UImessenger(), fAdePTConfiguration(adeptConfiguration)
{
  fDir = std::make_unique<G4UIdirectory>("/adept/");
  fDir->SetGuidance("adept configuration messenger");

  // NOTE: The ADEPT_DOCS_SECTION markers are parsed by
  // docs/scripts/generate_runtime_parameters.py. Keep the marker prefix and
  // section names stable unless the generator is updated.
  // ADEPT_DOCS_SECTION: Misc
  fSetVerbosityCmd = std::make_unique<G4UIcmdWithAnInteger>("/adept/setVerbosity", this);
  fSetVerbosityCmd->SetGuidance("Set verbosity level for the AdePT integration layer");

  fSetAdePTSeedCmd = std::make_unique<G4UIcmdWithAnInteger>("/adept/setSeed", this);
  fSetAdePTSeedCmd->SetGuidance("Set the base seed for the rng. Default: 1234567");

  // ADEPT_DOCS_SECTION: Setting Up the GPU
  fSetMillionsOfTrackSlotsCmd = std::make_unique<G4UIcmdWithADouble>("/adept/setMillionsOfTrackSlots", this);
  fSetMillionsOfTrackSlotsCmd->SetGuidance(
      "Set the total number of track slots that will be allocated on the GPU, in millions");

  fSetMillionsOfHitSlotsCmd = std::make_unique<G4UIcmdWithADouble>("/adept/setMillionsOfHitSlots", this);
  fSetMillionsOfHitSlotsCmd->SetGuidance(
      "Set the total number of hit slots that will be allocated on the GPU, in millions");

  fSetHitBufferFlushThresholdCmd = std::make_unique<G4UIcmdWithADouble>("/adept/setHitBufferThreshold", this);
  fSetHitBufferFlushThresholdCmd->SetGuidance(
      "Set the usage threshold at which the GPU steps are copied from the buffer and not taken "
      "directly by the G4 workers");
  fSetHitBufferFlushThresholdCmd->SetParameterName("HitBufferThreshold", false);
  fSetHitBufferFlushThresholdCmd->SetRange("HitBufferThreshold>=0.&&HitBufferThreshold<=1.");

  fSetCPUCapacityFactorCmd = std::make_unique<G4UIcmdWithADouble>("/adept/setCPUCapacityFactor", this);
  fSetCPUCapacityFactorCmd->SetGuidance(
      "Sets the CPUCapacity factor for returned GPU steps with respect to the GPU (see: /adept/setMillionsOfHitSlots). "
      "Must at least be 2.5");
  fSetCPUCapacityFactorCmd->SetParameterName("CPUCapacityFactor", false);
  fSetCPUCapacityFactorCmd->SetRange("CPUCapacityFactor>=2.5");

  fSetHitBufferSafetyFactorCmd = std::make_unique<G4UIcmdWithADouble>("/adept/setHitBufferSafetyFactor", this);
  fSetHitBufferSafetyFactorCmd->SetGuidance(
      "Sets the HitBuffer safety factor for stalling the GPU. If nParticlesInFlight * HitBufferSafetyFactor > "
      "NumHitSlotsLeft, the GPU will stall. Default: 1.5 ");

  fSetCUDAStackLimitCmd = std::make_unique<G4UIcmdWithAnInteger>("/adept/setCUDAStackLimit", this);
  fSetCUDAStackLimitCmd->SetGuidance("Set the CUDA device stack limit");

  fSetCUDAHeapLimitCmd = std::make_unique<G4UIcmdWithAnInteger>("/adept/setCUDAHeapLimit", this);
  fSetCUDAHeapLimitCmd->SetGuidance("Set the CUDA device heap limit");

  // ADEPT_DOCS_SECTION: Specify the regions where the GPU is used
  fSetTrackInAllRegionsCmd = std::make_unique<G4UIcmdWithABool>("/adept/setTrackInAllRegions", this);
  fSetTrackInAllRegionsCmd->SetGuidance("If true, particles are tracked on the GPU across the whole geometry");

  fAddRegionCmd = std::make_unique<G4UIcmdWithAString>("/adept/addGPURegion", this);
  fAddRegionCmd->SetGuidance("Add a region in which transport will be done on GPU");

  fRemoveRegionCmd = std::make_unique<G4UIcmdWithAString>("/adept/removeGPURegion", this);
  fRemoveRegionCmd->SetGuidance(
      "Remove a region in which transport will be done on GPU (so it will be done on the CPU)");

  // ADEPT_DOCS_SECTION: User Actions
  fSetCallUserActionsCmd = std::make_unique<G4UIcmdWithABool>("/adept/CallUserActions", this);
  fSetCallUserActionsCmd->SetGuidance(
      "If true, both UserTrackingAction and UserSteppingAction are called for returned GPU steps. This does not "
      "change which GPU steps are returned; combine it with /adept/returnFirstAndLastStep and/or "
      "/adept/returnAllSteps to make the required steps available on the host.");
  fSetCallUserActionsCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fSetCallUserSteppingActionCmd = std::make_unique<G4UIcmdWithABool>("/adept/CallUserSteppingAction", this);
  fSetCallUserSteppingActionCmd->SetGuidance(
      "If true, the UserSteppingAction is called for returned GPU steps. WARNING: The steps are currently not sorted, "
      "that means it is not guaranteed that the UserSteppingAction is called in order, i.e., it could get called on "
      "the secondary before the primary has finished its track. This does not change which GPU steps are returned; "
      "use /adept/returnAllSteps to return every GPU step.");
  fSetCallUserSteppingActionCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fSetCallUserTrackingActionCmd = std::make_unique<G4UIcmdWithABool>("/adept/CallUserTrackingAction", this);
  fSetCallUserTrackingActionCmd->SetGuidance(
      "If true, the UserTrackingAction is called for returned GPU track starts and ends. This does not change which "
      "GPU steps are returned; use /adept/returnFirstAndLastStep to return the first and last GPU steps.");
  fSetCallUserTrackingActionCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fSetReturnFirstAndLastStepCmd = std::make_unique<G4UIcmdWithABool>("/adept/returnFirstAndLastStep", this);
  fSetReturnFirstAndLastStepCmd->SetGuidance(
      "If true, the first and last GPU steps of each track are returned to the host. This only controls GPU step "
      "returning; use /adept/CallUserActions or the individual action commands to invoke user actions on them.");

  fSetReturnAllStepsCmd = std::make_unique<G4UIcmdWithABool>("/adept/returnAllSteps", this);
  fSetReturnAllStepsCmd->SetGuidance(
      "If true, every GPU step is returned to the host. This only controls GPU step returning; use "
      "/adept/CallUserActions or the individual action commands to invoke user actions on them.");

  // ADEPT_DOCS_SECTION: Special Settings
  fSetSpeedOfLightCmd = std::make_unique<G4UIcmdWithABool>("/adept/SpeedOfLight", this);
  fSetSpeedOfLightCmd->SetGuidance(
      "If true, all electrons, positrons, gammas handed over to AdePT are immediately killed. WARNING: Only to be used "
      "for testing the speed and fraction of EM, all results are wrong!");

  fSetGDMLCmd = std::make_unique<G4UIcmdWithAString>("/adept/setVecGeomGDML", this);
  fSetGDMLCmd->SetGuidance("Temporary method for setting the geometry to use with VecGeom");

  fSetCovfieFileCmd = std::make_unique<G4UIcmdWithAString>("/adept/setCovfieBfieldFile", this);
  fSetCovfieFileCmd->SetGuidance("Set the path the the covfie file for reading in an external magnetic field");

  fSetFinishOnCpuCmd = std::make_unique<G4UIcmdWithAnInteger>("/adept/FinishLastNParticlesOnCPU", this);
  fSetFinishOnCpuCmd->SetGuidance("Set N, the number of last N particles per event that are finished on CPU. Default: "
                                  "0. This is an important parameter for handling loopers in a magnetic field");

  fSetMaxChargedLooperCountCmd = std::make_unique<G4UIcmdWithAnInteger>("/adept/MaxChargedLooperCount", this);
  fSetMaxChargedLooperCountCmd->SetGuidance(
      "Set N, the maximum charged-particle looper count before killing an electron or positron. Default: 500. "
      "Set to 0 to disable this charged-looper kill.");
  fSetMaxChargedLooperCountCmd->SetParameterName("MaxChargedLooperCount", false);
  fSetMaxChargedLooperCountCmd->SetRange("MaxChargedLooperCount>=0&&MaxChargedLooperCount<=64535");

  fSetMaxWDTIterCmd = std::make_unique<G4UIcmdWithAnInteger>("/adept/MaxWDTIterations", this);
  fSetMaxWDTIterCmd->SetGuidance("Set N, the number of maximum Woodcock tracking iterations per step before giving the "
                                 "gamma back to the normal gamma kernel. Default: "
                                 "5. This can be used to optimize the performance in highly granular geometries");

  // ADEPT_DOCS_SECTION: Geometry Validation
  fValidateHandoffCmd = std::make_unique<G4UIcmdWithABool>("/adept/validateHandoff", this);
  fValidateHandoffCmd->SetGuidance(
      "If true, every track leaving the GPU regions is relocated by Geant4 as when it resumes CPU tracking, and the "
      "outcome is checked against the GPU navigation: the step end point must lie within Geant4's tolerance band of "
      "the crossed surfaces, and Geant4 must find the path reached on the GPU unless the final direction points back "
      "through a crossed surface or another volume starts on the same surface. Counters and the verdict "
      "(HandoffValidation: PASSED/FAILED) are printed at the end of the job. For validation only: it costs a Geant4 "
      "relocation per returned track.");

  fValidateCrossingsCmd = std::make_unique<G4UIcmdWithABool>("/adept/validateCrossings", this);
  fValidateCrossingsCmd->SetGuidance(
      "If true, the GPU transport checks every boundary crossing after relocating into the next volume: the step end "
      "point must lie within +-kTolerance of the crossed surfaces (not inside a volume being left, nor outside a "
      "volume "
      "being entered, beyond it), and the final direction is compared with the normal of the crossed surface to count "
      "bounce-backs. Off-band landings are recorded for replay (/adept/crossingRecordsFile). Counters and a depth "
      "histogram are printed at the end of the job (CrossingValidation lines). Solid navigation only; for validation "
      "and studies of the tolerance conventions, it costs several solid queries per crossing.");
  fValidateCrossingsCmd->AvailableForStates(G4State_PreInit);

  fCrossingRecordsFileCmd = std::make_unique<G4UIcmdWithAString>("/adept/crossingRecordsFile", this);
  fCrossingRecordsFileCmd->SetGuidance(
      "CSV file for the off-band landings found by /adept/validateCrossings: point and direction in the frame of the "
      "offending volume (for a replay on its solid) and in the global frame, depth, normal . direction, volumes.");
  fCrossingRecordsFileCmd->AvailableForStates(G4State_PreInit);

  fCrossingRecordCapacityCmd = std::make_unique<G4UIcmdWithAnInteger>("/adept/crossingRecordCapacity", this);
  fCrossingRecordCapacityCmd->SetGuidance("Maximum number of off-band landings recorded. Default: 10000");
  fCrossingRecordCapacityCmd->SetParameterName("CrossingRecordCapacity", false);
  fCrossingRecordCapacityCmd->SetRange("CrossingRecordCapacity>=0");
  fCrossingRecordCapacityCmd->AvailableForStates(G4State_PreInit);

  // ADEPT_DOCS_SECTION: Special settings for G4EmStandard_AdePT physics constructor
  fAddWDTRegionCmd = std::make_unique<G4UIcmdWithAString>("/adept/addWDTRegion", this);
  fAddWDTRegionCmd->SetGuidance("Add a region in which the gamma transport is done via Woodcock tracking. "
                                "NOTE: This ONLY applies to the AdePTPhysics, if the PhysicsList uses ANY other "
                                "physics (which is done in LHCb, CMS, ATLAS) then this will have NO effect!");

  fSetWDTKineticEnergyLimitCmd = std::make_unique<G4UIcmdWithADouble>("/adept/addWDTKineticEnergyLimit", this);
  fSetWDTKineticEnergyLimitCmd->SetGuidance(
      "Sets a kinetic energy limit above which the gamma transport is done via Woodcock tracking in the assigned "
      "regions. NOTE: This ONLY applies to the AdePTPhysics, if the PhysicsList uses ANY other physics (which is done "
      "in LHCb, CMS, ATLAS) then this will have NO effect!");

  fSetMultipleStepsInMSCWithTransportationCmd =
      std::make_unique<G4UIcmdWithABool>("/adept/SetMultipleStepsInMSCWithTransportation", this);
  fSetMultipleStepsInMSCWithTransportationCmd->SetGuidance(
      "If true, this configures G4HepEm to use multiple steps in MSC on CPU. This does not affect GPU transport");

  fSetEnergyLossFluctuationCmd = std::make_unique<G4UIcmdWithABool>("/adept/SetEnergyLossFluctuation", this);
  fSetEnergyLossFluctuationCmd->SetGuidance(
      "If true, this configures G4HepEm to use energy loss fluctuations. This affects both CPU and GPU transport"
      "NOTE: This is only true for the AdePTPhysics in the examples! In all other physics lists the setting is"
      " taken directly from Geant4 and this parameter does not change it.");
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

AdePTConfigurationMessenger::~AdePTConfigurationMessenger() = default;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void AdePTConfigurationMessenger::SetNewValue(G4UIcommand *command, G4String newValue)
{

  if (command == fSetTrackInAllRegionsCmd.get()) {
    if (!fAdePTConfiguration->SetTrackInAllRegions(fSetTrackInAllRegionsCmd->GetNewBoolValue(newValue))) {
      ReportLockedTransportInitializationOption(command);
    }
  } else if (command == fSetCallUserActionsCmd.get()) {
    fAdePTConfiguration->SetCallUserActions(fSetCallUserActionsCmd->GetNewBoolValue(newValue));
  } else if (command == fSetCallUserSteppingActionCmd.get()) {
    fAdePTConfiguration->SetCallUserSteppingAction(fSetCallUserSteppingActionCmd->GetNewBoolValue(newValue));
  } else if (command == fSetCallUserTrackingActionCmd.get()) {
    fAdePTConfiguration->SetCallUserTrackingAction(fSetCallUserTrackingActionCmd->GetNewBoolValue(newValue));
  } else if (command == fSetReturnFirstAndLastStepCmd.get()) {
    if (!fAdePTConfiguration->SetReturnFirstAndLastStep(fSetReturnFirstAndLastStepCmd->GetNewBoolValue(newValue))) {
      ReportLockedTransportInitializationOption(command);
    }
  } else if (command == fSetReturnAllStepsCmd.get()) {
    if (!fAdePTConfiguration->SetReturnAllSteps(fSetReturnAllStepsCmd->GetNewBoolValue(newValue))) {
      ReportLockedTransportInitializationOption(command);
    }
  } else if (command == fValidateCrossingsCmd.get()) {
#ifdef ADEPT_GEOMETRY_VALIDATION
    fAdePTConfiguration->SetValidateCrossings(fValidateCrossingsCmd->GetNewBoolValue(newValue));
#else
    G4cout << "AdePT: " << command->GetCommandPath() << " ignored, AdePT was built without ADEPT_GEOMETRY_VALIDATION"
           << G4endl;
#endif
  } else if (command == fCrossingRecordsFileCmd.get()) {
    fAdePTConfiguration->SetCrossingRecordsFile(newValue);
  } else if (command == fCrossingRecordCapacityCmd.get()) {
    fAdePTConfiguration->SetCrossingRecordCapacity(fCrossingRecordCapacityCmd->GetNewIntValue(newValue));
  } else if (command == fValidateHandoffCmd.get()) {
#ifdef ADEPT_GEOMETRY_VALIDATION
    fAdePTConfiguration->SetValidateHandoff(fValidateHandoffCmd->GetNewBoolValue(newValue));
#else
    G4cout << "AdePT: " << command->GetCommandPath() << " ignored, AdePT was built without ADEPT_GEOMETRY_VALIDATION"
           << G4endl;
#endif
  } else if (command == fSetSpeedOfLightCmd.get()) {
    fAdePTConfiguration->SetSpeedOfLight(fSetSpeedOfLightCmd->GetNewBoolValue(newValue));
  } else if (command == fSetMultipleStepsInMSCWithTransportationCmd.get()) {
    fAdePTConfiguration->SetMultipleStepsInMSCWithTransportation(
        fSetMultipleStepsInMSCWithTransportationCmd->GetNewBoolValue(newValue));
  } else if (command == fSetEnergyLossFluctuationCmd.get()) {
    fAdePTConfiguration->SetEnergyLossFluctuation(fSetEnergyLossFluctuationCmd->GetNewBoolValue(newValue));
  } else if (command == fAddRegionCmd.get()) {
    if (!fAdePTConfiguration->AddGPURegionName(newValue)) {
      ReportLockedTransportInitializationOption(command);
    }
  } else if (command == fAddWDTRegionCmd.get()) {
    fAdePTConfiguration->AddWDTRegionName(newValue);
  } else if (command == fRemoveRegionCmd.get()) {
    if (!fAdePTConfiguration->RemoveGPURegionName(newValue)) {
      ReportLockedTransportInitializationOption(command);
    }
  } else if (command == fSetVerbosityCmd.get()) {
    fAdePTConfiguration->SetVerbosity(fSetVerbosityCmd->GetNewIntValue(newValue));
  } else if (command == fSetMillionsOfTrackSlotsCmd.get()) {
    fAdePTConfiguration->SetMillionsOfTrackSlots(fSetMillionsOfTrackSlotsCmd->GetNewDoubleValue(newValue));
  } else if (command == fSetMillionsOfHitSlotsCmd.get()) {
    fAdePTConfiguration->SetMillionsOfHitSlots(fSetMillionsOfHitSlotsCmd->GetNewDoubleValue(newValue));
  } else if (command == fSetHitBufferFlushThresholdCmd.get()) {
    fAdePTConfiguration->SetHitBufferFlushThreshold(fSetHitBufferFlushThresholdCmd->GetNewDoubleValue(newValue));
  } else if (command == fSetCPUCapacityFactorCmd.get()) {
    fAdePTConfiguration->SetCPUCapacityFactor(fSetCPUCapacityFactorCmd->GetNewDoubleValue(newValue));
  } else if (command == fSetHitBufferSafetyFactorCmd.get()) {
    fAdePTConfiguration->SetHitBufferSafetyFactor(fSetHitBufferSafetyFactorCmd->GetNewDoubleValue(newValue));
  } else if (command == fSetGDMLCmd.get()) {
#ifdef VECGEOM_GDML_SUPPORT
    fAdePTConfiguration->SetVecGeomGDML(newValue);
#else
    throw std::runtime_error("Attempting to load geometry via VGDML, but VecGeom was compiled without GDML support, "
                             "unset /adept/setVecGeomGDML to load via g4vg");
#endif
  } else if (command == fSetCovfieFileCmd.get()) {
    fAdePTConfiguration->SetCovfieBfieldFile(newValue);
  } else if (command == fSetCUDAStackLimitCmd.get()) {
    fAdePTConfiguration->SetCUDAStackLimit(fSetCUDAStackLimitCmd->GetNewIntValue(newValue));
  } else if (command == fSetCUDAHeapLimitCmd.get()) {
    fAdePTConfiguration->SetCUDAHeapLimit(fSetCUDAHeapLimitCmd->GetNewIntValue(newValue));
  } else if (command == fSetAdePTSeedCmd.get()) {
    fAdePTConfiguration->SetAdePTSeed(fSetAdePTSeedCmd->GetNewIntValue(newValue));
  } else if (command == fSetFinishOnCpuCmd.get()) {
    fAdePTConfiguration->SetLastNParticlesOnCPU(fSetFinishOnCpuCmd->GetNewIntValue(newValue));
  } else if (command == fSetMaxChargedLooperCountCmd.get()) {
    fAdePTConfiguration->SetMaxChargedLooperCount(fSetMaxChargedLooperCountCmd->GetNewIntValue(newValue));
  } else if (command == fSetMaxWDTIterCmd.get()) {
    fAdePTConfiguration->SetMaxWDTIter(fSetMaxWDTIterCmd->GetNewIntValue(newValue));
  } else if (command == fSetWDTKineticEnergyLimitCmd.get()) {
    fAdePTConfiguration->SetWDTKineticEnergyLimit(fSetWDTKineticEnergyLimitCmd->GetNewDoubleValue(newValue));
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
