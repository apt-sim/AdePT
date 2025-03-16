// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0
/// \brief Implementation of the AdePTConfigurationMessenger class
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include <AdePT/integration/AdePTConfigurationMessenger.hh>
#include <AdePT/core/AdePTConfiguration.hh>

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4Tokenizer.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

AdePTConfigurationMessenger::AdePTConfigurationMessenger(AdePTConfiguration *adeptConfiguration)
    : G4UImessenger(), fAdePTConfiguration(adeptConfiguration)
{
  fDir = new G4UIdirectory("/adept/");
  fDir->SetGuidance("adept configuration messenger");

  fSetTrackInAllRegionsCmd = new G4UIcmdWithABool("/adept/setTrackInAllRegions", this);
  fSetTrackInAllRegionsCmd->SetGuidance("If true, particles are tracked on the GPU across the whole geometry");

  fSetCallUserSteppingActionCmd = new G4UIcmdWithABool("/adept/CallUserSteppingAction", this);
  fSetCallUserSteppingActionCmd->SetGuidance(
      "If true, the UserSteppingAction is called for on every step. WARNING: The steps are currently not sorted, that "
      "means it is not guaranteed that the UserSteppingAction is called in order, i.e., it could get called on the "
      "secondary before the primary has finished its track."
      " NOTE: This means that every single step is recorded on GPU and send back to CPU, which can impact performance");

  fSetCallPostUserTrackingActionCmd = new G4UIcmdWithABool("/adept/CallPostUserTrackingAction", this);
  fSetCallPostUserTrackingActionCmd->SetGuidance(
      "If true, the PostUserTrackingAction is called for on every track. NOTE: This "
      "means that the last step of every track is recorded on GPU and send back to CPU");

  fSetCallPostUserTrackingActionCmd = new G4UIcmdWithABool("/adept/CallPostUserTrackingAction", this);
  fSetCallPostUserTrackingActionCmd->SetGuidance(
      "If true, the PostUserTrackingAction is called for on every track. NOTE: This "
      "means that the last step of every track is recorded on GPU and send back to CPU");

  fAddRegionCmd = new G4UIcmdWithAString("/adept/addGPURegion", this);
  fAddRegionCmd->SetGuidance("Add a region in which transport will be done on GPU");

  fActivateAdePTCmd = new G4UIcmdWithABool("/adept/activateAdePT", this);
  fActivateAdePTCmd->SetGuidance("Set whether to use AdePT for transport, if false all transport is done by Geant4");

  fSetVerbosityCmd = new G4UIcmdWithAnInteger("/adept/setVerbosity", this);
  fSetVerbosityCmd->SetGuidance("Set verbosity level for the AdePT integration layer");

  fSetTransportBufferThresholdCmd = new G4UIcmdWithAnInteger("/adept/setTransportBufferThreshold", this);
  fSetTransportBufferThresholdCmd->SetGuidance(
      "Set number of particles to be buffered before triggering the transport on GPU");

  fSetMillionsOfTrackSlotsCmd = new G4UIcmdWithADouble("/adept/setMillionsOfTrackSlots", this);
  fSetMillionsOfTrackSlotsCmd->SetGuidance(
      "Set the total number of track slots that will be allocated on the GPU, in millions");

  fSetMillionsOfHitSlotsCmd = new G4UIcmdWithADouble("/adept/setMillionsOfHitSlots", this);
  fSetMillionsOfHitSlotsCmd->SetGuidance(
      "Set the total number of hit slots that will be allocated on the GPU, in millions");

  fSetHitBufferFlushThresholdCmd = new G4UIcmdWithADouble("/adept/setHitBufferThreshold", this);
  fSetHitBufferFlushThresholdCmd->SetGuidance(
      "Set the usage threshold at which the buffer of hits is copied back to the host from GPU");
  fSetHitBufferFlushThresholdCmd->SetParameterName("HitBufferThreshold", false);
  fSetHitBufferFlushThresholdCmd->SetRange("HitBufferThreshold>=0.&&HitBufferThreshold<=1.");

  fSetGDMLCmd = new G4UIcmdWithAString("/adept/setVecGeomGDML", this);
  fSetGDMLCmd->SetGuidance("Temporary method for setting the geometry to use with VecGeom");

  fSetCovfieFileCmd = new G4UIcmdWithAString("/adept/setCovfieBfieldFile", this);
  fSetCovfieFileCmd->SetGuidance("Set the path the the covfie file for reading in an external magnetic field");

  fSetFinishOnCpuCmd = new G4UIcmdWithAnInteger("/adept/FinishLastNParticlesOnCPU", this);
  fSetFinishOnCpuCmd->SetGuidance("Set N, the number of last N particles per event that are finished on CPU. Default: "
                                  "0. This is an important parameter for handling loopers in a magnetic field");

  fSetCUDAStackLimitCmd = new G4UIcmdWithAnInteger("/adept/setCUDAStackLimit", this);
  fSetCUDAStackLimitCmd->SetGuidance("Set the CUDA device stack limit");
  fSetCUDAHeapLimitCmd = new G4UIcmdWithAnInteger("/adept/setCUDAHeapLimit", this);
  fSetCUDAHeapLimitCmd->SetGuidance("Set the CUDA device heap limit");
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

AdePTConfigurationMessenger::~AdePTConfigurationMessenger()
{
  delete fDir;
  delete fSetCUDAStackLimitCmd;
  delete fSetCUDAHeapLimitCmd;
  delete fSetTrackInAllRegionsCmd;
  delete fSetCallUserSteppingActionCmd;
  delete fSetCallPostUserTrackingActionCmd;
  delete fAddRegionCmd;
  delete fActivateAdePTCmd;
  delete fSetVerbosityCmd;
  delete fSetTransportBufferThresholdCmd;
  delete fSetMillionsOfTrackSlotsCmd;
  delete fSetMillionsOfHitSlotsCmd;
  delete fSetHitBufferFlushThresholdCmd;
  delete fSetGDMLCmd;
  delete fSetCovfieFileCmd;
  delete fSetFinishOnCpuCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void AdePTConfigurationMessenger::SetNewValue(G4UIcommand *command, G4String newValue)
{

  if (command == fSetTrackInAllRegionsCmd) {
    fAdePTConfiguration->SetTrackInAllRegions(fSetTrackInAllRegionsCmd->GetNewBoolValue(newValue));
  } else if (command == fSetCallUserSteppingActionCmd) {
    fAdePTConfiguration->SetCallUserSteppingAction(newValue);
  } else if (command == fSetCallPostUserTrackingActionCmd) {
    fAdePTConfiguration->SetCallPostUserTrackingAction(newValue);
  } else if (command == fAddRegionCmd) {
    fAdePTConfiguration->AddGPURegionName(newValue);
  } else if (command == fActivateAdePTCmd) {
    fAdePTConfiguration->SetAdePTActivation(fActivateAdePTCmd->GetNewBoolValue(newValue));
  } else if (command == fSetVerbosityCmd) {
    fAdePTConfiguration->SetVerbosity(fSetVerbosityCmd->GetNewIntValue(newValue));
  } else if (command == fSetTransportBufferThresholdCmd) {
    fAdePTConfiguration->SetTransportBufferThreshold(fSetTransportBufferThresholdCmd->GetNewIntValue(newValue));
  } else if (command == fSetMillionsOfTrackSlotsCmd) {
    fAdePTConfiguration->SetMillionsOfTrackSlots(fSetMillionsOfTrackSlotsCmd->GetNewDoubleValue(newValue));
  } else if (command == fSetMillionsOfHitSlotsCmd) {
    fAdePTConfiguration->SetMillionsOfHitSlots(fSetMillionsOfHitSlotsCmd->GetNewDoubleValue(newValue));
  } else if (command == fSetHitBufferFlushThresholdCmd) {
    fAdePTConfiguration->SetHitBufferFlushThreshold(fSetHitBufferFlushThresholdCmd->GetNewDoubleValue(newValue));
  } else if (command == fSetGDMLCmd) {
    fAdePTConfiguration->SetVecGeomGDML(newValue);
  } else if (command == fSetCovfieFileCmd) {
    fAdePTConfiguration->SetCovfieBfieldFile(newValue);
  } else if (command == fSetCUDAStackLimitCmd) {
    fAdePTConfiguration->SetCUDAStackLimit(fSetCUDAStackLimitCmd->GetNewIntValue(newValue));
  } else if (command == fSetCUDAHeapLimitCmd) {
    fAdePTConfiguration->SetCUDAHeapLimit(fSetCUDAHeapLimitCmd->GetNewIntValue(newValue));
  } else if (command == fSetFinishOnCpuCmd) {
    fAdePTConfiguration->SetLastNParticlesOnCPU(fSetFinishOnCpuCmd->GetNewIntValue(newValue));
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
