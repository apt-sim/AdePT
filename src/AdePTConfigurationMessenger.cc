// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0
/// \brief Implementation of the AdePTConfigurationMessenger class
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include <AdePT/core/AdePTConfigurationMessenger.hh>
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

  fSetSeedCmd = new G4UIcmdWithAnInteger("/adept/setSeed", this);
  fSetSeedCmd->SetGuidance("Set a random seed for AdePT");

  fSetRegionCmd = new G4UIcmdWithAString("/adept/setRegion", this);
  fSetRegionCmd->SetGuidance("Set the region in which transport will be done on GPU");

  fActivateAdePTCmd = new G4UIcmdWithABool("/adept/activateAdePT", this);
  fActivateAdePTCmd->SetGuidance("Set whether to use AdePT for transport, if false all transport is done by Geant4");

  fSetVerbosityCmd = new G4UIcmdWithAnInteger("/adept/setVerbosity", this);
  fSetVerbosityCmd->SetGuidance("Set verbosity level for the AdePT integration layer");

  fSetTransportBufferThresholdCmd = new G4UIcmdWithAnInteger("/adept/setTransportBufferThreshold", this);
  fSetTransportBufferThresholdCmd->SetGuidance("Set number of particles to be buffered before triggering the transport on GPU");

  fSetMillionsOfTrackSlotsCmd = new G4UIcmdWithADouble("/adept/setMillionsOfTrackSlots", this);
  fSetMillionsOfTrackSlotsCmd->SetGuidance("Set the total number of track slots that will be allocated on the GPU, in millions");

  fSetMillionsOfHitSlotsCmd = new G4UIcmdWithADouble("/adept/setMillionsOfHitSlots", this);
  fSetMillionsOfHitSlotsCmd->SetGuidance("Set the total number of hit slots that will be allocated on the GPU, in millions");

  fSetHitBufferFlushThresholdCmd = new G4UIcmdWithADouble("/adept/setHitBufferThreshold", this);
  fSetHitBufferFlushThresholdCmd->SetGuidance("Set the usage threshold at which the buffer of hits is copied back to the host from GPU");
  fSetHitBufferFlushThresholdCmd->SetParameterName("HitBufferThreshold", false);
  fSetHitBufferFlushThresholdCmd->SetRange("HitBufferThreshold>=0.&&HitBufferThreshold<=1.");

  fSetGDMLCmd = new G4UIcmdWithAString("/adept/setVecGeomGDML", this);
  fSetGDMLCmd->SetGuidance("Temporary method for setting the geometry to use with VecGeom");
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

AdePTConfigurationMessenger::~AdePTConfigurationMessenger()
{
  delete fDir;
  delete fSetSeedCmd;
  delete fSetRegionCmd;
  delete fActivateAdePTCmd;
  delete fSetVerbosityCmd;
  delete fSetTransportBufferThresholdCmd;
  delete fSetMillionsOfTrackSlotsCmd;
  delete fSetMillionsOfHitSlotsCmd;
  delete fSetHitBufferFlushThresholdCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void AdePTConfigurationMessenger::SetNewValue(G4UIcommand *command, G4String newValue)
{

  if(command == fSetSeedCmd)
  {
    fAdePTConfiguration->SetRandomSeed(fSetSeedCmd->GetNewIntValue(newValue));
  }
  else if(command == fSetRegionCmd)
  {
    fAdePTConfiguration->SetGPURegionName(newValue);
  }
  else if(command == fActivateAdePTCmd)
  {
    fAdePTConfiguration->SetAdePTActivation(fActivateAdePTCmd->GetNewBoolValue(newValue));
  }
  else if(command == fSetVerbosityCmd)
  {
    fAdePTConfiguration->SetVerbosity(fSetVerbosityCmd->GetNewIntValue(newValue));
  }
  else if(command == fSetTransportBufferThresholdCmd)
  {
    fAdePTConfiguration->SetTransportBufferThreshold(fSetTransportBufferThresholdCmd->GetNewIntValue(newValue));
  }
  else if(command == fSetMillionsOfTrackSlotsCmd)
  {
    fAdePTConfiguration->SetMillionsOfTrackSlots(fSetMillionsOfTrackSlotsCmd->GetNewDoubleValue(newValue));
  }
  else if(command == fSetMillionsOfHitSlotsCmd)
  {
    fAdePTConfiguration->SetMillionsOfHitSlots(fSetMillionsOfHitSlotsCmd->GetNewDoubleValue(newValue));
  }
  else if(command == fSetHitBufferFlushThresholdCmd)
  {
    fAdePTConfiguration->SetHitBufferFlushThreshold(fSetHitBufferFlushThresholdCmd->GetNewDoubleValue(newValue));
  }
  else if(command == fSetGDMLCmd)
  {
    fAdePTConfiguration->SetVecGeomGDML(newValue);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......