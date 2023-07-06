// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
#include "DetectorMessenger.hh"
#include "DetectorConstruction.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

#include "Randomize.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorMessenger::DetectorMessenger(DetectorConstruction *aDetector) : G4UImessenger(), fDetector(aDetector)
{
  fExampleDir = new G4UIdirectory("/example17/");
  fExampleDir->SetGuidance("UI commands specific to this example");

  fDetectorDir = new G4UIdirectory("/example17/detector/");
  fDetectorDir->SetGuidance("Detector construction UI commands");

  fAdeptDir = new G4UIdirectory("/example17/adept/");
  fDetectorDir->SetGuidance("AdePT integration UI commands");

  fSetSeed = new G4UIcmdWithAnInteger("/example17/setSeed", this);
  fSetSeed->SetGuidance("Set the seed for the random number generator");

  fPrintCmd = new G4UIcmdWithoutParameter("/example17/detector/print", this);
  fPrintCmd->SetGuidance("Print current settings.");

  fFileNameCmd = new G4UIcmdWithAString("/example17/detector/filename", this);
  fFileNameCmd->SetGuidance("Set GDML file name");
  //
  fRegionNameCmd = new G4UIcmdWithAString("/example17/detector/regionname", this);
  fRegionNameCmd->SetGuidance("Set fast simulation region name");
  //
  fSensVolNameCmd = new G4UIcmdWithAString("/example17/detector/addsensitivevolume", this);
  fSensVolNameCmd->SetGuidance("Add a sensitive volume to the list");
  //
  fSensVolGroupCmd = new G4UIcmdWithAString("/example17/detector/sensitivegroup", this);
  fSensVolGroupCmd->SetGuidance("Define a wildcard for a sensitive volumes group");
  //
  fActivationCmd = new G4UIcmdWithABool("/example17/adept/activate", this);
  fActivationCmd->SetGuidance("(Activate AdePT");
  //
  fVerbosityCmd = new G4UIcmdWithAnInteger("/example17/adept/verbose", this);
  fVerbosityCmd->SetGuidance("Verbosity level for AdePT integration transport");

  fBufferThresholdCmd = new G4UIcmdWithAnInteger("/example17/adept/threshold", this);
  fBufferThresholdCmd->SetGuidance("Threshold for starting AdePT transport");

  fTrackSlotsCmd = new G4UIcmdWithADouble("/example17/adept/milliontrackslots", this);
  fTrackSlotsCmd->SetGuidance("Total number of allocated track slots per GPU");

  fFieldCmd = new G4UIcmdWith3VectorAndUnit("/example17/detector/setField", this);
  fFieldCmd->SetGuidance("Set the constant magenetic field vector.");
  fFieldCmd->SetUnitCategory("Magnetic flux density");
  fFieldCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
  fFieldCmd->SetToBeBroadcasted(false);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorMessenger::~DetectorMessenger()
{
  delete fSetSeed;
  delete fPrintCmd;
  delete fDetectorDir;
  delete fExampleDir;
  delete fAdeptDir;
  delete fFieldCmd;
  delete fSensVolNameCmd;
  delete fSensVolGroupCmd;
  delete fRegionNameCmd;
  delete fFileNameCmd;
  delete fActivationCmd;
  delete fVerbosityCmd;
  delete fBufferThresholdCmd;
  delete fTrackSlotsCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorMessenger::SetNewValue(G4UIcommand *aCommand, G4String aNewValue)
{
  if (aCommand == fSetSeed) {
    CLHEP::HepRandom::setTheSeed(abs(fSetSeed->GetNewIntValue(aNewValue)));
  } else if (aCommand == fPrintCmd) {
    fDetector->Print();
  } else if (aCommand == fFileNameCmd) {
    fDetector->SetGDMLFile(aNewValue);
  } else if (aCommand == fRegionNameCmd) {
    fDetector->SetRegionName(aNewValue);
  } else if (aCommand == fFieldCmd) {
    fDetector->SetMagField(fFieldCmd->GetNew3VectorValue(aNewValue));
  } else if (aCommand == fSensVolNameCmd) {
    fDetector->AddSensitiveVolume(aNewValue);
  } else if (aCommand == fSensVolGroupCmd) {
    fDetector->AddSensitiveGroup(aNewValue);
  } else if (aCommand == fActivationCmd) {
    fDetector->SetActivateAdePT(fActivationCmd->GetNewBoolValue(aNewValue));
  } else if (aCommand == fVerbosityCmd) {
    fDetector->SetVerbosity(fVerbosityCmd->GetNewIntValue(aNewValue));
  } else if (aCommand == fBufferThresholdCmd) {
    fDetector->SetBufferThreshold(fBufferThresholdCmd->GetNewIntValue(aNewValue));
  } else if (aCommand == fTrackSlotsCmd) {
    fDetector->SetTrackSlots(fTrackSlotsCmd->GetNewDoubleValue(aNewValue));
  }
}
