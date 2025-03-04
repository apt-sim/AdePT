// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
#include "DetectorMessenger.hh"
#include "DetectorConstruction.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorMessenger::DetectorMessenger(DetectorConstruction *aDetector) : G4UImessenger(), fDetector(aDetector)
{
  fDir = new G4UIdirectory("/detector/");
  fDir->SetGuidance("Detector construction UI commands");

  fPrintCmd = new G4UIcmdWithoutParameter("/detector/print", this);
  fPrintCmd->SetGuidance("Print current settings.");

  fFileNameCmd = new G4UIcmdWithAString("/detector/filename", this);
  fFileNameCmd->SetGuidance("Set GDML file name");

  fFieldCmd = new G4UIcmdWith3VectorAndUnit("/detector/setField", this);
  fFieldCmd->SetGuidance("Set the constant magenetic field vector.");
  fFieldCmd->SetUnitCategory("Magnetic flux density");
  fFieldCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
  fFieldCmd->SetToBeBroadcasted(false);

  fFieldFileNameCmd = new G4UIcmdWithAString("/detector/setCovfieBfieldFile", this);
  fFieldFileNameCmd->SetGuidance("Set covfie magnetic field file name");
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorMessenger::~DetectorMessenger()
{
  delete fPrintCmd;
  delete fDir;
  delete fFieldCmd;
  delete fFileNameCmd;
  delete fFieldFileNameCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorMessenger::SetNewValue(G4UIcommand *aCommand, G4String aNewValue)
{
  if (aCommand == fPrintCmd) {
    fDetector->Print();
  } else if (aCommand == fFileNameCmd) {
    fDetector->SetGDMLFile(aNewValue);
  } else if (aCommand == fFieldCmd) {
    fDetector->SetMagField(fFieldCmd->GetNew3VectorValue(aNewValue));
  } else if (aCommand == fFieldFileNameCmd) {
    fDetector->SetFieldFile(aNewValue);
  }
}
