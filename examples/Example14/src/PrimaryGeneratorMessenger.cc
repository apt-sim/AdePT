// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
/// \brief Implementation of the PAr04PrimaryGeneratorMessenger class
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "PrimaryGeneratorMessenger.hh"

#include "PrimaryGeneratorAction.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithADouble.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PrimaryGeneratorMessenger::PrimaryGeneratorMessenger(PrimaryGeneratorAction *Gun)
    : G4UImessenger(), fAction(Gun), fGunDir(0), fDefaultCmd(0), fRndmCmd(0)
{
  fGunDir = new G4UIdirectory("/example14/gun/");
  fGunDir->SetGuidance("gun control");

  fHepmcCmd = new G4UIcmdWithoutParameter("/example14/gun/hepmc", this);
  fHepmcCmd->SetGuidance("select hepmc input");
  fHepmcCmd->AvailableForStates(G4State_PreInit, G4State_Idle);


  fDefaultCmd = new G4UIcmdWithoutParameter("/example14/gun/setDefault", this);
  fDefaultCmd->SetGuidance("set/reset kinematic defined in PrimaryGenerator");
  fDefaultCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fPrintCmd = new G4UIcmdWithoutParameter("/example14/gun/print", this);
  fPrintCmd->SetGuidance("print gun kinematics in PrimaryGenerator");
  fPrintCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fRndmCmd = new G4UIcmdWithADouble("/example14/gun/rndm", this);
  fRndmCmd->SetGuidance("random lateral extension on the beam");
  fRndmCmd->SetParameterName("rBeam", false);
  fRndmCmd->SetRange("rBeam>=0.&&rBeam<=1.");
  fRndmCmd->AvailableForStates(G4State_Idle);

  fRndmDirCmd = new G4UIcmdWithADouble("/example14/gun/rndmDir", this);
  fRndmDirCmd->SetGuidance("random angular extension on the beam");
  fRndmDirCmd->SetParameterName("rBeamDir", false);
  fRndmDirCmd->SetRange("rBeamDir>=0.&&rBeamDir<=1.");
  fRndmDirCmd->AvailableForStates(G4State_Idle);

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PrimaryGeneratorMessenger::~PrimaryGeneratorMessenger()
{
  delete fHepmcCmd;
  delete fDefaultCmd;
  delete fPrintCmd;
  delete fRndmCmd;
  delete fRndmDirCmd;
  delete fGunDir;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PrimaryGeneratorMessenger::SetNewValue(G4UIcommand *command, G4String newValue)
{

  if (command == fHepmcCmd) {
    fAction->SetHepMC();
  }

  if (command == fDefaultCmd) {
    fAction->SetDefaultKinematic();
  }

  if (command == fPrintCmd) {
    fAction->Print();
  }

  if (command == fRndmCmd) {
    fAction->SetRndmBeam(fRndmCmd->GetNewDoubleValue(newValue));
  }

  if (command == fRndmDirCmd) {
    fAction->SetRndmDirection(fRndmDirCmd->GetNewDoubleValue(newValue));
  }

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
