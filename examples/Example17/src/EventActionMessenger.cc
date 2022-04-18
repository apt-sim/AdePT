// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
#include "EventActionMessenger.hh"
#include "EventAction.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAnInteger.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EventActionMessenger::EventActionMessenger(EventAction *aEventAction) : G4UImessenger(), fEventAction(aEventAction)
{
  fEventActionDir = new G4UIdirectory("/example17/event/");
  fEventActionDir->SetGuidance("UI commands for event actions");
  //
  fVerbosityCmd = new G4UIcmdWithAnInteger("/example17/event/verbose", this);
  fVerbosityCmd->SetGuidance("Scoring verbosity");
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EventActionMessenger::~EventActionMessenger()
{
  delete fEventActionDir;
  delete fVerbosityCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventActionMessenger::SetNewValue(G4UIcommand *aCommand, G4String aNewValue)
{
  if (aCommand == fVerbosityCmd) {
    fEventAction->SetVerbosity(fVerbosityCmd->GetNewIntValue(aNewValue));
  }
}
