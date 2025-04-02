// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include "SteppingAction.hh"
#include "G4Step.hh"

SteppingAction::SteppingAction() : G4UserSteppingAction() {}

SteppingAction::~SteppingAction() {}

void SteppingAction::UserSteppingAction(const G4Step *theStep)
{
  // Kill the particle if it has done over 1000 steps
  fNumSteps++;
  if (fNumSteps > 10000) {
    G4cout << "Warning: Killing track over 10000 steps" << G4endl;
    theStep->GetTrack()->SetTrackStatus(fStopAndKill);
  }
}
