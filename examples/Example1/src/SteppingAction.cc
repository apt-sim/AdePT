// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include "RunAction.hh"
#include "TrackingAction.hh"
#include "SteppingAction.hh"

#include "DetectorConstruction.hh"

#include "G4Step.hh"
#include "G4RunManager.hh"
#include "G4EventManager.hh"
#include "G4Track.hh"
#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include <AdePT/base/TestManager.h>

SteppingAction::SteppingAction(TrackingAction *aTrackingAction)
    : G4UserSteppingAction(), fTrackingAction(aTrackingAction)
{
}

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