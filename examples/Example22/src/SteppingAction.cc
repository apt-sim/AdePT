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
#include "Run.hh"
#include "TestManager.h"

SteppingAction::SteppingAction(DetectorConstruction *aDetector, RunAction *aRunAction, TrackingAction *aTrackingAction)
    : G4UserSteppingAction(), fDetector(aDetector), fRunAction(aRunAction), fTrackingAction(aTrackingAction)
{
}

SteppingAction::~SteppingAction() {}

void SteppingAction::UserSteppingAction(const G4Step *theStep)
{
  // Check if we moved to a new volume
  if (theStep->IsLastStepInVolume()) {
    G4VPhysicalVolume *nextVolume = theStep->GetTrack()->GetNextVolume();
    if (nextVolume != nullptr) {
      if (!fTrackingAction->getInsideEcal()) {
        // Check if the new volume is in the EM calorimeter region
        if (nextVolume->GetLogicalVolume()->GetRegion() == fTrackingAction->getGPURegion()) {
          // If it is, stop the timer for this track and store the result
          G4Track *aTrack = theStep->GetTrack();

          // Make sure this will only run on the first step when we enter the ECAL
          fTrackingAction->setInsideEcal(true);

          // We are only interested in the processing of e-, e+ and gammas in the EM calorimeter, for other
          // particles the time keeps running
          if (aTrack->GetDefinition() == G4Gamma::Gamma() || aTrack->GetDefinition() == G4Electron::Electron() ||
              aTrack->GetDefinition() == G4Positron::Positron()) {
            // Get the Run object associated to this thread and end the timer for this track
            Run *currentRun        = static_cast<Run *>(G4RunManager::GetRunManager()->GetNonConstCurrentRun());
            auto aTestManager = currentRun->GetTestManager();

            aTestManager->timerStop(Run::timers::NONEM);
            aTestManager->addToAccumulator(Run::accumulators::NONEM_EVT,
                                                aTestManager->getDurationSeconds(Run::timers::NONEM));
            aTestManager->removeTimer(Run::timers::NONEM);
          }
        }
      } else {
        // In case this track is exiting the EM calorimeter, start the timer
        if (nextVolume->GetLogicalVolume()->GetRegion() != fTrackingAction->getGPURegion()) {
          G4Track *aTrack = theStep->GetTrack();

          fTrackingAction->setInsideEcal(false);

          if (aTrack->GetDefinition() == G4Gamma::Gamma() || aTrack->GetDefinition() == G4Electron::Electron() ||
              aTrack->GetDefinition() == G4Positron::Positron()) {
            // Get the Run object associated to this thread and start the timer for this track
            Run *currentRun = static_cast<Run *>(G4RunManager::GetRunManager()->GetNonConstCurrentRun());
            currentRun->GetTestManager()->timerStart(Run::timers::NONEM);
          }
        }
      }
    }
  }
}