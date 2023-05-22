// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
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
#include "BenchmarkManager.h"

SteppingAction::SteppingAction(DetectorConstruction *aDetector, RunAction *aRunAction, TrackingAction *aTrackingAction)
    : G4UserSteppingAction(), fDetector(aDetector), fRunAction(aRunAction), fTrackingAction(aTrackingAction)
{
}

SteppingAction::~SteppingAction() {}

void SteppingAction::UserSteppingAction(const G4Step *theStep)
{

  // Check if we moved to a new volume
  G4VPhysicalVolume *previousVolume = theStep->GetPreStepPoint()->GetTouchableHandle()->GetVolume();
  G4VPhysicalVolume *currentVolume  = theStep->GetPostStepPoint()->GetTouchableHandle()->GetVolume();

  if (previousVolume != currentVolume) {
    if (!fTrackingAction->getInsideEcal()) {
      // Check if the new volume is in the EM calorimeter region, if it is stop the track timer
      if (currentVolume->GetLogicalVolume()->GetRegion() == fTrackingAction->getGPURegion()) {
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
          auto aBenchmarkManager = currentRun->getBenchmarkManager();

          aBenchmarkManager->timerStop(Run::timers::NONEM);
          aBenchmarkManager->addDurationSeconds(Run::timers::NONEM_EVT,
                                                aBenchmarkManager->getDurationSeconds(Run::timers::NONEM));
          aBenchmarkManager->removeTimer(Run::timers::NONEM);
        }
      }
    } else {
      // In case this track is exiting the EM calorimeter, start the timer
      if (currentVolume->GetLogicalVolume()->GetRegion() != fTrackingAction->getGPURegion()) {
        G4Track *aTrack = theStep->GetTrack();

        fTrackingAction->setInsideEcal(false);

        if (aTrack->GetDefinition() == G4Gamma::Gamma() || aTrack->GetDefinition() == G4Electron::Electron() ||
            aTrack->GetDefinition() == G4Positron::Positron()) {
          // Get the Run object associated to this thread and start the timer for this track
          Run *currentRun = static_cast<Run *>(G4RunManager::GetRunManager()->GetNonConstCurrentRun());
          currentRun->getBenchmarkManager()->timerStart(Run::timers::NONEM);
        }
      }
    }
  }
}