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
#include "BenchmarkManager.h"

SteppingAction::SteppingAction(DetectorConstruction *aDetector, RunAction *aRunAction, TrackingAction *aTrackingAction)
    : G4UserSteppingAction(), fDetector(aDetector), fRunAction(aRunAction), fTrackingAction(aTrackingAction)
{
}

SteppingAction::~SteppingAction() {}

void SteppingAction::UserSteppingAction(const G4Step *theStep)
{

  Run *currentRun = static_cast<Run *>(G4RunManager::GetRunManager()->GetNonConstCurrentRun());
  /*
  currentRun->getAuxBenchmarkManager()->setAccumulator("Step Length", theStep->GetStepLength());
  currentRun->getAuxBenchmarkManager()->setOutputFilename("example17_ecal_out");
  currentRun->getAuxBenchmarkManager()->exportCSV();
  currentRun->getAuxBenchmarkManager()->removeAccumulator("Step Length");
  */

  // G4cout << theStep->GetStepLength() << G4endl;

  //if (fTrackingAction->getInsideEcal()) {
    // Add track length to the accumulator if we are in ECAL
    //currentRun->getAuxBenchmarkManager()->addToAccumulator(std::to_string((long)theStep->GetTrack()->GetVolume()),
    currentRun->getAuxBenchmarkManager()->addToAccumulator(theStep->GetTrack()->GetVolume()->GetName(),
                                                           theStep->GetStepLength());
  //}

  // Check if we moved to a new volume
  if (theStep->IsLastStepInVolume()) {
    //Add to the counter for the volume
    //currentRun->getAuxBenchmarkManager()->addToAccumulator(theStep->GetTrack()->GetVolume()->GetName()+"_numtracks", 1);

    G4VPhysicalVolume *nextVolume = theStep->GetTrack()->GetNextVolume();
    if (!fTrackingAction->getInsideEcal()) {
      // Check if the new volume is in the EM calorimeter region
      if (nextVolume->GetLogicalVolume()->GetRegion() == fTrackingAction->getGPURegion()) {

        // If it is, stop the timer for this track and store the result
        G4Track *aTrack = theStep->GetTrack();

        aTrack->SetTrackStatus(fStopAndKill);

        // Make sure this will only run on the first step when we enter the ECAL
        fTrackingAction->setInsideEcal(true);

        // We are only interested in the processing of e-, e+ and gammas in the EM calorimeter, for other
        // particles the time keeps running
        if (aTrack->GetDefinition() == G4Gamma::Gamma() || aTrack->GetDefinition() == G4Electron::Electron() ||
            aTrack->GetDefinition() == G4Positron::Positron()) {
          // Get the Run object associated to this thread and end the timer for this track
          Run *currentRun        = static_cast<Run *>(G4RunManager::GetRunManager()->GetNonConstCurrentRun());
          auto aBenchmarkManager = currentRun->getBenchmarkManager();

          // currentRun->getAuxBenchmarkManager()->addToAccumulator("Energy in", aTrack->GetTotalEnergy());

          aBenchmarkManager->timerStop(Run::timers::NONEM);
          aBenchmarkManager->addToAccumulator(Run::accumulators::NONEM_EVT,
                                              aBenchmarkManager->getDurationSeconds(Run::timers::NONEM));
          aBenchmarkManager->removeTimer(Run::timers::NONEM);
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
          currentRun->getBenchmarkManager()->timerStart(Run::timers::NONEM);
          // currentRun->getAuxBenchmarkManager()->addToAccumulator("Energy out", aTrack->GetTotalEnergy());
          /*
          if (aTrack->GetDefinition() == G4Gamma::Gamma())
            currentRun->getAuxBenchmarkManager()->addToAccumulator("GAMMAS", 1);
          if (aTrack->GetDefinition() == G4Electron::Electron())
            currentRun->getAuxBenchmarkManager()->addToAccumulator("ELECTRONS", 1);
          if (aTrack->GetDefinition() == G4Positron::Positron())
            currentRun->getAuxBenchmarkManager()->addToAccumulator("POSITRONS", 1);
          */
        }
      }
    }
  }
}