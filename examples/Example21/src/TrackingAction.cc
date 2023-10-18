// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
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
/// \file electromagnetic/TestEm1/src/TrackingAction.cc
/// \brief Implementation of the TrackingAction class
//
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "TrackingAction.hh"

#include "G4RunManager.hh"
#include "G4Track.hh"
#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "Run.hh"
#include "TestManager.h"
#include "DetectorConstruction.hh"
#include "G4RegionStore.hh"

#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include "EventAction.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

TrackingAction::TrackingAction(DetectorConstruction* aDetector) : G4UserTrackingAction(), 
                                                                  fDetector(aDetector), 
                                                                  fCurrentRegion(nullptr), 
                                                                  fCurrentVolume(nullptr),
                                                                  fGPURegion(G4RegionStore::GetInstance()->GetRegion(aDetector->getRegionName())){}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void TrackingAction::PreUserTrackingAction(const G4Track *aTrack) 
{
  #if defined TEST
    //For leptons, get the Run object associated to this thread and start the timer for this track, only if it is outside 
    //the GPU region
    Run* currentRun = static_cast< Run* > ( G4RunManager::GetRunManager()->GetNonConstCurrentRun() );
    if(currentRun->GetDoBenchmark())
    {
      if (aTrack->GetDefinition() == G4Gamma::Gamma() || 
          aTrack->GetDefinition() == G4Electron::Electron() ||
          aTrack->GetDefinition() == G4Positron::Positron())
      {
        if(aTrack->GetVolume()->GetLogicalVolume()->GetRegion() != fGPURegion)
        {
          currentRun->GetTestManager()->timerStart(Run::timers::NONEM);
          setInsideEcal(false);
        }
        else
        {
          setInsideEcal(true);
        }
      }
      //For other particles we always count the time, get the Run object and start the timer
      else
      {
        currentRun->GetTestManager()->timerStart(Run::timers::NONEM);
      }
    }
  #endif
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void TrackingAction::PostUserTrackingAction(const G4Track *aTrack)
{
  #if defined TEST
    //Get the Run object associated to this thread and end the timer for this track
    Run* currentRun = static_cast< Run* > ( G4RunManager::GetRunManager()->GetNonConstCurrentRun() );
    if(currentRun->GetDoBenchmark())
    {
      //Timer may have been stopped in the stepping action
      if(!getInsideEcal())
      {
        const G4Event* currentEvent = G4EventManager::GetEventManager()->GetConstCurrentEvent();
        auto aTestManager = currentRun->GetTestManager();
        
        aTestManager->timerStop(Run::timers::NONEM);
        aTestManager->addToAccumulator(Run::accumulators::NONEM_EVT, aTestManager->getDurationSeconds(Run::timers::NONEM));
        aTestManager->removeTimer(Run::timers::NONEM);
      }
    }
  #endif

  // skip tracks coming from AdePT
  if (aTrack->GetParentID() == -99) return;

  // increase nb of processed tracks
  // count nb of steps of this track
  G4int nbSteps   = aTrack->GetCurrentStepNumber();
  G4double Trleng = aTrack->GetTrackLength();

  if (aTrack->GetDefinition() == G4Gamma::Gamma()) {
    dynamic_cast<EventAction *>(G4EventManager::GetEventManager()->GetUserEventAction())->number_gammas++;
  } else if (aTrack->GetDefinition() == G4Electron::Electron()) {
    dynamic_cast<EventAction *>(G4EventManager::GetEventManager()->GetUserEventAction())->number_electrons++;
  } else if (aTrack->GetDefinition() == G4Positron::Positron()) {
    dynamic_cast<EventAction *>(G4EventManager::GetEventManager()->GetUserEventAction())->number_positrons++;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
