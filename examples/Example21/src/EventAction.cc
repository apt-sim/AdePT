// SPDX-FileCopyrightText: 2022 CERN
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
#include "EventAction.hh"
#include "EventActionMessenger.hh"
#include "SimpleHit.hh"
#include "DetectorConstruction.hh"

#include "G4SDManager.hh"
#include "G4HCofThisEvent.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4RunManager.hh"

#include "G4GlobalFastSimulationManager.hh"
#include "AdeptIntegration.h"

#include "TestManager.h"
#include "TestManagerStore.h"
#include "Run.hh"

EventAction::EventAction(DetectorConstruction *aDetector)
    : G4UserEventAction(), fDetector(aDetector), fHitCollectionID(-1), fTimer()
{
  fMessenger = new EventActionMessenger(this);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EventAction::~EventAction() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventAction::BeginOfEventAction(const G4Event *)
{
  auto eventId         = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();

  fTimer.Start();

  #if defined TEST
    //Get the Run object associated to this thread and start the timer for this event
    Run* currentRun = static_cast< Run* > ( G4RunManager::GetRunManager()->GetNonConstCurrentRun() );
    currentRun->GetTestManager()->timerStart(Run::timers::EVENT);
  #endif

  // zero the counters
  number_electrons = 0;
  number_positrons = 0;
  number_gammas    = 0;
  number_killed    = 0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventAction::EndOfEventAction(const G4Event *aEvent)
{
  auto eventId         = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();

  fTimer.Stop();

  #if defined TEST
    //Get the Run object associated to this thread and stop the timer for this event
    Run* currentRun = static_cast< Run* > ( G4RunManager::GetRunManager()->GetNonConstCurrentRun() );
    auto aTestManager = currentRun->GetTestManager();
    if(currentRun->GetDoBenchmark())
    {
      aTestManager->timerStop(Run::timers::EVENT);
    }
  #endif

  // Get hits collection ID (only once)
  if (fHitCollectionID == -1) {
    fHitCollectionID = G4SDManager::GetSDMpointer()->GetCollectionID("hits");
  }
  // Get hits collection
  auto hitsCollection = static_cast<SimpleHitsCollection *>(aEvent->GetHCofThisEvent()->GetHC(fHitCollectionID));

  if (hitsCollection == nullptr) {
    G4ExceptionDescription msg;
    msg << "Cannot access hitsCollection ID " << fHitCollectionID;
    G4Exception("EventAction::GetHitsCollection()", "MyCode0001", FatalException, msg);
  }

  SimpleHit *hit       = nullptr;
  G4double hitEn       = 0;
  G4double totalEnergy = 0;
  
  // print number of secondaries std::setw(24) << std::fixed
  if (fVerbosity > 0) {
    G4cout << "EndOfEventAction " << eventId << ": electrons " << number_electrons << G4endl;
    G4cout << "EndOfEventAction " << eventId << ": positrons " << number_positrons << G4endl;
    G4cout << "EndOfEventAction " << eventId << ": gammas    " << number_gammas << G4endl;
    G4cout << "EndOfEventAction " << eventId << ": killed    " << number_killed << G4endl;
    G4cout << "EndOfEventAction " << eventId << ": real time " << fTimer.GetRealElapsed() << G4endl;
  }

  // Store the original IO precission and width
  auto aOriginalPrecission = std::cout.precision();
  auto aOriginalWidth = std::cout.width();

  auto &groups        = fDetector->GetSensitiveGroups();
  int ngroups         = groups.size();
  double *edep_groups = nullptr;
  if (ngroups > 0) edep_groups = new double[ngroups];
  for (auto i = 0; i < ngroups; ++i)
    edep_groups[i] = 0;

  for (size_t iHit = 0; iHit < hitsCollection->entries(); iHit++) {
    hit   = static_cast<SimpleHit *>(hitsCollection->GetHit(iHit));
    hitEn = hit->GetEdep();
    totalEnergy += hitEn;

    G4String vol_name = vecgeom::GeoManager::Instance().FindPlacedVolume(iHit)->GetLogicalVolume()->GetName();
    bool group_found  = false;
    for (int igroup = 0; igroup < ngroups; ++igroup) {
      if (vol_name.rfind(groups[igroup], 0) == 0) {
        edep_groups[igroup] += hitEn;
        group_found = true;
        break;
      }
    }
    if (group_found) continue;
    const char *type = (hit->GetType() == 1) ? "AdePT" : "G4";
    if (hitEn > 1 && fVerbosity > 1)
      G4cout << "EndOfEventAction " << eventId << " : id " << std::setw(5) << iHit << "  edep " << std::setprecision(2)
             << std::setw(12) << std::fixed << hitEn / MeV << " [MeV] logical " << vol_name << G4endl;
  }

  if (fVerbosity > 1) {
    for (int igroup = 0; igroup < ngroups; ++igroup) {
      G4cout << "EndOfEventAction " << eventId << " : group " << std::setw(5) << groups[igroup] << "  edep "
             << std::setprecision(2) << std::setw(12) << std::fixed << edep_groups[igroup] / MeV << " [MeV]\n";
    }
  }

  if (fVerbosity > 0) {
    G4cout << "EndOfEventAction " << eventId << "Total energy deposited: " << totalEnergy / MeV << " MeV" << G4endl;
  }

  #if defined TEST
    if(currentRun->GetDoValidation())
    {
      //Set an accumulator for each sensitive group, making sure the ID doesn't collide with the 
      //defined timers
      for (int igroup = 0; igroup < ngroups; ++igroup) {
        G4cout << "EndOfEventAction " << eventId << " : group " << std::setw(5) << groups[igroup] << "  edep "
                << std::setprecision(2) << std::setw(12) << std::fixed << edep_groups[igroup] / MeV << " [MeV]\n";
      }
      for (int igroup = 0; igroup < ngroups; ++igroup) {
        aTestManager->setAccumulator(igroup+Run::accumulators::NUM_ACCUMULATORS, edep_groups[igroup]);
      }

      //Record the current contents of the TestManager in order to be able to extract per-event data
      TestManagerStore<int>::GetInstance()->RecordState(aTestManager);
    }
    else if(currentRun->GetDoBenchmark())
    {
      //Get the timings
      double eventTime = aTestManager->getDurationSeconds(Run::timers::EVENT);
      double nonEMTime = aTestManager->getAccumulator(Run::accumulators::NONEM_EVT);
      double ecalTime = eventTime - nonEMTime;

      //Accumulate the results with the rest of events of this worker thread to provide global stats
      aTestManager->addToAccumulator(Run::accumulators::EVENT_SUM, eventTime);
      aTestManager->addToAccumulator(Run::accumulators::EVENT_SQ, eventTime*eventTime);
      aTestManager->addToAccumulator(Run::accumulators::NONEM_SUM, nonEMTime);
      aTestManager->addToAccumulator(Run::accumulators::NONEM_SQ, nonEMTime*nonEMTime);
      aTestManager->addToAccumulator(Run::accumulators::ECAL_SUM, ecalTime);
      aTestManager->addToAccumulator(Run::accumulators::ECAL_SQ, ecalTime*ecalTime);

      //Record the current contents of the TestManager in order to be able to extract per-event data
      TestManagerStore<int>::GetInstance()->RecordState(aTestManager);

      //Reset the timers for the next event
      aTestManager->removeTimer(Run::timers::EVENT);
      aTestManager->removeAccumulator(Run::accumulators::NONEM_EVT);
    }
  #endif
  
  delete[] edep_groups;

  // Restore the original IO precission
  G4cout << std::setprecision(aOriginalPrecission) << std::setw(aOriginalWidth) << std::defaultfloat;
}
