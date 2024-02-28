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

#include "G4SDManager.hh"
#include "G4HCofThisEvent.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4RunManager.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4SystemOfUnits.hh"

EventAction::EventAction() : G4UserEventAction(), fHitCollectionID(-1), fTimer()
{
  fMessenger = new EventActionMessenger(this);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EventAction::~EventAction() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventAction::BeginOfEventAction(const G4Event *)
{
  auto eventId = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();

  fTimer.Start();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventAction::EndOfEventAction(const G4Event *aEvent)
{
  auto eventId = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();

  fTimer.Stop();

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

  // Store the original IO precission and width
  auto aOriginalPrecission = std::cout.precision();
  auto aOriginalWidth      = std::cout.width();

  for (size_t iHit = 0; iHit < hitsCollection->entries(); iHit++) {
    hit   = static_cast<SimpleHit *>(hitsCollection->GetHit(iHit));
    hitEn = hit->GetEdep();
    totalEnergy += hitEn;
    G4String vol_name = hit->GetPhysicalVolumeName();

    if (hitEn > 1 && fVerbosity > 1)
      G4cout << "EndOfEventAction " << eventId << " : id " << std::setw(5) << iHit << "  edep " << std::setprecision(2)
             << std::setw(12) << std::fixed << hitEn / MeV << " [MeV] logical " << vol_name << G4endl;
  }

  if (fVerbosity > 0) {
    G4cout << "EndOfEventAction " << eventId << "Total energy deposited: " << totalEnergy / MeV << " MeV" << G4endl;
  }

  // Restore the original IO precission
  G4cout << std::setprecision(aOriginalPrecission) << std::setw(aOriginalWidth) << std::defaultfloat;
}
