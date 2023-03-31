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
#include "SensitiveDetector.hh"
#include "SimpleHit.hh"

#include "G4HCofThisEvent.hh"
#include "G4TouchableHistory.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4SDManager.hh"

SensitiveDetector::SensitiveDetector(G4String aName) : G4VSensitiveDetector(aName)
{
  collectionName.insert("hits");
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SensitiveDetector::SensitiveDetector(G4String aName, G4int numSensitive)
    : G4VSensitiveDetector(aName), fNumSensitive(numSensitive)
{
  collectionName.insert("hits");
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SensitiveDetector::~SensitiveDetector() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void SensitiveDetector::Initialize(G4HCofThisEvent *aHCE)
{
  fHitsCollection = new SimpleHitsCollection(SensitiveDetectorName, collectionName[0]);
  if (fHitCollectionID < 0) {
    fHitCollectionID = G4SDManager::GetSDMpointer()->GetCollectionID(fHitsCollection);
  }
  aHCE->AddHitsCollection(fHitCollectionID, fHitsCollection);

  // fill calorimeter hits with zero energy deposition
  for (G4int iz = 0; iz < fNumSensitive; iz++) {
    auto hit = new SimpleHit();
    fHitsCollection->insert(hit);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4bool SensitiveDetector::ProcessHits(G4Step *aStep, G4TouchableHistory *)
{
  G4double edep = aStep->GetTotalEnergyDeposit();
  if (edep == 0.) return true;

  G4TouchableHistory *aTouchable = (G4TouchableHistory *)(aStep->GetPreStepPoint()->GetTouchable());

  auto hit = RetrieveAndSetupHit(aTouchable);

  // Add energy deposit from G4Step
  hit->AddEdep(edep);

  // Fill time information from G4Step
  // If it's already filled, choose hit with earliest global time
  if (hit->GetTime() == -1 || hit->GetTime() > aStep->GetTrack()->GetGlobalTime())
    hit->SetTime(aStep->GetTrack()->GetGlobalTime());

  // Set hit type to full simulation (only if hit is not already marked as fast
  // sim)
  if (hit->GetType() != 1) hit->SetType(0);

  return true;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4bool SensitiveDetector::ProcessHits(const G4FastHit *aHit, const G4FastTrack *aTrack, G4TouchableHistory *aTouchable)
{

  G4double edep = aHit->GetEnergy();
  if (edep == 0.) return true;

  // I need to find the right hit based on the 'position'. For the moment, I use the position vector to store the id
  // of the cell.

  int hitID = (int)aHit->GetPosition().x();
  auto hit  = (*fHitsCollection)[hitID];

  // Add energy deposit from G4FastHit
  hit->AddEdep(edep);
  // set type to fast sim
  hit->SetType(1);

  /*
    // Fill time information from G4FastTrack
    // If it's already filled, choose hit with earliest global time
    if (hit->GetTime() == -1 || hit->GetTime() > aTrack->GetPrimaryTrack()->GetGlobalTime()) {
      hit->SetTime(aTrack->GetPrimaryTrack()->GetGlobalTime());
    }
  */

  return true;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4bool SensitiveDetector::ProcessHits(int hitID, double energy)
{

  if (energy == 0.) return true;

  auto hit = (*fHitsCollection)[hitID];

  // Add energy deposit from G4FastHit
  hit->AddEdep(energy);
  // set type to fast sim
  hit->SetType(1);

  return true;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SimpleHit *SensitiveDetector::RetrieveAndSetupHit(G4TouchableHistory *aTouchable)
{
  std::size_t hitID = (*fScoringMap)[aTouchable->GetHistory()->GetTopVolume()];
  assert(hitID < fNumSensitive);

  if (hitID >= fHitsCollection->entries()) {
    G4Exception("SensitiveDetector::RetrieveAndSetupHit()", "InvalidSetup", FatalException,
                "Size of hit collection in SensitiveDetector is smaller than the "
                "number of layers created in DetectorConstruction!");
  }

  return (*fHitsCollection)[hitID];
}