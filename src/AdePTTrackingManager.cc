// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/integration/AdePTTrackingManager.hh>

#include "G4Threading.hh"
#include "G4Track.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4RunManager.hh"

#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4Positron.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

AdePTTrackingManager::AdePTTrackingManager() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

AdePTTrackingManager::~AdePTTrackingManager()
{
  fAdept->Cleanup();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void AdePTTrackingManager::BuildPhysicsTable(const G4ParticleDefinition &part)
{

  // For tracking on CPU by Geant4, construct the physics tables for the processes of
  // particles taken by this tracking manager, since Geant4 won't do it anymore
  G4ProcessManager *pManager       = part.GetProcessManager();
  G4ProcessManager *pManagerShadow = part.GetMasterProcessManager();

  G4ProcessVector *pVector = pManager->GetProcessList();
  for (std::size_t j = 0; j < pVector->size(); ++j) {
    if (pManagerShadow == pManager) {
      (*pVector)[j]->BuildPhysicsTable(part);
    } else {
      (*pVector)[j]->BuildWorkerPhysicsTable(part);
    }
  }

  // For tracking on GPU by AdePT
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void AdePTTrackingManager::PreparePhysicsTable(const G4ParticleDefinition &part)
{

  // For tracking on CPU by Geant4, prepare the physics tables for the processes of
  // particles taken by this tracking manager, since Geant4 won't do it anymore
  G4ProcessManager *pManager       = part.GetProcessManager();
  G4ProcessManager *pManagerShadow = part.GetMasterProcessManager();

  G4ProcessVector *pVector = pManager->GetProcessList();
  for (std::size_t j = 0; j < pVector->size(); ++j) {
    if (pManagerShadow == pManager) {
      (*pVector)[j]->PreparePhysicsTable(part);
    } else {
      (*pVector)[j]->PrepareWorkerPhysicsTable(part);
    }
  }

  // For tracking on GPU by AdePT
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void AdePTTrackingManager::HandOverOneTrack(G4Track *aTrack)
{
  ProcessTrack(aTrack);
}

void AdePTTrackingManager::FlushEvent()
{

  if (fVerbosity > 0)
    G4cout << "No more particles on the stack, triggering shower to flush the AdePT buffer with "
           << fAdept->GetNtoDevice() << " particles left." << G4endl;

  fAdept->Shower(G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID());
}

void AdePTTrackingManager::ProcessTrack(G4Track *aTrack)
{
  /* From G4 Example RE07 */

  G4EventManager *eventManager       = G4EventManager::GetEventManager();
  G4TrackingManager *trackManager    = eventManager->GetTrackingManager();
  G4SteppingManager *steppingManager = trackManager->GetSteppingManager();
  G4TrackVector *secondaries         = trackManager->GimmeSecondaries();

  // Clear secondary particle vector
  for (std::size_t itr = 0; itr < secondaries->size(); ++itr) {
    delete (*secondaries)[itr];
  }
  secondaries->clear();

  steppingManager->SetInitialStep(aTrack);

  G4UserTrackingAction *userTrackingAction = trackManager->GetUserTrackingAction();
  if (userTrackingAction != nullptr) {
    userTrackingAction->PreUserTrackingAction(aTrack);
  }

  // Give SteppingManger the maxmimum number of processes
  steppingManager->GetProcessNumber();

  // Give track the pointer to the Step
  aTrack->SetStep(steppingManager->GetStep());

  // Inform beginning of tracking to physics processes
  aTrack->GetDefinition()->GetProcessManager()->StartTracking(aTrack);

  // Track the particle Step-by-Step while it is alive
  while ((aTrack->GetTrackStatus() == fAlive) || (aTrack->GetTrackStatus() == fStopButAlive)) {
    G4Region *region = aTrack->GetVolume()->GetLogicalVolume()->GetRegion();
    if (region == fPlaceholderRegion) {
      // If the track is in a GPU region, hand it over to AdePT

      auto particlePosition  = aTrack->GetPosition();
      auto particleDirection = aTrack->GetMomentumDirection();
      G4double energy        = aTrack->GetKineticEnergy();
      auto pdg               = aTrack->GetParticleDefinition()->GetPDGEncoding();

      fAdept->AddTrack(pdg, energy, particlePosition[0], particlePosition[1], particlePosition[2], particleDirection[0],
                       particleDirection[1], particleDirection[2]);
      
      // The track dies from the point of view of Geant4
      aTrack->SetTrackStatus(fStopAndKill);

    } else {
      // If the particle is not in a GPU region, track it on CPU

      StepInHostRegion(aTrack);
    }
  }
  // Inform end of tracking to physics processes
  aTrack->GetDefinition()->GetProcessManager()->EndTracking();

  if (userTrackingAction != nullptr) {
    userTrackingAction->PostUserTrackingAction(aTrack);
  }

  eventManager->StackTracks(secondaries);
  delete aTrack;
}

void AdePTTrackingManager::StepInHostRegion(G4Track *aTrack) 
{
  /* From G4 Example RE07 */

  G4EventManager* eventManager = G4EventManager::GetEventManager();
  G4TrackingManager* trackManager = eventManager->GetTrackingManager();
  G4SteppingManager* steppingManager = trackManager->GetSteppingManager();

  // Track the particle Step-by-Step while it is alive and outside of a GPU region
  while ((aTrack->GetTrackStatus() == fAlive) || (aTrack->GetTrackStatus() == fStopButAlive)) {
    aTrack->IncrementCurrentStepNumber();
    steppingManager->Stepping();

    if (aTrack->GetTrackStatus() != fStopAndKill) {
      // Switch the touchable to update the volume, which is checked in the
      // condition below and at the call site.
      aTrack->SetTouchableHandle(aTrack->GetNextTouchableHandle());
      G4Region* region = aTrack->GetVolume()->GetLogicalVolume()->GetRegion();
      if (region == fPlaceholderRegion) {
        return;
      }
    }
  }
}
