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

AdePTTrackingManager::AdePTTrackingManager() {
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

AdePTTrackingManager::~AdePTTrackingManager() {
   fAdept->Cleanup();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void AdePTTrackingManager::BuildPhysicsTable(const G4ParticleDefinition &part) {
  
  // For tracking on CPU by Geant4, construct the physics tables for the processes of 
  // particles taken by this tracking manager, since Geant4 won't do it anymore
  G4ProcessManager* pManager = part.GetProcessManager();
  G4ProcessManager* pManagerShadow = part.GetMasterProcessManager();

  G4ProcessVector* pVector = pManager->GetProcessList();
  for (std::size_t j = 0; j < pVector->size(); ++j) {
    if (pManagerShadow == pManager) {
      (*pVector)[j]->BuildPhysicsTable(part);
    }
    else {
      (*pVector)[j]->BuildWorkerPhysicsTable(part);
    }
  }

  // For tracking on GPU by AdePT

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void AdePTTrackingManager::PreparePhysicsTable(
    const G4ParticleDefinition &part) {

  // For tracking on CPU by Geant4, prepare the physics tables for the processes of 
  // particles taken by this tracking manager, since Geant4 won't do it anymore
  G4ProcessManager* pManager = part.GetProcessManager();
  G4ProcessManager* pManagerShadow = part.GetMasterProcessManager();

  G4ProcessVector* pVector = pManager->GetProcessList();
  for (std::size_t j = 0; j < pVector->size(); ++j) {
    if (pManagerShadow == pManager) {
      (*pVector)[j]->PreparePhysicsTable(part);
    }
    else {
      (*pVector)[j]->PrepareWorkerPhysicsTable(part);
    }
  }

  // For tracking on GPU by AdePT

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void AdePTTrackingManager::HandOverOneTrack(G4Track *aTrack) {

  auto particlePosition  = aTrack->GetPosition();
  auto particleDirection = aTrack->GetMomentumDirection();
  G4double energy = aTrack->GetKineticEnergy();
  auto pdg = aTrack->GetParticleDefinition()->GetPDGEncoding();

  fAdept->AddTrack(pdg, energy, particlePosition[0], particlePosition[1], particlePosition[2], particleDirection[0],
                   particleDirection[1], particleDirection[2]);

  aTrack->SetTrackStatus(fStopAndKill);
  delete aTrack;
}

void AdePTTrackingManager::FlushEvent() {

  if (fVerbosity > 0)
    G4cout << "No more particles on the stack, triggering shower to flush the AdePT buffer with "
           << fAdept->GetNtoDevice() << " particles left." << G4endl;


  fAdept->Shower(G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID());
}

