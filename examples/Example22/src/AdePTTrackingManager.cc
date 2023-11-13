// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include "AdePTTrackingManager.hh"

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
  if(!fInitDone)
  {
    fAdept = new AdeptIntegration;

    fAdept->SetDebugLevel(fVerbosity);
    fAdept->SetBufferThreshold(fBufferThreshold);
    fAdept->SetMaxBatch(2 * fBufferThreshold);

    G4RunManager::RMType rmType = G4RunManager::GetRunManager()->GetRunManagerType();
    bool sequential             = (rmType == G4RunManager::sequentialRM);

    fAdept->SetSensitiveVolumes(fSensitiveLogicalVolumes);
    fAdept->SetRegion(fRegion);

    auto tid = G4Threading::G4GetThreadId();
    if (tid < 0) {
      // This is supposed to set the max batching for Adept to allocate properly the memory
      int num_threads = G4RunManager::GetRunManager()->GetNumberOfThreads();
      int track_capacity    = 1024 * 1024 * fTrackSlotsGPU / num_threads;
      G4cout << "AdePT Allocated track capacity: " << track_capacity << " tracks" << G4endl;
      AdeptIntegration::SetTrackCapacity(track_capacity);
      int hit_buffer_capacity = 1024 * 1024 * fHitSlots / num_threads;
      G4cout << "AdePT Allocated hit buffer capacity: " << hit_buffer_capacity << " slots" << G4endl;
      AdeptIntegration::SetHitBufferCapacity(hit_buffer_capacity);
      fAdept->Initialize(true /*common_data*/);
      if (sequential) 
      {
        fAdept->Initialize();
      }
    } else {
      fAdept->Initialize();
    }
    fInitDone = true;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void AdePTTrackingManager::PreparePhysicsTable(
    const G4ParticleDefinition &part) {
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

