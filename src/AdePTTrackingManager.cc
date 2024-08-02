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
  fAdeptTransport->Cleanup();
}

void AdePTTrackingManager::InitializeAdePT()
{
  // AdePT needs to be initialized here, since we know all needed Geant4 initializations are already finished
  fAdeptTransport->SetDebugLevel(0);
  fAdeptTransport->SetBufferThreshold(fAdePTConfiguration->GetTransportBufferThreshold());
  fAdeptTransport->SetMaxBatch(2 * fAdePTConfiguration->GetTransportBufferThreshold());
  fAdeptTransport->SetTrackInAllRegions(fAdePTConfiguration->GetTrackInAllRegions());
  fAdeptTransport->SetGPURegionNames(fAdePTConfiguration->GetGPURegionNames());

  // Check if this is a sequential run
  G4RunManager::RMType rmType = G4RunManager::GetRunManager()->GetRunManagerType();
  bool sequential             = (rmType == G4RunManager::sequentialRM);

  // One thread initializes common elements
  auto tid = G4Threading::G4GetThreadId();
  if (tid < 0) {
    // Load the VecGeom world in memory
    AdePTGeant4Integration::CreateVecGeomWorld(fAdePTConfiguration->GetVecGeomGDML());

    // Track and Hit buffer capacities on GPU are split among threads
    int num_threads    = G4RunManager::GetRunManager()->GetNumberOfThreads();
    int track_capacity = 1024 * 1024 * fAdePTConfiguration->GetMillionsOfTrackSlots() / num_threads;
    G4cout << "AdePT Allocated track capacity: " << track_capacity << " tracks" << G4endl;
    fAdeptTransport->SetTrackCapacity(track_capacity);
    int hit_buffer_capacity = 1024 * 1024 * fAdePTConfiguration->GetMillionsOfHitSlots() / num_threads;
    G4cout << "AdePT Allocated hit buffer capacity: " << hit_buffer_capacity << " slots" << G4endl;
    fAdeptTransport->SetHitBufferCapacity(hit_buffer_capacity);

    // Initialize common data:
    // G4HepEM, Upload VecGeom geometry to GPU, Geometry check, Create volume auxiliary data
    fAdeptTransport->Initialize(true /*common_data*/);
    if (sequential) {
      // Initialize per-thread data (When in sequential mode)
      fAdeptTransport->Initialize();
    }
  } else {
    // Initialize per-thread data
    fAdeptTransport->Initialize();
  }

  // Initialize the GPU region list

  if (!fAdeptTransport->GetTrackInAllRegions()) {
    for (std::string regionName : *(fAdeptTransport->GetGPURegionNames())) {
      G4cout << "AdePTTrackingManager: Marking " << regionName << " as a GPU Region" << G4endl;
      G4Region *region = G4RegionStore::GetInstance()->GetRegion(regionName);
      if (region != nullptr)
        fGPURegions.insert(region);
      else
        G4Exception("AdePTTrackingManager", "Invalid parameter", FatalErrorInArgument,
                    ("Region given to /adept/addGPURegion: " + regionName + " Not found\n").c_str());
    }
  }

  fAdePTInitialized = true;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void AdePTTrackingManager::BuildPhysicsTable(const G4ParticleDefinition &part)
{
  if (!fAdePTInitialized) {
    InitializeAdePT();
  }

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
           << fAdeptTransport->GetNtoDevice() << " particles left." << G4endl;

  fAdeptTransport->Shower(G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID());
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
    // Check if the particle is in a GPU region
    bool isGPURegion = false;
    if (fAdeptTransport->GetTrackInAllRegions()) {
      isGPURegion = true;
    } else {
      if (fGPURegions.find(region) != fGPURegions.end()) {
        isGPURegion = true;
      }
    }

    if (isGPURegion) {
      // If the track is in a GPU region, hand it over to AdePT

      auto particlePosition  = aTrack->GetPosition();
      auto particleDirection = aTrack->GetMomentumDirection();
      G4double energy        = aTrack->GetKineticEnergy();
      G4double globalTime    = aTrack->GetGlobalTime();
      G4double localTime     = aTrack->GetLocalTime();
      G4double properTime    = aTrack->GetProperTime();
      auto pdg               = aTrack->GetParticleDefinition()->GetPDGEncoding();
      int id                 = aTrack->GetTrackID();

      fAdeptTransport->AddTrack(pdg, id, energy, particlePosition[0], particlePosition[1], particlePosition[2],
                                particleDirection[0], particleDirection[1], particleDirection[2], globalTime, localTime,
                                properTime);

      // The track dies from the point of view of Geant4
      aTrack->SetTrackStatus(fStopAndKill);

    } else {
      // If the particle is not in a GPU region, track it on CPU
      if (fGPURegions.empty()) {
        // If there are no GPU regions, track until the end in Geant4
        trackManager->ProcessOneTrack(aTrack);
        if (aTrack->GetTrackStatus() != fStopAndKill) {
          G4Exception("AdePTTrackingManager::HandOverOneTrack", "NotStopped", FatalException, "track was not stopped");
        }
      } else {
        // Track the particle step by step until it dies or enters a GPU region
        StepInHostRegion(aTrack);
      }
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

  G4EventManager *eventManager       = G4EventManager::GetEventManager();
  G4TrackingManager *trackManager    = eventManager->GetTrackingManager();
  G4SteppingManager *steppingManager = trackManager->GetSteppingManager();
  G4Region *previousRegion           = aTrack->GetVolume()->GetLogicalVolume()->GetRegion();

  // Track the particle Step-by-Step while it is alive and outside of a GPU region
  while ((aTrack->GetTrackStatus() == fAlive) || (aTrack->GetTrackStatus() == fStopButAlive)) {
    aTrack->IncrementCurrentStepNumber();
    steppingManager->Stepping();

    if (aTrack->GetTrackStatus() != fStopAndKill) {
      // Switch the touchable to update the volume, which is checked in the
      // condition below and at the call site.
      aTrack->SetTouchableHandle(aTrack->GetNextTouchableHandle());
      G4Region *region = aTrack->GetVolume()->GetLogicalVolume()->GetRegion();
      // This should never be true if this flag is set, as all particles would be sent to AdePT
      assert(fAdeptTransport->GetTrackInAllRegions() == false);
      // If the region changed, check whether the particle has entered a GPU region
      if (region != previousRegion) {
        previousRegion = region;
        if (fGPURegions.find(region) != fGPURegions.end()) {
          return;
        }
      }
    }
  }
}
