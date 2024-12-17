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

#include <algorithm>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

AdePTTrackingManager::AdePTTrackingManager() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

AdePTTrackingManager::~AdePTTrackingManager()
{
  if (fAdeptTransport) fAdeptTransport->Cleanup();
}

void AdePTTrackingManager::InitializeAdePT()
{
  // Check if this is a sequential run
  G4RunManager::RMType rmType = G4RunManager::GetRunManager()->GetRunManagerType();
  bool sequential             = (rmType == G4RunManager::sequentialRM);

  // One thread initializes common elements
  auto tid = G4Threading::G4GetThreadId();
  if (tid < 0) {
    // Only the master thread knows the actual number of threads, the worker threads will return "1"
    // This value is stored here by the master in a static variable, and used by each thread to pass the 
    // correct number to their AdePTConfiguration instance
    fNumThreads = G4RunManager::GetRunManager()->GetNumberOfThreads();
    fAdePTConfiguration->SetNumThreads(fNumThreads);

    // Load the VecGeom world in memory
    AdePTGeant4Integration::CreateVecGeomWorld(fAdePTConfiguration->GetVecGeomGDML());

    // Create an instance of an AdePT transport engine. This can either be one engine per thread or a shared engine for
    // all threads.
    #ifndef ASYNC_MODE
    fAdeptTransport = std::make_shared<AdePTTransport<AdePTGeant4Integration>>(*fAdePTConfiguration);
    #else
    fAdeptTransport = std::make_shared<AsyncAdePT::AsyncAdePTTransport<AdePTGeant4Integration>>(*fAdePTConfiguration);
    #endif

    // Initialize common data:
    // G4HepEM, Upload VecGeom geometry to GPU, Geometry check, Create volume auxiliary data
    fAdeptTransport->Initialize(true /*common_data*/);
    if (sequential) {
      // Initialize per-thread data (When in sequential mode)
      fAdeptTransport->Initialize();
    }
  } else {
    // Create an instance of an AdePT transport engine. This can either be one engine per thread or a shared engine for
    // all threads.
    fAdePTConfiguration->SetNumThreads(fNumThreads);
    #ifndef ASYNC_MODE
    fAdeptTransport = std::make_shared<AdePTTransport<AdePTGeant4Integration>>(*fAdePTConfiguration);
    #else
    fAdeptTransport = std::make_shared<AsyncAdePT::AsyncAdePTTransport<AdePTGeant4Integration>>(*fAdePTConfiguration);
    #endif
    // Initialize per-thread data
    fAdeptTransport->Initialize();
  }

  // Initialize the GPU region list

  if (!fAdePTConfiguration->GetTrackInAllRegions()) {
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
  if (fGPURegions.empty() && !fAdeptTransport->GetTrackInAllRegions()) {
    G4EventManager *eventManager       = G4EventManager::GetEventManager();
    G4TrackingManager *trackManager    = eventManager->GetTrackingManager();
    // If there are no GPU regions, track until the end in Geant4
    trackManager->ProcessOneTrack(aTrack);
    if (aTrack->GetTrackStatus() != fStopAndKill) {
      G4Exception("AdePTTrackingManager::HandOverOneTrack", "NotStopped", FatalException, "track was not stopped");
    }
    G4TrackVector* secondaries = trackManager->GimmeSecondaries();
    eventManager->StackTracks(secondaries);
    delete aTrack;
    return;
  }
  ProcessTrack(aTrack);
}

void AdePTTrackingManager::FlushEvent()
{
  if (fVerbosity > 0)
    G4cout << "No more particles on the stack, triggering shower to flush the AdePT buffer." << G4endl;

  fAdeptTransport->Shower(G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID(),
                          G4Threading::G4GetThreadId());
}

void AdePTTrackingManager::ProcessTrack(G4Track *aTrack)
{
  /* From G4 Example RE07 */

  G4EventManager *eventManager       = G4EventManager::GetEventManager();
  G4TrackingManager *trackManager    = eventManager->GetTrackingManager();
  G4SteppingManager *steppingManager = trackManager->GetSteppingManager();
  G4TrackVector *secondaries         = trackManager->GimmeSecondaries();
  const bool trackInAllRegions       = fAdeptTransport->GetTrackInAllRegions();

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
    G4Region const *region = aTrack->GetVolume()->GetLogicalVolume()->GetRegion();

    // Check if the particle is in a GPU region
    const bool isGPURegion = trackInAllRegions || fGPURegions.find(region) != fGPURegions.end();

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
      const auto eventID     = eventManager->GetConstCurrentEvent()->GetEventID();
      if (fCurrentEventID != eventID) {
        // Do this to reproducibly seed the AdePT random numbers:
        fCurrentEventID = eventID;
        fTrackCounter   = 0;
      }

      fAdeptTransport->AddTrack(pdg, id, energy, particlePosition[0], particlePosition[1], particlePosition[2],
                                particleDirection[0], particleDirection[1], particleDirection[2], globalTime, localTime,
                                properTime, G4Threading::G4GetThreadId(), eventID, fTrackCounter++);

      // The track dies from the point of view of Geant4
      aTrack->SetTrackStatus(fStopAndKill);

    } else { // If the particle is not in a GPU region, track it on CPU
      // Track the particle step by step until it dies or enters a GPU region
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

  G4EventManager *eventManager       = G4EventManager::GetEventManager();
  G4TrackingManager *trackManager    = eventManager->GetTrackingManager();
  G4SteppingManager *steppingManager = trackManager->GetSteppingManager();
  G4Region const *previousRegion       = aTrack->GetVolume()->GetLogicalVolume()->GetRegion();

  // Track the particle Step-by-Step while it is alive and outside of a GPU region
  while ((aTrack->GetTrackStatus() == fAlive || aTrack->GetTrackStatus() == fStopButAlive)) {
    aTrack->IncrementCurrentStepNumber();
    steppingManager->Stepping();

    if (aTrack->GetTrackStatus() != fStopAndKill) {
      // Switch the touchable to update the volume, which is checked in the
      // condition below and at the call site.
      aTrack->SetTouchableHandle(aTrack->GetNextTouchableHandle());
      G4Region const *region = aTrack->GetVolume()->GetLogicalVolume()->GetRegion();

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
