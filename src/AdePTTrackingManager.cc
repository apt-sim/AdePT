// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/integration/AdePTTrackingManager.hh>

#include "G4Threading.hh"
#include "G4Track.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4RunManager.hh"
#include "G4MTRunManager.hh"
#include "G4TransportationManager.hh"
#include "G4EmParameters.hh"

#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4Positron.hh"

#include <algorithm>

#ifdef ASYNC_MODE
std::shared_ptr<AdePTTransportInterface> InstantiateAdePT(AdePTConfiguration &conf,
                                                          G4HepEmTrackingManagerSpecialized *hepEmTM)
{
  static std::shared_ptr<AsyncAdePT::AsyncAdePTTransport<AdePTGeant4Integration>> AdePT{
      new AsyncAdePT::AsyncAdePTTransport<AdePTGeant4Integration>(conf, hepEmTM)};
  return AdePT;
}
#endif

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

AdePTTrackingManager::AdePTTrackingManager()
{
  fHepEmTrackingManager = std::make_unique<G4HepEmTrackingManagerSpecialized>();
}

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

  auto tid = G4Threading::G4GetThreadId();

  // Master thread cannot initialize AdePT as the number of G4 worker threads may not yet be known in applications.
  // In sequential mode, the tid is -2, so there we need to continue
  if (tid == -1) return;

  // a condition variable and a mutex is used for the initialization:
  // The first G4 worker that reaches the initialization, needs to initialize AdePT.
  // At the same time, all other G4 workers must wait for the common initialization to be finished, before they are
  // allowed to continue
  static std::once_flag onceFlag;
  static std::mutex initMutex;
  static std::condition_variable initCV;
  static bool commonInitDone = false;

  // Global initialization: only done once by the first worker thread
  std::call_once(onceFlag, [&]() {
    fNumThreads = G4MTRunManager::GetMasterRunManager()->GetNumberOfThreads();
    std::cout << " NUM OF THREADS ACCORDING TO G4: " << fNumThreads << std::endl;
    fAdePTConfiguration->SetNumThreads(fNumThreads);

    // Load the VecGeom world using G4VG if we don't have a GDML file, VGDML otherwise
    if (fAdePTConfiguration->GetVecGeomGDML().empty()) {
      auto *tman  = G4TransportationManager::GetTransportationManager();
      auto *world = tman->GetNavigatorForTracking()->GetWorldVolume();
      std::cout << "Loading geometry via G4VG\n";
      AdePTGeant4Integration::CreateVecGeomWorld(world);
    } else {
      std::cout << "Loading geometry via VGDML\n";
      AdePTGeant4Integration::CreateVecGeomWorld(fAdePTConfiguration->GetVecGeomGDML());
    }

#ifdef ADEPT_USE_EXT_BFIELD
    std::cout << "Reading in covfie file for magnetic field: " << fAdePTConfiguration->GetCovfieBfieldFile()
              << std::endl;
    if (fAdePTConfiguration->GetCovfieBfieldFile() == "") std::cout << "No magnetic field file provided!" << std::endl;
#endif

// Create an instance of an AdePT transport engine. This can either be one engine per thread or a shared engine for
// all threads.
#ifdef ASYNC_MODE
    fAdeptTransport = InstantiateAdePT(*fAdePTConfiguration, fHepEmTrackingManager.get());
#else
    fAdeptTransport =
        std::make_unique<AdePTTransport<AdePTGeant4Integration>>(*fAdePTConfiguration, fHepEmTrackingManager.get());

    // Initialize common data:
    // G4HepEM, Upload VecGeom geometry to GPU, Geometry check, Create volume auxiliary data
    fAdeptTransport->Initialize(fHepEmTrackingManager->GetConfig(), true /*common_data*/);
#endif

    // common init done, can notify other workers to proceed their initialization
    {
      std::lock_guard<std::mutex> lock(initMutex);
      commonInitDone = true;
    }
    initCV.notify_all();
  });

  // All other G4 worker threads must wait for common init to complete
  {
    std::unique_lock<std::mutex> lock(initMutex);
    initCV.wait(lock, [&] { return commonInitDone; });
  }

  // Now the fNumThreads is known and all workers can initialize
  fAdePTConfiguration->SetNumThreads(fNumThreads);

#ifdef ASYNC_MODE
  // the first G4 worker has already initialized the GPU worker, the other G4 workers need to get their shared pointer
  // here and need to initialize the integration layer with their own G4HepEmTracking manager (this is required for
  // nuclear processes)
  fAdeptTransport = InstantiateAdePT(*fAdePTConfiguration, fHepEmTrackingManager.get());
  fAdeptTransport->SetIntegrationLayerForThread(tid, fHepEmTrackingManager.get());
#else
  if (!sequential) { // if sequential, the instance is already created
    fAdeptTransport =
        std::make_unique<AdePTTransport<AdePTGeant4Integration>>(*fAdePTConfiguration, fHepEmTrackingManager.get());
  }
  // Initialize per-thread data
  fAdeptTransport->Initialize(fHepEmTrackingManager->GetConfig());
#endif

  // Initialize the GPU region list
  if (!fAdePTConfiguration->GetTrackInAllRegions()) {
    // Case 1: GPU regions are explicitly listed, and CPU regions must not overlap
    for (const std::string &regionName : *(fAdeptTransport->GetGPURegionNames())) {
      G4Region *region = G4RegionStore::GetInstance()->GetRegion(regionName);
      if (!region) {
        G4Exception("AdePTTrackingManager", "Invalid parameter", FatalErrorInArgument,
                    ("Region given to /adept/addGPURegion: " + regionName + " not found\n").c_str());
      }

      // Check for conflict with CPURegionNames
      for (const std::string &cpuRegionName : *(fAdeptTransport->GetCPURegionNames())) {
        if (regionName == cpuRegionName) {
          G4Exception("AdePTTrackingManager", "Conflicting region assignment", FatalErrorInArgument,
                      ("Region '" + regionName + "' is defined in both /adept/addGPURegion and /adept/removeGPURegion")
                          .c_str());
        }
      }

      G4cout << "AdePTTrackingManager: Marking " << regionName << " as a GPU Region" << G4endl;
      fGPURegions.insert(region);
    }

    fHepEmTrackingManager->SetTrackInAllRegions(false);

  } else if (!fAdeptTransport->GetCPURegionNames()->empty()) {
    // Case 2: Track everywhere except explicitly listed CPU regions
    const auto &cpuRegionNames = *(fAdeptTransport->GetCPURegionNames());

    // First mark all regions as GPU regions
    for (G4Region *region : *G4RegionStore::GetInstance()) {
      if (region) {
        fGPURegions.insert(region);
      }
    }

    // Then remove explicitly listed CPU regions
    for (const std::string &cpuRegionName : cpuRegionNames) {
      G4Region *region = G4RegionStore::GetInstance()->GetRegion(cpuRegionName);
      if (!region) {
        G4Exception("AdePTTrackingManager", "Invalid parameter", FatalErrorInArgument,
                    ("Region given to /adept/removeGPURegion: " + cpuRegionName + " not found\n").c_str());
      }

      G4cout << "AdePTTrackingManager: Removing " << cpuRegionName << " from GPU Regions" << G4endl;
      fGPURegions.erase(region);
    }

    fHepEmTrackingManager->SetTrackInAllRegions(false);
  } else {
    // Case 3: Track everywhere, no CPU overrides
    fHepEmTrackingManager->SetTrackInAllRegions(true);
  }
  // initialize special G4HepEmTrackingManager
  fHepEmTrackingManager->SetGPURegions(fGPURegions);
  fHepEmTrackingManager->ResetFinishEventOnCPUSize(fNumThreads);

  fSpeedOfLight = fAdePTConfiguration->GetSpeedOfLight();

  fAdePTInitialized = true;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void AdePTTrackingManager::BuildPhysicsTable(const G4ParticleDefinition &part)
{
  if (!fAdePTInitialized) {
    InitializeAdePT();
  }

  // Bulid PhysicsTable for G4HepEm
  fHepEmTrackingManager->BuildPhysicsTable(part);

  // For tracking on GPU by AdePT
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void AdePTTrackingManager::PreparePhysicsTable(const G4ParticleDefinition &part)
{
  // Prepare PhysicsTable for G4HepEm
  fHepEmTrackingManager->PreparePhysicsTable(part);

  // For tracking on GPU by AdePT
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void AdePTTrackingManager::HandOverOneTrack(G4Track *aTrack)
{

  // Speed of light: kill all e-+/gamma immediately
  if (fSpeedOfLight) {
    delete aTrack;
    return;
  }

  if (fGPURegions.empty() && !fAdeptTransport->GetTrackInAllRegions()) {
    // if no GPU regions, hand over directly to G4HepEmTrackingManager
    fHepEmTrackingManager->HandOverOneTrack(aTrack);
    if (aTrack->GetTrackStatus() != fStopAndKill) {
      throw std::logic_error(
          "Error: Although there is no GPU region, the G4HepEmTrackingManager did not finish tracking.");
    }
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

  G4EventManager *eventManager       = G4EventManager::GetEventManager();
  G4TrackingManager *trackManager    = eventManager->GetTrackingManager();
  G4SteppingManager *steppingManager = trackManager->GetSteppingManager();
  const bool trackInAllRegions       = fAdeptTransport->GetTrackInAllRegions();

  const auto eventID = eventManager->GetConstCurrentEvent()->GetEventID();

  // Check for GPU steps, to alleviate pressure on the GPU step buffer
  G4int threadId = G4Threading::G4GetThreadId();
  fAdeptTransport->ProcessGPUSteps(threadId, eventID);

  // new event detected, reset
  if (fHepEmTrackingManager->GetFinishEventOnCPU(threadId) >= 0 &&
      fHepEmTrackingManager->GetFinishEventOnCPU(threadId) != eventID) {
    fHepEmTrackingManager->SetFinishEventOnCPU(threadId, -1);
  }

  // first leaked particle detected, let's finish this event on CPU
  if (fHepEmTrackingManager->GetFinishEventOnCPU(threadId) < 0 && aTrack->GetTrackStatus() == fStopButAlive) {
    fHepEmTrackingManager->SetFinishEventOnCPU(threadId, eventID);
  }

  // setup touchable to be able to get region from GetNextVolume
  steppingManager->SetInitialStep(aTrack);

  // Track the particle Step-by-Step while it is alive
  while ((aTrack->GetTrackStatus() == fAlive) || (aTrack->GetTrackStatus() == fStopButAlive)) {
    G4Region const *region = aTrack->GetNextVolume()->GetLogicalVolume()->GetRegion();

    // Check if the particle is in a GPU region
    const bool isGPURegion = trackInAllRegions || fGPURegions.find(region) != fGPURegions.end();

    if (isGPURegion && (fHepEmTrackingManager->GetFinishEventOnCPU(threadId) < 0)) {
      // If the track is in a GPU region, hand it over to AdePT
      auto particlePosition  = aTrack->GetPosition();
      auto particleDirection = aTrack->GetMomentumDirection();
      G4double energy        = aTrack->GetKineticEnergy();
      G4double globalTime    = aTrack->GetGlobalTime();
      G4double localTime     = aTrack->GetLocalTime();
      G4double properTime    = aTrack->GetProperTime();
      G4double weight        = aTrack->GetWeight();
      auto pdg               = aTrack->GetParticleDefinition()->GetPDGEncoding();
      int id                 = aTrack->GetTrackID();
      if (fCurrentEventID != eventID) {
        // Do this to reproducibly seed the AdePT random numbers:
        fCurrentEventID = eventID;
        fTrackCounter   = 0;
      }

      // Get VecGeom Navigation state from G4History
      vecgeom::NavigationState converted = GetVecGeomFromG4State(*aTrack);

      // Get the VecGeom Navigation state at the track's origin
      vecgeom::NavigationState convertedOrigin;
      if (aTrack->GetParentID() == 0 && aTrack->GetCurrentStepNumber() == 0) {
        // For the first step of primary tracks, the origin touchable handle is not set,
        // so we need to use the track's current position
        // If the vertex is not in a GPU region, the origin touchable handle will be set by the HepEmTrackingManager
        convertedOrigin = GetVecGeomFromG4State(*aTrack->GetTouchable()->GetHistory());
      } else {
        // For secondary tracks, the origin touchable handle is set when they are stacked
        convertedOrigin = GetVecGeomFromG4State(*aTrack->GetOriginTouchableHandle()->GetHistory());
      }

      // Get the vertex information
      G4ThreeVector vertexPosition;
      G4ThreeVector vertexDirection;
      G4double vertexEnergy;
      if (aTrack->GetCurrentStepNumber() == 0) {
        // If it's the first step of the track these values are not set
        vertexPosition  = aTrack->GetPosition();
        vertexDirection = aTrack->GetMomentumDirection();
        vertexEnergy    = aTrack->GetKineticEnergy();
      } else {
        vertexPosition  = aTrack->GetVertexPosition();
        vertexDirection = aTrack->GetVertexMomentumDirection();
        vertexEnergy    = aTrack->GetVertexKineticEnergy();
      }

      fAdeptTransport->AddTrack(pdg, id, energy, vertexEnergy, particlePosition[0], particlePosition[1],
                                particlePosition[2], particleDirection[0], particleDirection[1], particleDirection[2],
                                vertexPosition[0], vertexPosition[1], vertexPosition[2], vertexDirection[0],
                                vertexDirection[1], vertexDirection[2], globalTime, localTime, properTime, weight,
                                G4Threading::G4GetThreadId(), eventID, fTrackCounter++, std::move(converted),
                                std::move(convertedOrigin));

      // The track dies from the point of view of Geant4
      aTrack->SetTrackStatus(fStopAndKill);

    } else { // If the particle is not in a GPU region, track it on CPU
             // Track the particle step by step until it dies or enters a GPU region in the (specialized)
             // G4HepEmTrackingManager
      fHepEmTrackingManager->HandOverOneTrack(aTrack);
    }
  }

  // delete track after finishing offloading to AdePT or finished tracking in G4HepEmTrackingManager
  delete aTrack;
}

const vecgeom::NavigationState AdePTTrackingManager::GetVecGeomFromG4State(
    const G4NavigationHistory &aG4NavigationHistory)
{

  // get history and depth from track
  auto aG4HistoryDepth = aG4NavigationHistory.GetDepth();

  // Initialize the NavState to be filled and push the world to it
  vecgeom::NavigationState aNavState;
  auto current_volume = vecgeom::GeoManager::Instance().GetWorld();
  aNavState.Push(current_volume);

  bool found_volume;
  // we pushed already the world, so we can start at level 1
  for (unsigned int level = 1; level <= aG4HistoryDepth; ++level) {

    found_volume = false;

    // Get current G4 volume and parent volume.
    const G4VPhysicalVolume *g4Volume_parent = aG4NavigationHistory.GetVolume(level - 1);
    const G4VPhysicalVolume *g4Volume        = aG4NavigationHistory.GetVolume(level);

    // The index of the VecGeom volume on this level (that we need to push the NavState to)
    // is the same as the G4 volume. The index of the G4 volume is found by matching it against
    // the daughters of the parent volume, since the G4 volume itself has no index.
    for (size_t id = 0; id < g4Volume_parent->GetLogicalVolume()->GetNoDaughters(); ++id) {
      if (g4Volume == g4Volume_parent->GetLogicalVolume()->GetDaughter(id)) {
        auto daughter = current_volume->GetLogicalVolume()->GetDaughters()[id];
        aNavState.Push(daughter);
        current_volume = daughter;
        found_volume   = true;
        break;
      }
    }

    if (!found_volume) {
      throw std::runtime_error("Fatal: G4 To VecGeom Geometry matching failed: G4 Volume name " +
                               std::string(g4Volume->GetLogicalVolume()->GetName()) +
                               " was not found in VecGeom Parent volume " +
                               std::string(current_volume->GetLogicalVolume()->GetName()));
    }
  }

  // Set boundary status
  aNavState.SetBoundaryState(false);

  return aNavState;
}

const vecgeom::NavigationState AdePTTrackingManager::GetVecGeomFromG4State(const G4Track &aG4Track)
{
  auto aNavState = GetVecGeomFromG4State(*aG4Track.GetNextTouchableHandle()->GetHistory());

  // Set boundary status based on the track
  if (aG4Track.GetStep() != nullptr) { // at initialization, the G4Step is not set yet, then we put OnBoundary to false
    if (aG4Track.GetStep()->GetPostStepPoint()->GetStepStatus() == fGeomBoundary) {
      aNavState.SetBoundaryState(true);
    } else {
      aNavState.SetBoundaryState(false);
    }
  } else {
    aNavState.SetBoundaryState(false);
  }

  return aNavState;
}
