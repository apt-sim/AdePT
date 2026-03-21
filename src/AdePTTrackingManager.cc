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

#include <VecGeom/management/GeoManager.h>

#include <algorithm>

#ifdef ENABLE_POWER_METER
#include <power_meter.hh>
#endif

namespace {
using AdePTTransport = AdePTTrackingManager::AdePTTransport;

// Store only a weak reference here so the transport lifetime is still owned by
// the thread-local AdePTTrackingManager instances. A static owning shared_ptr
// would keep the transport alive until very late process teardown.
std::weak_ptr<AdePTTransport> &SharedAdePTTransportStorage()
{
  static std::weak_ptr<AdePTTransport> transport;
  return transport;
}

std::shared_ptr<AdePTTransport> GetSharedAdePTTransport(
    AdePTConfiguration &conf, std::unique_ptr<AsyncAdePT::AdePTG4HepEmState> adeptG4HepEmState,
    adeptint::VolAuxData *auxData, const adeptint::WDTHostPacked &wdtPacked,
    const std::vector<float> &uniformFieldValues)
{
  auto &transport = SharedAdePTTransportStorage();
  // weak_ptr::lock() promotes the stored weak reference to a shared_ptr if the
  // shared transport is still alive. This is not a mutex lock.
  if (auto existing = transport.lock()) {
    return existing;
  }

  auto created =
      std::make_shared<AdePTTransport>(conf, std::move(adeptG4HepEmState), auxData, wdtPacked, uniformFieldValues);
  transport = created;
  return created;
}

std::shared_ptr<AdePTTransport> GetSharedAdePTTransport()
{
  // weak_ptr::lock() promotes the weak reference held in static storage. The
  // actual ownership remains with the AdePTTrackingManager instances.
  auto transport = SharedAdePTTransportStorage().lock();
  if (!transport) {
    throw std::runtime_error("Shared AdePT transport is not available.");
  }
  return transport;
}
} // namespace

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

AdePTTrackingManager::AdePTTrackingManager(AdePTConfiguration *config, int verbosity)
    : fHepEmTrackingManager(std::make_unique<G4HepEmTrackingManagerSpecialized>()), fAdePTConfiguration(config),
      fVerbosity(verbosity)
{
  fGeant4Integration.SetHepEmTrackingManager(fHepEmTrackingManager.get());
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

AdePTTrackingManager::~AdePTTrackingManager()
{
#ifdef ENABLE_POWER_METER
  // NOTE: Prior to Geant4 11.3 the destructor for the specialized tracking managers was not
  // called. In this case the loop will not be stopped and we will get an error at the end of
  // the run. This should not cause a memory leak however as terminate will be called on the thread
  if (fPowerMeterRunning) power_meter::stop_monitoring_loop();
#endif
}

void AdePTTrackingManager::InitializeSharedAdePTTransport()
{
#ifdef ADEPT_USE_EXT_BFIELD
  std::cout << "Reading in covfie file for magnetic field: " << fAdePTConfiguration->GetCovfieBfieldFile() << std::endl;
  if (fAdePTConfiguration->GetCovfieBfieldFile() == "") std::cout << "No magnetic field file provided!" << std::endl;
#endif
  const auto uniformFieldValues = fGeant4Integration.GetUniformField();
  auto adeptG4HepEmState        = std::make_unique<AsyncAdePT::AdePTG4HepEmState>(fHepEmTrackingManager->GetConfig());

  // Check VecGeom geometry matches Geant4 before deriving any geometry metadata for transport.
  fGeant4Integration.CheckGeometry(adeptG4HepEmState->GetData());

  // Initialize auxiliary per-LV data and collect the raw WDT metadata on the Geant4 side.
  auto *auxData = new adeptint::VolAuxData[vecgeom::GeoManager::Instance().GetRegisteredVolumesCount()];
  adeptint::WDTHostRaw wdtRaw;
  fGeant4Integration.InitVolAuxData(auxData, adeptG4HepEmState->GetData(), fHepEmTrackingManager.get(),
                                    fAdePTConfiguration->GetTrackInAllRegions(),
                                    fAdePTConfiguration->GetGPURegionNames(), wdtRaw);
  adeptint::WDTHostPacked wdtPacked = adeptint::PackWDT(wdtRaw);

  // Build the AdePT-owned G4HepEm state on the Geant4 side, then move that
  // ownership into the shared transport once all host-side preparation is complete.
  fAdeptTransport = GetSharedAdePTTransport(*fAdePTConfiguration, std::move(adeptG4HepEmState), auxData, wdtPacked,
                                            uniformFieldValues);
}

void AdePTTrackingManager::InitializeAdePT()
{
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
    // get number of threads from config, if available
    if (fNumThreads <= 0) {
      auto nThreads = fAdePTConfiguration->GetNumThreads();
      if (nThreads > 0) fNumThreads = nThreads;
    }

    // otherwise, try to get the threads frm the G4RunManager
    if (fNumThreads <= 0) {
      if (G4MTRunManager::GetMasterRunManager()) {
        fNumThreads = G4MTRunManager::GetMasterRunManager()->GetNumberOfThreads();
      } else {
        throw std::runtime_error(
            "Number of G4 workers is neither passed via the AdeptConfig nor is the G4MasterRunManager available");
      }
    }

    std::cout << " NUM OF THREADS ACCORDING TO G4: " << fNumThreads << std::endl;
    fAdePTConfiguration->SetNumThreads(fNumThreads);

    // Load the VecGeom world using G4VG if we don't have a GDML file, VGDML otherwise
    if (fAdePTConfiguration->GetVecGeomGDML().empty()) {
      auto *tman  = G4TransportationManager::GetTransportationManager();
      auto *world = tman->GetNavigatorForTracking()->GetWorldVolume();
      std::cout << "Loading geometry via G4VG\n";
      AdePTGeant4Integration::CreateVecGeomWorld(world);
    } else {
#ifdef VECGEOM_GDML_SUPPORT
      std::cout << "Loading geometry via VGDML\n";
      AdePTGeant4Integration::CreateVecGeomWorld(fAdePTConfiguration->GetVecGeomGDML());
#endif
    }

    InitializeSharedAdePTTransport();

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

  // Set the fNumThreads in case it was retrieved from G4RunManager
  // Now the fNumThreads is known and all workers can initialize
  fAdePTConfiguration->SetNumThreads(fNumThreads);

  // The shared AdePT transport was already created and initialized by the first worker.
  // The remaining workers only retrieve the shared pointer here.
  fAdeptTransport = GetSharedAdePTTransport();

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

#ifdef ENABLE_POWER_METER
  // Start measuring power consumption from here
  static std::once_flag onceFlagPower;
  std::call_once(onceFlagPower, [&]() {
    fPowerMeterRunning = true;
    try {
      // Start measuring consumption from this point
      power_meter::launch_monitoring_loop(1000);
    } catch (const std::exception &ex) {
      fPowerMeterRunning = false;
      std::cerr << "\033[31m" << ex.what() << "\033[0m" << std::endl;
      printf("\033[31m ERROR: Power meter could not be initialized, consumption will not be recorded for this run "
             "\033[0m\n");
    }
  });
#endif
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void AdePTTrackingManager::BuildPhysicsTable(const G4ParticleDefinition &part)
{

  // Build PhysicsTable for G4HepEm
  // Note: the G4HepEm physics table must be build first, such that the Woodcock tracking helper
  // is initialized when AdePT gets its data from there
  fHepEmTrackingManager->BuildPhysicsTable(part);

  if (!fAdePTInitialized) {
    InitializeAdePT();
  }

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
  if (fVerbosity > 1)
    G4cout << "No more particles on the stack, triggering shower to flush the AdePT buffer." << G4endl;

  fAdeptTransport->Flush(G4Threading::G4GetThreadId(),
                         G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID(), fGeant4Integration);
}

void AdePTTrackingManager::ProcessTrack(G4Track *aTrack)
{

  G4EventManager *eventManager       = G4EventManager::GetEventManager();
  G4TrackingManager *trackManager    = eventManager->GetTrackingManager();
  G4SteppingManager *steppingManager = trackManager->GetSteppingManager();
  const bool trackInAllRegions       = fAdeptTransport->GetTrackInAllRegions();
  const bool callUserActions         = fAdeptTransport->GetCallUserActions();

  const auto eventID = eventManager->GetConstCurrentEvent()->GetEventID();

  // Check for GPU steps, to alleviate pressure on the GPU step buffer
  G4int threadId = G4Threading::G4GetThreadId();
  fAdeptTransport->ProcessGPUSteps(threadId, eventID, fGeant4Integration);
  auto &trackMapper = fGeant4Integration.GetHostTrackDataMapper();

  if (fCurrentEventID != eventID) trackMapper.beginEvent(eventID);

  // new event detected, reset
  if (fHepEmTrackingManager->GetFinishEventOnCPU(threadId) >= 0 &&
      fHepEmTrackingManager->GetFinishEventOnCPU(threadId) != eventID) {
    fHepEmTrackingManager->SetFinishEventOnCPU(threadId, -1);
  }

  // first leaked particle detected, let's finish this event on CPU
  if (fHepEmTrackingManager->GetFinishEventOnCPU(threadId) < 0 && aTrack->GetTrackStatus() == fStopButAlive) {
    fHepEmTrackingManager->SetFinishEventOnCPU(threadId, eventID);
  }

  // If this is the first step, set touchable and next touchable via SetInitialStep
  // All other tracks should have a current and next touchable being set
  if (aTrack->GetCurrentStepNumber() == 0) {
    steppingManager->SetInitialStep(aTrack);
  }

  // Track the particle Step-by-Step while it is alive
  while ((aTrack->GetTrackStatus() == fAlive) || (aTrack->GetTrackStatus() == fStopButAlive)) {
    G4Region const *region = aTrack->GetNextVolume()->GetLogicalVolume()->GetRegion();

    // Check if the particle is in a GPU region
    const bool isGPURegion = trackInAllRegions || fGPURegions.find(region) != fGPURegions.end();

    if (isGPURegion && (fHepEmTrackingManager->GetFinishEventOnCPU(threadId) < 0)) {

      // Speed of light: kill all e-+/gamma immediately
      if (fSpeedOfLight) {
        delete aTrack;
        return;
      }

      // If the track is in a GPU region, hand it over to AdePT
      auto pdg = aTrack->GetParticleDefinition()->GetPDGEncoding();

      // Get GPU ID: either it already exists, then the previously used GPU id is given back to the G4 id, or a new one
      // is created (by casting the int G4 id into a uint64 GPU id) Then, the existing hostTrackData is returned or a
      // new one is created in case it either never existed or was retired after the track left the GPU region
      uint64_t gpuTrackID;
      HostTrackData dummy; // default constructed dummy if no advanced information is available
      bool entryExists = trackMapper.getGPUId(aTrack->GetTrackID(), gpuTrackID);
      HostTrackData &hostTrackData =
          callUserActions ? trackMapper.activateForGPU(gpuTrackID, aTrack->GetTrackID(), entryExists) : dummy;

      // fill hostTracKData if being used:
      if (callUserActions) {
        // set pointers and G4 parent id
        hostTrackData.primary        = aTrack->GetDynamicParticle()->GetPrimaryParticle();
        hostTrackData.creatorProcess = const_cast<G4VProcess *>(aTrack->GetCreatorProcess());
        hostTrackData.g4parentid     = aTrack->GetParentID();

        // Set the vertex information
        if (aTrack->GetCurrentStepNumber() == 0) {
          // If it's the first step of the track these values are not set
          hostTrackData.vertexPosition          = aTrack->GetPosition();
          hostTrackData.vertexMomentumDirection = aTrack->GetMomentumDirection();
          hostTrackData.vertexKineticEnergy     = aTrack->GetKineticEnergy();
          hostTrackData.logicalVolumeAtVertex   = aTrack->GetTouchableHandle()->GetVolume()->GetLogicalVolume();
        } else {
          hostTrackData.vertexPosition          = aTrack->GetVertexPosition();
          hostTrackData.vertexMomentumDirection = aTrack->GetVertexMomentumDirection();
          hostTrackData.vertexKineticEnergy     = aTrack->GetVertexKineticEnergy();
          hostTrackData.logicalVolumeAtVertex   = const_cast<G4LogicalVolume *>(aTrack->GetLogicalVolumeAtVertex());
        }

        // set the particle type
        if (pdg == 11) {
          hostTrackData.particleType = ParticleType::Electron;
        } else if (pdg == -11) {
          hostTrackData.particleType = ParticleType::Positron;
        } else if (pdg == 22) {
          hostTrackData.particleType = ParticleType::Gamma;
        }
      }

      // if there has been no step, call PreUserTrackingAction and try to attach UserInformation
      if (aTrack->GetCurrentStepNumber() == 0) {
        auto *userTrackingAction = eventManager->GetUserTrackingAction();
        if (userTrackingAction) {

          // this assumes that the UserTrackInformation is attached to the track in the PreUserTrackingAction
          userTrackingAction->PreUserTrackingAction(aTrack);
          hostTrackData.userTrackInfo = aTrack->GetUserInformation();
        }
      } else {
        // not the initializing step, just attach user information in case it is there
        hostTrackData.userTrackInfo = aTrack->GetUserInformation();
      }

      uint64_t gpuParentID;
      trackMapper.getGPUId(aTrack->GetParentID(), gpuParentID);

      auto particlePosition     = aTrack->GetPosition();
      auto particleDirection    = aTrack->GetMomentumDirection();
      G4double energy           = aTrack->GetKineticEnergy();
      G4double globalTime       = aTrack->GetGlobalTime();
      G4double localTime        = aTrack->GetLocalTime();
      G4double properTime       = aTrack->GetProperTime();
      G4double weight           = aTrack->GetWeight();
      unsigned short stepNumber = static_cast<unsigned short>(aTrack->GetCurrentStepNumber());

      if (fCurrentEventID != eventID) {
        // Do this to reproducibly seed the AdePT random numbers:
        fCurrentEventID = eventID;
        fTrackCounter   = 0;
      }

      // Get VecGeom Navigation state from G4History
      vecgeom::NavigationState converted = GetVecGeomFromG4State(*aTrack);

      // The VecGeom NavState is stored in the hostTrackData; in principle, the G4TouchableHandle could also be stored
      // directly, but it has proven to be very expensive to create new G4TouchableHandle objects for each track in the
      // HostTrackData. Instead, it was much cheaper to just store the vecgeom::NavState and create the
      // G4TouchableHandle only when the track is returned from the GPU
      if (callUserActions) {
        if (aTrack->GetParentID() == 0 && aTrack->GetCurrentStepNumber() == 0) {
          // For the first step of primary tracks, the origin touchable handle is not set,
          // so we need to use the track's current position
          // If the vertex is not in a GPU region, the origin touchable handle will be set by the HepEmTrackingManager
          hostTrackData.originNavState = converted;
        } else {
          // For secondary tracks, the origin touchable handle is set when they are stacked
          vecgeom::NavigationState convertedOrigin =
              GetVecGeomFromG4State(*aTrack, aTrack->GetOriginTouchableHandle()->GetHistory());
          hostTrackData.originNavState = convertedOrigin;
        }
      }

      fAdeptTransport->AddTrack(pdg, gpuTrackID, gpuParentID, energy, particlePosition[0], particlePosition[1],
                                particlePosition[2], particleDirection[0], particleDirection[1], particleDirection[2],
                                globalTime, localTime, properTime, weight, stepNumber, G4Threading::G4GetThreadId(),
                                eventID, std::move(converted));

      fTrackCounter++; // increment the track counter for AdePT

      // The track dies from the point of view of Geant4
      aTrack->SetTrackStatus(fStopAndKill);

      // After the track has been offloaded to the GPU, it can be deleted on the CPU.
      // However, the HostTrackData is now owning and therefore responsible for deleting the TrackUserInfo data
      // To avoid deletion of the TrackUserInfo data when the track is deleted, the pointer must be reset.
      // Then, the underlying data is either deleted via hostTrackData.removeTrack in case the track is finished on GPU,
      // or the ownership is transferred back to G4, when the track is given back to the CPU
      aTrack->SetUserInformation(nullptr);

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

const vecgeom::NavigationState AdePTTrackingManager::GetVecGeomFromG4State(
    const G4Track &aG4Track, const G4NavigationHistory *aG4NavigationHistory)
{

  if (!aG4NavigationHistory) {
    aG4NavigationHistory = aG4Track.GetNextTouchableHandle()->GetHistory();
  }

  auto aNavState = GetVecGeomFromG4State(*aG4NavigationHistory);

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
