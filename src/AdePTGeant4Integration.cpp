// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/integration/AdePTGeant4Integration.hh>
#include <AdePT/integration/AdePTGeometryBridge.hh>

#include <VecGeom/navigation/NavigationState.h>

#include <G4ios.hh>
#include <G4SystemOfUnits.hh>
#include <G4TransportationManager.hh>
#include <G4VSensitiveDetector.hh>
#include <G4UniformMagField.hh>
#include <G4FieldManager.hh>
#include <G4TouchableHandle.hh>
#include <G4StepStatus.hh>

#include <G4HepEmNoProcess.hh>

#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4Gamma.hh"

#include <new>
#include <type_traits>

namespace AdePTGeant4Integration_detail {
/// This struct holds temporary scoring objects that are needed to send hits to Geant4.
/// These are allocated as members or using placement new to go around G4's pool allocators,
/// which cause a destruction order fiasco when the pools are destroyed before the objects,
/// and then the objects' destructors are called.
/// This also keeps all these objects much closer in memory.
struct ScoringObjects {

  static constexpr int kMaxSecElectrons     = 1;
  static constexpr int kMaxSecPositrons     = 1;
  static constexpr int kMaxSecGammas        = 3;
  static constexpr int kMaxTotalSecondaries = kMaxSecElectrons + kMaxSecPositrons + kMaxSecGammas;
  static constexpr int kMaxSecondaries      = 3;

  G4NavigationHistory fPreG4NavigationHistory;
  G4NavigationHistory fPostG4NavigationHistory;

  // Create local storage for placement new of step points:
  std::aligned_storage<sizeof(G4StepPoint), alignof(G4StepPoint)>::type stepPointStorage[2];
  std::aligned_storage<sizeof(G4Step), alignof(G4Step)>::type stepStorage;
  std::aligned_storage<sizeof(G4TouchableHistory), alignof(G4TouchableHistory)>::type toucheableHistoryStorage[2];
  std::aligned_storage<sizeof(G4TouchableHandle), alignof(G4TouchableHandle)>::type toucheableHandleStorage[2];
  // Cannot run G4Step's and TouchableHandle's destructors, since their internal reference-counted handles
  // are in memory pools. Therefore, we construct them as pointer to local memory.
  G4Step *fG4Step = nullptr;

  G4TouchableHandle *fPreG4TouchableHistoryHandle  = nullptr;
  G4TouchableHandle *fPostG4TouchableHistoryHandle = nullptr;

  // We need the dynamic particle associated to the track to have the correct particle definition, however this can
  // only be set at construction time. Similarly, we can only set the dynamic particle for a track when creating it
  // For this reason we create one track per particle type for the parent track and up to kMaxTotalSecondaries tracks
  // for the secondary tracks that could be processed for each parent step, to be reused We set position to nullptr and
  // kinetic energy to 0 for the dynamic particle since they need to be updated per hit The same goes for the G4Track
  // global time and position
  std::aligned_storage<sizeof(G4DynamicParticle), alignof(G4DynamicParticle)>::type dynParticleStorage[3];
  std::aligned_storage<sizeof(G4DynamicParticle), alignof(G4DynamicParticle)>::type secDynStorage[kMaxTotalSecondaries];
  std::aligned_storage<sizeof(G4Track), alignof(G4Track)>::type trackStorage[3];
  std::aligned_storage<sizeof(G4Track), alignof(G4Track)>::type secTrackStorage[kMaxTotalSecondaries];

  G4Track *fElectronTrack = nullptr;
  G4Track *fPositronTrack = nullptr;
  G4Track *fGammaTrack    = nullptr;

  G4DynamicParticle *secDyn[kMaxTotalSecondaries] = {nullptr};
  G4Track *secTrk[kMaxTotalSecondaries]           = {nullptr};

  std::aligned_storage<sizeof(G4TrackVector), alignof(G4TrackVector)>::type secondaryStorage;
  G4TrackVector *fSecondaryVector = nullptr;

  // Indices for each type inside the flat pool
  int secEBase = 0;            // size 1
  int secPBase = secEBase + 1; // size 1
  int secGBase = secPBase + 1; // size 3

  int secEUsed = 0;
  int secPUsed = 0;
  int secGUsed = 0;

  void ResetSecondaryTracks()
  {
    secEUsed = 0;
    secPUsed = 0;
    secGUsed = 0;
  }

  ScoringObjects()
  {

    // Cache particle definitions once
    auto *particleTable = G4ParticleTable::GetParticleTable();
    auto *electronDef   = particleTable->FindParticle("e-");
    auto *positronDef   = particleTable->FindParticle("e+");
    auto *gammaDef      = particleTable->FindParticle("gamma");

    // Assign step points in local storage and take ownership of the StepPoints
    fG4Step = new (&stepStorage) G4Step;
    fG4Step->SetPreStepPoint(::new (&stepPointStorage[0]) G4StepPoint());
    fG4Step->SetPostStepPoint(::new (&stepPointStorage[1]) G4StepPoint());

    fSecondaryVector = ::new (&secondaryStorage) G4TrackVector;
    fSecondaryVector->reserve(kMaxSecondaries);
    fG4Step->SetSecondary(fSecondaryVector);

    // Touchable handles
    fPreG4TouchableHistoryHandle =
        ::new (&toucheableHandleStorage[0]) G4TouchableHandle{::new (&toucheableHistoryStorage[0]) G4TouchableHistory};
    fPostG4TouchableHistoryHandle =
        ::new (&toucheableHandleStorage[1]) G4TouchableHandle{::new (&toucheableHistoryStorage[1]) G4TouchableHistory};

    // Tracks
    fElectronTrack = ::new (&trackStorage[0])
        G4Track{::new (&dynParticleStorage[0]) G4DynamicParticle{electronDef, G4ThreeVector(0, 0, 0), 0}, 0,
                G4ThreeVector(0, 0, 0)};
    fPositronTrack = ::new (&trackStorage[1])
        G4Track{::new (&dynParticleStorage[1]) G4DynamicParticle{positronDef, G4ThreeVector(0, 0, 0), 0}, 0,
                G4ThreeVector(0, 0, 0)};
    fGammaTrack = ::new (&trackStorage[2])
        G4Track{::new (&dynParticleStorage[2]) G4DynamicParticle{gammaDef, G4ThreeVector(0, 0, 0), 0}, 0,
                G4ThreeVector(0, 0, 0)};

    // Secondary electron track(s)
    for (int i = 0; i < kMaxSecElectrons; ++i) {
      const int slot = secEBase + i;
      secDyn[slot]   = ::new (&secDynStorage[slot]) G4DynamicParticle{electronDef, G4ThreeVector(0, 0, 0), 0.0};
      secTrk[slot]   = ::new (&secTrackStorage[slot]) G4Track{secDyn[slot], 0.0, G4ThreeVector(0, 0, 0)};
    }

    // Secondary positron track(s)
    for (int i = 0; i < kMaxSecPositrons; ++i) {
      const int slot = secPBase + i;
      secDyn[slot]   = ::new (&secDynStorage[slot]) G4DynamicParticle{positronDef, G4ThreeVector(0, 0, 0), 0.0};
      secTrk[slot]   = ::new (&secTrackStorage[slot]) G4Track{secDyn[slot], 0.0, G4ThreeVector(0, 0, 0)};
    }

    // Secondary gamma track(s)
    for (int i = 0; i < kMaxSecGammas; ++i) {
      const int slot = secGBase + i;
      secDyn[slot]   = ::new (&secDynStorage[slot]) G4DynamicParticle{gammaDef, G4ThreeVector(0, 0, 0), 0.0};
      secTrk[slot]   = ::new (&secTrackStorage[slot]) G4Track{secDyn[slot], 0.0, G4ThreeVector(0, 0, 0)};
    }
  }

  // Note: no destructor needed since we’re intentionally *not* calling dtors on the placement-new'ed objects
};

void Deleter::operator()(ScoringObjects *ptr)
{
  delete ptr;
}

} // namespace AdePTGeant4Integration_detail

AdePTGeant4Integration::~AdePTGeant4Integration() {}

G4Track *AdePTGeant4Integration::ConstructSecondaryTrackInPlace(GPUHit const *secHit) const
{
  auto &so = *fScoringObjects;

  const ParticleType ptype = secHit->fParticleType;

  if (ptype == ParticleType::Electron) {
    if (so.secEUsed >= so.kMaxSecElectrons) std::abort();
    return so.secTrk[so.secEBase + so.secEUsed++];
  }

  if (ptype == ParticleType::Positron) {
    if (so.secPUsed >= so.kMaxSecPositrons) std::abort();
    return so.secTrk[so.secPBase + so.secPUsed++];
  }

  if (ptype == ParticleType::Gamma) {
    if (so.secGUsed >= so.kMaxSecGammas) std::abort();
    return so.secTrk[so.secGBase + so.secGUsed++];
  }

  std::abort();
}

void AdePTGeant4Integration::ProcessGPUStep(std::span<const GPUHit> gpuSteps, bool const callUserSteppingAction,
                                            bool const callUserTrackingAction)
{
  if (!fScoringObjects) {
    fScoringObjects.reset(new AdePTGeant4Integration_detail::ScoringObjects());
  }

  // Reset secondary tracks, as their dynamic properties change from returned step to step
  fScoringObjects->ResetSecondaryTracks();
  // Clear the persistent secondary vector
  fScoringObjects->fSecondaryVector->clear();

  // first step in the span is the parent step
  const GPUHit &parentStep = gpuSteps[0];

  assert(gpuSteps.size() == 1 + parentStep.fNumSecondaries);

  // Reconstruct G4NavigationHistory and G4Step, and call the SD code for each hit
  vecgeom::NavigationState const &preNavState = parentStep.fPreStepPoint.fNavigationState;
  // Reconstruct Pre-Step point G4NavigationHistory
  FillG4NavigationHistory(preNavState, fScoringObjects->fPreG4NavigationHistory);
  (*fScoringObjects->fPreG4TouchableHistoryHandle)
      ->UpdateYourself(fScoringObjects->fPreG4NavigationHistory.GetTopVolume(),
                       &fScoringObjects->fPreG4NavigationHistory);
  // Reconstruct Post-Step point G4NavigationHistory
  vecgeom::NavigationState const &postNavState = parentStep.fPostStepPoint.fNavigationState;
  if (!postNavState.IsOutside()) {
    FillG4NavigationHistory(postNavState, fScoringObjects->fPostG4NavigationHistory);
    (*fScoringObjects->fPostG4TouchableHistoryHandle)
        ->UpdateYourself(fScoringObjects->fPostG4NavigationHistory.GetTopVolume(),
                         &fScoringObjects->fPostG4NavigationHistory);
  }

  // Get step status
  // NOTE: Currently, we only check for the geometrical boundary, other statuses are set as unknown for now
  G4StepStatus preStepStatus{G4StepStatus::fUndefined};
  G4StepStatus postStepStatus{G4StepStatus::fUndefined};
  if (preNavState.IsOnBoundary()) {
    preStepStatus = G4StepStatus::fGeomBoundary;
  }
  if (postNavState.IsOnBoundary()) {
    postStepStatus = G4StepStatus::fGeomBoundary;
  }
  // if the track has left the world, set to
  if (postNavState.IsOutside()) {
    postStepStatus = G4StepStatus::fWorldBoundary;
  }

  // Reconstruct G4Step
  switch (parentStep.fParticleType) {
  case ParticleType::Electron:
    fScoringObjects->fG4Step->SetTrack(fScoringObjects->fElectronTrack);
    break;
  case ParticleType::Positron:
    fScoringObjects->fG4Step->SetTrack(fScoringObjects->fPositronTrack);
    break;
  case ParticleType::Gamma:
    fScoringObjects->fG4Step->SetTrack(fScoringObjects->fGammaTrack);
    break;
  default:
    std::cerr << "Error: unknown particle type " << static_cast<int>(parentStep.fParticleType) << "\n";
    std::abort();
  }

  // For all steps, the HostTrackData Mapper must already exist:
  // - for particles created on CPU, it was created in the AdePTTrackingManager when offloading to GPU,
  // - for particles createdon GPU, it was created in InitSecondaryHostTrackDataFromParent below

  HostTrackData dummy; // default constructed dummy if no advanced information is available

  // if the userActions are used, advanced track information is available
  const bool actions = (callUserTrackingAction || callUserSteppingAction);

  // Bind a reference *without* touching the mapper unless actions==true
  HostTrackData &parentTData = actions ? fHostTrackDataMapper->get(parentStep.fTrackID) : dummy;

  // Fill the G4Step, fill the G4Track, and intertwine them
  FillG4Step(&parentStep, fScoringObjects->fG4Step, parentTData, *fScoringObjects->fPreG4TouchableHistoryHandle,
             *fScoringObjects->fPostG4TouchableHistoryHandle, preStepStatus, postStepStatus, callUserTrackingAction,
             callUserSteppingAction);
  FillG4Track(&parentStep, fScoringObjects->fG4Step->GetTrack(), parentTData,
              *fScoringObjects->fPreG4TouchableHistoryHandle, *fScoringObjects->fPostG4TouchableHistoryHandle);
  fScoringObjects->fG4Step->GetTrack()->SetStep(fScoringObjects->fG4Step);

  // Create and attach secondaries.
  // User tracking callbacks for the secondaries are delayed until after the
  // parent step callbacks to match Geant4 ordering
  {
    // Attention!!! The reference parentTData to the hostTrackDataMapper will be invalidated by inserting a new element
    // via create()! Therefore, the g4id that is needed as a parent ID for the secondaries must be saved before!
    const auto parentID = parentTData.g4id;

    // the steps after the first one in the span are the initializing steps for the secondaries
    std::span<const GPUHit> secondaries = gpuSteps.subspan(1);
    // Loop over secondaries, create and fill info, and attach to secondary vector of the GPUStep
    for (const GPUHit &secStep : secondaries) {

      // 1. Create HostTrackData
      HostTrackData &secTData = fHostTrackDataMapper->create(secStep.fTrackID);

      // 2. Initialize from parent
      InitSecondaryHostTrackDataFromParent(&secStep, secTData, parentID,
                                           *fScoringObjects->fPreG4TouchableHistoryHandle);

      // 3. Construct G4Track in place
      G4Track *secTrack = ConstructSecondaryTrackInPlace(&secStep);

      // 4. Fill data for secondary track
      FillG4Track(&secStep, secTrack, secTData, *fScoringObjects->fPreG4TouchableHistoryHandle,
                  *fScoringObjects->fPostG4TouchableHistoryHandle);

      // 5. Attach secondaries to G4Step: the fSecondaryVector is the persistent storage for the G4Step->SecondaryVector
      fScoringObjects->fSecondaryVector->push_back(secTrack);
    }
  }

  // Now, the G4Step is fully initialized and also contains the secondaries created in that step.

  // Call scoring if SD is defined and it is not the initializing step.
  // As in G4, this is called before the SteppingAction
  G4VSensitiveDetector *aSensitiveDetector =
      fScoringObjects->fPreG4NavigationHistory.GetVolume(fScoringObjects->fPreG4NavigationHistory.GetDepth())
          ->GetLogicalVolume()
          ->GetSensitiveDetector();

  if (aSensitiveDetector != nullptr && parentStep.fStepCounter != 0) {
    aSensitiveDetector->Hit(fScoringObjects->fG4Step);
  }

  if (callUserSteppingAction) {
    auto *evtMgr             = G4EventManager::GetEventManager();
    auto *userSteppingAction = evtMgr->GetUserSteppingAction();
    if (userSteppingAction) userSteppingAction->UserSteppingAction(fScoringObjects->fG4Step);
  }

  // call UserTrackingAction if required
  if (parentStep.fLastStepOfTrack && (callUserTrackingAction)) {
    auto *evtMgr             = G4EventManager::GetEventManager();
    auto *userTrackingAction = evtMgr->GetUserTrackingAction();
    if (userTrackingAction) userTrackingAction->PostUserTrackingAction(fScoringObjects->fG4Step->GetTrack());
  }

  // Secondaries only get the PreUserTracking callback after the parent step has been
  // fully processed. Now the secondaries are ready to process their next step that could arrive
  if (callUserTrackingAction || callUserSteppingAction) {
    auto *evtMgr             = G4EventManager::GetEventManager();
    auto *userTrackingAction = evtMgr->GetUserTrackingAction();
    if (userTrackingAction) {
      std::span<const GPUHit> secondaries = gpuSteps.subspan(1);
      for (size_t i = 0; i < secondaries.size(); ++i) {
        auto *secondary = (*fScoringObjects->fSecondaryVector)[i];
        if (secondary == nullptr) continue;

        userTrackingAction->PreUserTrackingAction(secondary);

        auto &secTData = fHostTrackDataMapper->get(secondaries[i].fTrackID);
        if (secTData.userTrackInfo == nullptr && secondary->GetUserInformation() != nullptr) {
          secTData.userTrackInfo = secondary->GetUserInformation();
        }
      }
    }
  }

  // If this was the last step of a track, the hostTrackData of that track can be safely deleted.
  // Note: This deletes the AdePT-owned UserTrackInfo data
  // Exception: gamma-nuclear steps (fParticleType == 2 and fStepLimProcessId == 3):
  // Steps are processed before the leaked tracks, and the hostTrackData is still used for invoking
  // gamma-nuclear when the track is returned to the host (although it will die then). Thus, the hostTrackData must be
  // kept alive until the invokation of the gamma-nuclear reaction
  if (parentStep.fLastStepOfTrack &&
      !(parentStep.fParticleType == ParticleType::Gamma && parentStep.fStepLimProcessId == 3)) {
    fHostTrackDataMapper->removeTrack(parentStep.fTrackID);
  }
}

void AdePTGeant4Integration::FillG4NavigationHistory(const vecgeom::NavigationState &aNavState,
                                                     G4NavigationHistory &aG4NavigationHistory) const
{
  // Get the current depth of the history (corresponding to the previous reconstructed touchable)
  auto aG4HistoryDepth = aG4NavigationHistory.GetDepth();
  // Get the depth of the navigation state we want to reconstruct
  auto aVecGeomLevel = aNavState.GetLevel();

  unsigned int aLevel{0};
  G4VPhysicalVolume const *pvol{nullptr};
  G4VPhysicalVolume *pnewvol{nullptr};

  for (aLevel = 0; aLevel <= aVecGeomLevel; aLevel++) {
    // While we are in levels shallower than the history depth, it may be that we already
    // have the correct volume in the history
    assert(aNavState.At(aLevel));
    pnewvol = const_cast<G4VPhysicalVolume *>(AdePTGeometryBridge::GetG4PhysicalVolume(aNavState.At(aLevel)));
    assert(pnewvol != nullptr);
    if (!pnewvol) throw std::runtime_error("VecGeom volume not found in G4 mapping!");

    if (aG4HistoryDepth && (aLevel <= aG4HistoryDepth)) {
      pvol = aG4NavigationHistory.GetVolume(aLevel);
      // If they match we do not need to update the history at this level
      if (pvol == pnewvol) continue;
      // Once we find two non-matching volumes, we need to update the touchable history from this level on
      if (aLevel) {
        // If we are not in the top level
        aG4NavigationHistory.BackLevel(aG4HistoryDepth - aLevel + 1);
        // Update the current level
        aG4NavigationHistory.NewLevel(pnewvol, kNormal, pnewvol->GetCopyNo());
      } else {
        // Update the top level
        aG4NavigationHistory.BackLevel(aG4HistoryDepth);
        aG4NavigationHistory.SetFirstEntry(pnewvol);
      }
      // Now we are overwriting the history, so set the depth to the current depth
      aG4HistoryDepth = aLevel;
    } else {
      // If the navigation state is deeper than the current history we need to add the new levels
      if (aLevel) {
        aG4NavigationHistory.NewLevel(pnewvol, kNormal, pnewvol->GetCopyNo());
        aG4HistoryDepth++;
      } else {
        aG4NavigationHistory.SetFirstEntry(pnewvol);
      }
    }
  }
  // Once finished, remove the extra levels if the current state is shallower than the previous history
  if (aG4HistoryDepth >= aLevel) aG4NavigationHistory.BackLevel(aG4HistoryDepth - aLevel + 1);
}

G4TouchableHandle AdePTGeant4Integration::MakeTouchableFromNavState(vecgeom::NavigationState const &navState) const
{
  // Reconstruct the origin touchable history from a VecGeom NavigationState
  // - We can't update the track's navigation history in place as it is a const member
  // - For the same reason, we need to fill a navigation history and then create a touchable history from it
  auto navigationHistory = std::make_unique<G4NavigationHistory>();
  FillG4NavigationHistory(navState, *navigationHistory);

  // G4TouchableHistory constructor does a shallow copy of the navigation history
  // There is no way to transfer ownership of this pointer to the G4TouchableHistory, as the other available method,
  // UpdateYourself() does a shallow copy as well.
  // The only way to avoid a memory leak is to do the shallow copy and then allow our instance to be deleted, which will
  // call G4NavigationHistoryPool::DeRegister()
  auto touchableHistory = std::make_unique<G4TouchableHistory>(*navigationHistory);

  // Give ownership of the touchable history to a newly created touchable handle, which will now manage its lifetime
  return G4TouchableHandle(touchableHistory.release());
}

void AdePTGeant4Integration::InitSecondaryHostTrackDataFromParent(GPUHit const *secHit, HostTrackData &secTData,
                                                                  int g4ParentID, G4TouchableHandle &preTouchable) const
{
#ifdef DEBUG
  if (secHit->fStepCounter != 0) {
    std::cerr << "\033[1;31mERROR: secondary init called with stepCounter != 0\033[0m\n";
    std::abort();
  }
  if (secHit->fParentID == 0) {
    std::cerr << "\033[1;31mERROR: secondary init called with parentID == 0\033[0m\n";
    std::abort();
  }
  if (secHit->fStepCounter == 0 && fHostTrackDataMapper->contains(secHit->fTrackID)) {
    std::cerr << "\033[1;31mERROR: TRACK ALREADY HAS AN ENTRY (trackID = " << secHit->fTrackID
              << ", parentID = " << secHit->fParentID << ") "
              << " stepLimProcessId " << secHit->fStepLimProcessId << " pdg charge "
              << static_cast<int>(secHit->fParticleType) << " stepCounter " << secHit->fStepCounter << "\033[0m"
              << std::endl;
    std::abort();
  }
#endif

  secTData.particleType = secHit->fParticleType;
  secTData.g4parentid   = g4ParentID;

  secTData.logicalVolumeAtVertex = preTouchable->GetVolume()->GetLogicalVolume();
  secTData.vertexPosition = G4ThreeVector(secHit->fPostStepPoint.fPosition.x(), secHit->fPostStepPoint.fPosition.y(),
                                          secHit->fPostStepPoint.fPosition.z());
  secTData.vertexMomentumDirection =
      G4ThreeVector(secHit->fPostStepPoint.fMomentumDirection.x(), secHit->fPostStepPoint.fMomentumDirection.y(),
                    secHit->fPostStepPoint.fMomentumDirection.z());
  secTData.vertexKineticEnergy = secHit->fPostStepPoint.fEKin;

  // For the initializing step, the step defining process ID is the creator process
  const int stepId = secHit->fStepLimProcessId;
  assert(stepId >= 0);
  const ParticleType ptype = secTData.particleType;
  if (ptype == ParticleType::Electron || ptype == ParticleType::Positron) {
    secTData.creatorProcess = fHepEmTrackingManager->GetElectronNoProcessVector()[stepId];
  } else if (ptype == ParticleType::Gamma) {
    secTData.creatorProcess = fHepEmTrackingManager->GetGammaNoProcessVector()[stepId];
  }
}

void AdePTGeant4Integration::FillG4Track(GPUHit const *aGPUHit, G4Track *aTrack, const HostTrackData &hostTData,
                                         G4TouchableHandle &aPreG4TouchableHandle,
                                         G4TouchableHandle &aPostG4TouchableHandle) const
{

  const G4ThreeVector aPostStepPointMomentumDirection(aGPUHit->fPostStepPoint.fMomentumDirection.x(),
                                                      aGPUHit->fPostStepPoint.fMomentumDirection.y(),
                                                      aGPUHit->fPostStepPoint.fMomentumDirection.z());
  const G4ThreeVector aPostStepPointPosition(aGPUHit->fPostStepPoint.fPosition.x(),
                                             aGPUHit->fPostStepPoint.fPosition.y(),
                                             aGPUHit->fPostStepPoint.fPosition.z());

  // must const-cast as GetDynamicParticle only returns const
  aTrack->SetCreatorProcess(hostTData.creatorProcess);
  auto *dyn = const_cast<G4DynamicParticle *>(aTrack->GetDynamicParticle());
  dyn->SetPrimaryParticle(hostTData.primary);

  aTrack->SetTrackID(hostTData.g4id);          // Real data
  aTrack->SetParentID(hostTData.g4parentid);   // ID of the initial particle that entered AdePT
  aTrack->SetPosition(aPostStepPointPosition); // Real data
  aTrack->SetGlobalTime(aGPUHit->fGlobalTime); // Real data
  aTrack->SetLocalTime(aGPUHit->fLocalTime);   // Real data
  // aTrack->SetProperTime(0);                                                                // Missing data
  if (const auto preVolume = aPreG4TouchableHandle->GetVolume();
      preVolume != nullptr) {                          // protect against nullptr if NavState is outside
    aTrack->SetTouchableHandle(aPreG4TouchableHandle); // Real data
  }
  if (const auto postVolume = aPostG4TouchableHandle->GetVolume();
      postVolume != nullptr) {                              // protect against nullptr if postNavState is outside
    aTrack->SetNextTouchableHandle(aPostG4TouchableHandle); // Real data
  }
  // aTrack->SetOriginTouchableHandle(nullptr);                                               // Missing data
  aTrack->SetKineticEnergy(aGPUHit->fPostStepPoint.fEKin);       // Real data
  aTrack->SetMomentumDirection(aPostStepPointMomentumDirection); // Real data
  // aTrack->SetVelocity(0);                                                                  // Missing data
  // aTrack->SetPolarization(); // Missing Data data
  // aTrack->SetTrackStatus(G4TrackStatus::fAlive);                                           // Missing data
  // aTrack->SetBelowThresholdFlag(false);                                                    // Missing data
  // aTrack->SetGoodForTrackingFlag(false);                                                   // Missing data
  aTrack->SetStepLength(aGPUHit->fStepLength);                           // Real data
  aTrack->SetVertexPosition(hostTData.vertexPosition);                   // Real data
  aTrack->SetVertexMomentumDirection(hostTData.vertexMomentumDirection); // Real data
  aTrack->SetVertexKineticEnergy(hostTData.vertexKineticEnergy);         // Real data
  aTrack->SetLogicalVolumeAtVertex(hostTData.logicalVolumeAtVertex);     // Real data
  // aTrack->SetCreatorModelID(0);                                                            // Missing data
  // aTrack->SetParentResonanceDef(nullptr);                                                  // Missing data
  // aTrack->SetParentResonanceID(0);                                                         // Missing data
  aTrack->SetWeight(aGPUHit->fTrackWeight);
  // if it exists, add UserTrackInfo
  aTrack->SetUserInformation(hostTData.userTrackInfo); // Real data
  // aTrack->SetAuxiliaryTrackInformation(0, nullptr);                                        // Missing data

  // adjust for gamma-nuclear steps:
  // As the steps are processed before the leaked tracks, the step has not undergone the gamma-nuclear reaction yet.
  // Therefore, the kinetic energy for the track and postStepPoint must be set by hand to 0
  if (aGPUHit->fLastStepOfTrack && aGPUHit->fParticleType == ParticleType::Gamma && aGPUHit->fStepLimProcessId == 3) {
    aTrack->SetKineticEnergy(0.);
    // aTrack->SetVelocity(0.);              // Not set in other cases, so also not set here
  }
}

void AdePTGeant4Integration::FillG4Step(GPUHit const *aGPUHit, G4Step *aG4Step, const HostTrackData &hostTData,
                                        G4TouchableHandle &aPreG4TouchableHandle,
                                        G4TouchableHandle &aPostG4TouchableHandle, G4StepStatus aPreStepStatus,
                                        G4StepStatus aPostStepStatus, bool callUserTrackingAction,
                                        bool callUserSteppingAction) const
{
  const G4ThreeVector aPostStepPointMomentumDirection(aGPUHit->fPostStepPoint.fMomentumDirection.x(),
                                                      aGPUHit->fPostStepPoint.fMomentumDirection.y(),
                                                      aGPUHit->fPostStepPoint.fMomentumDirection.z());
  const G4ThreeVector aPostStepPointPosition(aGPUHit->fPostStepPoint.fPosition.x(),
                                             aGPUHit->fPostStepPoint.fPosition.y(),
                                             aGPUHit->fPostStepPoint.fPosition.z());

  // G4Step
  aG4Step->SetStepLength(aGPUHit->fStepLength);                 // Real data
  aG4Step->SetTotalEnergyDeposit(aGPUHit->fTotalEnergyDeposit); // Real data
  // aG4Step->SetNonIonizingEnergyDeposit(0);                      // Missing data
  // aG4Step->SetControlFlag(G4SteppingControl::NormalCondition);  // Missing data
  if (aGPUHit->fStepCounter == 1) aG4Step->SetFirstStepFlag(); // Real data
  if (aGPUHit->fLastStepOfTrack) aG4Step->SetLastStepFlag();   // Real data
  // aG4Step->SetPointerToVectorOfAuxiliaryPoints(nullptr);        // Missing data

  // G4Track
  G4Track *aTrack = aG4Step->GetTrack();

  // set the step-defining process for non-initializing steps
  G4VProcess *stepDefiningProcess = nullptr;
  if (aGPUHit->fStepCounter != 0) {
    // not an initial step, therefore setting the step defining process:
    const int stepId         = aGPUHit->fStepLimProcessId;
    const ParticleType ptype = hostTData.particleType;

    if (ptype == ParticleType::Electron || ptype == ParticleType::Positron) {
      if (stepId == 10)
        stepDefiningProcess = fHepEmTrackingManager->GetTransportNoProcess(); // set to transportation
      else if (stepId == -2)
        stepDefiningProcess = fHepEmTrackingManager->GetElectronNoProcessVector()[3]; // MSC
      else if (stepId == -1)
        stepDefiningProcess = fHepEmTrackingManager->GetElectronNoProcessVector()[0]; // dE/dx due to ionization
      else if (stepId == 3) {
        if (ptype == ParticleType::Electron)
          stepDefiningProcess = fHepEmTrackingManager->GetElectronNoProcessVector()[4]; // e- nuclear
        if (ptype == ParticleType::Positron)
          stepDefiningProcess = fHepEmTrackingManager->GetElectronNoProcessVector()[5]; // e+ nuclear
      } else {
        stepDefiningProcess = fHepEmTrackingManager->GetElectronNoProcessVector()[stepId]; // discrete interactions
      }
    } else if (ptype == ParticleType::Gamma) {
      stepDefiningProcess = (stepId == 10)
                                ? fHepEmTrackingManager->GetTransportNoProcess()            // transportation
                                : fHepEmTrackingManager->GetGammaNoProcessVector()[stepId]; // discrete interactions
    }
  }

  // Pre-Step Point
  G4StepPoint *aPreStepPoint = aG4Step->GetPreStepPoint();
  aPreStepPoint->SetPosition(G4ThreeVector(aGPUHit->fPreStepPoint.fPosition.x(), aGPUHit->fPreStepPoint.fPosition.y(),
                                           aGPUHit->fPreStepPoint.fPosition.z())); // Real data
  // aPreStepPoint->SetLocalTime(0);                                                                // Missing data
  aPreStepPoint->SetGlobalTime(aGPUHit->fPreGlobalTime); // Real data
  // aPreStepPoint->SetProperTime(0);                                                               // Missing data
  aPreStepPoint->SetMomentumDirection(G4ThreeVector(aGPUHit->fPreStepPoint.fMomentumDirection.x(),
                                                    aGPUHit->fPreStepPoint.fMomentumDirection.y(),
                                                    aGPUHit->fPreStepPoint.fMomentumDirection.z())); // Real data
  aPreStepPoint->SetKineticEnergy(aGPUHit->fPreStepPoint.fEKin);                                     // Real data
  // aPreStepPoint->SetVelocity(0);                                                                 // Missing data
  aPreStepPoint->SetTouchableHandle(aPreG4TouchableHandle);                                          // Real data
  aPreStepPoint->SetMaterial(aPreG4TouchableHandle->GetVolume()->GetLogicalVolume()->GetMaterial()); // Real data
  aPreStepPoint->SetMaterialCutsCouple(aPreG4TouchableHandle->GetVolume()->GetLogicalVolume()->GetMaterialCutsCouple());
  // aPreStepPoint->SetSensitiveDetector(nullptr);                                                  // Missing data
  // aPreStepPoint->SetSafety(0);                                                                   // Missing data
  // aPreStepPoint->SetPolarization(); // Missing data
  aPreStepPoint->SetStepStatus(aPreStepStatus); // Missing data
  // aPreStepPoint->SetProcessDefinedStep(nullptr);                                                 // Missing data
  // aPreStepPoint->SetMass(0);                                                                     // Missing data
  aPreStepPoint->SetCharge(aTrack->GetParticleDefinition()->GetPDGCharge()); // Real data
  // aPreStepPoint->SetMagneticMoment(0);                                                           // Missing data
  // aPreStepPoint->SetWeight(0);                                                                   // Missing data

  // Post-Step Point
  G4StepPoint *aPostStepPoint = aG4Step->GetPostStepPoint();
  aPostStepPoint->SetPosition(aPostStepPointPosition); // Real data
  aPostStepPoint->SetLocalTime(aGPUHit->fLocalTime);   // Real data
  aPostStepPoint->SetGlobalTime(aGPUHit->fGlobalTime); // Real data
  // aPostStepPoint->SetProperTime(0);                                                                // Missing data
  aPostStepPoint->SetMomentumDirection(aPostStepPointMomentumDirection); // Real data
  aPostStepPoint->SetKineticEnergy(aGPUHit->fPostStepPoint.fEKin);       // Real data
  // aPostStepPoint->SetVelocity(0);                                                                  // Missing data
  if (const auto postVolume = aPostG4TouchableHandle->GetVolume();
      postVolume != nullptr) {                                  // protect against nullptr if postNavState is outside
    aPostStepPoint->SetTouchableHandle(aPostG4TouchableHandle); // Real data
    aPostStepPoint->SetMaterial(postVolume->GetLogicalVolume()->GetMaterial()); // Real data
    aPostStepPoint->SetMaterialCutsCouple(postVolume->GetLogicalVolume()->GetMaterialCutsCouple());
  }
  // aPostStepPoint->SetSensitiveDetector(nullptr);                                                   // Missing data
  // aPostStepPoint->SetSafety(0);                                                                    // Missing data
  // aPostStepPoint->SetPolarization(); // Missing data
  aPostStepPoint->SetStepStatus(aPostStepStatus);             // Missing data
  aPostStepPoint->SetProcessDefinedStep(stepDefiningProcess); // Real data
  // aPostStepPoint->SetMass(0);                                                                      // Missing data
  aPostStepPoint->SetCharge(aTrack->GetParticleDefinition()->GetPDGCharge()); // Real data
  // aPostStepPoint->SetMagneticMoment(0);                                                            // Missing data
  // aPostStepPoint->SetWeight(0);                                                                    // Missing data

  // adjust for gamma-nuclear steps:
  // As the steps are processed before the leaked tracks, the step has not undergone the gamma-nuclear reaction yet.
  // Therefore, the kinetic energy for the track and postStepPoint must be set by hand to 0
  if (aGPUHit->fLastStepOfTrack && aGPUHit->fParticleType == ParticleType::Gamma && aGPUHit->fStepLimProcessId == 3) {
    aPostStepPoint->SetKineticEnergy(0.);
    // aPostStepPoint->SetVelocity(0.);      // Not set in other cases, so also not set here
  }
}

void AdePTGeant4Integration::ReturnTrack(adeptint::TrackData const &track, unsigned int trackIndex, int debugLevel,
                                         bool callUserActions) const
{
  constexpr double tolerance = 10. * vecgeom::kTolerance;

  // Build the secondaries and put them back on the Geant4 stack
  if (debugLevel > 6) {
    std::cout << "[" << GetThreadID() << "] fromDevice[ " << trackIndex << "]: pdg " << track.pdg << " parent id "
              << track.parentId << " kinetic energy " << track.eKin << " position " << track.position[0] << " "
              << track.position[1] << " " << track.position[2] << " direction " << track.direction[0] << " "
              << track.direction[1] << " " << track.direction[2] << " global time, local time, proper time: "
              << "(" << track.globalTime << ", " << track.localTime << ", " << track.properTime << ")" << " LeakStatus "
              << static_cast<int>(track.leakStatus) << std::endl;
  }
  G4ParticleMomentum direction(track.direction[0], track.direction[1], track.direction[2]);

  G4DynamicParticle *dynamic =
      new G4DynamicParticle(G4ParticleTable::GetParticleTable()->FindParticle(track.pdg), direction, track.eKin);

  G4ThreeVector posi(track.position[0], track.position[1], track.position[2]);
  // The returned track will be located by Geant4. For now we need to
  // push it to make sure it is not relocated again in the GPU region
  posi += tolerance * direction;

  if (track.stepCounter == 0) {
    std::cerr << "\033[1;31mERROR: Leaked track with stepCounter == 0 detected, this should never be the case! "
              << " (trackID = " << track.trackId << ", parentID = " << track.parentId << ") "
              << " pdg " << track.pdg << " stepCounter " << track.stepCounter << "\033[0m" << std::endl;
  }

  HostTrackData dummy; // default constructed dummy if no advanced information is available

  // Bind a reference *without* touching the mapper unless callUserActions==true
  // When the userActions are enabled, the entry must exist
  HostTrackData &hostTData = callUserActions ? fHostTrackDataMapper->get(track.trackId) : dummy;

  dynamic->SetPrimaryParticle(hostTData.primary);

  // Create track
  G4Track *leakedTrack = new G4Track(dynamic, track.globalTime, posi);

  // G4 does not allow to set the current step number directly, only to increment it.
  // For now, it is sufficient to increment just once, to distinguish from the 0th step
  leakedTrack->IncrementCurrentStepNumber();

  leakedTrack->SetTrackID(hostTData.g4id);
  leakedTrack->SetParentID(hostTData.g4parentid);

  leakedTrack->SetUserInformation(hostTData.userTrackInfo);
  leakedTrack->SetCreatorProcess(hostTData.creatorProcess);

  // Set time information
  leakedTrack->SetLocalTime(track.localTime);
  leakedTrack->SetProperTime(track.properTime);

  // Set weight
  leakedTrack->SetWeight(track.weight);

  // Set vertex information
  leakedTrack->SetVertexPosition(hostTData.vertexPosition);
  leakedTrack->SetVertexMomentumDirection(hostTData.vertexMomentumDirection);
  leakedTrack->SetVertexKineticEnergy(hostTData.vertexKineticEnergy);
  leakedTrack->SetLogicalVolumeAtVertex(hostTData.logicalVolumeAtVertex);

  // ------ Handle leaked tracks according to their status, if not LeakStatus::OutOfGPURegion ---------

  // sanity check
  if (track.leakStatus == LeakStatus::NoLeak) throw std::runtime_error("Leaked track with status NoLeak detected!");

  // We set the status of leaked tracks to fStopButAlive to be able to distinguish them to keep
  // them on the CPU until they are done
  if (track.leakStatus == LeakStatus::FinishEventOnCPU) {
    // FIXME: previous approach was broken in sync AdePT, therefore it was removed
    // to be fixed with a different approach e.g., negative track ID.
    leakedTrack->SetTrackStatus(fStopButAlive);
  }

  // Set the Touchable and NextTouchable handle. This is always needed, as either
  // gamma-/lepton-nuclear need it, or potentially any user stacking action, as this is called
  // before the track is handed back to AdePT
  auto TouchableHandle = MakeTouchableFromNavState(track.navState);
  leakedTrack->SetTouchableHandle(TouchableHandle);
  leakedTrack->SetNextTouchableHandle(TouchableHandle);

  // handle gamma- and lepton-nuclear directly in G4HepEm
  if (track.leakStatus == LeakStatus::GammaNuclear || track.leakStatus == LeakStatus::LeptonNuclear) {

    // create a new step
    G4Step *step = new G4Step();
    step->NewSecondaryVector();

    // initialize preStepPoint values
    step->InitializeStep(leakedTrack);

    // entangle our newly created step and the leaked track
    step->SetTrack(leakedTrack);
    leakedTrack->SetStep(step);

    // get fresh secondary vector
    G4TrackVector *secondariesPtr = step->GetfSecondary();
    if (!secondariesPtr) throw std::runtime_error("Failed to allocate secondary vector");
    G4TrackVector &secondaries = *secondariesPtr;

    if (fHepEmTrackingManager) {

      if (track.leakStatus == LeakStatus::GammaNuclear) {

        auto GNucProcess = fHepEmTrackingManager->GetGammaNuclearProcess();
        if (GNucProcess != nullptr) {

          // need to call StartTracking to set the particle type in the hadronic process (see G4HadronicProcess.cc)
          GNucProcess->StartTracking(leakedTrack);

          // perform gamma nuclear from G4HepEmTrackingManager
          // ApplyCuts must be false, as we are not in a tracking loop here and could not deposit the energy
          fHepEmTrackingManager->PerformNuclear(leakedTrack, step, /*particleID=*/2, /*isApplyCuts=*/false);

          // Give secondaries to G4
          G4EventManager::GetEventManager()->StackTracks(&secondaries);

          // As Gamma-nuclear kills the track, and the track is owned by AdePT, it has to be deleted. This includes:
          // 1. deleting the HostTrackData, which also deletes the UserTrackInfo
          // 2. Setting the UserInformation pointer in the track to nullptr (as the data was just deleted)
          // 3. deleting the track
          // Note that it is safe to remove the track and the hostTrackData here, as the hits are always handled
          // before the leaks and hits with Gamma-nuclear are not marked as last step.
          fHostTrackDataMapper->removeTrack(track.trackId);
          // as the UserTrackInfo was just deleted by removeTrack, the pointer must be set to null to avoid double
          // deletion
          leakedTrack->SetUserInformation(nullptr);
          // Now the track and step can be safely deleted
          delete leakedTrack;
          delete step;
        } else {
          // no gamma nuclear process attached, just give back the track to G4 to put it back on GPU
          G4EventManager::GetEventManager()->GetStackManager()->PushOneTrack(leakedTrack);
          delete step;
          // Track is how handled by CPU
          fHostTrackDataMapper->retireToCPU(track.trackId);
        }
      } else {
        // case LeakStatus::LeptonNuclear
        const double charge   = leakedTrack->GetParticleDefinition()->GetPDGCharge();
        const bool isElectron = (charge < 0.0);
        int particleID        = isElectron ? 0 : 1;

        // Invoke the electron/positron-nuclear process start tracking interace (if any)
        G4VProcess *theNucProcess = isElectron ? fHepEmTrackingManager->GetElectronNuclearProcess()
                                               : fHepEmTrackingManager->GetPositronNuclearProcess();
        if (theNucProcess != nullptr) {

          // perform lepton nuclear from G4HepEmTrackingManager
          // ApplyCuts must be false, as we are not in a tracking loop here and could not deposit the energy
          fHepEmTrackingManager->PerformNuclear(leakedTrack, step, particleID, /*isApplyCuts=*/false);

          // Give secondaries to G4 - they are already stored from G4HepEm in the G4Step, now we need to pass them to G4
          // itself
          G4EventManager::GetEventManager()->StackTracks(&secondaries);
          // Give updated primary after lepton nuclear reacton to G4
          G4EventManager::GetEventManager()->GetStackManager()->PushOneTrack(leakedTrack);
        } else {
          // no lepton nuclear process attached, just give back the track to G4 to put it back on GPU
          G4EventManager::GetEventManager()->GetStackManager()->PushOneTrack(leakedTrack);
          delete step;
        }
        // Track is how handled by CPU
        fHostTrackDataMapper->retireToCPU(track.trackId);
      }
    } else {
      throw std::runtime_error("Specialized HepEmTrackingManager not longer valid in integration!");
    }

  } else {

    // LeakStatus::OutOfGPURegion: just give track back to G4
    G4EventManager::GetEventManager()->GetStackManager()->PushOneTrack(leakedTrack);

    // The track is now handled on CPU. To reduce the map lookup time, the hostTrackData can be safely deleted because
    // the leaks are guaranteed to be handled after the processing of steps. Only the g4idToGPUid mapping cannot be
    // deleted as the GPU id needs to be the same for the reproducibility if the track returns to the GPU, as the
    // trackID is used for seeding the rng
    fHostTrackDataMapper->retireToCPU(track.trackId);
  }
}

std::vector<float> AdePTGeant4Integration::GetUniformField() const
{
  std::vector<float> Bfield({0., 0., 0.});

  G4MagneticField *field =
      (G4MagneticField *)G4TransportationManager::GetTransportationManager()->GetFieldManager()->GetDetectorField();

  if (field) {
    G4double origin[3] = {0., 0., 0.};
    G4double temp[3]   = {0., 0., 0.};
    field->GetFieldValue(origin, temp);
    Bfield = {static_cast<float>(temp[0]), static_cast<float>(temp[1]), static_cast<float>(temp[2])};
  }

  return Bfield;
}
