// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/integration/AdeptIntegration.h>

#include <VecGeom/management/BVHManager.h>
#include "VecGeom/management/GeoManager.h"
#include <VecGeom/gdml/Frontend.h>

#include "CopCore/SystemOfUnits.h"

#include <G4RunManager.hh>
#include <G4Threading.hh>
#include <G4Proton.hh>
#include <G4Region.hh>
#include <G4SDManager.hh>
#include <G4MaterialCutsCouple.hh>
#include <G4ProductionCutsTable.hh>
#include <G4TransportationManager.hh>
#include <G4StepPoint.hh>

#include <G4HepEmData.hh>
#include <G4HepEmState.hh>
#include <G4HepEmStateInit.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmMatCutData.hh>

#include "SensitiveDetector.hh"
#include "EventAction.hh"

#include <AdePT/base/TestManager.h>
#include <AdePT/base/TestManagerStore.h>

// Static members definition
std::unordered_map<size_t, size_t> AdeptIntegration::fglobal_volume_to_hit_map;
std::unordered_map<size_t, const G4VPhysicalVolume *> AdeptIntegration::fglobal_vecgeom_to_g4_map;
int AdeptIntegration::fGlobalNumSensitive = 0;

AdeptIntegration::~AdeptIntegration()
{
  delete fScoring;
}

void AdeptIntegration::CreateVecGeomWorld()
{
  // Import the gdml file into VecGeom
  vecgeom::GeoManager::Instance().SetTransformationCacheDepth(0);
  vgdml::Parser vgdmlParser;
  //auto middleWare = vgdmlParser.Load(fGDML_file.c_str(), false, copcore::units::mm);
  auto middleWare = vgdmlParser.Load("cms2018_sd.gdml", false, copcore::units::mm);
  if (middleWare == nullptr) {
    std::cerr << "Failed to read geometry from GDML file '" << "cms2018_sd.gdml" << "'" << G4endl;
    return;
  }

  const vecgeom::VPlacedVolume *world = vecgeom::GeoManager::Instance().GetWorld();
  if (world == nullptr) {
    std::cerr << "GeoManager world volume is nullptr" << G4endl;
    return;
  }
}

void AdeptIntegration::AddTrack(int pdg, double energy, double x, double y, double z, double dirx, double diry,
                                double dirz)
{
  fBuffer.toDevice.emplace_back(pdg, energy, x, y, z, dirx, diry, dirz);
  if (pdg == 11)
    fBuffer.nelectrons++;
  else if (pdg == -11)
    fBuffer.npositrons++;
  else if (pdg == 22)
    fBuffer.ngammas++;

  if (fBuffer.toDevice.size() >= fBufferThreshold) {
    if (fDebugLevel > 0)
      G4cout << "Reached the threshold of " << fBufferThreshold << " triggering the shower" << G4endl;
    this->Shower(G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID());
  }
}

void AdeptIntegration::Initialize(bool common_data)
{
  if (fInit) return;
  if (fMaxBatch <= 0) throw std::runtime_error("AdeptIntegration::Initialize: Maximum batch size not set.");

  fNumVolumes = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();

  if (fNumVolumes == 0) throw std::runtime_error("AdeptIntegration::Initialize: Number of geometry volumes is zero.");

  if (common_data) {
    G4cout << "=== AdeptIntegration: initializing geometry and physics\n";
    // Initialize geometry on device
    if (!vecgeom::GeoManager::Instance().IsClosed())
      throw std::runtime_error("AdeptIntegration::Initialize: VecGeom geometry not closed.");

    const vecgeom::cxx::VPlacedVolume *world = vecgeom::GeoManager::Instance().GetWorld();
    if (!InitializeGeometry(world))
      throw std::runtime_error("AdeptIntegration::Initialize: Cannot initialize geometry on GPU");

    // Initialize G4HepEm
    if (!InitializePhysics()) throw std::runtime_error("AdeptIntegration::Initialize cannot initialize physics on GPU");

    // Do the material-cut couple index mapping once
    // as well as set flags for sensitive volumes and region
    // Also set the mappings from sensitive volumes to hits and VecGeom to G4 indices
    int *sensitive_volumes = nullptr;

    VolAuxData *auxData = CreateVolAuxData(
        G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume(),
        vecgeom::GeoManager::Instance().GetWorld(), *fg4hepem_state);

    // Initialize volume auxiliary data on device
    VolAuxArray::GetInstance().fNumVolumes = fNumVolumes;
    VolAuxArray::GetInstance().fAuxData    = auxData;
    VolAuxArray::GetInstance().InitializeOnGPU();

    // Print some settings
    G4cout << "=== AdeptIntegration: buffering " << fBufferThreshold << " particles for transport on the GPU" << G4endl;
    G4cout << "=== AdeptIntegration: maximum number of GPU track slots per thread: " << kCapacity << G4endl;
    return;
  }

  G4cout << "=== AdeptIntegration: initializing transport engine for thread: " << G4Threading::G4GetThreadId()
         << G4endl;

  // Initialize user scoring data
  // fScoring     = new AdeptScoring(fGlobalNumSensitive, &fglobal_volume_to_hit_map,
  // vecgeom::GeoManager::Instance().GetPlacedVolumesCount());
  fScoring     = new AdeptScoring(kHitBufferCapacity);
  fScoring_dev = fScoring->InitializeOnGPU();

  // Initialize the transport engine for the current thread
  InitializeGPU();

  fInit = true;
}
void AdeptIntegration::InitBVH()
{
  vecgeom::cxx::BVHManager::Init();
  vecgeom::cxx::BVHManager::DeviceInit();
}

void AdeptIntegration::Cleanup()
{
  if (!fInit) return;
  AdeptIntegration::FreeGPU();
  fScoring->FreeGPU(fScoring_dev);
  delete[] fBuffer.fromDeviceBuff;
}

void AdeptIntegration::Shower(int event)
{
  constexpr double tolerance = 10. * vecgeom::kTolerance;

  int tid = G4Threading::G4GetThreadId();
  if (fDebugLevel > 0 && fBuffer.toDevice.size() == 0) {
    G4cout << "[" << tid << "] AdeptIntegration::Shower: No more particles in buffer. Exiting.\n";
    return;
  }

  if (event != fBuffer.eventId) {
    fBuffer.eventId    = event;
    fBuffer.startTrack = 0;
  } else {
    fBuffer.startTrack += fBuffer.toDevice.size();
  }

  int itr   = 0;
  int nelec = 0, nposi = 0, ngamma = 0;
  if (fDebugLevel > 0) {
    G4cout << "[" << tid << "] toDevice: " << fBuffer.nelectrons << " elec, " << fBuffer.npositrons << " posi, "
           << fBuffer.ngammas << " gamma\n";
  }
  if (fDebugLevel > 1) {
    for (auto &track : fBuffer.toDevice) {
      G4cout << "[" << tid << "] toDevice[ " << itr++ << "]: pdg " << track.pdg << " energy " << track.energy
             << " position " << track.position[0] << " " << track.position[1] << " " << track.position[2]
             << " direction " << track.direction[0] << " " << track.direction[1] << " " << track.direction[2] << G4endl;
    }
  }

  AdeptIntegration::ShowerGPU(event, fBuffer);

  // if (fDebugLevel > 1) fScoring->Print();
  for (auto const &track : fBuffer.fromDevice) {
    if (track.pdg == 11)
      nelec++;
    else if (track.pdg == -11)
      nposi++;
    else if (track.pdg == 22)
      ngamma++;
  }
  if (fDebugLevel > 0) {
    G4cout << "[" << tid << "] fromDevice: " << nelec << " elec, " << nposi << " posi, " << ngamma << " gamma\n";
  }

  // Build the secondaries and put them back on the Geant4 stack
  int i = 0;
  for (auto const &track : fBuffer.fromDevice) {
    if (fDebugLevel > 1) {
      G4cout << "[" << tid << "] fromDevice[ " << i++ << "]: pdg " << track.pdg << " energy " << track.energy
             << " position " << track.position[0] << " " << track.position[1] << " " << track.position[2]
             << " direction " << track.direction[0] << " " << track.direction[1] << " " << track.direction[2] << G4endl;
    }
    G4ParticleMomentum direction(track.direction[0], track.direction[1], track.direction[2]);

    G4DynamicParticle *dynamique =
        new G4DynamicParticle(G4ParticleTable::GetParticleTable()->FindParticle(track.pdg), direction, track.energy);

    G4ThreeVector posi(track.position[0], track.position[1], track.position[2]);
    // The returned track will be located by Geant4. For now we need to
    // push it to make sure it is not relocated again in the GPU region
    posi += tolerance * direction;

    G4Track *secondary = new G4Track(dynamique, 0, posi);
    secondary->SetParentID(-99);

    G4EventManager::GetEventManager()->GetStackManager()->PushOneTrack(secondary);
  }

  // Count the number of particles produced
  EventAction *evAct      = dynamic_cast<EventAction *>(G4EventManager::GetEventManager()->GetUserEventAction());
  evAct->number_gammas    = evAct->number_gammas + fScoring->fGlobalCounters_host->numGammas;
  evAct->number_electrons = evAct->number_electrons + fScoring->fGlobalCounters_host->numElectrons;
  evAct->number_positrons = evAct->number_positrons + fScoring->fGlobalCounters_host->numPositrons;
  evAct->number_killed    = evAct->number_killed + fScoring->fGlobalCounters_host->numKilled;


  fBuffer.Clear();

  // fScoring->ClearGPU();
}

adeptint::VolAuxData *AdeptIntegration::CreateVolAuxData(const G4VPhysicalVolume *g4world,
                                                         const vecgeom::VPlacedVolume *world,
                                                         const G4HepEmState &hepEmState)
{
  const int *g4tohepmcindex = hepEmState.fData->fTheMatCutData->fG4MCIndexToHepEmMCIndex;

  // - FIND vecgeom::LogicalVolume corresponding to each and every G4LogicalVolume
  int nphysical      = 0;
  int nlogical_sens  = 0;
  int nphysical_sens = 0;
  int ninregion      = 0;

  int nvolumes        = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();
  VolAuxData *auxData = new VolAuxData[nvolumes];
  // Used to keep track of the current vecgeom history while visiting the tree
  std::vector<vecgeom::VPlacedVolume const *> aCurrentVecgeomHistory;
  // Used to keep track of the current geant4 history while visiting the tree
  std::vector<G4VPhysicalVolume const *> aCurrentGeant4History;
  // Used to optimize the process of marking sensitive VecGeom volumes
  std::vector<G4LogicalVolume *> aUnmarkedSensitiveLogicalVolumes(*fSensitiveLogicalVolumes);

  // recursive geometry visitor lambda matching one by one Geant4 and VecGeom logical volumes
  // (we need to make sure we set the right MCC index to the right volume)
  typedef std::function<void(G4VPhysicalVolume const *, vecgeom::VPlacedVolume const *)> func_t;
  func_t visitAndSetMCindex = [&](G4VPhysicalVolume const *g4_pvol, vecgeom::VPlacedVolume const *vg_pvol) {
    const auto g4_lvol = g4_pvol->GetLogicalVolume();
    const auto vg_lvol = vg_pvol->GetLogicalVolume();
    int nd             = g4_lvol->GetNoDaughters();
    auto daughters     = vg_lvol->GetDaughters();

    aCurrentVecgeomHistory.push_back(vg_pvol);
    aCurrentGeant4History.push_back(g4_pvol);

    if (nd != daughters.size()) throw std::runtime_error("Fatal: CreateVolAuxData: Mismatch in number of daughters");
    // Check if transformations are matching
    auto g4trans = g4_pvol->GetTranslation();
    auto g4rot   = g4_pvol->GetRotation();
    G4RotationMatrix idrot;
    auto vgtransformation  = vg_pvol->GetTransformation();
    constexpr double epsil = 1.e-8;
    for (int i = 0; i < 3; ++i) {
      if (std::abs(g4trans[i] - vgtransformation->Translation(i)) > epsil)
        throw std::runtime_error(
            std::string("Fatal: CreateVolAuxData: Mismatch between Geant4 translation for physical volume") +
            vg_pvol->GetName());
    }

    // check if VecGeom and Geant4 (local) transformations are matching. Not optimized, this will re-check
    // already checked placed volumes when re-visiting the same volumes in different branches
    if (!g4rot) g4rot = &idrot;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        int i = row + 3 * col;
        if (std::abs((*g4rot)(row, col) - vgtransformation->Rotation(i)) > epsil)
          throw std::runtime_error(
              std::string("Fatal: CreateVolAuxData: Mismatch between Geant4 rotation for physical volume") +
              vg_pvol->GetName());
      }
    }

    // Check the couples
    if (g4_lvol->GetMaterialCutsCouple() == nullptr)
      throw std::runtime_error("Fatal: CreateVolAuxData: G4LogicalVolume " + std::string(g4_lvol->GetName()) +
                               std::string(" has no material-cuts couple"));
    int g4mcindex    = g4_lvol->GetMaterialCutsCouple()->GetIndex();
    int hepemmcindex = g4tohepmcindex[g4mcindex];
    // Check consistency with G4HepEm data
    if (hepEmState.fData->fTheMatCutData->fMatCutData[hepemmcindex].fG4MatCutIndex != g4mcindex)
      throw std::runtime_error(
          "Fatal: CreateVolAuxData: Mismatch between Geant4 mcindex and corresponding G4HepEm index");
    if (vg_lvol->id() >= nvolumes)
      throw std::runtime_error("Fatal: CreateVolAuxData: Volume id larger than number of volumes");

    // All OK, now fill the MCC index in the array
    auxData[vg_lvol->id()].fMCIndex = hepemmcindex;
    nphysical++;

    // Check if the volume belongs to the interesting region
    // I am commenting out this 'if' because (for the moment) we don't want any particles leaking out from AdePT
    // if (g4_lvol->GetRegion() == fRegion) {
    auxData[vg_lvol->id()].fGPUregion = 1;
    ninregion++;
    //}

    // Check if the logical volume is sensitive
    bool sens = false;
    //for (G4LogicalVolume* sensvol : aUnmarkedSensitiveLogicalVolumes) {
    for (auto iter = aUnmarkedSensitiveLogicalVolumes.begin();
          iter != aUnmarkedSensitiveLogicalVolumes.end();
          iter++) {

      G4LogicalVolume* sensvol = *iter;
        
      if (vg_lvol->GetName() == sensvol->GetName() ||
          std::string(vg_lvol->GetName()).rfind(sensvol->GetName() + "0x", 0) == 0) {
        sens = true;

        // We know the VecGeom LV is sensitive because it matches a G4 LV in the list
        // Make sure the LV corresponding to the PV we are visiting is indeed sensitive
        if (g4_lvol->GetSensitiveDetector() == nullptr)
          throw std::runtime_error("Fatal: CreateVolAuxData: G4LogicalVolume " + std::string(g4_lvol->GetName()) +
                                   " not sensitive while VecGeom one " + std::string(vg_lvol->GetName()) + " is.");
        // Mark the LV as sensitive in its auxiliary data 
        if (auxData[vg_lvol->id()].fSensIndex < 0) 
        {
          nlogical_sens++;
          G4cout << "VecGeom: Making " << vg_lvol->GetName() << " sensitive" << G4endl;
        }
        auxData[vg_lvol->id()].fSensIndex = 1;

        aUnmarkedSensitiveLogicalVolumes.erase(iter);
        break;
      }
    }

    // If the volume is sensitive:
    if(auxData[vg_lvol->id()].fSensIndex == 1)
    {
      nphysical_sens++;

      // Initialize mapping of Vecgeom sensitive PlacedVolume IDs to G4 PhysicalVolume IDs
      // In order to be able to reconstruct navigation histories based on a VecGeom Navigation State Index,
      // we need to map not only the sensitive volume, but also the ones leading up to here
      for (uint i = 0; i < aCurrentVecgeomHistory.size() - 1; i++) {
        fglobal_vecgeom_to_g4_map.insert(
            std::pair<int, const G4VPhysicalVolume *>(aCurrentVecgeomHistory[i]->id(), aCurrentGeant4History[i]));
      }
      bool new_pvol =
          fglobal_vecgeom_to_g4_map.insert(std::pair<int, const G4VPhysicalVolume *>(vg_pvol->id(), g4_pvol)).second;
      if (new_pvol) {
        // If we found a new placed volume, map its ID to a hit
        // This is done because we score per placed volume at the moment, and not per touchable
        fglobal_volume_to_hit_map[vg_pvol->id()] = fGlobalNumSensitive;
        fGlobalNumSensitive++;
      }
    }

    // Make sure that if the G4 LV is sensitive, the VecGeom LV has been marked as well
    if (auxData[vg_lvol->id()].fSensIndex == 0 && g4_lvol->GetSensitiveDetector() != nullptr)
      throw std::runtime_error("Fatal: CreateVolAuxData: G4LogicalVolume " + std::string(g4_lvol->GetName()) +
                               " sensitive while VecGeom one " + std::string(vg_lvol->GetName()) + " isn't.");

    // Now do the daughters
    for (int id = 0; id < nd; ++id) {
      auto g4pvol_d = g4_lvol->GetDaughter(id);
      auto pvol_d   = daughters[id];

      // VecGeom does not strip pointers from logical volume names
      if (std::string(pvol_d->GetLogicalVolume()->GetName()).rfind(g4pvol_d->GetLogicalVolume()->GetName(), 0) != 0)
        throw std::runtime_error("Fatal: CreateVolAuxData: Volume names " +
                                 std::string(pvol_d->GetLogicalVolume()->GetName()) + " and " +
                                 std::string(g4pvol_d->GetLogicalVolume()->GetName()) + " mismatch");
      visitAndSetMCindex(g4pvol_d, pvol_d);
    }

    aCurrentVecgeomHistory.pop_back();
    aCurrentGeant4History.pop_back();
  };

  visitAndSetMCindex(g4world, world);

  G4cout << "Visited " << nphysical << " matching physical volumes\n";
  G4cout << "Number of logical sensitive:      " << nlogical_sens << "\n";
  G4cout << "Number of physical sensitive:     " << nphysical_sens << "\n";
  G4cout << "Number of physical in GPU region: " << ninregion << "\n";
  return auxData;
}

void AdeptIntegration::ProcessGPUHits(HostScoring::Stats &aStats)
{
  // For sequential processing of hits we only need one instance of each object
  G4NavigationHistory *fPreG4NavigationHistory = new G4NavigationHistory();
  G4NavigationHistory *fPostG4NavigationHistory = new G4NavigationHistory();
  G4Step *fG4Step                           = new G4Step();
  G4TouchableHandle fPreG4TouchableHistoryHandle = new G4TouchableHistory();
  G4TouchableHandle fPostG4TouchableHistoryHandle = new G4TouchableHistory();
  fG4Step->SetPreStepPoint(new G4StepPoint());
  fG4Step->SetPostStepPoint(new G4StepPoint());
  fG4Step->SetTrack(new G4Track());

  // Reconstruct G4NavigationHistory and G4Step, and call the SD code for each hit
  for (size_t i = aStats.fBufferStart; i < aStats.fBufferStart + aStats.fUsedSlots; i++) {
    // Get Hit index (Circular buffer)
    int aHitIdx = i % fScoring->fBufferCapacity;

    int aNavindex = fScoring->fGPUHitsBuffer_host[aHitIdx].fPreStepPoint.fNavigationStateIndex;
    // Reconstruct Pre-Step point G4NavigationHistory
    FillG4NavigationHistory(aNavindex, fPreG4NavigationHistory);
    ((G4TouchableHistory *)fPreG4TouchableHistoryHandle())
        ->UpdateYourself(fPreG4NavigationHistory->GetTopVolume(), fPreG4NavigationHistory);
    // Reconstruct Post-Step point G4NavigationHistory
    FillG4NavigationHistory(aNavindex, fPostG4NavigationHistory);
    ((G4TouchableHistory *)fPostG4TouchableHistoryHandle())
        ->UpdateYourself(fPostG4NavigationHistory->GetTopVolume(), fPostG4NavigationHistory);

    // Reconstruct G4Step
    FillG4Step(&(fScoring->fGPUHitsBuffer_host[aHitIdx]), fG4Step, fPreG4TouchableHistoryHandle, fPostG4TouchableHistoryHandle);

    // Call SD code
    SensitiveDetector *aSensitiveDetector =
        (SensitiveDetector *)fPreG4NavigationHistory->GetVolume(fPreG4NavigationHistory->GetDepth())
            ->GetLogicalVolume()
            ->GetSensitiveDetector();

    // Double check, a nullptr here can indicate an issue reconstructing the navigation history
    assert(aSensitiveDetector != nullptr);

    aSensitiveDetector->ProcessHits(fG4Step, (G4TouchableHistory *)fPreG4TouchableHistoryHandle());
  }

  delete fPreG4NavigationHistory;
  delete fPostG4NavigationHistory;
  delete fG4Step;
}

/// @brief Reconstruct G4TouchableHistory from a VecGeom Navigation index
void AdeptIntegration::FillG4NavigationHistory(unsigned int aNavIndex, G4NavigationHistory *aG4NavigationHistory)
{
  // Get the current depth of the history (corresponding to the previous reconstructed touchable)
  auto aG4HistoryDepth = aG4NavigationHistory->GetDepth();
  // Get the depth of the navigation state we want to reconstruct
  vecgeom::NavigationState vgState(aNavIndex);
  auto aVecGeomLevel = vgState.GetLevel();

  unsigned int aLevel{0};
  G4VPhysicalVolume *pvol{nullptr}, *pnewvol{nullptr};

  for (aLevel = 0; aLevel <= aVecGeomLevel; aLevel++) {
    // While we are in levels shallower than the history depth, it may be that we already
    // have the correct volume in the history
    pnewvol = const_cast<G4VPhysicalVolume *>(fglobal_vecgeom_to_g4_map[vgState.At(aLevel)->id()]);

    if (aLevel < aG4HistoryDepth) {
      // In G4NavigationHistory, the initial volume has the index 1
      pvol = aG4NavigationHistory->GetVolume(aLevel + 1);

      // If they match we do not need to update the history at this level
      if (pvol == pnewvol) continue;
      // Once we find two non-matching volumes, we need to update the touchable history from this level on
      if (aLevel) {
        // If we are not in the top level
        aG4NavigationHistory->BackLevel(aG4HistoryDepth - aLevel);
        // Update the current level
        aG4NavigationHistory->NewLevel(pnewvol, kNormal, pnewvol->GetCopyNo());
      } else {
        // Update the top level
        aG4NavigationHistory->BackLevel(aG4HistoryDepth);
        aG4NavigationHistory->SetFirstEntry(pnewvol);
      }
      // Now we are overwriting the history, so set the depth to the current depth
      aG4HistoryDepth = aLevel;
    } else {
      // If the navigation state is deeper than the current history we need to add the new levels
      aG4NavigationHistory->NewLevel(pnewvol, kNormal, pnewvol->GetCopyNo());
      aG4HistoryDepth++;
    }
  }
  // Once finished, remove the extra levels if the current state is shallower than the previous history
  if (aG4HistoryDepth >= aLevel) aG4NavigationHistory->BackLevel(aG4HistoryDepth - aLevel);
}

void AdeptIntegration::FillG4Step(GPUHit *aGPUHit, 
                                  G4Step *aG4Step, 
                                  G4TouchableHandle &aPreG4TouchableHandle,
                                  G4TouchableHandle &aPostG4TouchableHandle)
{
  // G4Step
  aG4Step->SetStepLength(aGPUHit->fStepLength);                    // Real data
  aG4Step->SetTotalEnergyDeposit(aGPUHit->fTotalEnergyDeposit);    // Real data
  // aG4Step->SetNonIonizingEnergyDeposit(0);                      // Missing data
  // aG4Step->SetControlFlag(G4SteppingControl::NormalCondition);  // Missing data
  // aG4Step->SetFirstStepFlag();                                  // Missing data
  // aG4Step->SetLastStepFlag();                                   // Missing data
  // aG4Step->SetPointerToVectorOfAuxiliaryPoints(nullptr);        // Missing data
  // aG4Step->SetSecondary(nullptr);                               // Missing data

  // G4Track
  G4Track *aTrack = aG4Step->GetTrack();
  // aTrack->SetTrackID(0);                                                                   // Missing data
  // aTrack->SetParentID(0);                                                                  // Missing data
  aTrack->SetPosition(G4ThreeVector(aGPUHit->fPostStepPoint.fPosition));                      // Real data
  // aTrack->SetGlobalTime(0);                                                                // Missing data
  // aTrack->SetLocalTime(0);                                                                 // Missing data
  // aTrack->SetProperTime(0);                                                                // Missing data
  // aTrack->SetTouchableHandle(aTrackTouchableHistory);                                      // Missing data
  // aTrack->SetNextTouchableHandle(nullptr);                                                 // Missing data
  // aTrack->SetOriginTouchableHandle(nullptr);                                               // Missing data
  // aTrack->SetKineticEnergy(aGPUHit->fPostStepPoint.fEKin);                                 // Real data
  aTrack->SetMomentumDirection(G4ThreeVector(aGPUHit->fPostStepPoint.fMomentumDirection));    // Real data
  // aTrack->SetVelocity(0);                                                                  // Missing data
  aTrack->SetPolarization(G4ThreeVector(aGPUHit->fPostStepPoint.fPolarization));              // Real data
  // aTrack->SetTrackStatus(G4TrackStatus::fAlive);                                           // Missing data
  // aTrack->SetBelowThresholdFlag(false);                                                    // Missing data
  // aTrack->SetGoodForTrackingFlag(false);                                                   // Missing data
  aTrack->SetStep(aG4Step);                                                                   // Real data
  aTrack->SetStepLength(aGPUHit->fStepLength);                                                // Real data
  // aTrack->SetVertexPosition(G4ThreeVector(0, 0, 0));                                       // Missing data
  // aTrack->SetVertexMomentumDirection(G4ThreeVector(0, 0, 0));                              // Missing data
  // aTrack->SetVertexKineticEnergy(0);                                                       // Missing data
  // aTrack->SetLogicalVolumeAtVertex(nullptr);                                               // Missing data
  // aTrack->SetCreatorProcess(nullptr);                                                      // Missing data
  // aTrack->SetCreatorModelID(0);                                                            // Missing data
  // aTrack->SetParentResonanceDef(nullptr);                                                  // Missing data
  // aTrack->SetParentResonanceID(0);                                                         // Missing data
  // aTrack->SetWeight(0);                                                                    // Missing data
  // aTrack->SetUserInformation(nullptr);                                                     // Missing data
  // aTrack->SetAuxiliaryTrackInformation(0, nullptr);                                        // Missing data

  // Pre-Step Point
  G4StepPoint *aPreStepPoint = aG4Step->GetPreStepPoint();
  aPreStepPoint->SetPosition(G4ThreeVector(aGPUHit->fPreStepPoint.fPosition));                      // Real data
  // aPreStepPoint->SetLocalTime(0);                                                                // Missing data
  // aPreStepPoint->SetGlobalTime(0);                                                               // Missing data
  // aPreStepPoint->SetProperTime(0);                                                               // Missing data
  aPreStepPoint->SetMomentumDirection(G4ThreeVector(aGPUHit->fPreStepPoint.fMomentumDirection));    // Real data
  aPreStepPoint->SetKineticEnergy(aGPUHit->fPreStepPoint.fEKin);                                    // Real data
  // aPreStepPoint->SetVelocity(0);                                                                 // Missing data
  // aPreStepPoint->SetMaterial(nullptr);                                                           // Missing data
  aPreStepPoint->SetTouchableHandle(aPreG4TouchableHandle);                                         // Real data
  // aPreStepPoint->SetMaterialCutsCouple(nullptr);                                                 // Missing data
  // aPreStepPoint->SetSensitiveDetector(nullptr);                                                  // Missing data
  // aPreStepPoint->SetSafety(0);                                                                   // Missing data
  aPreStepPoint->SetPolarization(G4ThreeVector(aGPUHit->fPreStepPoint.fPolarization));              // Real data
  // aPreStepPoint->SetStepStatus(G4StepStatus::fUndefined);                                        // Missing data
  // aPreStepPoint->SetProcessDefinedStep(nullptr);                                                 // Missing data
  // aPreStepPoint->SetMass(0);                                                                     // Missing data
  aPreStepPoint->SetCharge(aGPUHit->fPreStepPoint.fCharge);                                         // Real data
  // aPreStepPoint->SetMagneticMoment(0);                                                           // Missing data
  // aPreStepPoint->SetWeight(0);                                                                   // Missing data

  // Post-Step Point
  G4StepPoint *aPostStepPoint = aG4Step->GetPostStepPoint();
  aPostStepPoint->SetPosition(G4ThreeVector(aGPUHit->fPostStepPoint.fPosition));                      // Real data
  // aPostStepPoint->SetLocalTime(0);                                                                 // Missing data
  // aPostStepPoint->SetGlobalTime(0);                                                                // Missing data
  // aPostStepPoint->SetProperTime(0);                                                                // Missing data
  aPostStepPoint->SetMomentumDirection(G4ThreeVector(aGPUHit->fPostStepPoint.fMomentumDirection));    // Real data
  aPostStepPoint->SetKineticEnergy(aGPUHit->fPostStepPoint.fEKin);                                    // Real data
  // aPostStepPoint->SetVelocity(0);                                                                  // Missing data
  aPostStepPoint->SetTouchableHandle(aPostG4TouchableHandle);                                         // Real data
  // aPostStepPoint->SetMaterial(nullptr);                                                            // Missing data
  // aPostStepPoint->SetMaterialCutsCouple(nullptr);                                                  // Missing data
  // aPostStepPoint->SetSensitiveDetector(nullptr);                                                   // Missing data
  // aPostStepPoint->SetSafety(0);                                                                    // Missing data
  aPostStepPoint->SetPolarization(G4ThreeVector(aGPUHit->fPostStepPoint.fPolarization));              // Real data
  // aPostStepPoint->SetStepStatus(G4StepStatus::fUndefined);                                         // Missing data
  // aPostStepPoint->SetProcessDefinedStep(nullptr);                                                  // Missing data
  // aPostStepPoint->SetMass(0);                                                                      // Missing data
  aPostStepPoint->SetCharge(aGPUHit->fPostStepPoint.fCharge);                                         // Real data
  // aPostStepPoint->SetMagneticMoment(0);                                                            // Missing data
  // aPostStepPoint->SetWeight(0);                                                                    // Missing data
}