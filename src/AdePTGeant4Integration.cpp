// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/integration/AdePTGeant4Integration.hh>

#include <VecGeom/management/GeoManager.h>
#include <VecGeom/gdml/Frontend.h>
#include <VecGeom/navigation/NavigationState.h>

#include <G4ios.hh>
#include <G4SystemOfUnits.hh>
#include <G4GeometryManager.hh>
#include <G4TransportationManager.hh>
#include <G4MaterialCutsCouple.hh>
#include <G4VSensitiveDetector.hh>
#include <G4UniformMagField.hh>
#include <G4FieldManager.hh>
#include <G4RegionStore.hh>
#include <G4TouchableHandle.hh>
#include <G4PVReplica.hh>
#include <G4ReplicaNavigation.hh>
#include <G4StepStatus.hh>

#include <G4HepEmState.hh>
#include <G4HepEmData.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmNoProcess.hh>

#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4Gamma.hh"

#include <G4VG.hh>

#include <new>
#include <type_traits>

namespace AdePTGeant4Integration_detail {
/// This struct holds temporary scoring objects that are needed to send hits to Geant4.
/// These are allocated as members or using placement new to go around G4's pool allocators,
/// which cause a destruction order fiasco when the pools are destroyed before the objects,
/// and then the objects' destructors are called.
/// This also keeps all these objects much closer in memory.
struct ScoringObjects {
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
  // For this reason we create one track per particle type, to be reused
  // We set position to nullptr and kinetic energy to 0 for the dynamic particle since they need to be updated per hit
  // The same goes for the G4Track global time and position
  std::aligned_storage<sizeof(G4DynamicParticle), alignof(G4DynamicParticle)>::type dynParticleStorage[3];
  std::aligned_storage<sizeof(G4Track), alignof(G4Track)>::type trackStorage[3];

  G4Track *fElectronTrack = nullptr;
  G4Track *fPositronTrack = nullptr;
  G4Track *fGammaTrack    = nullptr;

  ScoringObjects()
  {
    // Assign step points in local storage and take ownership of the StepPoints
    fG4Step = new (&stepStorage) G4Step;
    fG4Step->SetPreStepPoint(::new (&stepPointStorage[0]) G4StepPoint());
    fG4Step->SetPostStepPoint(::new (&stepPointStorage[1]) G4StepPoint());

    // Touchable handles
    fPreG4TouchableHistoryHandle =
        ::new (&toucheableHandleStorage[0]) G4TouchableHandle{::new (&toucheableHistoryStorage[0]) G4TouchableHistory};
    fPostG4TouchableHistoryHandle =
        ::new (&toucheableHandleStorage[1]) G4TouchableHandle{::new (&toucheableHistoryStorage[1]) G4TouchableHistory};

    // Tracks
    fElectronTrack = ::new (&trackStorage[0]) G4Track{
        ::new (&dynParticleStorage[0])
            G4DynamicParticle{G4ParticleTable::GetParticleTable()->FindParticle("e-"), G4ThreeVector(0, 0, 0), 0},
        0, G4ThreeVector(0, 0, 0)};
    fPositronTrack = ::new (&trackStorage[1]) G4Track{
        ::new (&dynParticleStorage[1])
            G4DynamicParticle{G4ParticleTable::GetParticleTable()->FindParticle("e+"), G4ThreeVector(0, 0, 0), 0},
        0, G4ThreeVector(0, 0, 0)};
    fGammaTrack = ::new (&trackStorage[2]) G4Track{
        ::new (&dynParticleStorage[2])
            G4DynamicParticle{G4ParticleTable::GetParticleTable()->FindParticle("gamma"), G4ThreeVector(0, 0, 0), 0},
        0, G4ThreeVector(0, 0, 0)};
  }

  // Note: no destructor needed since we’re intentionally *not* calling dtors on the placement-new'ed objects
};

void Deleter::operator()(ScoringObjects *ptr)
{
  delete ptr;
}

} // namespace AdePTGeant4Integration_detail

std::vector<G4VPhysicalVolume const *> AdePTGeant4Integration::fglobal_vecgeom_pv_to_g4_map;
std::vector<G4LogicalVolume const *> AdePTGeant4Integration::fglobal_vecgeom_lv_to_g4_map;

AdePTGeant4Integration::~AdePTGeant4Integration() {}

void AdePTGeant4Integration::MapVecGeomToG4(std::vector<G4VPhysicalVolume const *> &vecgeomPvToG4Map,
                                            std::vector<G4LogicalVolume const *> &vecgeomLvToG4Map)
{
  const G4VPhysicalVolume *g4world =
      G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
  const vecgeom::VPlacedVolume *vecgeomWorld = vecgeom::GeoManager::Instance().GetWorld();

  // recursive geometry visitor lambda matching one by one Geant4 and VecGeom logical volumes
  typedef std::function<void(G4VPhysicalVolume const *, vecgeom::VPlacedVolume const *)> func_t;
  func_t visitGeometry = [&](G4VPhysicalVolume const *g4_pvol, vecgeom::VPlacedVolume const *vg_pvol) {
    const auto g4_lvol = g4_pvol->GetLogicalVolume();
    const auto vg_lvol = vg_pvol->GetLogicalVolume();

    // Initialize mapping of Vecgeom PlacedVolume IDs to G4 PhysicalVolume*
    vecgeomPvToG4Map.resize(std::max<std::size_t>(vecgeomPvToG4Map.size(), vg_pvol->id() + 1), nullptr);
    vecgeomPvToG4Map[vg_pvol->id()] = g4_pvol;
    // Initialize mapping of Vecgeom LogicalVolume IDs to G4 LogicalVolume*
    vecgeomLvToG4Map.resize(std::max<std::size_t>(vecgeomLvToG4Map.size(), vg_lvol->id() + 1), nullptr);
    vecgeomLvToG4Map[vg_lvol->id()] = g4_lvol;

    // Now do the daughters
    for (size_t id = 0; id < g4_lvol->GetNoDaughters(); ++id) {
      auto g4pvol_d = g4_lvol->GetDaughter(id);
      auto pvol_d   = vg_lvol->GetDaughters()[id];

      visitGeometry(g4pvol_d, pvol_d);
    }
  };
  visitGeometry(g4world, vecgeomWorld);
}

void AdePTGeant4Integration::CreateVecGeomWorld(std::string filename)
{
  // Import the gdml file into VecGeom
  vecgeom::GeoManager::Instance().SetTransformationCacheDepth(0);
  vgdml::Parser vgdmlParser;
  auto middleWare = vgdmlParser.Load(filename, false, mm);
  if (middleWare == nullptr) {
    std::cerr << "Failed to read geometry from GDML file '" << filename << "'" << G4endl;
    return;
  }

  // Generate the mapping of VecGeom volume IDs to Geant4 physical volumes
  MapVecGeomToG4(fglobal_vecgeom_pv_to_g4_map, fglobal_vecgeom_lv_to_g4_map);

  const vecgeom::VPlacedVolume *vecgeomWorld = vecgeom::GeoManager::Instance().GetWorld();
  if (vecgeomWorld == nullptr) {
    std::cerr << "GeoManager vecgeomWorld volume is nullptr" << G4endl;
    return;
  }
}

void AdePTGeant4Integration::CreateVecGeomWorld(G4VPhysicalVolume const *physvol)
{
  // EXPECT: a non-null input volume
  if (physvol == nullptr) {
    throw std::runtime_error("AdePTGeant4Integration::CreateVecGeomWorld : Input Geant4 Physical Volume is nullptr");
  }

  vecgeom::GeoManager::Instance().SetTransformationCacheDepth(0);
  g4vg::Options options;
  options.reflection_factory = false;
  auto conversion            = g4vg::convert(physvol, options);
  vecgeom::GeoManager::Instance().SetWorldAndClose(conversion.world);

  // Get the mapping of VecGeom volume IDs to Geant4 physical volumes from g4vg
  fglobal_vecgeom_pv_to_g4_map = conversion.physical_volumes;
  fglobal_vecgeom_lv_to_g4_map = conversion.logical_volumes;

  // EXPECT: we finish with a non-null VecGeom host geometry
  vecgeom::VPlacedVolume const *vecgeomWorld = vecgeom::GeoManager::Instance().GetWorld();
  if (vecgeomWorld == nullptr) {
    throw std::runtime_error("AdePTGeant4Integration::CreateVecGeomWorld : Output VecGeom Physical Volume is nullptr");
  }
}

namespace {
struct VisitContext {
  const int *g4tohepmcindex;
  std::size_t nvolumes;
  G4HepEmState const *hepEmState;
};

/// Recursive geometry visitor matching one by one Geant4 and VecGeom logical volumes
void visitGeometry(G4VPhysicalVolume const *g4_pvol, vecgeom::VPlacedVolume const *vg_pvol, const VisitContext &context)
{
  const auto g4_lvol = g4_pvol->GetLogicalVolume();
  const auto vg_lvol = vg_pvol->GetLogicalVolume();

  // Geant4 Parameterised/Replica volumes are represented with direct placements in VecGeom
  // To accurately compare the number of daughters, we need to sum multiplicity on Geant4 side
  const size_t nd     = g4_lvol->GetNoDaughters();
  size_t nd_converted = 0;
  for (size_t daughter_id = 0; daughter_id < nd; ++daughter_id) {
    const G4VPhysicalVolume *daughter_pvol = g4_lvol->GetDaughter(daughter_id);
    nd_converted += daughter_pvol->GetMultiplicity();
  }

  const auto daughters = vg_lvol->GetDaughters();

  if (nd_converted != daughters.size())
    throw std::runtime_error("Fatal: CheckGeometry: Mismatch in number of daughters");

  // Check if transformations are matching
  // As above, with Parameterized/Replica volumes, we need to compare the transforms between
  // the VG direct placement and that for the Parameterised/Replicated volume given the same copy
  // number as that of the VG physical volume.
  // NOTE:
  // 1. Nasty const_cast as currently known way to get transform is to directly transform the
  //    Geant4 phys vol before extracting, which is non-const....
  // 2. ...this does modify the physical volume, but this is _probably_ o.k. as actual navigation
  //    will reset things o.k.
  if (G4VPVParameterisation *param = g4_pvol->GetParameterisation()) {
    param->ComputeTransformation(vg_pvol->GetCopyNo(), const_cast<G4VPhysicalVolume *>(g4_pvol));
  } else if (auto *replica = dynamic_cast<G4PVReplica *>(const_cast<G4VPhysicalVolume *>(g4_pvol))) {
    G4ReplicaNavigation nav;
    nav.ComputeTransformation(vg_pvol->GetCopyNo(), replica);
  }

  const auto g4trans            = g4_pvol->GetTranslation();
  const G4RotationMatrix *g4rot = g4_pvol->GetRotation();
  G4RotationMatrix idrot;
  const auto vgtransformation = vg_pvol->GetTransformation();
  constexpr double epsil      = 1.e-8;

  for (int i = 0; i < 3; ++i) {
    if (std::abs(g4trans[i] - vgtransformation->Translation(i)) > epsil)
      throw std::runtime_error(
          std::string("Fatal: CheckGeometry: Mismatch between Geant4 translation for physical volume") +
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
            std::string("Fatal: CheckGeometry: Mismatch between Geant4 rotation for physical volume") +
            vg_pvol->GetName());
    }
  }

  // Check the couples
  if (g4_lvol->GetMaterialCutsCouple() == nullptr)
    throw std::runtime_error("Fatal: CheckGeometry: G4LogicalVolume " + std::string(g4_lvol->GetName()) +
                             std::string(" has no material-cuts couple"));
  const int g4mcindex    = g4_lvol->GetMaterialCutsCouple()->GetIndex();
  const int hepemmcindex = context.g4tohepmcindex[g4mcindex];
  // Check consistency with G4HepEm data
  if (context.hepEmState->fData->fTheMatCutData->fMatCutData[hepemmcindex].fG4MatCutIndex != g4mcindex)
    throw std::runtime_error("Fatal: CheckGeometry: Mismatch between Geant4 mcindex and corresponding G4HepEm index");
  if (vg_lvol->id() >= context.nvolumes)
    throw std::runtime_error("Fatal: CheckGeometry: Volume id larger than number of volumes");

  // Now do the daughters
  for (size_t id = 0; id < g4_lvol->GetNoDaughters(); ++id) {
    const auto g4pvol_d = g4_lvol->GetDaughter(id);
    const auto pvol_d   = vg_lvol->GetDaughters()[id];

    visitGeometry(g4pvol_d, pvol_d, context);
  }
}
} // namespace

void AdePTGeant4Integration::CheckGeometry(G4HepEmState *hepEmState)
{
  const G4VPhysicalVolume *g4world =
      G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
  const vecgeom::VPlacedVolume *vecgeomWorld = vecgeom::GeoManager::Instance().GetWorld();
  const int *g4tohepmcindex                  = hepEmState->fData->fTheMatCutData->fG4MCIndexToHepEmMCIndex;
  const auto nvolumes                        = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();

  std::cout << "Visiting geometry ...\n";
  const VisitContext context{g4tohepmcindex, nvolumes, hepEmState};
  visitGeometry(g4world, vecgeomWorld, context);
  std::cout << "Visiting geometry done\n";
}

void AdePTGeant4Integration::InitVolAuxData(adeptint::VolAuxData *volAuxData, G4HepEmState *hepEmState,
                                            G4HepEmTrackingManagerSpecialized *hepEmTM, bool trackInAllRegions,
                                            std::vector<std::string> const *gpuRegionNames,
                                            adeptint::WDTHostRaw &wdtRaw)
{

  // Note: the hepEmTM must be passed as an argument despite the member fHepEmTrackingManager,
  // as InitVolAuxData is a static function and cannot call member variables
  wdtRaw.ekinMin = (float)hepEmTM->GetWDTKineticEnergyLimit();

  const G4VPhysicalVolume *g4world =
      G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
  const vecgeom::VPlacedVolume *vecgeomWorld = vecgeom::GeoManager::Instance().GetWorld();
  const int *g4tohepmcindex                  = hepEmState->fData->fTheMatCutData->fG4MCIndexToHepEmMCIndex;

  // We need to go from region names to G4Region
  std::vector<G4Region *> gpuRegions{};
  if (!trackInAllRegions) {
    for (std::string regionName : *(gpuRegionNames)) {
      G4Region *region = G4RegionStore::GetInstance()->GetRegion(regionName);
      gpuRegions.push_back(region);
    }
  }

  // recursive geometry visitor lambda matching one by one Geant4 and VecGeom logical volumes
  typedef std::function<void(G4VPhysicalVolume const *, vecgeom::VPlacedVolume const *, vecgeom::NavigationState)>
      func_t;
  func_t visitGeometry = [&](G4VPhysicalVolume const *g4_pvol, vecgeom::VPlacedVolume const *vg_pvol,
                             vecgeom::NavigationState currentNavState) {
    const auto g4_lvol = g4_pvol->GetLogicalVolume();
    const auto vg_lvol = vg_pvol->GetLogicalVolume();

    // Push this placed volume into the running NavState
    currentNavState.Push(vg_pvol);
    currentNavState.SetBoundaryState(false);

    // Fill the MCC index in the array
    int g4mcindex                      = g4_lvol->GetMaterialCutsCouple()->GetIndex();
    int hepemmcindex                   = g4tohepmcindex[g4mcindex];
    volAuxData[vg_lvol->id()].fMCIndex = hepemmcindex;

    const int regionId = g4_lvol->GetRegion()->GetInstanceID();

    // Check if the region is a Woodcock tracking region in G4HepEm
    if (hepEmTM->IsWDTRegion(regionId)) {
      // check if this logical volume is one of the declared WDT root LVs for this region
      const int rootIMC = hepEmTM->GetWDTCoupleHepEmIndex(regionId, g4_lvol->GetInstanceID());
      if (rootIMC >= 0) {
        // this placed volume belongs to a WDT root LV -> record a WDTRoot
        int idx = (int)wdtRaw.roots.size();
        wdtRaw.roots.push_back(adeptint::WDTRoot{currentNavState, rootIMC});
        wdtRaw.regionToRootIndices[regionId].push_back(idx);
      }
    }

    // Check if the volume belongs to a GPU region
    if (!trackInAllRegions) {
      for (G4Region *gpuRegion : gpuRegions) {
        if (g4_lvol->GetRegion() == gpuRegion) {
          volAuxData[vg_lvol->id()].fGPUregionId = g4_lvol->GetRegion()->GetInstanceID();
        }
      }
    } else {
      volAuxData[vg_lvol->id()].fGPUregionId = g4_lvol->GetRegion()->GetInstanceID();
    }

    if (g4_lvol->GetSensitiveDetector() != nullptr) {
      if (volAuxData[vg_lvol->id()].fSensIndex < 0) {
        G4cout << "VecGeom: Making " << vg_lvol->GetName() << " sensitive" << G4endl;
      }
      volAuxData[vg_lvol->id()].fSensIndex = 1;
    }
    // Now do the daughters
    for (size_t id = 0; id < g4_lvol->GetNoDaughters(); ++id) {
      auto g4pvol_d = g4_lvol->GetDaughter(id);
      auto pvol_d   = vg_lvol->GetDaughters()[id];

      visitGeometry(g4pvol_d, pvol_d, currentNavState);
    }

    // Pop NavState before before returning
    currentNavState.Pop();
  };

  // Initialize root NavState
  vecgeom::NavigationState rootNavState;

  visitGeometry(g4world, vecgeomWorld, rootNavState);

  auto findRegionName = [](int rid) -> std::string {
    for (auto *r : *G4RegionStore::GetInstance()) {
      if (r && r->GetInstanceID() == rid) return r->GetName();
    }
    return std::string("<unknown>");
  };

  std::cout << "\n=== Woodcock tracking summary (host) ===\n";
  std::cout << "KineticEnergyLimit = " << wdtRaw.ekinMin << " [G4 units]\n";
  std::cout << "Total WDT roots found: " << wdtRaw.roots.size() << std::endl;
  std::cout << "Regions with WDT: " << wdtRaw.regionToRootIndices.size() << std::endl;

  if (wdtRaw.regionToRootIndices.empty()) {
    std::cout << "  (none)\n";
  } else {
    for (const auto &kv : wdtRaw.regionToRootIndices) {
      const int rid    = kv.first;
      const auto &idxs = kv.second;
      std::cout << "\nRegionID " << rid << "  (" << findRegionName(rid) << "): " << idxs.size()
                << " root placed-volume(s)\n";

      for (size_t i = 0; i < idxs.size(); ++i) {
        const int rootIdx = idxs[i];
        const auto &root  = wdtRaw.roots[rootIdx];

        std::cout << "  [" << i << "] hepemIMC=" << root.hepemIMC << "\n";
        std::cout << "      NavState (level=" << root.root.GetLevel() << "):\n";
        // vecgeom::NavigationState::Print() prints the full stack
        root.root.Print();
      }
    }
  }
  std::cout << "=== End Woodcock tracking summary ===\n\n";
}

void AdePTGeant4Integration::ProcessGPUStep(GPUHit const &hit, bool const callUserSteppingAction,
                                            bool const callUserTrackingAction)
{
  if (!fScoringObjects) {
    fScoringObjects.reset(new AdePTGeant4Integration_detail::ScoringObjects());
  }

  // Reconstruct G4NavigationHistory and G4Step, and call the SD code for each hit
  vecgeom::NavigationState const &preNavState = hit.fPreStepPoint.fNavigationState;
  // Reconstruct Pre-Step point G4NavigationHistory
  FillG4NavigationHistory(preNavState, fScoringObjects->fPreG4NavigationHistory);
  (*fScoringObjects->fPreG4TouchableHistoryHandle)
      ->UpdateYourself(fScoringObjects->fPreG4NavigationHistory.GetTopVolume(),
                       &fScoringObjects->fPreG4NavigationHistory);
  // Reconstruct Post-Step point G4NavigationHistory
  vecgeom::NavigationState const &postNavState = hit.fPostStepPoint.fNavigationState;
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
  switch (hit.fParticleType) {
  case 0:
    fScoringObjects->fG4Step->SetTrack(fScoringObjects->fElectronTrack);
    break;
  case 1:
    fScoringObjects->fG4Step->SetTrack(fScoringObjects->fPositronTrack);
    break;
  case 2:
    fScoringObjects->fG4Step->SetTrack(fScoringObjects->fGammaTrack);
    break;
  default:
    std::cerr << "Error: unknown particle type " << static_cast<int>(static_cast<unsigned char>(hit.fParticleType))
              << "\n";
    std::abort();
  }
  FillG4Step(&hit, fScoringObjects->fG4Step, *fScoringObjects->fPreG4TouchableHistoryHandle,
             *fScoringObjects->fPostG4TouchableHistoryHandle, preStepStatus, postStepStatus, callUserTrackingAction,
             callUserSteppingAction);

  // Call SD code
  G4VSensitiveDetector *aSensitiveDetector =
      fScoringObjects->fPreG4NavigationHistory.GetVolume(fScoringObjects->fPreG4NavigationHistory.GetDepth())
          ->GetLogicalVolume()
          ->GetSensitiveDetector();

  // Call scoring if SD is defined and it is not the initializing step
  if (aSensitiveDetector != nullptr && hit.fStepCounter != 0) {
    aSensitiveDetector->Hit(fScoringObjects->fG4Step);
  }

  // cleanup of the secondary vector that is created in FillG4Step above
  fScoringObjects->fG4Step->DeleteSecondaryVector();

  // If this was the last step of a track, the hostTrackData of that track can be safely deleted.
  // Note: This deletes the AdePT-owned UserTrackInfo data
  if (hit.fLastStepOfTrack) {
    fHostTrackDataMapper->removeTrack(hit.fTrackID);
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
    pnewvol = const_cast<G4VPhysicalVolume *>(fglobal_vecgeom_pv_to_g4_map[aNavState.At(aLevel)->id()]);
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

void AdePTGeant4Integration::FillG4Step(GPUHit const *aGPUHit, G4Step *aG4Step,
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
  // initialize secondary vector (although it is empty for now)
  // Note: we own this vector, we are responsible for deleting it!
  aG4Step->NewSecondaryVector();
  // aG4Step->SetSecondary(nullptr);                               // Missing data

  // G4Track
  G4Track *aTrack = aG4Step->GetTrack();

  HostTrackData dummy; // default constructed dummy if no advanced information is available

  // if the userActions are used, advanced track information is available
  const bool actions = (callUserTrackingAction || callUserSteppingAction);

#ifdef DEBUG
  if (aGPUHit->fStepCounter == 0 && actions && fHostTrackDataMapper->contains(aGPUHit->fTrackID)) {
    std::cerr << "\033[1;31mERROR: TRACK ALREADY HAS AN ENTRY (trackID = " << aGPUHit->fTrackID
              << ", parentID = " << aGPUHit->fParentID << ") "
              << " stepLimProcessId " << aGPUHit->fStepLimProcessId << " pdg charge "
              << static_cast<int>(aGPUHit->fParticleType) << " stepCounter " << aGPUHit->fStepCounter << "\033[0m"
              << std::endl;
    std::abort();
  }
#endif

  // Bind a reference *without* touching the mapper unless actions==true
  HostTrackData &hostTData =
      actions ? (aGPUHit->fStepCounter == 0
                     ? fHostTrackDataMapper->create(aGPUHit->fTrackID) // new trackData for initializing step
                     : fHostTrackDataMapper->get(aGPUHit->fTrackID))   // existing trackData for later steps
              : dummy;                                                 // no map access, just use the dummy

  if (actions) {
    // When the user Actions are used, the 0th and the last step is returned. The 0th step is used to initialize the
    // hostTrackData, which includes all the host only pointers to the creator process, primary particle, G4VUserActions
    // etc.

    // initializing step
    if (aGPUHit->fStepCounter == 0) {
      hostTData.particleType = aGPUHit->fParticleType;

      if (aGPUHit->fParentID != 0) {
        // the parent must exist in the mapper, as the parent must have been created by AdePT,
        // as e-/e+/gamma created by G4 never arrive with a stepCounter = 0

#ifdef DEBUG
        if (aGPUHit->fStepCounter == 0 && actions && !fHostTrackDataMapper->contains(aGPUHit->fParentID)) {
          std::cerr << "\033[1;31mERROR: PARENT TRACK ID NOT FOUND (trackID = " << aGPUHit->fTrackID
                    << ", parentID = " << aGPUHit->fParentID << ") "
                    << " stepLimProcessId " << aGPUHit->fStepLimProcessId << " pdg charge "
                    << static_cast<int>(aGPUHit->fParticleType) << " stepCounter " << aGPUHit->fStepCounter
                    << " steplength " << aGPUHit->fStepLength << " global time " << aGPUHit->fGlobalTime
                    << " local time " << aGPUHit->fLocalTime << " isLasteStep " << aGPUHit->fLastStepOfTrack
                    << " thread id " << aGPUHit->threadId << " eventid " << aGPUHit->fEventId << "\033[0m" << std::endl;
          std::abort();
        }
#endif

        HostTrackData &p     = fHostTrackDataMapper->get(aGPUHit->fParentID);
        hostTData.g4parentid = p.g4id;
      } else {
        hostTData.g4parentid = 0;
      }

      hostTData.logicalVolumeAtVertex = aPreG4TouchableHandle->GetVolume()->GetLogicalVolume();
      hostTData.vertexPosition =
          G4ThreeVector{aGPUHit->fPostStepPoint.fPosition.x(), aGPUHit->fPostStepPoint.fPosition.y(),
                        aGPUHit->fPostStepPoint.fPosition.z()};
      hostTData.vertexMomentumDirection =
          G4ThreeVector{aGPUHit->fPostStepPoint.fMomentumDirection.x(), aGPUHit->fPostStepPoint.fMomentumDirection.y(),
                        aGPUHit->fPostStepPoint.fMomentumDirection.z()};
      hostTData.vertexKineticEnergy = aGPUHit->fPostStepPoint.fEKin;
      if (hostTData.originTouchableHandle) {
        hostTData.originTouchableHandle =
            std::make_unique<G4TouchableHandle>(MakeTouchableFromNavState(aGPUHit->fPostStepPoint.fNavigationState));
      }

      // For the initializing step, the step defining process ID is the creator process
      const int stepId = aGPUHit->fStepLimProcessId;
      const int ptype  = static_cast<int>(hostTData.particleType);
      if (aGPUHit->fParentID != 0) {
        if (ptype == 0 || ptype == 1) {
          hostTData.creatorProcess = fHepEmTrackingManager->GetElectronNoProcessVector()[stepId];
        } else if (ptype == 2) {
          hostTData.creatorProcess = fHepEmTrackingManager->GetGammaNoProcessVector()[stepId];
        }
      } else {
        hostTData.creatorProcess = nullptr; // primary
      }
    }

    // Now the hostTrackData is surely initialized and we can set the creator process and the dynamic particle

    // must const-cast as GetDynamicParticle only returns const
    aG4Step->GetTrack()->SetCreatorProcess(hostTData.creatorProcess);
    auto *dyn = const_cast<G4DynamicParticle *>(aG4Step->GetTrack()->GetDynamicParticle());
    dyn->SetPrimaryParticle(hostTData.primary);
  }

  // set the step-defining process for non-initializing steps
  G4VProcess *stepDefiningProcess = nullptr;
  if (aGPUHit->fStepCounter != 0) {
    // not an initial step, therefore setting the step defining process:
    const int stepId = aGPUHit->fStepLimProcessId;
    const int ptype  = static_cast<int>(hostTData.particleType);

    if (ptype == 0 || ptype == 1) { // e- or e+
      if (stepId == 10)
        stepDefiningProcess = fHepEmTrackingManager->GetTransportNoProcess(); // set to transportation
      else if (stepId == -2)
        stepDefiningProcess = fHepEmTrackingManager->GetElectronNoProcessVector()[3]; // MSC
      else if (stepId == -1)
        stepDefiningProcess = fHepEmTrackingManager->GetElectronNoProcessVector()[0]; // dE/dx due to ionization
      else if (stepId == 3) {
        if (ptype == 0) stepDefiningProcess = fHepEmTrackingManager->GetElectronNoProcessVector()[4]; // e- nuclear
        if (ptype == 1) stepDefiningProcess = fHepEmTrackingManager->GetElectronNoProcessVector()[5]; // e+ nuclear
      } else {
        stepDefiningProcess = fHepEmTrackingManager->GetElectronNoProcessVector()[stepId]; // discrete interactions
      }
    } else if (ptype == 2) {
      stepDefiningProcess = (stepId == 10)
                                ? fHepEmTrackingManager->GetTransportNoProcess()            // transportation
                                : fHepEmTrackingManager->GetGammaNoProcessVector()[stepId]; // discrete interactions
    }
  }

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
  aTrack->SetStep(aG4Step);                                              // Real data
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

  // Last, call tracking and stepping actions
  // call UserSteppingAction if required

  if (aGPUHit->fStepCounter == 0 && (callUserTrackingAction || callUserSteppingAction)) {
    auto *evtMgr             = G4EventManager::GetEventManager();
    auto *userTrackingAction = evtMgr->GetUserTrackingAction();
    if (userTrackingAction) {
      userTrackingAction->PreUserTrackingAction(aTrack);

      // if userTrackInfo didn't exist before but exists now, update the map as a new TrackInfo was attached for
      // the first time
      if (hostTData.userTrackInfo == nullptr && aTrack->GetUserInformation() != nullptr) {
        hostTData.userTrackInfo = aTrack->GetUserInformation();
      }
    }
  }

  if (callUserSteppingAction) {
    auto *evtMgr             = G4EventManager::GetEventManager();
    auto *userSteppingAction = evtMgr->GetUserSteppingAction();
    if (userSteppingAction) userSteppingAction->UserSteppingAction(aG4Step);
  }

  // call UserTrackingAction if required
  if (aGPUHit->fLastStepOfTrack && (callUserTrackingAction || callUserSteppingAction)) {
    auto *evtMgr             = G4EventManager::GetEventManager();
    auto *userTrackingAction = evtMgr->GetUserTrackingAction();
    if (userTrackingAction) userTrackingAction->PostUserTrackingAction(aTrack);
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
  if (hostTData.originTouchableHandle) {
    leakedTrack->SetOriginTouchableHandle(*hostTData.originTouchableHandle);
  }

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
