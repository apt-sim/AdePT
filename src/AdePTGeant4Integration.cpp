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
                                            bool trackInAllRegions, std::vector<std::string> const *gpuRegionNames)
{
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
  typedef std::function<void(G4VPhysicalVolume const *, vecgeom::VPlacedVolume const *)> func_t;
  func_t visitGeometry = [&](G4VPhysicalVolume const *g4_pvol, vecgeom::VPlacedVolume const *vg_pvol) {
    const auto g4_lvol = g4_pvol->GetLogicalVolume();
    const auto vg_lvol = vg_pvol->GetLogicalVolume();
    // Fill the MCC index in the array
    int g4mcindex                      = g4_lvol->GetMaterialCutsCouple()->GetIndex();
    int hepemmcindex                   = g4tohepmcindex[g4mcindex];
    volAuxData[vg_lvol->id()].fMCIndex = hepemmcindex;

    // Check if the volume belongs to a GPU region
    if (!trackInAllRegions) {
      for (G4Region *gpuRegion : gpuRegions) {
        if (g4_lvol->GetRegion() == gpuRegion) {
          volAuxData[vg_lvol->id()].fGPUregion = 1;
        }
      }
    } else {
      volAuxData[vg_lvol->id()].fGPUregion = 1;
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

      visitGeometry(g4pvol_d, pvol_d);
    }
  };

  visitGeometry(g4world, vecgeomWorld);
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
  }
  FillG4Step(&hit, fScoringObjects->fG4Step, *fScoringObjects->fPreG4TouchableHistoryHandle,
             *fScoringObjects->fPostG4TouchableHistoryHandle, preStepStatus, postStepStatus, callUserTrackingAction,
             callUserSteppingAction);

  // Call SD code
  G4VSensitiveDetector *aSensitiveDetector =
      fScoringObjects->fPreG4NavigationHistory.GetVolume(fScoringObjects->fPreG4NavigationHistory.GetDepth())
          ->GetLogicalVolume()
          ->GetSensitiveDetector();

  // Call scoring if SD is defined
  if (aSensitiveDetector != nullptr) {
    aSensitiveDetector->Hit(fScoringObjects->fG4Step);
  }
}

void AdePTGeant4Integration::FillG4NavigationHistory(vecgeom::NavigationState aNavState,
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

void AdePTGeant4Integration::FillG4Step(GPUHit const *aGPUHit, G4Step *aG4Step,
                                        G4TouchableHandle &aPreG4TouchableHandle,
                                        G4TouchableHandle &aPostG4TouchableHandle, G4StepStatus aPreStepStatus,
                                        G4StepStatus aPostStepStatus, bool callUserTrackingAction,
                                        bool callUserSteppingAction) const
{
  const G4ThreeVector aPostStepPointMomentumDirection(aGPUHit->fPostStepPoint.fMomentumDirection.x(),
                                                      aGPUHit->fPostStepPoint.fMomentumDirection.y(),
                                                      aGPUHit->fPostStepPoint.fMomentumDirection.z());
  const G4ThreeVector aPostStepPointPolarization(aGPUHit->fPostStepPoint.fPolarization.x(),
                                                 aGPUHit->fPostStepPoint.fPolarization.y(),
                                                 aGPUHit->fPostStepPoint.fPolarization.z());
  const G4ThreeVector aPostStepPointPosition(aGPUHit->fPostStepPoint.fPosition.x(),
                                             aGPUHit->fPostStepPoint.fPosition.y(),
                                             aGPUHit->fPostStepPoint.fPosition.z());

  const G4ThreeVector aVertexPosition(aGPUHit->fVertexPosition.x(), aGPUHit->fVertexPosition.y(),
                                      aGPUHit->fVertexPosition.z());

  // G4Step
  aG4Step->SetStepLength(aGPUHit->fStepLength);                 // Real data
  aG4Step->SetTotalEnergyDeposit(aGPUHit->fTotalEnergyDeposit); // Real data
  // aG4Step->SetNonIonizingEnergyDeposit(0);                      // Missing data
  // aG4Step->SetControlFlag(G4SteppingControl::NormalCondition);  // Missing data
  if (aGPUHit->fFirstStepOfTrack) aG4Step->SetFirstStepFlag(); // Real data
  if (aGPUHit->fLastStepOfTrack) aG4Step->SetLastStepFlag();   // Real data
  // aG4Step->SetPointerToVectorOfAuxiliaryPoints(nullptr);        // Missing data
  // aG4Step->SetSecondary(nullptr);                               // Missing data

  // G4Track
  int g4TrackID  = fTrackIDMapper->mapGpuToG4(aGPUHit->fTrackID);
  int g4ParentID = (aGPUHit->fParentID != 0) ? fTrackIDMapper->mapGpuToG4(aGPUHit->fParentID) : 0;

  G4Track *aTrack = aG4Step->GetTrack();

  G4PrimaryParticle *primary = fTrackIDMapper->getPrimaryForG4ID(g4TrackID);

  if (aGPUHit->fCreatorProcessID == -1) {
    aTrack->SetCreatorProcess(fTrackIDMapper->getCreatorProcessForG4ID(g4TrackID));
  } else {
    const G4ParticleDefinition *part = aTrack->GetParticleDefinition();
    if (part == G4Electron::Definition() || part == G4Positron::Definition()) {
      aTrack->SetCreatorProcess(fHepEmTrackingManager->GetElectronNoProcessVector()[aGPUHit->fCreatorProcessID]);
    } else if (part == G4Gamma::Definition()) {
      aTrack->SetCreatorProcess(fHepEmTrackingManager->GetGammaNoProcessVector()[aGPUHit->fCreatorProcessID]);
    }
  }

  // G4DynamicParticle* dynamicParticle = aTrack->GetDynamicParticle();
  // must const-cast as GetDynamicParticle only returns const
  G4DynamicParticle *dynamicParticle = const_cast<G4DynamicParticle *>(aTrack->GetDynamicParticle());

  dynamicParticle->SetPrimaryParticle(primary);

  if (g4ParentID == 0 && primary == nullptr) {
    fTrackIDMapper->mapGpuToG4(aGPUHit->fTrackID);
  }

  aTrack->SetTrackID(g4TrackID);               // Missing data
  aTrack->SetParentID(g4ParentID);             // ID of the initial particle that entered AdePT
  aTrack->SetPosition(aPostStepPointPosition); // Real data
  aTrack->SetGlobalTime(aGPUHit->fGlobalTime); // Real data
  aTrack->SetLocalTime(aGPUHit->fLocalTime);   // Real data
  // aTrack->SetProperTime(0);                                                                // Missing data
  // aTrack->SetTouchableHandle(aTrackTouchableHistory);                                      // Missing data
  // aTrack->SetNextTouchableHandle(nullptr);                                                 // Missing data
  // aTrack->SetOriginTouchableHandle(nullptr);                                               // Missing data
  aTrack->SetKineticEnergy(aGPUHit->fPostStepPoint.fEKin);       // Real data
  aTrack->SetMomentumDirection(aPostStepPointMomentumDirection); // Real data
  // aTrack->SetVelocity(0);                                                                  // Missing data
  aTrack->SetPolarization(aPostStepPointPolarization); // Real data
  // aTrack->SetTrackStatus(G4TrackStatus::fAlive);                                           // Missing data
  // aTrack->SetBelowThresholdFlag(false);                                                    // Missing data
  // aTrack->SetGoodForTrackingFlag(false);                                                   // Missing data
  aTrack->SetStep(aG4Step);                    // Real data
  aTrack->SetStepLength(aGPUHit->fStepLength); // Real data
  aTrack->SetVertexPosition(aVertexPosition);  // Real data
  // aTrack->SetVertexMomentumDirection(G4ThreeVector(0, 0, 0));                              // Missing data
  // aTrack->SetVertexKineticEnergy(0);                                                       // Missing data
  // aTrack->SetLogicalVolumeAtVertex(nullptr);                                               // Missing data
  // aTrack->SetCreatorProcess(nullptr);                                                      // Missing data
  // aTrack->SetCreatorModelID(0);                                                            // Missing data
  // aTrack->SetParentResonanceDef(nullptr);                                                  // Missing data
  // aTrack->SetParentResonanceID(0);                                                         // Missing data
  aTrack->SetWeight(aGPUHit->fTrackWeight);
  // if it exists, add UserTrackInfo
  aTrack->SetUserInformation(fTrackIDMapper->getUserTrackInfoForG4ID(g4TrackID)); // Real data for primaries
  // aTrack->SetAuxiliaryTrackInformation(0, nullptr);                                        // Missing data

  // Pre-Step Point
  G4StepPoint *aPreStepPoint = aG4Step->GetPreStepPoint();
  aPreStepPoint->SetPosition(G4ThreeVector(aGPUHit->fPreStepPoint.fPosition.x(), aGPUHit->fPreStepPoint.fPosition.y(),
                                           aGPUHit->fPreStepPoint.fPosition.z())); // Real data
  // aPreStepPoint->SetLocalTime(0);                                                                // Missing data
  // aPreStepPoint->SetGlobalTime(0);                                                               // Missing data
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
  aPreStepPoint->SetPolarization(G4ThreeVector(aGPUHit->fPreStepPoint.fPolarization.x(),
                                               aGPUHit->fPreStepPoint.fPolarization.y(),
                                               aGPUHit->fPreStepPoint.fPolarization.z())); // Real data
  aPreStepPoint->SetStepStatus(aPreStepStatus);                                            // Missing data
  // aPreStepPoint->SetProcessDefinedStep(nullptr);                                                 // Missing data
  // aPreStepPoint->SetMass(0);                                                                     // Missing data
  aPreStepPoint->SetCharge(aGPUHit->fPreStepPoint.fCharge); // Real data
  // aPreStepPoint->SetMagneticMoment(0);                                                           // Missing data
  // aPreStepPoint->SetWeight(0);                                                                   // Missing data

  // Post-Step Point
  G4StepPoint *aPostStepPoint = aG4Step->GetPostStepPoint();
  aPostStepPoint->SetPosition(aPostStepPointPosition); // Real data
  // aPostStepPoint->SetLocalTime(0);                                                                 // Missing data
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
  aPostStepPoint->SetPolarization(aPostStepPointPolarization); // Real data
  aPostStepPoint->SetStepStatus(aPostStepStatus);              // Missing data
  // aPostStepPoint->SetProcessDefinedStep(nullptr);                                                  // Missing data
  // aPostStepPoint->SetMass(0);                                                                      // Missing data
  aPostStepPoint->SetCharge(aGPUHit->fPostStepPoint.fCharge); // Real data
  // aPostStepPoint->SetMagneticMoment(0);                                                            // Missing data
  // aPostStepPoint->SetWeight(0);                                                                    // Missing data

  // Last, call tracking and stepping actions
  // call UserSteppingAction if required

  if (aGPUHit->fFirstStepOfTrack && callUserTrackingAction) {
    auto *evtMgr             = G4EventManager::GetEventManager();
    auto *userTrackingAction = evtMgr->GetUserTrackingAction();
    if (userTrackingAction) {
      userTrackingAction->PreUserTrackingAction(aTrack);

      // To do, if userTrackInfo didn't exist before but exists now, update the map as a new TrackInfo was attached for
      // the first time
      if (fTrackIDMapper->getUserTrackInfoForG4ID(g4TrackID) == nullptr && aTrack->GetUserInformation() != nullptr) {
        fTrackIDMapper->setUserTrackInfoForG4ID(g4TrackID, aTrack->GetUserInformation());
      }
    }
  }

  if (callUserSteppingAction) {
    auto *evtMgr             = G4EventManager::GetEventManager();
    auto *userSteppingAction = evtMgr->GetUserSteppingAction();
    if (userSteppingAction) userSteppingAction->UserSteppingAction(aG4Step);
  }

  // call UserTrackingAction if required
  if (aGPUHit->fLastStepOfTrack && callUserTrackingAction) {
    auto *evtMgr             = G4EventManager::GetEventManager();
    auto *userTrackingAction = evtMgr->GetUserTrackingAction();
    if (userTrackingAction) userTrackingAction->PostUserTrackingAction(aTrack);
  }
}

void AdePTGeant4Integration::ReturnTrack(adeptint::TrackData const &track, unsigned int trackIndex,
                                         int debugLevel) const
{
  constexpr double tolerance = 10. * vecgeom::kTolerance;

  // Build the secondaries and put them back on the Geant4 stack
  if (debugLevel > 1) {
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

  int g4TrackID              = fTrackIDMapper->mapGpuToG4(track.trackId);
  G4PrimaryParticle *primary = fTrackIDMapper->getPrimaryForG4ID(g4TrackID);
  // G4VUserTrackInformation* userTrackInfo = fTrackIDMapper->getPrimaryForG4ID(g4TrackID);
  int g4ParentID = (track.parentId != 0) ? fTrackIDMapper->mapGpuToG4(track.parentId) : 0;
  dynamic->SetPrimaryParticle(primary);

  // Create track
  G4Track *leakedTrack = new G4Track(dynamic, track.globalTime, posi);

  if (track.creatorProcessId == -1) {
    leakedTrack->SetCreatorProcess(fTrackIDMapper->getCreatorProcessForG4ID(g4TrackID));
  } else {
    const G4ParticleDefinition *part = leakedTrack->GetParticleDefinition();
    if (part == G4Electron::Definition() || part == G4Positron::Definition()) {
      leakedTrack->SetCreatorProcess(fHepEmTrackingManager->GetElectronNoProcessVector()[track.creatorProcessId]);
    } else if (part == G4Gamma::Definition()) {
      leakedTrack->SetCreatorProcess(fHepEmTrackingManager->GetGammaNoProcessVector()[track.creatorProcessId]);
    }
  }

  leakedTrack->SetTrackID(g4TrackID);
  leakedTrack->SetParentID(g4ParentID);

  // auto* userInfo = static_cast<G4VUserTrackInformation*>(track.userTrackInfo);
  leakedTrack->SetUserInformation(fTrackIDMapper->getUserTrackInfoForG4ID(g4TrackID));

  // Set time information
  leakedTrack->SetLocalTime(track.localTime);
  leakedTrack->SetProperTime(track.properTime);
  leakedTrack->SetParentID(track.parentId);

  // Set weight
  leakedTrack->SetWeight(track.weight);

  // Set vertex information
  leakedTrack->SetVertexPosition(
      G4ThreeVector(track.vertexPosition[0], track.vertexPosition[1], track.vertexPosition[2]));
  leakedTrack->SetVertexMomentumDirection(G4ThreeVector(
      track.vertexMomentumDirection[0], track.vertexMomentumDirection[1], track.vertexMomentumDirection[2]));
  leakedTrack->SetVertexKineticEnergy(track.vertexEkin);
  leakedTrack->SetLogicalVolumeAtVertex(
      fglobal_vecgeom_lv_to_g4_map[track.originNavState.Top()->GetLogicalVolume()->id()]);

  // Reconstruct the origin touchable history and update it on the track
  // - We can't update the track's navigation history in place as it is a const member
  // - For the same reason, we need to fill a navigation history and then create a touchable history from it
  auto originNavigationHistory = std::make_unique<G4NavigationHistory>();
  FillG4NavigationHistory(track.originNavState, *originNavigationHistory);

  // G4TouchableHistory constructor does a shallow copy of the navigation history
  // There is no way to transfer ownership of this pointer to the G4TouchableHistory, as the other available method,
  // UpdateYourself() does a shallow copy as well.
  // The only way to avoid a memory leak is to do the shallow copy and then allow our instance to be deleted, which will
  // call G4NavigationHistoryPool::DeRegister()
  auto originTouchableHistory = std::make_unique<G4TouchableHistory>(*originNavigationHistory);

  // Give ownership of the touchable history to the touchable handle, which will now manage its lifetime
  G4TouchableHandle originTouchableHandle(originTouchableHistory.release() /* Now owned by G4TouchableHandle */);
  leakedTrack->SetOriginTouchableHandle(originTouchableHandle);

  // ------ Handle leaked tracks according to their status, if not LeakStatus::OutOfGPURegion ---------

  // sanity check
  if (track.leakStatus == LeakStatus::NoLeak) throw std::runtime_error("Leaked track with status NoLeak detected!");

  // We set the status of leaked tracks to fStopButAlive to be able to distinguish them to keep
  // them on the CPU until they are done
  if (track.leakStatus == LeakStatus::FinishEventOnCPU) {
    // FIXME: previous approach was broken in sync AdePT, therefore it was removed
    // to be fixed with a different approach e.g., negative track ID.
    // leakedTrack->SetTrackStatus(fStopAndAlive);
  }

  // handle gamma- and lepton-nuclear directly in G4HepEm
  if (track.leakStatus == LeakStatus::GammaNuclear || track.leakStatus == LeakStatus::LeptonNuclear) {

    // AS ABOVE the current touchable is set as it is needed for gamma-/lepton- nuclear
    auto NavigationHistory = std::make_unique<G4NavigationHistory>();
    FillG4NavigationHistory(track.navState, *NavigationHistory);
    auto TouchableHistory = std::make_unique<G4TouchableHistory>(*NavigationHistory);
    auto topVolume        = NavigationHistory->GetTopVolume();
    G4TouchableHandle TouchableHandle(TouchableHistory.release() /* Now owned by G4TouchableHandle */);
    leakedTrack->SetTouchableHandle(TouchableHandle);

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

          // Since we do not give back the track to G4, we have to delete it here
          delete leakedTrack;
          delete step;
        } else {
          // no gamma nuclear process attached, just give back the track to G4 to put it back on GPU
          G4EventManager::GetEventManager()->GetStackManager()->PushOneTrack(leakedTrack);
          delete step;
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
      }
    } else {
      throw std::runtime_error("Specialized HepEmTrackingManager not longer valid in integration!");
    }

  } else {

    // LeakStatus::OutOfGPURegion: just give track back to G4
    G4EventManager::GetEventManager()->GetStackManager()->PushOneTrack(leakedTrack);
  }
}

vecgeom::Vector3D<float> AdePTGeant4Integration::GetUniformField() const
{
  vecgeom::Vector3D<float> Bfield(0., 0., 0.);

  G4MagneticField *field =
      (G4MagneticField *)G4TransportationManager::GetTransportationManager()->GetFieldManager()->GetDetectorField();

  if (field) {
    G4double origin[3] = {0., 0., 0.};
    G4double temp[3]   = {0., 0., 0.};
    field->GetFieldValue(origin, temp);
    Bfield.Set(temp[0], temp[1], temp[2]);
  }

  return Bfield;
}
