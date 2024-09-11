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

#include <G4HepEmData.hh>
#include <G4HepEmMatCutData.hh>

AdePTGeant4Integration::~AdePTGeant4Integration()
{
  delete fPreG4NavigationHistory;
  delete fPostG4NavigationHistory;
  delete fG4Step;
  delete fElectronTrack;
  delete fPositronTrack;
  delete fGammaTrack;
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

  const vecgeom::VPlacedVolume *vecgeomWorld = vecgeom::GeoManager::Instance().GetWorld();
  if (vecgeomWorld == nullptr) {
    std::cerr << "GeoManager vecgeomWorld volume is nullptr" << G4endl;
    return;
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

  const int nd         = g4_lvol->GetNoDaughters();
  const auto daughters = vg_lvol->GetDaughters();

  if (nd != daughters.size()) throw std::runtime_error("Fatal: CheckGeometry: Mismatch in number of daughters");
  // Check if transformations are matching
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
  for (int id = 0; id < g4_lvol->GetNoDaughters(); ++id) {
    const auto g4pvol_d = g4_lvol->GetDaughter(id);
    const auto pvol_d   = vg_lvol->GetDaughters()[id];

    // VecGeom does not strip pointers from logical volume names
    if (std::string(pvol_d->GetLogicalVolume()->GetName()).rfind(g4pvol_d->GetLogicalVolume()->GetName(), 0) != 0)
      throw std::runtime_error("Fatal: CheckGeometry: Volume names " +
                               std::string(pvol_d->GetLogicalVolume()->GetName()) + " and " +
                               std::string(g4pvol_d->GetLogicalVolume()->GetName()) + " mismatch");
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
                                            bool trackInAllRegions, std::vector<std::string> *gpuRegionNames)
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

    // Check if the logical volume is sensitive
    bool sens = false;

    if (g4_lvol->GetSensitiveDetector() != nullptr) {
      if (volAuxData[vg_lvol->id()].fSensIndex < 0) {
        G4cout << "VecGeom: Making " << vg_lvol->GetName() << " sensitive" << G4endl;
      }
      volAuxData[vg_lvol->id()].fSensIndex = 1;
    }
    // Now do the daughters
    for (int id = 0; id < g4_lvol->GetNoDaughters(); ++id) {
      auto g4pvol_d = g4_lvol->GetDaughter(id);
      auto pvol_d   = vg_lvol->GetDaughters()[id];

      // VecGeom does not strip pointers from logical volume names
      if (std::string(pvol_d->GetLogicalVolume()->GetName()).rfind(g4pvol_d->GetLogicalVolume()->GetName(), 0) != 0)
        throw std::runtime_error("Fatal: CheckGeometry: Volume names " +
                                 std::string(pvol_d->GetLogicalVolume()->GetName()) + " and " +
                                 std::string(g4pvol_d->GetLogicalVolume()->GetName()) + " mismatch");
      visitGeometry(g4pvol_d, pvol_d);
    }
  };

  visitGeometry(g4world, vecgeomWorld);
}

void AdePTGeant4Integration::InitScoringData(adeptint::VolAuxData *volAuxData)
{
  const G4VPhysicalVolume *g4world =
      G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
  const vecgeom::VPlacedVolume *vecgeomWorld = vecgeom::GeoManager::Instance().GetWorld();

  // Used to keep track of the current vecgeom history while visiting the tree
  std::vector<vecgeom::VPlacedVolume const *> aCurrentVecgeomHistory;
  // Used to keep track of the current geant4 history while visiting the tree
  std::vector<G4VPhysicalVolume const *> aCurrentGeant4History;

  // recursive geometry visitor lambda matching one by one Geant4 and VecGeom logical volumes
  typedef std::function<void(G4VPhysicalVolume const *, vecgeom::VPlacedVolume const *)> func_t;
  func_t visitGeometry = [&](G4VPhysicalVolume const *g4_pvol, vecgeom::VPlacedVolume const *vg_pvol) {
    const auto g4_lvol = g4_pvol->GetLogicalVolume();
    const auto vg_lvol = vg_pvol->GetLogicalVolume();

    aCurrentVecgeomHistory.push_back(vg_pvol);
    aCurrentGeant4History.push_back(g4_pvol);

    // If the volume is sensitive:
    if (volAuxData[vg_lvol->id()].fSensIndex == 1) {
      // Initialize mapping of Vecgeom sensitive PlacedVolume IDs to G4 PhysicalVolume IDs
      // In order to be able to reconstruct navigation histories based on a VecGeom Navigation State Index,
      // we need to map not only the sensitive volume, but also the ones leading up to here
      for (uint i = 0; i < aCurrentVecgeomHistory.size() - 1; i++) {
        fglobal_vecgeom_to_g4_map.insert(
            std::pair<int, const G4VPhysicalVolume *>(aCurrentVecgeomHistory[i]->id(), aCurrentGeant4History[i]));
      }
      bool new_pvol =
          fglobal_vecgeom_to_g4_map.insert(std::pair<int, const G4VPhysicalVolume *>(vg_pvol->id(), g4_pvol)).second;
    }
    // Now do the daughters
    for (int id = 0; id < g4_lvol->GetNoDaughters(); ++id) {
      auto g4pvol_d = g4_lvol->GetDaughter(id);
      auto pvol_d   = vg_lvol->GetDaughters()[id];

      // VecGeom does not strip pointers from logical volume names
      if (std::string(pvol_d->GetLogicalVolume()->GetName()).rfind(g4pvol_d->GetLogicalVolume()->GetName(), 0) != 0)
        throw std::runtime_error("Fatal: CheckGeometry: Volume names " +
                                 std::string(pvol_d->GetLogicalVolume()->GetName()) + " and " +
                                 std::string(g4pvol_d->GetLogicalVolume()->GetName()) + " mismatch");
      visitGeometry(g4pvol_d, pvol_d);
    }
    aCurrentVecgeomHistory.pop_back();
    aCurrentGeant4History.pop_back();
  };
  visitGeometry(g4world, vecgeomWorld);
}

void AdePTGeant4Integration::ProcessGPUHits(HostScoring &aScoring, HostScoring::Stats &aStats)
{
  if (!fScoringObjectsInitialized) {
    // For sequential processing of hits we only need one instance of each object
    fPreG4NavigationHistory       = new G4NavigationHistory();
    fPostG4NavigationHistory      = new G4NavigationHistory();
    fG4Step                       = new G4Step();
    fPreG4TouchableHistoryHandle  = new G4TouchableHistory();
    fPostG4TouchableHistoryHandle = new G4TouchableHistory();
    fG4Step->SetPreStepPoint(new G4StepPoint());
    fG4Step->SetPostStepPoint(new G4StepPoint());

    // We need the dynamic particle associated to the track to have the correct particle definition, however this can
    // only be set at construction time. Similarly, we can only set the dynamic particle for a track when creating it
    // For this reason we create one track per particle type, to be reused
    // We set position to nullptr and kinetic energy to 0 for the dynamic particle since they need to be updated per hit
    // The same goes for the G4Track global time and position
    fElectronTrack = new G4Track(
        new G4DynamicParticle(G4ParticleTable::GetParticleTable()->FindParticle("e-"), G4ThreeVector(0, 0, 0), 0), 0,
        G4ThreeVector(0, 0, 0));
    fPositronTrack = new G4Track(
        new G4DynamicParticle(G4ParticleTable::GetParticleTable()->FindParticle("e+"), G4ThreeVector(0, 0, 0), 0), 0,
        G4ThreeVector(0, 0, 0));
    fGammaTrack = new G4Track(
        new G4DynamicParticle(G4ParticleTable::GetParticleTable()->FindParticle("gamma"), G4ThreeVector(0, 0, 0), 0), 0,
        G4ThreeVector(0, 0, 0));
    fScoringObjectsInitialized = true;
  }

  // Reconstruct G4NavigationHistory and G4Step, and call the SD code for each hit
  for (size_t i = aStats.fBufferStart; i < aStats.fBufferStart + aStats.fUsedSlots; i++) {
    // Get Hit index (Circular buffer)
    int aHitIdx = i % aScoring.fBufferCapacity;

    vecgeom::NavigationState &aNavState = aScoring.fGPUHitsBuffer_host[aHitIdx].fPreStepPoint.fNavigationState;
    // Reconstruct Pre-Step point G4NavigationHistory
    FillG4NavigationHistory(aNavState, fPreG4NavigationHistory);
    ((G4TouchableHistory *)fPreG4TouchableHistoryHandle())
        ->UpdateYourself(fPreG4NavigationHistory->GetTopVolume(), fPreG4NavigationHistory);
    // Reconstruct Post-Step point G4NavigationHistory
    FillG4NavigationHistory(aNavState, fPostG4NavigationHistory);
    ((G4TouchableHistory *)fPostG4TouchableHistoryHandle())
        ->UpdateYourself(fPostG4NavigationHistory->GetTopVolume(), fPostG4NavigationHistory);

    // Reconstruct G4Step
    switch (aScoring.fGPUHitsBuffer_host[aHitIdx].fParticleType) {
    case 0:
      fG4Step->SetTrack(fElectronTrack);
      break;
    case 1:
      fG4Step->SetTrack(fPositronTrack);
      break;
    case 2:
      fG4Step->SetTrack(fGammaTrack);
      break;
    }
    FillG4Step(&(aScoring.fGPUHitsBuffer_host[aHitIdx]), fG4Step, fPreG4TouchableHistoryHandle,
               fPostG4TouchableHistoryHandle);

    // Call SD code
    G4VSensitiveDetector *aSensitiveDetector = fPreG4NavigationHistory->GetVolume(fPreG4NavigationHistory->GetDepth())
                                                   ->GetLogicalVolume()
                                                   ->GetSensitiveDetector();

    // Double check, a nullptr here can indicate an issue reconstructing the navigation history
    assert(aSensitiveDetector != nullptr);

    aSensitiveDetector->Hit(fG4Step);
  }
}

void AdePTGeant4Integration::FillG4NavigationHistory(vecgeom::NavigationState aNavState, G4NavigationHistory *aG4NavigationHistory)
{
  // Get the current depth of the history (corresponding to the previous reconstructed touchable)
  auto aG4HistoryDepth = aG4NavigationHistory->GetDepth();
  // Get the depth of the navigation state we want to reconstruct
  auto aVecGeomLevel = aNavState.GetLevel();

  unsigned int aLevel{0};
  G4VPhysicalVolume *pvol{nullptr}, *pnewvol{nullptr};

  for (aLevel = 0; aLevel <= aVecGeomLevel; aLevel++) {
    // While we are in levels shallower than the history depth, it may be that we already
    // have the correct volume in the history
    pnewvol = const_cast<G4VPhysicalVolume *>(fglobal_vecgeom_to_g4_map[aNavState.At(aLevel)->id()]);

    if (aG4HistoryDepth && (aLevel <= aG4HistoryDepth)) {
      pvol = aG4NavigationHistory->GetVolume(aLevel);
      // If they match we do not need to update the history at this level
      if (pvol == pnewvol) continue;
      // Once we find two non-matching volumes, we need to update the touchable history from this level on
      if (aLevel) {
        // If we are not in the top level
        aG4NavigationHistory->BackLevel(aG4HistoryDepth - aLevel + 1);
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
      if(aLevel)
      { 
        aG4NavigationHistory->NewLevel(pnewvol, kNormal, pnewvol->GetCopyNo());
        aG4HistoryDepth++;
      }
      else
      {
        aG4NavigationHistory->SetFirstEntry(pnewvol);
      }
    }
  }
  // Once finished, remove the extra levels if the current state is shallower than the previous history
  if (aG4HistoryDepth >= aLevel) aG4NavigationHistory->BackLevel(aG4HistoryDepth - aLevel + 1);
}

void AdePTGeant4Integration::FillG4Step(GPUHit *aGPUHit, G4Step *aG4Step, G4TouchableHandle &aPreG4TouchableHandle,
                                        G4TouchableHandle &aPostG4TouchableHandle)
{
  const G4ThreeVector *aPostStepPointMomentumDirection = new G4ThreeVector(aGPUHit->fPostStepPoint.fMomentumDirection.x(),
                                                                           aGPUHit->fPostStepPoint.fMomentumDirection.y(),
                                                                           aGPUHit->fPostStepPoint.fMomentumDirection.z());
  const G4ThreeVector *aPostStepPointPolarization      = new G4ThreeVector(aGPUHit->fPostStepPoint.fPolarization.x(),
                                                                           aGPUHit->fPostStepPoint.fPolarization.y(),
                                                                           aGPUHit->fPostStepPoint.fPolarization.z());
  const G4ThreeVector *aPostStepPointPosition          = new G4ThreeVector(aGPUHit->fPostStepPoint.fPosition.x(),
                                                                           aGPUHit->fPostStepPoint.fPosition.y(),
                                                                           aGPUHit->fPostStepPoint.fPosition.z());

  // G4Step
  aG4Step->SetStepLength(aGPUHit->fStepLength);                 // Real data
  aG4Step->SetTotalEnergyDeposit(aGPUHit->fTotalEnergyDeposit); // Real data
  // aG4Step->SetNonIonizingEnergyDeposit(0);                      // Missing data
  // aG4Step->SetControlFlag(G4SteppingControl::NormalCondition);  // Missing data
  // aG4Step->SetFirstStepFlag();                                  // Missing data
  // aG4Step->SetLastStepFlag();                                   // Missing data
  // aG4Step->SetPointerToVectorOfAuxiliaryPoints(nullptr);        // Missing data
  // aG4Step->SetSecondary(nullptr);                               // Missing data

  // G4Track
  G4Track *aTrack = aG4Step->GetTrack();
  aTrack->SetTrackID(aGPUHit->fParentID);                                                                   // Missing data
  aTrack->SetParentID(aGPUHit->fParentID); // ID of the initial particle that entered AdePT
  aTrack->SetPosition(*aPostStepPointPosition); // Real data
  // aTrack->SetGlobalTime(0);                                                                // Missing data
  // aTrack->SetLocalTime(0);                                                                 // Missing data
  // aTrack->SetProperTime(0);                                                                // Missing data
  // aTrack->SetTouchableHandle(aTrackTouchableHistory);                                      // Missing data
  // aTrack->SetNextTouchableHandle(nullptr);                                                 // Missing data
  // aTrack->SetOriginTouchableHandle(nullptr);                                               // Missing data
  // aTrack->SetKineticEnergy(aGPUHit->fPostStepPoint.fEKin);                                 // Real data
  aTrack->SetMomentumDirection(*aPostStepPointMomentumDirection); // Real data
  // aTrack->SetVelocity(0);                                                                  // Missing data
  aTrack->SetPolarization(*aPostStepPointPolarization); // Real data
  // aTrack->SetTrackStatus(G4TrackStatus::fAlive);                                           // Missing data
  // aTrack->SetBelowThresholdFlag(false);                                                    // Missing data
  // aTrack->SetGoodForTrackingFlag(false);                                                   // Missing data
  aTrack->SetStep(aG4Step);                    // Real data
  aTrack->SetStepLength(aGPUHit->fStepLength); // Real data
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
  aPreStepPoint->SetPosition(G4ThreeVector(aGPUHit->fPreStepPoint.fPosition.x(),
                                          aGPUHit->fPreStepPoint.fPosition.y(),
                                          aGPUHit->fPreStepPoint.fPosition.z())); // Real data
  // aPreStepPoint->SetLocalTime(0);                                                                // Missing data
  // aPreStepPoint->SetGlobalTime(0);                                                               // Missing data
  // aPreStepPoint->SetProperTime(0);                                                               // Missing data
  aPreStepPoint->SetMomentumDirection(G4ThreeVector(aGPUHit->fPreStepPoint.fMomentumDirection.x(),
                                                    aGPUHit->fPreStepPoint.fMomentumDirection.y(),
                                                    aGPUHit->fPreStepPoint.fMomentumDirection.z())); // Real data
  aPreStepPoint->SetKineticEnergy(aGPUHit->fPreStepPoint.fEKin);                                 // Real data
  // aPreStepPoint->SetVelocity(0);                                                                 // Missing data
  aPreStepPoint->SetTouchableHandle(aPreG4TouchableHandle);                                          // Real data
  aPreStepPoint->SetMaterial(aPreG4TouchableHandle->GetVolume()->GetLogicalVolume()->GetMaterial()); // Real data
  aPreStepPoint->SetMaterialCutsCouple(aPreG4TouchableHandle->GetVolume()->GetLogicalVolume()->GetMaterialCutsCouple());
  // aPreStepPoint->SetSensitiveDetector(nullptr);                                                  // Missing data
  // aPreStepPoint->SetSafety(0);                                                                   // Missing data
  aPreStepPoint->SetPolarization(G4ThreeVector(aGPUHit->fPreStepPoint.fPolarization.x(),
                                              aGPUHit->fPreStepPoint.fPolarization.y(),
                                              aGPUHit->fPreStepPoint.fPolarization.z())); // Real data
  // aPreStepPoint->SetStepStatus(G4StepStatus::fUndefined);                                        // Missing data
  // aPreStepPoint->SetProcessDefinedStep(nullptr);                                                 // Missing data
  // aPreStepPoint->SetMass(0);                                                                     // Missing data
  aPreStepPoint->SetCharge(aGPUHit->fPreStepPoint.fCharge); // Real data
  // aPreStepPoint->SetMagneticMoment(0);                                                           // Missing data
  // aPreStepPoint->SetWeight(0);                                                                   // Missing data

  // Post-Step Point
  G4StepPoint *aPostStepPoint = aG4Step->GetPostStepPoint();
  aPostStepPoint->SetPosition(*aPostStepPointPosition); // Real data
  // aPostStepPoint->SetLocalTime(0);                                                                 // Missing data
  // aPostStepPoint->SetGlobalTime(0);                                                                // Missing data
  // aPostStepPoint->SetProperTime(0);                                                                // Missing data
  aPostStepPoint->SetMomentumDirection(*aPostStepPointMomentumDirection); // Real data
  aPostStepPoint->SetKineticEnergy(aGPUHit->fPostStepPoint.fEKin);        // Real data
  // aPostStepPoint->SetVelocity(0);                                                                  // Missing data
  aPostStepPoint->SetTouchableHandle(aPostG4TouchableHandle);                                          // Real data
  aPostStepPoint->SetMaterial(aPostG4TouchableHandle->GetVolume()->GetLogicalVolume()->GetMaterial()); // Real data
  aPostStepPoint->SetMaterialCutsCouple(aPostG4TouchableHandle->GetVolume()->GetLogicalVolume()->GetMaterialCutsCouple());
  // aPostStepPoint->SetSensitiveDetector(nullptr);                                                   // Missing data
  // aPostStepPoint->SetSafety(0);                                                                    // Missing data
  aPostStepPoint->SetPolarization(*aPostStepPointPolarization); // Real data
  // aPostStepPoint->SetStepStatus(G4StepStatus::fUndefined);                                         // Missing data
  // aPostStepPoint->SetProcessDefinedStep(nullptr);                                                  // Missing data
  // aPostStepPoint->SetMass(0);                                                                      // Missing data
  aPostStepPoint->SetCharge(aGPUHit->fPostStepPoint.fCharge); // Real data
  // aPostStepPoint->SetMagneticMoment(0);                                                            // Missing data
  // aPostStepPoint->SetWeight(0);                                                                    // Missing data
}

void AdePTGeant4Integration::ReturnTracks(std::vector<adeptint::TrackData> *tracksFromDevice, int debugLevel)
{
  // debugLevel = 2;
  if (debugLevel > 1) {
    G4cout << "Returning " << tracksFromDevice->size() << " tracks from device" << G4endl;
  }

  constexpr double tolerance = 10. * vecgeom::kTolerance;
  int tid                    = GetThreadID();

  // Build the secondaries and put them back on the Geant4 stack
  int i = 0;
  for (auto const &track : *tracksFromDevice) {
    if (debugLevel > 1) {
      std::cout << "[" << tid << "] fromDevice[ " << i++ << "]: pdg " << track.pdg << " parent id " << track.parentID
                << " kinetic energy " << track.eKin << " position " << track.position[0] << " " << track.position[1]
                << " " << track.position[2] << " direction " << track.direction[0] << " " << track.direction[1] << " "
                << track.direction[2] << " global time, local time, proper time: "
                << "(" << track.globalTime << ", " << track.localTime << ", " << track.properTime << ")" << std::endl;
    }
    G4ParticleMomentum direction(track.direction[0], track.direction[1], track.direction[2]);

    G4DynamicParticle *dynamique =
        new G4DynamicParticle(G4ParticleTable::GetParticleTable()->FindParticle(track.pdg), direction, track.eKin);

    G4ThreeVector posi(track.position[0], track.position[1], track.position[2]);
    // The returned track will be located by Geant4. For now we need to
    // push it to make sure it is not relocated again in the GPU region
    posi += tolerance * direction;

    G4Track *secondary = new G4Track(dynamique, track.globalTime, posi);
    secondary->SetLocalTime(track.localTime);
    secondary->SetProperTime(track.properTime);
    secondary->SetParentID(track.parentID);

    G4EventManager::GetEventManager()->GetStackManager()->PushOneTrack(secondary);
  }
}

double AdePTGeant4Integration::GetUniformFieldZ()
{
  G4UniformMagField *field =
      (G4UniformMagField *)G4TransportationManager::GetTransportationManager()->GetFieldManager()->GetDetectorField();
  if (field)
    return field->GetConstantFieldValue()[2];
  else
    return 0;
}
