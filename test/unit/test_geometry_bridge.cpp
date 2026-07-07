// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <G4Event.hh>
#include <G4EventManager.hh>
#include <G4NavigationHistory.hh>
#include <G4TouchableHistory.hh>
#include <G4Track.hh>
#include <G4VTrackingManager.hh>
#include <globals.hh>

#include <array>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#define private public
#include <G4HepEmTrackingManager.hh>
#undef private

#include <AdePT/g4integration/geometry/AdePTGeometryBridge.hh>
#include <AdePT/g4integration/returned_steps/HostTrackDataMapper.hh>
#include <AdePT/g4integration/tracking_managers/G4HepEmTrackingManagerSpecialized.hh>
#include <AdePT/transport/steps/GPUStep.hh>

#define private public
#include <AdePT/g4integration/returned_steps/AdePTGeant4Integration.hh>
#undef private

#define private public
#include <AdePT/g4integration/AdePTTrackingManager.hh>
#undef private

#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/UnplacedBox.h>

#include <G4HepEmData.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmWoodcockHelper.hh>

#include <G4Box.hh>
#include <G4GeometryManager.hh>
#include <G4LogicalVolume.hh>
#include <G4MaterialCutsCouple.hh>
#include <G4NistManager.hh>
#include <G4PVParameterised.hh>
#include <G4PVPlacement.hh>
#include <G4PVReplica.hh>
#include <G4VPVParameterisation.hh>
#include <G4ProductionCuts.hh>
#include <G4Region.hh>
#include <G4ReplicaNavigation.hh>
#include <G4SystemOfUnits.hh>
#include <G4TransportationManager.hh>

#include <gtest/gtest.h>

namespace {

struct GeometryCleanup {
  ~GeometryCleanup()
  {
    G4GeometryManager::GetInstance()->OpenGeometry();
    vecgeom::GeoManager::Instance().Clear();
  }
};

class SingleCopyOffsetParameterisation : public G4VPVParameterisation {
public:
  explicit SingleCopyOffsetParameterisation(G4ThreeVector translation) : fTranslation(translation) {}

  void ComputeTransformation(const G4int copyNo, G4VPhysicalVolume *physicalVolume) const override
  {
    physicalVolume->SetTranslation(fTranslation);
    physicalVolume->SetRotation(nullptr);
    physicalVolume->SetCopyNo(copyNo);
  }

private:
  G4ThreeVector fTranslation;
};

struct SingleMatCutHepEmData {
  int g4ToHepEm[1]                  = {0};
  G4HepEmMCCData hepEmMatCutData[1] = {G4HepEmMCCData{}};
  G4HepEmMatCutData matCutData{};
  G4HepEmData hepEmData{};

  SingleMatCutHepEmData()
  {
    hepEmMatCutData[0].fG4MatCutIndex   = 0;
    matCutData.fNumG4MatCuts            = 1;
    matCutData.fNumMatCutData           = 1;
    matCutData.fG4MCIndexToHepEmMCIndex = g4ToHepEm;
    matCutData.fMatCutData              = hepEmMatCutData;
    hepEmData.fTheMatCutData            = &matCutData;
  }
};

vecgeom::Transformation3D MakeVecGeomRotation(G4RotationMatrix const &rotation)
{
  return vecgeom::Transformation3D(0.0, 0.0, 0.0, rotation(0, 0), rotation(1, 0), rotation(2, 0), rotation(0, 1),
                                   rotation(1, 1), rotation(2, 1), rotation(0, 2), rotation(1, 2), rotation(2, 2));
}

void Require(bool condition, std::string const &message)
{
  if (!condition) throw std::runtime_error(message);
}

struct HandoffNavStateSummary {
  int copyNo = -1;
  std::array<double, 3> translation{};
};

HandoffNavStateSummary SummarizeTopPlacement(vecgeom::NavigationState const &state)
{
  Require(state.GetLevel() == 1, "converted handoff state does not have exactly one daughter level");
  auto const *top = state.Top();
  Require(top != nullptr, "converted handoff state has no top placement");

  auto const *transform = top->GetTransformation();
  return HandoffNavStateSummary{top->GetCopyNo(),
                                {{transform->Translation(0), transform->Translation(1), transform->Translation(2)}}};
}

} // namespace

// Flattened replica traversal: G4VG expands one Geant4 replica daughter into
// multiple concrete VecGeom placements. InitVolAuxData must visit each copy so
// WDT root collection does not silently keep only replica copy 0.
TEST(AdePTGeometryBridge, InitVolAuxDataIncludesAllReplicatedWdtRoots)
{
  GeometryCleanup cleanup;
  G4GeometryManager::GetInstance()->OpenGeometry();
  vecgeom::GeoManager::Instance().Clear();

  auto *material = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
  ASSERT_NE(material, nullptr);

  auto *worldSolid = new G4Box("replica_world_solid", 150.0 * mm, 10.0 * mm, 10.0 * mm);
  auto *worldLV    = new G4LogicalVolume(worldSolid, material, "replica_world_lv");
  auto *worldPV    = new G4PVPlacement(nullptr, G4ThreeVector(), worldLV, "replica_world", nullptr, false, 0, false);

  auto *sliceSolid = new G4Box("slice_solid", 50.0 * mm, 10.0 * mm, 10.0 * mm);
  auto *sliceLV    = new G4LogicalVolume(sliceSolid, material, "slice_lv");
  new G4PVReplica("slice", sliceLV, worldLV, kXAxis, 3, 100.0 * mm);

  auto *couple = new G4MaterialCutsCouple(material, new G4ProductionCuts);
  couple->SetIndex(0);
  couple->SetUseFlag(true);
  worldLV->SetMaterialCutsCouple(couple);
  sliceLV->SetMaterialCutsCouple(couple);

  G4Region worldRegion("geometry_bridge_world_region");
  worldRegion.AddRootLogicalVolume(worldLV);
  worldRegion.RegisterMaterialCouplePair(material, couple);

  G4Region wdtRegion("geometry_bridge_wdt_region");
  wdtRegion.AddRootLogicalVolume(sliceLV);
  wdtRegion.RegisterMaterialCouplePair(material, couple);
  ASSERT_EQ(wdtRegion.FindCouple(material), couple);

  auto *transport = G4TransportationManager::GetTransportationManager();
  transport->SetWorldForTracking(worldPV);
  G4GeometryManager::GetInstance()->CloseGeometry(true);

  AdePTGeometryBridge::CreateVecGeomWorld(worldPV);
  auto const *vgWorld = vecgeom::GeoManager::Instance().GetWorld();
  ASSERT_NE(vgWorld, nullptr);

  auto const &vgDaughters = vgWorld->GetLogicalVolume()->GetDaughters();
  ASSERT_EQ(vgDaughters.size(), 3u);

  int g4ToHepEm[]                   = {0};
  G4HepEmMCCData hepEmMatCutData[]  = {G4HepEmMCCData{}};
  hepEmMatCutData[0].fG4MatCutIndex = 0;
  hepEmMatCutData[0].fG4RegionIndex = wdtRegion.GetInstanceID();

  G4HepEmMatCutData matCutData;
  matCutData.fNumG4MatCuts            = 1;
  matCutData.fNumMatCutData           = 1;
  matCutData.fG4MCIndexToHepEmMCIndex = g4ToHepEm;
  matCutData.fMatCutData              = hepEmMatCutData;

  G4HepEmData hepEmData;
  hepEmData.fTheMatCutData = &matCutData;

  G4HepEmTrackingManagerSpecialized hepEmTM;
  hepEmTM.fWDTHelper = new G4HepEmWoodcockHelper;
  std::vector<std::string> wdtRegionNames{wdtRegion.GetName()};
  ASSERT_TRUE(hepEmTM.fWDTHelper->Initialize(wdtRegionNames, &matCutData, worldPV));

  std::vector<adeptint::VolAuxData> volAuxData(vecgeom::GeoManager::Instance().GetRegisteredVolumesCount());
  std::vector<std::string> gpuRegionNames;
  std::vector<std::string> deadRegionNames;
  adeptint::WDTHostRaw wdtRaw;

  AdePTGeometryBridge::InitVolAuxData(volAuxData.data(), &hepEmData, &hepEmTM, true, &gpuRegionNames, deadRegionNames,
                                      wdtRaw);

  const auto regionRoots = wdtRaw.regionToRootIndices.find(wdtRegion.GetInstanceID());
  ASSERT_NE(regionRoots, wdtRaw.regionToRootIndices.end());

  // The G4VG conversion flattens the three Geant4 replica copies into three
  // VecGeom placements. The previous one-for-one recursion visited only copy 0,
  // so this collected one WDT root instead of all three.
  EXPECT_EQ(regionRoots->second.size(), 3u);
  EXPECT_EQ(wdtRaw.roots.size(), 3u);
  for (auto const &root : wdtRaw.roots) {
    EXPECT_EQ(root.hepemIMC, 0);
  }
}

// G4-to-VecGeom handoff for flattened replicas: a Geant4 history in replica
// copy 2 must choose the same VecGeom slice as an equivalent explicit-placement
// geometry. Matching only by Geant4 daughter index incorrectly selected copy 0.
TEST(AdePTGeometryBridge, G4HandoffReplicaCopyMatchesEquivalentExplicitPlacement)
{
  constexpr int selectedCopy = 2;

  auto convertReplicaCopy = [](int copyNo) {
    GeometryCleanup cleanup;
    G4GeometryManager::GetInstance()->OpenGeometry();
    vecgeom::GeoManager::Instance().Clear();

    auto *material = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
    Require(material != nullptr, "G4_AIR material was not available");

    auto *worldSolid = new G4Box("handoff_replica_world_solid", 150.0 * mm, 10.0 * mm, 10.0 * mm);
    auto *worldLV    = new G4LogicalVolume(worldSolid, material, "handoff_replica_world_lv");
    auto *worldPV =
        new G4PVPlacement(nullptr, G4ThreeVector(), worldLV, "handoff_replica_world", nullptr, false, 0, false);

    auto *sliceSolid = new G4Box("handoff_replica_slice_solid", 50.0 * mm, 10.0 * mm, 10.0 * mm);
    auto *sliceLV    = new G4LogicalVolume(sliceSolid, material, "handoff_replica_slice_lv");
    auto *replicaPV  = new G4PVReplica("handoff_replica_slice", sliceLV, worldLV, kXAxis, 3, 100.0 * mm);

    auto *transport = G4TransportationManager::GetTransportationManager();
    transport->SetWorldForTracking(worldPV);
    G4GeometryManager::GetInstance()->CloseGeometry(true);

    AdePTGeometryBridge::CreateVecGeomWorld(worldPV);
    auto const *vgWorld = vecgeom::GeoManager::Instance().GetWorld();
    Require(vgWorld != nullptr, "VecGeom replica world was not created");
    Require(vgWorld->GetLogicalVolume()->GetDaughters().size() == 3, "replica conversion did not produce 3 copies");

    G4ReplicaNavigation replicaNavigation;
    replicaNavigation.ComputeTransformation(copyNo, replicaPV);
    replicaPV->SetCopyNo(copyNo);

    G4NavigationHistory history;
    history.SetFirstEntry(worldPV);
    history.NewLevel(replicaPV, kReplica, copyNo);

    AdePTTrackingManager trackingManager(nullptr);
    return SummarizeTopPlacement(trackingManager.GetVecGeomFromG4State(history));
  };

  auto convertExplicitCopy = [](int copyNo) {
    GeometryCleanup cleanup;
    G4GeometryManager::GetInstance()->OpenGeometry();
    vecgeom::GeoManager::Instance().Clear();

    auto *material = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
    Require(material != nullptr, "G4_AIR material was not available");

    auto *worldSolid = new G4Box("handoff_explicit_world_solid", 150.0 * mm, 10.0 * mm, 10.0 * mm);
    auto *worldLV    = new G4LogicalVolume(worldSolid, material, "handoff_explicit_world_lv");
    auto *worldPV =
        new G4PVPlacement(nullptr, G4ThreeVector(), worldLV, "handoff_explicit_world", nullptr, false, 0, false);

    auto *sliceSolid = new G4Box("handoff_explicit_slice_solid", 50.0 * mm, 10.0 * mm, 10.0 * mm);
    auto *sliceLV    = new G4LogicalVolume(sliceSolid, material, "handoff_explicit_slice_lv");
    std::array<G4VPhysicalVolume *, 3> slicePVs{};
    for (int id = 0; id < 3; ++id) {
      const G4double x = -100.0 * mm + id * 100.0 * mm;
      slicePVs[id] = new G4PVPlacement(nullptr, G4ThreeVector(x, 0.0, 0.0), sliceLV, "handoff_explicit_slice", worldLV,
                                       false, id, false);
    }

    auto *transport = G4TransportationManager::GetTransportationManager();
    transport->SetWorldForTracking(worldPV);
    G4GeometryManager::GetInstance()->CloseGeometry(true);

    AdePTGeometryBridge::CreateVecGeomWorld(worldPV);
    auto const *vgWorld = vecgeom::GeoManager::Instance().GetWorld();
    Require(vgWorld != nullptr, "VecGeom explicit world was not created");
    Require(vgWorld->GetLogicalVolume()->GetDaughters().size() == 3,
            "explicit placement conversion did not produce 3 copies");

    G4NavigationHistory history;
    history.SetFirstEntry(worldPV);
    history.NewLevel(slicePVs[copyNo], kNormal, slicePVs[copyNo]->GetCopyNo());

    AdePTTrackingManager trackingManager(nullptr);
    return SummarizeTopPlacement(trackingManager.GetVecGeomFromG4State(history));
  };

  const auto replicaSummary  = convertReplicaCopy(selectedCopy);
  const auto explicitSummary = convertExplicitCopy(selectedCopy);

  EXPECT_EQ(replicaSummary.copyNo, explicitSummary.copyNo);
  for (std::size_t i = 0; i < replicaSummary.translation.size(); ++i) {
    EXPECT_NEAR(replicaSummary.translation[i], explicitSummary.translation[i], 1.e-9);
  }
}

// Single-copy parameterised traversal: GetMultiplicity() is one, but the
// copy-0 transform still comes from the Geant4 parameterisation. CheckGeometry
// must compute that transform instead of comparing the mutable prototype state.
TEST(AdePTGeometryBridge, CheckGeometryUsesCopyTransformForSingleCopyParameterisation)
{
  GeometryCleanup cleanup;
  G4GeometryManager::GetInstance()->OpenGeometry();
  vecgeom::GeoManager::Instance().Clear();

  auto *material = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
  ASSERT_NE(material, nullptr);

  auto *worldSolid = new G4Box("single_param_world_solid", 100.0 * mm, 20.0 * mm, 20.0 * mm);
  auto *worldLV    = new G4LogicalVolume(worldSolid, material, "single_param_world_lv");
  auto *worldPV = new G4PVPlacement(nullptr, G4ThreeVector(), worldLV, "single_param_world", nullptr, false, 0, false);

  auto *sliceSolid = new G4Box("single_param_slice_solid", 5.0 * mm, 5.0 * mm, 5.0 * mm);
  auto *sliceLV    = new G4LogicalVolume(sliceSolid, material, "single_param_slice_lv");
  const G4ThreeVector copyTranslation(35.0 * mm, 0.0, 0.0);
  auto *parameterisation = new SingleCopyOffsetParameterisation(copyTranslation);
  auto *parameterisedPV =
      new G4PVParameterised("single_param_slice", sliceLV, worldLV, kXAxis, 1, parameterisation, false);

  auto *couple = new G4MaterialCutsCouple(material, new G4ProductionCuts);
  couple->SetIndex(0);
  couple->SetUseFlag(true);
  worldLV->SetMaterialCutsCouple(couple);
  sliceLV->SetMaterialCutsCouple(couple);

  auto *transport = G4TransportationManager::GetTransportationManager();
  transport->SetWorldForTracking(worldPV);

  auto *vgWorldSolid = new vecgeom::UnplacedBox(100.0, 20.0, 20.0);
  auto *vgWorldLV    = new vecgeom::LogicalVolume("single_param_world_lv", vgWorldSolid);
  auto *vgSliceSolid = new vecgeom::UnplacedBox(5.0, 5.0, 5.0);
  auto *vgSliceLV    = new vecgeom::LogicalVolume("single_param_slice_lv", vgSliceSolid);

  vecgeom::Transformation3D childTransform(35.0, 0.0, 0.0);
  auto const *vgSlicePV = vgWorldLV->PlaceDaughter("single_param_slice", vgSliceLV, &childTransform);
  const_cast<vecgeom::VPlacedVolume *>(vgSlicePV)->SetCopyNo(0);
  vecgeom::Transformation3D identity;
  auto *vgWorldPV = vgWorldLV->Place("single_param_world", &identity);
  vecgeom::GeoManager::Instance().SetWorldAndClose(vgWorldPV);

  auto const *vgWorld = vecgeom::GeoManager::Instance().GetWorld();
  ASSERT_NE(vgWorld, nullptr);
  ASSERT_EQ(vgWorld->GetLogicalVolume()->GetDaughters().size(), 1u);
  EXPECT_EQ(parameterisedPV->VolumeType(), kParameterised);
  EXPECT_EQ(parameterisedPV->GetMultiplicity(), 1);
  EXPECT_NE(parameterisedPV->GetTranslation(), copyTranslation);

  SingleMatCutHepEmData hepEm;
  EXPECT_NO_THROW(AdePTGeometryBridge::CheckGeometry(&hepEm.hepEmData));
  EXPECT_EQ(parameterisedPV->GetCopyNo(), 0);
  EXPECT_EQ(parameterisedPV->GetTranslation(), copyTranslation);
}

// Single-copy replica traversal: a kPhi replica can have one copy and still
// require G4ReplicaNavigation to stamp a non-identity copy-0 rotation. The
// traversal flag must follow the Geant4 volume type, not only multiplicity > 1.
TEST(AdePTGeometryBridge, CheckGeometryUsesCopyTransformForSingleCopyReplica)
{
  GeometryCleanup cleanup;
  G4GeometryManager::GetInstance()->OpenGeometry();
  vecgeom::GeoManager::Instance().Clear();

  auto *material = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
  ASSERT_NE(material, nullptr);

  auto *worldSolid = new G4Box("single_replica_world_solid", 100.0 * mm, 20.0 * mm, 20.0 * mm);
  auto *worldLV    = new G4LogicalVolume(worldSolid, material, "single_replica_world_lv");
  auto *worldPV =
      new G4PVPlacement(nullptr, G4ThreeVector(), worldLV, "single_replica_world", nullptr, false, 0, false);

  auto *sliceSolid = new G4Box("single_replica_slice_solid", 5.0 * mm, 5.0 * mm, 5.0 * mm);
  auto *sliceLV    = new G4LogicalVolume(sliceSolid, material, "single_replica_slice_lv");
  auto *replicaPV  = new G4PVReplica("single_replica_slice", sliceLV, worldLV, kPhi, 1, 0.5, 0.25);

  auto *couple = new G4MaterialCutsCouple(material, new G4ProductionCuts);
  couple->SetIndex(0);
  couple->SetUseFlag(true);
  worldLV->SetMaterialCutsCouple(couple);
  sliceLV->SetMaterialCutsCouple(couple);

  auto *transport = G4TransportationManager::GetTransportationManager();
  transport->SetWorldForTracking(worldPV);

  G4ReplicaNavigation replicaNavigation;
  replicaNavigation.ComputeTransformation(0, replicaPV);
  ASSERT_NE(replicaPV->GetRotation(), nullptr);
  const G4RotationMatrix copyRotation = *replicaPV->GetRotation();
  *replicaPV->GetRotation()           = G4RotationMatrix{};
  ASSERT_GT(std::abs(copyRotation(0, 1) - (*replicaPV->GetRotation())(0, 1)), 1.e-12);

  auto *vgWorldSolid = new vecgeom::UnplacedBox(100.0, 20.0, 20.0);
  auto *vgWorldLV    = new vecgeom::LogicalVolume("single_replica_world_lv", vgWorldSolid);
  auto *vgSliceSolid = new vecgeom::UnplacedBox(5.0, 5.0, 5.0);
  auto *vgSliceLV    = new vecgeom::LogicalVolume("single_replica_slice_lv", vgSliceSolid);

  auto childTransform   = MakeVecGeomRotation(copyRotation);
  auto const *vgSlicePV = vgWorldLV->PlaceDaughter("single_replica_slice", vgSliceLV, &childTransform);
  const_cast<vecgeom::VPlacedVolume *>(vgSlicePV)->SetCopyNo(0);
  vecgeom::Transformation3D identity;
  auto *vgWorldPV = vgWorldLV->Place("single_replica_world", &identity);
  vecgeom::GeoManager::Instance().SetWorldAndClose(vgWorldPV);

  auto const *vgWorld = vecgeom::GeoManager::Instance().GetWorld();
  ASSERT_NE(vgWorld, nullptr);
  ASSERT_EQ(vgWorld->GetLogicalVolume()->GetDaughters().size(), 1u);
  EXPECT_EQ(replicaPV->VolumeType(), kReplica);
  EXPECT_EQ(replicaPV->GetMultiplicity(), 1);

  SingleMatCutHepEmData hepEm;
  EXPECT_NO_THROW(AdePTGeometryBridge::CheckGeometry(&hepEm.hepEmData));
  EXPECT_NEAR((*replicaPV->GetRotation())(0, 1), copyRotation(0, 1), 1.e-12);
}

// Returned-touchable reconstruction for flattened replicas: multiple VecGeom
// placements map back to one mutable Geant4 replica physical volume. Reusing a
// G4NavigationHistory level must compare and stamp the replica copy identity.
TEST(AdePTGeometryBridge, ReconstructedTouchableKeepsReplicaCopyNumber)
{
  GeometryCleanup cleanup;
  G4GeometryManager::GetInstance()->OpenGeometry();
  vecgeom::GeoManager::Instance().Clear();

  auto *material = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
  ASSERT_NE(material, nullptr);

  auto *worldSolid = new G4Box("touchable_replica_world_solid", 150.0 * mm, 10.0 * mm, 10.0 * mm);
  auto *worldLV    = new G4LogicalVolume(worldSolid, material, "touchable_replica_world_lv");
  auto *worldPV =
      new G4PVPlacement(nullptr, G4ThreeVector(), worldLV, "touchable_replica_world", nullptr, false, 0, false);

  auto *sliceSolid = new G4Box("touchable_slice_solid", 50.0 * mm, 10.0 * mm, 10.0 * mm);
  auto *sliceLV    = new G4LogicalVolume(sliceSolid, material, "touchable_slice_lv");
  new G4PVReplica("touchable_slice", sliceLV, worldLV, kXAxis, 3, 100.0 * mm);

  auto *transport = G4TransportationManager::GetTransportationManager();
  transport->SetWorldForTracking(worldPV);
  G4GeometryManager::GetInstance()->CloseGeometry(true);

  AdePTGeometryBridge::CreateVecGeomWorld(worldPV);
  auto const *vgWorld = vecgeom::GeoManager::Instance().GetWorld();
  ASSERT_NE(vgWorld, nullptr);

  auto const &vgDaughters = vgWorld->GetLogicalVolume()->GetDaughters();
  ASSERT_EQ(vgDaughters.size(), 3u);
  ASSERT_EQ(vgDaughters[0]->GetCopyNo(), 0);
  ASSERT_EQ(vgDaughters[2]->GetCopyNo(), 2);
  ASSERT_EQ(AdePTGeometryBridge::GetG4PhysicalVolume(vgDaughters[0]),
            AdePTGeometryBridge::GetG4PhysicalVolume(vgDaughters[2]));

  const auto makeNavState = [&](std::size_t replicaCopy) {
    vecgeom::NavigationState state;
    state.Push(vgWorld);
    state.Push(vgDaughters[replicaCopy]);
    state.SetBoundaryState(false);
    return state;
  };

  AdePTGeant4Integration integration;
  G4NavigationHistory history;

  integration.FillG4NavigationHistory(makeNavState(0), history);
  ASSERT_EQ(history.GetDepth(), 1u);
  EXPECT_EQ(history.GetVolumeType(1), kReplica);
  EXPECT_EQ(history.GetReplicaNo(1), 0);

  G4TouchableHistory touchableCopy0(history);
  auto *replicaVolume = touchableCopy0.GetVolume(0);
  const auto copy0    = touchableCopy0.GetCopyNumber(0);
  EXPECT_EQ(copy0, 0);
  EXPECT_EQ(replicaVolume->GetCopyNo(), 0);

  integration.FillG4NavigationHistory(makeNavState(2), history);
  ASSERT_EQ(history.GetDepth(), 1u);
  EXPECT_EQ(history.GetVolume(1), replicaVolume);
  EXPECT_EQ(history.GetVolumeType(1), kReplica);
  EXPECT_EQ(history.GetReplicaNo(1), 2);

  G4TouchableHistory touchableCopy2(history);
  EXPECT_EQ(touchableCopy2.GetVolume(0), replicaVolume);
  EXPECT_EQ(touchableCopy2.GetCopyNumber(0), 2);
  EXPECT_EQ(replicaVolume->GetCopyNo(), 2);
  EXPECT_NE(copy0, touchableCopy2.GetCopyNumber(0));
}

// Non-equivalent unexpanded replica guard: a single ordinary VecGeom
// placement for a multi-copy G4PVReplica is not a valid preserved replica
// representation. Traversal must reject it instead of visiting copy 0 only.
TEST(AdePTGeometryBridge, RejectsUnexpandedReplicaWithoutVecGeomCopies)
{
  GeometryCleanup cleanup;
  G4GeometryManager::GetInstance()->OpenGeometry();
  vecgeom::GeoManager::Instance().Clear();

  auto *material = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
  ASSERT_NE(material, nullptr);

  auto *worldSolid = new G4Box("unexpanded_replica_world_solid", 150.0 * mm, 10.0 * mm, 10.0 * mm);
  auto *worldLV    = new G4LogicalVolume(worldSolid, material, "unexpanded_replica_world_lv");
  auto *worldPV =
      new G4PVPlacement(nullptr, G4ThreeVector(), worldLV, "unexpanded_replica_world", nullptr, false, 0, false);

  auto *sliceSolid = new G4Box("unexpanded_slice_solid", 50.0 * mm, 10.0 * mm, 10.0 * mm);
  auto *sliceLV    = new G4LogicalVolume(sliceSolid, material, "unexpanded_slice_lv");
  new G4PVReplica("unexpanded_slice", sliceLV, worldLV, kXAxis, 3, 100.0 * mm);

  auto *couple = new G4MaterialCutsCouple(material, new G4ProductionCuts);
  couple->SetIndex(0);
  couple->SetUseFlag(true);
  worldLV->SetMaterialCutsCouple(couple);
  sliceLV->SetMaterialCutsCouple(couple);

  G4Region worldRegion("geometry_bridge_unexpanded_world_region");
  worldRegion.AddRootLogicalVolume(worldLV);
  worldRegion.RegisterMaterialCouplePair(material, couple);

  G4Region wdtRegion("geometry_bridge_unexpanded_wdt_region");
  wdtRegion.AddRootLogicalVolume(sliceLV);
  wdtRegion.RegisterMaterialCouplePair(material, couple);
  ASSERT_EQ(wdtRegion.FindCouple(material), couple);

  auto *transport = G4TransportationManager::GetTransportationManager();
  transport->SetWorldForTracking(worldPV);
  G4GeometryManager::GetInstance()->CloseGeometry(true);

  auto *vgWorldSolid = new vecgeom::UnplacedBox(150.0, 10.0, 10.0);
  auto *vgWorldLV    = new vecgeom::LogicalVolume("unexpanded_replica_world_lv", vgWorldSolid);
  auto *vgSliceSolid = new vecgeom::UnplacedBox(50.0, 10.0, 10.0);
  auto *vgSliceLV    = new vecgeom::LogicalVolume("unexpanded_slice_lv", vgSliceSolid);

  vecgeom::Transformation3D identity;
  vgWorldLV->PlaceDaughter("unexpanded_slice", vgSliceLV, &identity);
  auto *vgWorldPV = vgWorldLV->Place("unexpanded_replica_world", &identity);
  vecgeom::GeoManager::Instance().SetWorldAndClose(vgWorldPV);

  auto const *vgWorld = vecgeom::GeoManager::Instance().GetWorld();
  ASSERT_NE(vgWorld, nullptr);

  auto const &vgDaughters = vgWorld->GetLogicalVolume()->GetDaughters();
  ASSERT_EQ(vgDaughters.size(), 1u);

  int g4ToHepEm[]                   = {0};
  G4HepEmMCCData hepEmMatCutData[]  = {G4HepEmMCCData{}};
  hepEmMatCutData[0].fG4MatCutIndex = 0;
  hepEmMatCutData[0].fG4RegionIndex = wdtRegion.GetInstanceID();

  G4HepEmMatCutData matCutData;
  matCutData.fNumG4MatCuts            = 1;
  matCutData.fNumMatCutData           = 1;
  matCutData.fG4MCIndexToHepEmMCIndex = g4ToHepEm;
  matCutData.fMatCutData              = hepEmMatCutData;

  G4HepEmData hepEmData;
  hepEmData.fTheMatCutData = &matCutData;

  EXPECT_THROW(AdePTGeometryBridge::CheckGeometry(&hepEmData), std::runtime_error);

  G4HepEmTrackingManagerSpecialized hepEmTM;
  hepEmTM.fWDTHelper = new G4HepEmWoodcockHelper;
  std::vector<std::string> wdtRegionNames{wdtRegion.GetName()};
  ASSERT_TRUE(hepEmTM.fWDTHelper->Initialize(wdtRegionNames, &matCutData, worldPV));

  std::vector<adeptint::VolAuxData> volAuxData(vecgeom::GeoManager::Instance().GetRegisteredVolumesCount());
  std::vector<std::string> gpuRegionNames;
  std::vector<std::string> deadRegionNames;
  adeptint::WDTHostRaw wdtRaw;

  EXPECT_THROW(AdePTGeometryBridge::InitVolAuxData(volAuxData.data(), &hepEmData, &hepEmTM, true, &gpuRegionNames,
                                                   deadRegionNames, wdtRaw),
               std::runtime_error);
}
