// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <G4Event.hh>
#include <G4EventManager.hh>
#include <G4NavigationHistory.hh>
#include <G4TouchableHistory.hh>
#include <G4Track.hh>
#include <G4VTrackingManager.hh>
#include <globals.hh>

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
#include <G4PVPlacement.hh>
#include <G4PVReplica.hh>
#include <G4ProductionCuts.hh>
#include <G4Region.hh>
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
