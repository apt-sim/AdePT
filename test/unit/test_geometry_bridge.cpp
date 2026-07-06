// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <G4EventManager.hh>
#include <G4VTrackingManager.hh>
#include <globals.hh>

#include <string>
#include <vector>

#define private public
#include <G4HepEmTrackingManager.hh>
#undef private

#include <AdePT/g4integration/geometry/AdePTGeometryBridge.hh>
#include <AdePT/g4integration/tracking_managers/G4HepEmTrackingManagerSpecialized.hh>

#include <VecGeom/management/GeoManager.h>
#include <VecGeom/volumes/LogicalVolume.h>

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
