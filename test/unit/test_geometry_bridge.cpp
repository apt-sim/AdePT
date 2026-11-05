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
#include <algorithm>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <type_traits>
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
  ASSERT_TRUE(hepEmTM.fWDTHelper->Initialize(wdtRegionNames, &matCutData));

  std::vector<adeptint::VolAuxData> volAuxData(vecgeom::GeoManager::Instance().GetRegisteredVolumesCount());
  std::set<G4Region const *> gpuRegions{&worldRegion, &wdtRegion};
  CMSSteppingActionConfig cmsConfig;
  adeptint::WDTHostRaw wdtRaw;

  AdePTGeometryBridge::InitVolAuxData(volAuxData.data(), &hepEmData, &hepEmTM, gpuRegions, cmsConfig, wdtRaw);

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

namespace {
// A has two root LVs, the first placed twice. B/C occur below those roots and
// B has another placement outside A. Material-cut indices deliberately differ
// from the G4HepEm indices so inherited reference and actual couples cannot be confused.
struct NestedWdtGeometry {
  GeometryCleanup cleanup;
  // Geant4's RegionStore also owns regions at process exit (including death-test exits).
  std::unique_ptr<G4Region> worldOwner{new G4Region("nested_wdt_world")}, aOwner{new G4Region("nested_wdt_A")};
  std::unique_ptr<G4Region> bOwner{new G4Region("nested_wdt_B")}, cOwner{new G4Region("nested_wdt_C")};
  G4Region &worldRegion{*worldOwner}, &regionA{*aOwner}, &regionB{*bOwner}, &regionC{*cOwner};
  G4LogicalVolume *worldLV{}, *rootLV{}, *otherRootLV{}, *regionBLV{}, *regionCLV{};
  int g4ToHepEm[5] = {4, 3, 2, 1, 0};
  G4HepEmMCCData couples[5]{};
  G4HepEmMatCutData matCutData{};
  G4HepEmData hepEmData{};
  std::unique_ptr<G4HepEmTrackingManagerSpecialized> trackingManager;

  explicit NestedWdtGeometry(bool nestInSecondRoot = true)
  {
    G4GeometryManager::GetInstance()->OpenGeometry();
    vecgeom::GeoManager::Instance().Clear();
    auto *nist      = G4NistManager::Instance();
    auto *air       = nist->FindOrBuildMaterial("G4_AIR");
    auto *aluminium = nist->FindOrBuildMaterial("G4_Al");
    auto *lead      = nist->FindOrBuildMaterial("G4_Pb");
    auto *iron      = nist->FindOrBuildMaterial("G4_Fe");
    auto makeVolume = [](const char *name, double halfLength, G4Material *material) {
      return new G4LogicalVolume(new G4Box(name, halfLength * mm, halfLength * mm, halfLength * mm), material, name);
    };
    worldLV       = makeVolume("nested_wdt_world_lv", 200., air);
    rootLV        = makeVolume("nested_wdt_root_lv", 20., air);
    otherRootLV   = makeVolume("nested_wdt_other_root_lv", 20., iron);
    regionBLV     = makeVolume("nested_wdt_B_lv", 5., aluminium);
    regionCLV     = makeVolume("nested_wdt_C_lv", 1., lead);
    auto *worldPV = new G4PVPlacement(nullptr, {}, worldLV, "nested_wdt_world", nullptr, false, 0, false);
    new G4PVPlacement(nullptr, {-100. * mm, 0., 0.}, rootLV, "nested_wdt_root_0", worldLV, false, 0, false);
    new G4PVPlacement(nullptr, {-40. * mm, 0., 0.}, rootLV, "nested_wdt_root_1", worldLV, false, 1, false);
    new G4PVPlacement(nullptr, {40. * mm, 0., 0.}, otherRootLV, "nested_wdt_other_root", worldLV, false, 2, false);
    new G4PVPlacement(nullptr, {}, regionBLV, "nested_wdt_B", rootLV, false, 0, false);
    if (nestInSecondRoot) new G4PVPlacement(nullptr, {}, regionBLV, "nested_wdt_B", otherRootLV, false, 0, false);
    new G4PVPlacement(nullptr, {100. * mm, 0., 0.}, regionBLV, "nested_wdt_B_outside", worldLV, false, 3, false);
    new G4PVPlacement(nullptr, {}, regionCLV, "nested_wdt_C", regionBLV, false, 0, false);

    worldRegion.AddRootLogicalVolume(worldLV);
    regionA.AddRootLogicalVolume(rootLV);
    regionA.AddRootLogicalVolume(otherRootLV);
    regionB.AddRootLogicalVolume(regionBLV);
    regionC.AddRootLogicalVolume(regionCLV);
    const std::array<G4LogicalVolume *, 5> volumes{{worldLV, rootLV, regionBLV, regionCLV, otherRootLV}};
    const std::array<G4Region *, 5> regions{{&worldRegion, &regionA, &regionB, &regionC, &regionA}};
    for (int g4Index = 0; g4Index < 5; ++g4Index) {
      auto *cuts = new G4ProductionCuts;
      cuts->SetProductionCut((g4Index + 1) * 0.1 * mm);
      auto *couple = new G4MaterialCutsCouple(volumes[g4Index]->GetMaterial(), cuts);
      couple->SetIndex(g4Index);
      couple->SetUseFlag(true);
      volumes[g4Index]->SetMaterialCutsCouple(couple);
      regions[g4Index]->RegisterMaterialCouplePair(volumes[g4Index]->GetMaterial(), couple);
      const int hepemIndex               = g4ToHepEm[g4Index];
      couples[hepemIndex].fG4MatCutIndex = g4Index;
      couples[hepemIndex].fG4RegionIndex = regions[g4Index]->GetInstanceID();
    }
    matCutData.fNumG4MatCuts = matCutData.fNumMatCutData = 5;
    matCutData.fG4MCIndexToHepEmMCIndex                  = g4ToHepEm;
    matCutData.fMatCutData                               = couples;
    hepEmData.fTheMatCutData                             = &matCutData;
    G4TransportationManager::GetTransportationManager()->SetWorldForTracking(worldPV);
    G4GeometryManager::GetInstance()->CloseGeometry(true);
    trackingManager             = std::make_unique<G4HepEmTrackingManagerSpecialized>();
    trackingManager->fWDTHelper = new G4HepEmWoodcockHelper;
    Require(trackingManager->fWDTHelper->Initialize({regionA.GetName()}, &matCutData), "WDT initialization failed");
    AdePTGeometryBridge::CreateVecGeomWorld(worldPV);
  }

  adeptint::WDTHostRaw Collect(std::vector<adeptint::VolAuxData> &aux)
  {
    return Collect(aux, {&worldRegion, &regionA, &regionB, &regionC});
  }

  adeptint::WDTHostRaw Collect(std::vector<adeptint::VolAuxData> &aux, std::set<G4Region const *> const &gpuRegions)
  {
    aux.resize(vecgeom::GeoManager::Instance().GetRegisteredVolumesCount());
    adeptint::WDTHostRaw raw;
    AdePTGeometryBridge::InitVolAuxData(aux.data(), &hepEmData, trackingManager.get(), gpuRegions, {}, raw);
    return raw;
  }
};
} // namespace

TEST(AdePTGeometryBridge, InheritsWdtRootsAcrossNestedCutRegionsAndPlacements)
{
  NestedWdtGeometry geometry;
  std::vector<adeptint::VolAuxData> aux;
  const auto raw = geometry.Collect(aux);
  ASSERT_EQ(raw.roots.size(), 3u);
  const auto &aRoots = raw.regionToRootIndices.at(geometry.regionA.GetInstanceID());
  EXPECT_EQ(aRoots.size(), 3u);
  EXPECT_EQ(raw.regionToRootIndices.at(geometry.regionB.GetInstanceID()), aRoots);
  EXPECT_EQ(raw.regionToRootIndices.at(geometry.regionC.GetInstanceID()), aRoots);
  std::set<std::decay_t<decltype(raw.roots.front().root.GetState())>> rootIndices;
  for (const auto &root : raw.roots) {
    EXPECT_EQ(root.root.GetLevel(), 1);
    EXPECT_EQ(root.hepemIMC, 1); // The maximum-density material exists only in C.
    rootIndices.insert(root.root.GetState());
  }
  EXPECT_EQ(rootIndices.size(), 3u);

  const auto *vgWorld = vecgeom::GeoManager::Instance().GetWorld();
  const auto *vgRoot  = vgWorld->GetLogicalVolume()->GetDaughters()[0];
  const auto *vgB     = vgRoot->GetLogicalVolume()->GetDaughters()[0];
  const auto *vgC     = vgB->GetLogicalVolume()->GetDaughters()[0];
  EXPECT_EQ(aux[vgB->GetLogicalVolume()->id()].fMCIndex, 2);
  EXPECT_EQ(aux[vgC->GetLogicalVolume()->id()].fMCIndex, 1);
  EXPECT_EQ(aux[vgB->GetLogicalVolume()->id()].fGPUregionId, geometry.regionB.GetInstanceID());
  EXPECT_EQ(aux[vgC->GetLogicalVolume()->id()].fGPUregionId, geometry.regionC.GetInstanceID());

  const auto packed = AdePTGeometryBridge::PackWDT(raw);
  ASSERT_EQ(packed.roots.size(), 3u);
  const int governingBucket = packed.regionToWDT[geometry.regionA.GetInstanceID()];
  ASSERT_GE(governingBucket, 0);
  const int governingOffset = packed.regions[governingBucket].offset;
  for (const auto *region : {&geometry.regionA, &geometry.regionB, &geometry.regionC}) {
    const int bucket = packed.regionToWDT[region->GetInstanceID()];
    ASSERT_GE(bucket, 0);
    const auto &header = packed.regions[bucket];
    ASSERT_EQ(header.count, 3);
    EXPECT_EQ(header.offset, governingOffset);
    for (int i = 0; i < header.count; ++i) {
      const auto &root = packed.roots[header.offset + i];
      EXPECT_EQ(root.root.GetLevel(), 1);
      EXPECT_EQ(root.hepemIMC, 1);
    }
  }
  // B's outside placement is still a GPU volume, but never becomes a WDT root.
  const auto *outsideB = vgWorld->GetLogicalVolume()->GetDaughters()[3];
  EXPECT_EQ(outsideB->GetLogicalVolume(), vgB->GetLogicalVolume());
  EXPECT_EQ(raw.roots.size(), 3u);
}

TEST(AdePTGeometryBridge, SharesIdenticalPackedWdtRootListsButKeepsDistinctEnvelopes)
{
  NestedWdtGeometry geometry;
  std::vector<adeptint::VolAuxData> aux;
  auto raw = geometry.Collect(aux);
  // A and B share two placements; C has a separate envelope. Equal reference
  // couples must not cause physically distinct roots to be merged.
  raw.regionToRootIndices[geometry.regionA.GetInstanceID()] = {0, 1};
  raw.regionToRootIndices[geometry.regionB.GetInstanceID()] = {0, 1};
  raw.regionToRootIndices[geometry.regionC.GetInstanceID()] = {2};

  const auto packed = AdePTGeometryBridge::PackWDT(raw);
  ASSERT_EQ(packed.roots.size(), 3u);
  const auto &a = packed.regions[packed.regionToWDT[geometry.regionA.GetInstanceID()]];
  const auto &b = packed.regions[packed.regionToWDT[geometry.regionB.GetInstanceID()]];
  const auto &c = packed.regions[packed.regionToWDT[geometry.regionC.GetInstanceID()]];
  EXPECT_EQ(a.offset, b.offset);
  ASSERT_EQ(a.count, 2);
  EXPECT_EQ(b.count, 2);
  ASSERT_EQ(c.count, 1);
  EXPECT_NE(a.offset, c.offset);
  for (int i = 0; i < a.count; ++i) {
    EXPECT_EQ(packed.roots[a.offset + i].root.GetState(), raw.roots[i].root.GetState());
    EXPECT_EQ(packed.roots[a.offset + i].hepemIMC, raw.roots[i].hepemIMC);
  }
  EXPECT_EQ(packed.roots[c.offset].root.GetState(), raw.roots[2].root.GetState());
  EXPECT_EQ(packed.roots[c.offset].hepemIMC, raw.roots[2].hepemIMC);
}

TEST(AdePTGeometryBridge, RejectsWdtEnvelopesContainingCPURegions)
{
  NestedWdtGeometry geometry(false);
  std::vector<adeptint::VolAuxData> aux;
  // C is CPU-only. Omitting its GPU envelope would still let the CPU WDT helper
  // cross back into GPU B without the specialized tracking manager seeing a boundary.
  EXPECT_DEATH(geometry.Collect(aux, {&geometry.worldRegion, &geometry.regionA, &geometry.regionB}),
               "WDTEnvelopeContainsCPURegion");
}

TEST(AdePTGeometryBridge, RejectsCPUWdtRootsContainingGPURegions)
{
  NestedWdtGeometry geometry;
  std::vector<adeptint::VolAuxData> aux;
  // The inverse arrangement is equally unsafe: the CPU WDT root must not skip a GPU child.
  EXPECT_DEATH(geometry.Collect(aux, {&geometry.worldRegion, &geometry.regionB, &geometry.regionC}),
               "WDTEnvelopeContainsCPURegion");
}

TEST(AdePTGeometryBridge, OmitsEntirelyCPUWdtEnvelopesFromGPUData)
{
  NestedWdtGeometry geometry;
  std::vector<adeptint::VolAuxData> aux;
  const auto raw = geometry.Collect(aux, {&geometry.worldRegion});
  EXPECT_TRUE(raw.roots.empty());
  EXPECT_TRUE(raw.regionToRootIndices.empty());
}

TEST(AdePTGeometryBridge, RejectsExplicitWdtConfigurationOfAnInheritedRegion)
{
  NestedWdtGeometry geometry;
  // This is the G4HepEm configuration restriction, also used by AdePT initialization.
  EXPECT_EXIT(geometry.trackingManager->fWDTHelper->Initialize({geometry.regionA.GetName(), geometry.regionB.GetName()},
                                                               &geometry.matCutData),
              ::testing::ExitedWithCode(1), "Sub-regions of Woodcock tracking regions");
}

TEST(AdePTGeometryBridge, UsesResolvedGPURegionSet)
{
  GeometryCleanup cleanup;
  G4GeometryManager::GetInstance()->OpenGeometry();
  vecgeom::GeoManager::Instance().Clear();

  auto *material = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
  ASSERT_NE(material, nullptr);

  auto *worldSolid = new G4Box("cpu_override_world_solid", 10.0 * mm, 10.0 * mm, 10.0 * mm);
  auto *worldLV    = new G4LogicalVolume(worldSolid, material, "cpu_override_world_lv");
  auto *worldPV = new G4PVPlacement(nullptr, G4ThreeVector(), worldLV, "cpu_override_world", nullptr, false, 0, false);

  auto *cpuSolid = new G4Box("cpu_override_solid", 1.0 * mm, 1.0 * mm, 1.0 * mm);
  auto *cpuLV    = new G4LogicalVolume(cpuSolid, material, "cpu_override_lv");
  new G4PVPlacement(nullptr, G4ThreeVector(), cpuLV, "cpu_override", worldLV, false, 0, false);

  auto *couple = new G4MaterialCutsCouple(material, new G4ProductionCuts);
  couple->SetIndex(0);
  couple->SetUseFlag(true);
  worldLV->SetMaterialCutsCouple(couple);
  cpuLV->SetMaterialCutsCouple(couple);

  G4Region worldRegion("cpu_override_world_region");
  worldRegion.AddRootLogicalVolume(worldLV);
  worldRegion.RegisterMaterialCouplePair(material, couple);

  G4Region cpuRegion("cpu_override_region");
  cpuRegion.AddRootLogicalVolume(cpuLV);
  cpuRegion.RegisterMaterialCouplePair(material, couple);

  auto *transport = G4TransportationManager::GetTransportationManager();
  transport->SetWorldForTracking(worldPV);
  G4GeometryManager::GetInstance()->CloseGeometry(true);

  AdePTGeometryBridge::CreateVecGeomWorld(worldPV);
  auto const *vgWorld = vecgeom::GeoManager::Instance().GetWorld();
  ASSERT_NE(vgWorld, nullptr);
  ASSERT_EQ(vgWorld->GetLogicalVolume()->GetDaughters().size(), 1u);
  auto const *vgCPUVolume = vgWorld->GetLogicalVolume()->GetDaughters()[0];

  SingleMatCutHepEmData hepEmData;
  G4HepEmTrackingManagerSpecialized hepEmTM;
  std::set<G4Region const *> gpuRegions{&worldRegion};
  CMSSteppingActionConfig cmsConfig;
  adeptint::WDTHostRaw wdtRaw;
  std::vector<adeptint::VolAuxData> volAuxData(vecgeom::GeoManager::Instance().GetRegisteredVolumesCount());

  AdePTGeometryBridge::InitVolAuxData(volAuxData.data(), &hepEmData.hepEmData, &hepEmTM, gpuRegions, cmsConfig, wdtRaw);

  EXPECT_EQ(volAuxData[vgWorld->GetLogicalVolume()->id()].fGPUregionId, worldRegion.GetInstanceID());
  EXPECT_EQ(volAuxData[vgCPUVolume->GetLogicalVolume()->id()].fGPUregionId, -1);
}

#if defined(ADEPT_STEPACTION_TYPE) && (ADEPT_STEPACTION_TYPE == 1)
namespace {
// Minimal two-volume geometry for the CMS stepping action configuration tests. The returned pair holds the VecGeom
// logical-volume ids of the world and the inner volume.
std::pair<int, int> BuildCMSConfigGeometry(G4Region &worldRegion, G4Region &innerRegion, const char *prefix)
{
  G4GeometryManager::GetInstance()->OpenGeometry();
  vecgeom::GeoManager::Instance().Clear();

  auto *material = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
  Require(material != nullptr, "G4_AIR material was not available");

  const std::string p{prefix};
  auto *worldSolid = new G4Box(p + "_world_solid", 10.0 * mm, 10.0 * mm, 10.0 * mm);
  auto *worldLV    = new G4LogicalVolume(worldSolid, material, p + "_world_lv");
  auto *worldPV    = new G4PVPlacement(nullptr, G4ThreeVector(), worldLV, p + "_world", nullptr, false, 0, false);

  auto *innerSolid = new G4Box(p + "_inner_solid", 1.0 * mm, 1.0 * mm, 1.0 * mm);
  auto *innerLV    = new G4LogicalVolume(innerSolid, material, p + "_inner_lv");
  new G4PVPlacement(nullptr, G4ThreeVector(), innerLV, p + "_inner", worldLV, false, 0, false);

  auto *couple = new G4MaterialCutsCouple(material, new G4ProductionCuts);
  couple->SetIndex(0);
  couple->SetUseFlag(true);
  worldLV->SetMaterialCutsCouple(couple);
  innerLV->SetMaterialCutsCouple(couple);

  worldRegion.AddRootLogicalVolume(worldLV);
  worldRegion.RegisterMaterialCouplePair(material, couple);
  innerRegion.AddRootLogicalVolume(innerLV);
  innerRegion.RegisterMaterialCouplePair(material, couple);

  auto *transport = G4TransportationManager::GetTransportationManager();
  transport->SetWorldForTracking(worldPV);
  G4GeometryManager::GetInstance()->CloseGeometry(true);

  AdePTGeometryBridge::CreateVecGeomWorld(worldPV);
  auto const *vgWorld = vecgeom::GeoManager::Instance().GetWorld();
  Require(vgWorld != nullptr, "VecGeom world was not created");
  auto const *vgInner = vgWorld->GetLogicalVolume()->GetDaughters()[0];
  return {vgWorld->GetLogicalVolume()->id(), vgInner->GetLogicalVolume()->id()};
}
} // namespace

// CMS stepping action: the configuration's region names are resolved into the per-volume flags and the gamma Russian
// roulette survival probability. Regions missing from the geometry are ignored, except for the dead regions.
TEST(AdePTGeometryBridge, FillsCMSSteppingActionVolAuxData)
{
  GeometryCleanup cleanup;
  G4Region worldRegion("cms_aux_world_region");
  G4Region innerRegion("cms_aux_inner_region");
  const auto [worldId, innerId] = BuildCMSConfigGeometry(worldRegion, innerRegion, "cms_aux");

  SingleMatCutHepEmData hepEmData;
  G4HepEmTrackingManagerSpecialized hepEmTM;
  std::set<G4Region const *> gpuRegions{&worldRegion};
  adeptint::WDTHostRaw wdtRaw;
  std::vector<adeptint::VolAuxData> volAuxData(vecgeom::GeoManager::Instance().GetRegisteredVolumesCount());

  CMSSteppingActionConfig cmsConfig;
  cmsConfig.lowDensityRegions                     = {"cms_aux_inner_region", "cms_aux_missing_region"};
  cmsConfig.gammaRussianRoulette                  = {{"cms_aux_world_region", 0.3}, {"cms_aux_missing_region", 0.5}};
  cmsConfig.gammaRussianRouletteRestrictedRegions = {"cms_aux_world_region"};

  AdePTGeometryBridge::InitVolAuxData(volAuxData.data(), &hepEmData.hepEmData, &hepEmTM, gpuRegions, cmsConfig, wdtRaw);

  EXPECT_FALSE(volAuxData[worldId].fCMSDeadRegion);
  EXPECT_FALSE(volAuxData[worldId].fCMSLowDensityRegion);
  EXPECT_TRUE(volAuxData[worldId].fCMSGammaRussianRouletteRestricted);
  EXPECT_DOUBLE_EQ(volAuxData[worldId].fCMSGammaRussianRouletteProbability, 0.3);
  EXPECT_TRUE(volAuxData[innerId].fCMSLowDensityRegion);
  EXPECT_FALSE(volAuxData[innerId].fCMSGammaRussianRouletteRestricted);
  EXPECT_DOUBLE_EQ(volAuxData[innerId].fCMSGammaRussianRouletteProbability, 1.0);
}

// A survival probability so close to one that the reweighted float weight would not stay above one must be rejected,
// as CMSSW would otherwise play Russian roulette again with the replayed secondary.
TEST(AdePTGeometryBridge, RejectsCMSGammaRouletteProbabilityTooCloseToOne)
{
  GeometryCleanup cleanup;
  G4Region worldRegion("cms_rr_world_region");
  G4Region innerRegion("cms_rr_inner_region");
  BuildCMSConfigGeometry(worldRegion, innerRegion, "cms_rr");

  SingleMatCutHepEmData hepEmData;
  G4HepEmTrackingManagerSpecialized hepEmTM;
  std::set<G4Region const *> gpuRegions{&worldRegion};
  adeptint::WDTHostRaw wdtRaw;
  std::vector<adeptint::VolAuxData> volAuxData(vecgeom::GeoManager::Instance().GetRegisteredVolumesCount());

  CMSSteppingActionConfig cmsConfig;
  cmsConfig.gammaRussianRoulette = {{"cms_rr_world_region", 1.0 - 1e-9}};

  EXPECT_THROW(AdePTGeometryBridge::InitVolAuxData(volAuxData.data(), &hepEmData.hepEmData, &hepEmTM, gpuRegions,
                                                   cmsConfig, wdtRaw),
               std::runtime_error);
}
#endif // CMS stepping action

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

  const auto navState0  = makeNavState(0);
  const auto navState2  = makeNavState(2);
  auto const &instance0 = AdePTGeometryBridge::LookupMappedVolumeInstance(vgDaughters[0]);
  auto const &instance2 = AdePTGeometryBridge::LookupMappedVolumeInstance(vgDaughters[2]);
  EXPECT_EQ(instance0.g4Volume, instance2.g4Volume);
  EXPECT_EQ(instance0.type, kReplica);
  EXPECT_EQ(instance2.type, kReplica);
  EXPECT_EQ(instance0.copyNo, 0);
  EXPECT_EQ(instance2.copyNo, 2);
  EXPECT_EQ(&instance0, &AdePTGeometryBridge::LookupMappedVolumeInstance(vgDaughters[0]));

  AdePTGeant4Integration integration;
  G4NavigationHistory history;

  integration.FillG4NavigationHistory(navState0, history);
  ASSERT_EQ(history.GetDepth(), 1u);
  EXPECT_EQ(history.GetVolumeType(1), kReplica);
  EXPECT_EQ(history.GetReplicaNo(1), 0);

  G4TouchableHistory touchableCopy0(history);
  auto *replicaVolume = touchableCopy0.GetVolume(0);
  const auto copy0    = touchableCopy0.GetCopyNumber(0);
  EXPECT_EQ(copy0, 0);
  EXPECT_EQ(replicaVolume->GetCopyNo(), 0);

  integration.FillG4NavigationHistory(navState2, history);
  ASSERT_EQ(history.GetDepth(), 1u);
  EXPECT_EQ(history.GetVolume(1), replicaVolume);
  EXPECT_EQ(history.GetVolumeType(1), kReplica);
  EXPECT_EQ(history.GetReplicaNo(1), 2);

  G4TouchableHistory touchableCopy2(history);
  EXPECT_EQ(touchableCopy2.GetVolume(0), replicaVolume);
  EXPECT_EQ(touchableCopy2.GetCopyNumber(0), 2);
  EXPECT_EQ(replicaVolume->GetCopyNo(), 2);
  EXPECT_NE(copy0, touchableCopy2.GetCopyNumber(0));

  // The paired path must produce exactly the same histories as two independent
  // calls while looking up the shared world prefix only once.
  G4NavigationHistory referencePre;
  G4NavigationHistory referencePost;
  integration.FillG4NavigationHistory(navState0, referencePre);
  integration.FillG4NavigationHistory(navState2, referencePost);

  G4NavigationHistory pairedPre;
  G4NavigationHistory pairedPost;
  integration.FillG4NavigationHistories(navState0, navState2, pairedPre, pairedPost);

  const auto expectSameHistory = [](G4NavigationHistory const &actual, G4NavigationHistory const &expected) {
    ASSERT_EQ(actual.GetDepth(), expected.GetDepth());
    for (std::size_t level = 0; level <= actual.GetDepth(); ++level) {
      EXPECT_EQ(actual.GetVolume(static_cast<G4int>(level)), expected.GetVolume(static_cast<G4int>(level)));
      EXPECT_EQ(actual.GetVolumeType(static_cast<G4int>(level)), expected.GetVolumeType(static_cast<G4int>(level)));
      EXPECT_EQ(actual.GetReplicaNo(static_cast<G4int>(level)), expected.GetReplicaNo(static_cast<G4int>(level)));
      EXPECT_EQ(actual.GetTransform(static_cast<G4int>(level)).NetTranslation(),
                expected.GetTransform(static_cast<G4int>(level)).NetTranslation());
      EXPECT_EQ(actual.GetTransform(static_cast<G4int>(level)).NetRotation(),
                expected.GetTransform(static_cast<G4int>(level)).NetRotation());
    }
  };
  expectSameHistory(pairedPre, referencePre);
  expectSameHistory(pairedPost, referencePost);

  const auto touchable = integration.MakeTouchableFromNavState(navState2);
  ASSERT_NE(touchable(), nullptr);
  EXPECT_EQ(touchable->GetHistoryDepth(), 1);
  EXPECT_EQ(touchable->GetVolume(), replicaVolume);
  EXPECT_EQ(touchable->GetCopyNumber(), 2);
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
  ASSERT_TRUE(hepEmTM.fWDTHelper->Initialize(wdtRegionNames, &matCutData));

  std::vector<adeptint::VolAuxData> volAuxData(vecgeom::GeoManager::Instance().GetRegisteredVolumesCount());
  std::set<G4Region const *> gpuRegions;
  CMSSteppingActionConfig cmsConfig;
  adeptint::WDTHostRaw wdtRaw;

  EXPECT_THROW(
      AdePTGeometryBridge::InitVolAuxData(volAuxData.data(), &hepEmData, &hepEmTM, gpuRegions, cmsConfig, wdtRaw),
      std::runtime_error);
}
