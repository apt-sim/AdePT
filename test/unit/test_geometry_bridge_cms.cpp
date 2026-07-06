// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/g4integration/geometry/AdePTGeometryBridge.hh>

#include <VecGeom/management/GeoManager.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include <G4GDMLParser.hh>
#include <G4GeometryManager.hh>
#include <G4TransportationManager.hh>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#ifndef ADEPT_CMS2018_GDML
#define ADEPT_CMS2018_GDML "cms2018_sd.gdml"
#endif

namespace {

struct GeometryCleanup {
  ~GeometryCleanup()
  {
    G4GeometryManager::GetInstance()->OpenGeometry();
    vecgeom::GeoManager::Instance().Clear();
  }
};

struct VecGeomTreeStats {
  std::size_t placements                = 0;
  std::size_t edges                     = 0;
  std::size_t leaves                    = 0;
  std::size_t maxDepth                  = 0;
  std::vector<std::size_t> daughterTree = {};
};

void AccumulateVecGeomTreeStats(vecgeom::VPlacedVolume const *placed, std::size_t depth, VecGeomTreeStats &stats)
{
  if (placed == nullptr) {
    throw std::runtime_error("Cannot collect VecGeom tree stats from a null placed volume");
  }

  ++stats.placements;
  stats.maxDepth = std::max(stats.maxDepth, depth);

  auto const &daughters = placed->GetLogicalVolume()->GetDaughters();
  stats.edges += daughters.size();
  stats.daughterTree.push_back(daughters.size());
  if (daughters.size() == 0) ++stats.leaves;

  for (auto const *daughter : daughters) {
    AccumulateVecGeomTreeStats(daughter, depth + 1, stats);
  }
}

VecGeomTreeStats CollectVecGeomTreeStats()
{
  auto const *world = vecgeom::GeoManager::Instance().GetWorld();
  if (world == nullptr) {
    throw std::runtime_error("Cannot collect VecGeom tree stats before the world is initialized");
  }

  VecGeomTreeStats stats;
  AccumulateVecGeomTreeStats(world, 0, stats);
  return stats;
}

} // namespace

TEST(AdePTGeometryBridge, CMS2018G4VGAndVGDMLHaveSamePlacementTree)
{
  GeometryCleanup cleanup;
  G4GeometryManager::GetInstance()->OpenGeometry();
  vecgeom::GeoManager::Instance().Clear();

  G4GDMLParser parser;
  parser.Read(ADEPT_CMS2018_GDML, false);
  auto *world = parser.GetWorldVolume();
  ASSERT_NE(world, nullptr);

  auto *transport = G4TransportationManager::GetTransportationManager();
  transport->SetWorldForTracking(world);
  G4GeometryManager::GetInstance()->CloseGeometry(true);

  AdePTGeometryBridge::CreateVecGeomWorld(world);
  const auto g4vgStats = CollectVecGeomTreeStats();

  vecgeom::GeoManager::Instance().Clear();

#ifdef VECGEOM_GDML_SUPPORT
  AdePTGeometryBridge::CreateVecGeomWorld(ADEPT_CMS2018_GDML);
#else
  GTEST_SKIP() << "VecGeom was built without GDML support";
#endif
  const auto vgdmlStats = CollectVecGeomTreeStats();

  EXPECT_EQ(vgdmlStats.placements, g4vgStats.placements);
  EXPECT_EQ(vgdmlStats.edges, g4vgStats.edges);
  EXPECT_EQ(vgdmlStats.leaves, g4vgStats.leaves);
  EXPECT_EQ(vgdmlStats.maxDepth, g4vgStats.maxDepth);
  EXPECT_EQ(vgdmlStats.daughterTree, g4vgStats.daughterTree);
}
