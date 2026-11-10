// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <VecGeom/management/GeoManager.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/UnplacedBox.h>
#include <array>
#include <cstdint>
#include <initializer_list>

void BuildWoodcockTestGeometry(std::array<std::uint64_t, 7> &indices, std::array<int, 4> &logicalIds)
{
  auto *worldLV  = new vecgeom::LogicalVolume("wdt_world", new vecgeom::UnplacedBox(100., 100., 100.));
  auto *aLV      = new vecgeom::LogicalVolume("wdt_A", new vecgeom::UnplacedBox(10., 10., 10.));
  auto *bLV      = new vecgeom::LogicalVolume("wdt_B", new vecgeom::UnplacedBox(2., 2., 2.));
  auto *cLV      = new vecgeom::LogicalVolume("wdt_C", new vecgeom::UnplacedBox(1., 1., 1.));
  auto *identity = new vecgeom::Transformation3D;
  bLV->PlaceDaughter("C", cLV, identity);
  aLV->PlaceDaughter("B", bLV, identity);
  worldLV->PlaceDaughter("A_left", aLV, new vecgeom::Transformation3D(-30., 0., 0.));
  worldLV->PlaceDaughter("A_right", aLV, new vecgeom::Transformation3D(30., 0., 0.));
  worldLV->PlaceDaughter("B_outside", bLV, new vecgeom::Transformation3D(70., 0., 0.));
  auto *world = worldLV->Place("world", identity);
  vecgeom::GeoManager::Instance().SetWorldAndClose(world);
  auto const &daughters = worldLV->GetDaughters();
  const auto *b         = aLV->GetDaughters()[0];
  const auto *c         = bLV->GetDaughters()[0];
  auto index            = [&](std::initializer_list<const vecgeom::VPlacedVolume *> path) {
    vecgeom::NavigationState state;
    state.Push(world);
    for (const auto *volume : path)
      state.Push(volume);
    return state.GetNavIndex();
  };
  indices    = {{index({daughters[0]}), index({daughters[1]}), index({daughters[0], b}), index({daughters[0], b, c}),
                 index({daughters[1], b}), index({daughters[2]}), index({})}};
  logicalIds = {{static_cast<int>(worldLV->id()), static_cast<int>(aLV->id()), static_cast<int>(bLV->id()),
                 static_cast<int>(cLV->id())}};
}
