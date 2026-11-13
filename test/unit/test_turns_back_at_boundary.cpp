// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

// Regression tests for TurnsBackAtBoundary: a track on the boundary of the GPU regions is handed to Geant4 only if
// its final direction really leaves the volume it is in, including on edges, where no single normal decides.

#include <AdePT/transport/navigation/AdePTNavigator.h>

#include <VecGeom/management/GeoManager.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/UnplacedBox.h>

#include <gtest/gtest.h>

namespace {

using Vec = vecgeom::Vector3D<double>;

// World box with a unit box at the origin.
struct Geometry {
  vecgeom::NavigationState world, box;
  Geometry()
  {
    auto *worldLV = new vecgeom::LogicalVolume("world", new vecgeom::UnplacedBox(10., 10., 10.));
    auto *boxLV   = new vecgeom::LogicalVolume("box", new vecgeom::UnplacedBox(1., 1., 1.));
    worldLV->PlaceDaughter("box", boxLV, new vecgeom::Transformation3D);
    auto *placedWorld = worldLV->Place("world", new vecgeom::Transformation3D);
    vecgeom::GeoManager::Instance().SetWorldAndClose(placedWorld);
    world.Push(placedWorld);
    box.Push(placedWorld);
    box.Push(worldLV->GetDaughters()[0]);
  }
};

const Geometry &Geo()
{
  static const Geometry geometry;
  return geometry;
}

// Leaving the box: the track turns back only if its direction goes back into the box.
TEST(TurnsBackAtBoundary, ExitFace)
{
  const Vec pos(1., .3, 0.);
  EXPECT_TRUE(TurnsBackAtBoundary(Geo().box, Geo().world, pos, Vec(-.0126, 1., 0.).Unit()));
  EXPECT_FALSE(TurnsBackAtBoundary(Geo().box, Geo().world, pos, Vec(.0126, 1., 0.).Unit()));
  EXPECT_FALSE(TurnsBackAtBoundary(Geo().box, Geo().world, pos, Vec(1., 0., 0.)));
}

// On the edge at (1,1,0), (-2,1,0) goes back across x = 1 but leaves through y = 1: it leaves the box. The averaged
// edge normal (1,1,0)/sqrt2 has a negative product with it and would keep the track.
TEST(TurnsBackAtBoundary, ExitEdge)
{
  const Vec edge(1., 1., 0.);
  EXPECT_FALSE(TurnsBackAtBoundary(Geo().box, Geo().world, edge, Vec(-2., 1., 0.).Unit()));
  EXPECT_TRUE(TurnsBackAtBoundary(Geo().box, Geo().world, edge, Vec(-1., -1., 0.).Unit()));
}

// Entering the box from the world: the track turns back only if its direction does not go into the box.
TEST(TurnsBackAtBoundary, EntryFaceAndEdge)
{
  EXPECT_FALSE(TurnsBackAtBoundary(Geo().world, Geo().box, Vec(1., .3, 0.), Vec(-1., .1, 0.).Unit()));
  EXPECT_TRUE(TurnsBackAtBoundary(Geo().world, Geo().box, Vec(1., .3, 0.), Vec(1., .1, 0.).Unit()));
  // On the edge, (-2,1,0) does not enter: the averaged normal would hand the track over.
  EXPECT_TRUE(TurnsBackAtBoundary(Geo().world, Geo().box, Vec(1., 1., 0.), Vec(-2., 1., 0.).Unit()));
  EXPECT_FALSE(TurnsBackAtBoundary(Geo().world, Geo().box, Vec(1., 1., 0.), Vec(-1., -1., 0.).Unit()));
}

} // namespace
