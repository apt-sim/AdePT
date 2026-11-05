// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <VecGeom/management/GeoManager.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/UnplacedBox.h>

#include <cstdint>

// Geometry construction uses host C++ because VecGeom's default namespace in
// CUDA translation units denotes the device volume classes.
std::uint64_t BuildSplitElectronMscGeometry(int &logicalId)
{
  auto *logical = new vecgeom::LogicalVolume("msc_world", new vecgeom::UnplacedBox(10., 10., 10.));
  auto *world   = logical->Place("world", new vecgeom::Transformation3D);
  vecgeom::GeoManager::Instance().SetWorldAndClose(world);
  vecgeom::NavigationState state;
  state.Push(world);
  logicalId = logical->id();
  return state.GetNavIndex();
}
