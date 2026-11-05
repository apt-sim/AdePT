// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <VecGeom/management/GeoManager.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/UnplacedBox.h>

#include <cstdint>

// Geometry construction uses host C++ because VecGeom's default namespace in
// CUDA translation units denotes the device volume classes.
// Supply a valid world navigation state and logical-volume ID for ElectronHowFar.
// Its box has 10 mm half-lengths, so the fixture positions remain well inside it.
// The test sets smaller conservative cached safeties explicitly; the box's actual
// safety is not used to select the MSC branches.
std::uint64_t BuildSplitElectronMscGeometry(int &logicalId)
{
  auto *logical = new vecgeom::LogicalVolume("msc_world", new vecgeom::UnplacedBox(10., 10., 10.));
  auto *world   = logical->Place("world", new vecgeom::Transformation3D);
  vecgeom::GeoManager::Instance().SetWorldAndClose(world);
  vecgeom::NavigationState state;
  state.Push(world);
  // The caller sizes the auxiliary-volume array with this ID and synchronizes
  // the navigation table so the kernel can read the track's logical volume.
  logicalId = logical->id();
  return state.GetNavIndex();
}
