// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file VolumeDispatcher.h
 * @brief Dispatching queries to volumes.
 */

#ifndef VOLUME_DISPATCHER_H_
#define VOLUME_DISPATCHER_H_

#include <CopCore/Global.h>

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/volumes/PlacedVolume.h>

class VolumeDispatcher {

  using VolumePtr = vecgeom::VPlacedVolume const *;
  using Precision = vecgeom::Precision;
  using Vector    = vecgeom::Vector3D<Precision>;

public:
  __host__ __device__ static bool UnplacedContains(VolumePtr vol, Vector const &point)
  {
    return vol->UnplacedContains(point);
  }

  __host__ __device__ static bool Contains(VolumePtr vol, Vector const &point, Vector &localPoint)
  {
    return vol->Contains(point, localPoint);
  }

  __host__ __device__ static Precision DistanceToIn(VolumePtr vol, Vector const &position, Vector const &direction,
                                                    Precision stepMax)
  {
    return vol->DistanceToIn(position, direction, stepMax);
  }

  __host__ __device__ static Precision DistanceToOut(VolumePtr vol, Vector const &position, Vector const &direction,
                                                     Precision stepMax)
  {
    return vol->DistanceToOut(position, direction, stepMax);
  }
};

#endif
