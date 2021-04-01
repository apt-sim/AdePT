// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file VolumeDispatcher.h
 * @brief Dispatching queries to volumes without virtual calls.
 */

#ifndef VOLUME_DISPATCHER_H_
#define VOLUME_DISPATCHER_H_

#include <CopCore/Global.h>

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/volumes/PlacedVolume.h>
#include <VecGeom/volumes/VolumeTypes.h>

#include <VecGeom/volumes/Box.h>
#include <VecGeom/volumes/Trd.h>
#include <VecGeom/volumes/Tube.h>

#define DISPATCH_TO_PLACED_VOLUME_KIND(vol, kind, class, impl, type, func, ...)                \
  case vecgeom::VolumeTypes::kind: {                                                           \
    type output;                                                                               \
    const auto unplacedStruct = static_cast<const vecgeom::class *>(vol)->GetUnplacedStruct(); \
    vecgeom::impl::func(*unplacedStruct, __VA_ARGS__, output);                                 \
    return output;                                                                             \
  }

#define DISPATCH_TO_PLACED_VOLUME(vol, type, func, ...)                                                            \
  switch (vol->GetType()) {                                                                                        \
    DISPATCH_TO_PLACED_VOLUME_KIND(vol, kBox, PlacedBox, BoxImplementation, type, func, __VA_ARGS__);              \
    DISPATCH_TO_PLACED_VOLUME_KIND(vol, kTrd, PlacedTrd, TrdImplementation<vecgeom::TrdTypes::UniversalTrd>, type, \
                                   func, __VA_ARGS__);                                                             \
    DISPATCH_TO_PLACED_VOLUME_KIND(vol, kTube, PlacedTube, TubeImplementation<vecgeom::TubeTypes::UniversalTube>,  \
                                   type, func, __VA_ARGS__);                                                       \
  default:                                                                                                         \
    break;                                                                                                         \
  }

class VolumeDispatcher {

  using VolumePtr         = vecgeom::VPlacedVolume const *;
  using Precision         = vecgeom::Precision;
  using TransformationPtr = vecgeom::Transformation3D const *;
  using Vector            = vecgeom::Vector3D<Precision>;

public:
  __host__ __device__ static bool UnplacedContains(VolumePtr vol, Vector const &point)
  {
    DISPATCH_TO_PLACED_VOLUME(vol, bool, Contains, point);
    COPCORE_EXCEPTION("No matching type found to dispatch Contains!");
    return false;
  }

  __host__ __device__ static bool Contains(VolumePtr vol, Vector const &point, Vector &localPoint)
  {
    TransformationPtr tr = vol->GetTransformation();
    localPoint           = tr->Transform(point);
    return UnplacedContains(vol, localPoint);
  }

  __host__ __device__ static Precision DistanceToIn(VolumePtr vol, Vector const &position, Vector const &direction,
                                                    Precision stepMax)
  {
    TransformationPtr tr  = vol->GetTransformation();
    Vector localPosition  = tr->Transform(position);
    Vector localDirection = tr->TransformDirection(direction);
    DISPATCH_TO_PLACED_VOLUME(vol, Precision, DistanceToIn, localPosition, localDirection, stepMax);
    COPCORE_EXCEPTION("No matching type found to dispatch DistanceToIn!");
    return 0;
  }

  __host__ __device__ static Precision DistanceToOut(VolumePtr vol, Vector const &position, Vector const &direction,
                                                     Precision stepMax)
  {
    DISPATCH_TO_PLACED_VOLUME(vol, Precision, DistanceToOut, position, direction, stepMax);
    COPCORE_EXCEPTION("No matching type found to dispatch DistanceToOut!");
    return 0;
  }
};

#undef DISPATCH_TO_PLACED_VOLUME_KIND
#undef DISPATCH_TO_PLACED_VOLUME

#endif
