// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file AdePTNavigator.h
 * @brief Top switch for different navigators
 */

#pragma once

#include <AdePT/AdePTBuildFlags.hh>
#include <AdePT/transport/support/Global.h>

#include <AdePT/transport/navigation/BVHNavigator.h>
#ifdef ADEPT_USE_SURF
#include <AdePT/transport/navigation/SurfNavigator.h>
#endif

// inline namespace COPCORE_IMPL {
#ifdef ADEPT_USE_SURF
#ifdef ADEPT_MIXED_PRECISION
using AdePTNavigator = SurfNavigator<float>;
#else
using AdePTNavigator = SurfNavigator<double>;
#endif
#else
using AdePTNavigator = BVHNavigator;
#endif
// } // End namespace COPCORE_IMPL

/// Whether a track whose step ends on a boundary leading out of `currentState` turns back with its final direction:
/// the ray stays in the exited volume, or does not go into the entered daughter. Geant4 relocates once with that
/// direction, so such a track has only touched the surface. The solid decides from the direction, which also holds on
/// edges and corners, where no single normal does.
__host__ __device__ inline bool TurnsBackAtBoundary(vecgeom::NavigationState const &currentState,
                                                    vecgeom::NavigationState const &nextState,
                                                    vecgeom::Vector3D<double> const &pos,
                                                    vecgeom::Vector3D<double> const &dir)
{
#ifdef ADEPT_USE_SURF
  return false;
#else
  const bool entersDaughter             = nextState.IsDescendent(currentState.GetState());
  vecgeom::NavigationState surfaceState = entersDaughter ? nextState : currentState;
  while (entersDaughter && surfaceState.GetLevel() > currentState.GetLevel() + 1) {
    surfaceState.Pop();
  }
  vecgeom::Transformation3D m;
  surfaceState.TopMatrix(m);
  const auto localPos = m.Transform(pos);
  const auto localDir = m.TransformDirection(dir);
  auto const *solid   = surfaceState.Top()->GetUnplacedVolume();
  // A positive distance means the ray moves away from the side it would be handed over to.
  return entersDaughter ? solid->DistanceToIn(localPos, localDir, vecgeom::kInfLength) > 0.
                        : solid->DistanceToOut(localPos, localDir, vecgeom::kInfLength) > 0.;
#endif
}
