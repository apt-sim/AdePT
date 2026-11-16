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

/// Whether the final direction at a boundary keeps a track that would leave `currentState` on the side of
/// `currentState`: it goes back into the exited volume, or out of the entered daughter. Geant4 relocates once with that
/// direction, so such a track has only touched the surface.
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
  vecgeom::Vector3D<double> normal;
  if (!surfaceState.Top()->GetUnplacedVolume()->Normal(m.Transform(pos), normal)) return false;
  const double nd = normal.Dot(m.TransformDirection(dir));
  return entersDaughter ? nd > 0. : nd < 0.;
#endif
}
