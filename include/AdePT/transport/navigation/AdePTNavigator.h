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

/// Whether the final direction at a boundary keeps a track that would leave `from` on the side of `from`: it goes back
/// into the exited volume, or out of the entered daughter. Geant4 relocates once with that direction, so such a track
/// has only touched the surface.
__host__ __device__ inline bool TurnsBackAtBoundary(vecgeom::NavigationState const &from,
                                                    vecgeom::NavigationState const &next,
                                                    vecgeom::Vector3D<double> const &pos,
                                                    vecgeom::Vector3D<double> const &dir)
{
#ifdef ADEPT_USE_SURF
  return false;
#else
  const bool entered               = next.IsDescendent(from.GetState());
  vecgeom::NavigationState crossed = entered ? next : from;
  while (entered && crossed.GetLevel() > from.GetLevel() + 1)
    crossed.Pop();
  vecgeom::Transformation3D m;
  crossed.TopMatrix(m);
  vecgeom::Vector3D<double> normal;
  if (!crossed.Top()->GetUnplacedVolume()->Normal(m.Transform(pos), normal)) return false;
  const double nd = normal.Dot(m.TransformDirection(dir));
  return entered ? nd > 0. : nd < 0.;
#endif
}
