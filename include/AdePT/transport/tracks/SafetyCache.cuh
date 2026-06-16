// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <VecGeom/base/Vector3D.h>

#include <VecCore/VecMath.h>

/// Cached isotropic safety value together with the position where it was
/// computed. SafetyAt() returns the conservative remaining safety at another
/// point by subtracting the displacement from the cached origin.
struct SafetyCache {
  vecgeom::Vector3D<float> origin; ///< last position where the safety was computed
  float value{0.f};                ///< safety value at origin

  __host__ __device__ SafetyCache() = default;

  /// @brief Get the conservative remaining safety at a given track position.
  /// @param position Track position.
  /// @param accurateLimit Only return non-zero if the remaining safety is larger than this limit.
  /// @return Remaining safety at @p position.
  __host__ __device__ VECGEOM_FORCE_INLINE float SafetyAt(vecgeom::Vector3D<double> const &position,
                                                          float accurateLimit = 0.f) const
  {
    float remaining = value - accurateLimit;
    if (remaining <= 0.f) return 0.f;
    vecgeom::Vector3D<float> pos{static_cast<float>(position[0]), static_cast<float>(position[1]),
                                 static_cast<float>(position[2])};
    float distSq = (pos - origin).Mag2();
    if (remaining * remaining < distSq) return 0.f;
    return value - vecCore::math::Sqrt(distSq);
  }

  /// @brief Store a safety value computed at a new point.
  /// @param position Position where the safety was computed.
  /// @param safety Safety value.
  /// @return Stored non-negative safety value.
  __host__ __device__ VECGEOM_FORCE_INLINE float Refresh(vecgeom::Vector3D<double> const &position, double safety)
  {
    origin.Set(static_cast<float>(position[0]), static_cast<float>(position[1]), static_cast<float>(position[2]));
    value = vecCore::math::Max(static_cast<float>(safety), 0.f);
    return value;
  }
};
