// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <VecGeom/base/Vector3D.h>

#include <type_traits>

/// @brief Wraps a field so that every lookup is shifted by a fixed origin.
///
/// Used for mixed-precision field integration. The RK driver integrates the
/// distance moved since the start of the chord, not the absolute position, so
/// the float values stay small. The track position itself stays in double in
/// the caller. This wrapper adds the origin back, in double, whenever the
/// field is evaluated.
///
/// This is needed because float cannot resolve small steps on top of large
/// coordinates: at r = 1 m, neighboring float values are about 1e-4 mm apart,
/// so steps below that would simply be lost if positions were stored in float.
///
/// @tparam Field_t Underlying field type providing Evaluate(x, y, z)
template <class Field_t>
class OffsetMagneticField {
public:
  /// @brief Construct an adapter around @p field with lookups shifted by @p origin
  __host__ __device__ OffsetMagneticField(const Field_t &field, const vecgeom::Vector3D<double> &origin)
      : fField(field), fOrigin(origin)
  {
  }

  /// @brief Evaluate the field at origin + (x, y, z), with the addition done in double
  template <typename Real_t>
  __host__ __device__ auto Evaluate(Real_t x, Real_t y, Real_t z) const -> vecgeom::Vector3D<Real_t>
  {
    const auto value = fField.Evaluate(fOrigin[0] + double(x), fOrigin[1] + double(y), fOrigin[2] + double(z));
    return {Real_t(value[0]), Real_t(value[1]), Real_t(value[2])};
  }

  /// @brief Evaluate the field at origin + position, with the addition done in double
  template <typename Real_t>
  __host__ __device__ auto Evaluate(const vecgeom::Vector3D<Real_t> &position) const -> vecgeom::Vector3D<Real_t>
  {
    return Evaluate(position[0], position[1], position[2]);
  }

private:
  const Field_t &fField;             ///< Underlying absolute-coordinate field
  vecgeom::Vector3D<double> fOrigin; ///< Double-precision origin added to every lookup
};

/// @brief Field type the RK driver is instantiated with, for a given integration precision.
///
/// Double-precision integration uses the plain field and absolute coordinates.
/// Single precision (mixed-precision builds) integrates relative to the chord start and
/// therefore sees the field through OffsetMagneticField. This alias keeps the kernels
/// consistent with the dispatch in fieldPropagatorRungeKutta::ComputeStepAndNextVolume.
template <class Field_t, typename Real_t>
using RkFieldType = std::conditional_t<std::is_same_v<Real_t, double>, Field_t, OffsetMagneticField<Field_t>>;
