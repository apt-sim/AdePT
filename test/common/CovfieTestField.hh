// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <ostream>

#include <covfie/core/algebra/affine.hpp>
#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/parameter_pack.hpp>

namespace adept_test {

// Match the IO field type consumed by AdePT's GeneralMagneticField:
// an affine transform from world coordinates into a linearly interpolated,
// strided 3D array storing float3 magnetic-field values.
using CovfieTestInnerBackend =
    covfie::backend::strided<covfie::vector::size3, covfie::backend::array<covfie::vector::float3>>;
using CovfieTestBackend = covfie::backend::affine<covfie::backend::linear<CovfieTestInnerBackend>>;
using CovfieTestField   = covfie::field<CovfieTestBackend>;

/// @brief Sampled covfie map over [minExtent, maxExtent] on each axis.
/// @param value Callable giving the field (std::array<float, 3>) at sample indices (x, y, z)
/// @param minExtent Lower world coordinate of the map on each axis
/// @param maxExtent Upper world coordinate of the map on each axis
/// @param samples Grid dimensions along x, y and z (each at least two)
template <class Value>
inline CovfieTestField MakeSampledCovfieField(Value value, float minExtent = -10000.f, float maxExtent = 10000.f,
                                              std::array<std::size_t, 3> samples = {{2, 2, 2}})
{
  covfie::field<CovfieTestInnerBackend> inner(
      covfie::make_parameter_pack(CovfieTestInnerBackend::configuration_t{samples[0], samples[1], samples[2]}));
  covfie::field<CovfieTestInnerBackend>::view_t innerView(inner);

  for (std::size_t x = 0; x < samples[0]; ++x) {
    for (std::size_t y = 0; y < samples[1]; ++y) {
      for (std::size_t z = 0; z < samples[2]; ++z) {
        const std::array<float, 3> v = value(x, y, z);
        innerView.at(x, y, z)        = {v[0], v[1], v[2]};
      }
    }
  }

  // Convert world coordinates in [minExtent, maxExtent] along each axis into
  // each axis's index range [0, samples[axis] - 1] used by the strided backend.
  // Values outside the range are clamped by the CPU-side Covfie wrapper used
  // in MagneticFields.hh; the GPU texture backend also clamps accesses.
  const auto translation = covfie::algebra::affine<3>::translation(-minExtent, -minExtent, -minExtent);
  const auto scaleX      = static_cast<float>(samples[0] - 1) / (maxExtent - minExtent);
  const auto scaleY      = static_cast<float>(samples[1] - 1) / (maxExtent - minExtent);
  const auto scaleZ      = static_cast<float>(samples[2] - 1) / (maxExtent - minExtent);
  const auto scaling     = covfie::algebra::affine<3>::scaling(scaleX, scaleY, scaleZ);

  return CovfieTestField(covfie::make_parameter_pack(CovfieTestBackend::configuration_t(scaling * translation),
                                                     CovfieTestBackend::backend_t::configuration_t{},
                                                     std::move(inner.backend())));
}

/// @brief 2x2x2 covfie map with the same field value at every sample, so constant everywhere.
inline CovfieTestField MakeConstantCovfieField(const std::array<float, 3> &fieldValue, float minExtent = -10000.f,
                                               float maxExtent = 10000.f)
{
  // Every sample has the same value, so interpolation anywhere represents a constant field
  return MakeSampledCovfieField([&](std::size_t, std::size_t, std::size_t) { return fieldValue; }, minExtent,
                                maxExtent);
}

/// @brief Write a 2x3x5 map over [-10000, 10000] with Bx, By and Bz rising along x, y and z.
/// @param output Stream receiving the covfie dump
/// @param fieldScale Field increase per grid index; upper edge values are fieldScale * (1, 2, 4)
inline void WriteGradientCovfieField(std::ostream &output, float fieldScale)
{
  // Unequal dimensions expose axis-index mistakes. Power-of-two index ranges keep the midpoint values exact.
  auto field = MakeSampledCovfieField(
      [&](std::size_t x, std::size_t y, std::size_t z) {
        return std::array<float, 3>{{fieldScale * x, fieldScale * y, fieldScale * z}};
      },
      -10000.f, 10000.f, {{2, 3, 5}});
  field.dump(output);
}

inline void WriteConstantCovfieField(std::ostream &output, const std::array<float, 3> &fieldValue,
                                     float minExtent = -10000.f, float maxExtent = 10000.f)
{
  // Keep the helper header-only so tests and tiny generators can share exactly
  // the same fixture creation code without introducing another library target.
  auto field = MakeConstantCovfieField(fieldValue, minExtent, maxExtent);
  field.dump(output);
}

} // namespace adept_test
