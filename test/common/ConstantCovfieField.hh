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
using ConstantCovfieInnerBackend =
    covfie::backend::strided<covfie::vector::size3, covfie::backend::array<covfie::vector::float3>>;
using ConstantCovfieBackend = covfie::backend::affine<covfie::backend::linear<ConstantCovfieInnerBackend>>;
using ConstantCovfieField   = covfie::field<ConstantCovfieBackend>;

/// @brief 2x2x2 covfie map over [minExtent, maxExtent] on each axis.
/// @param value Callable giving the field (std::array<float, 3>) at sample indices (x, y, z)
/// @param minExtent Lower world coordinate of the map on each axis
/// @param maxExtent Upper world coordinate of the map on each axis
template <class Value>
inline ConstantCovfieField MakeSampledCovfieField(Value value, float minExtent = -10000.f, float maxExtent = 10000.f)
{
  constexpr std::size_t kSamplesPerAxis = 2;

  covfie::field<ConstantCovfieInnerBackend> inner(covfie::make_parameter_pack(
      ConstantCovfieInnerBackend::configuration_t{kSamplesPerAxis, kSamplesPerAxis, kSamplesPerAxis}));
  covfie::field<ConstantCovfieInnerBackend>::view_t innerView(inner);

  for (std::size_t x = 0; x < kSamplesPerAxis; ++x) {
    for (std::size_t y = 0; y < kSamplesPerAxis; ++y) {
      for (std::size_t z = 0; z < kSamplesPerAxis; ++z) {
        const std::array<float, 3> v = value(x, y, z);
        innerView.at(x, y, z)        = {v[0], v[1], v[2]};
      }
    }
  }

  // Convert world coordinates in [minExtent, maxExtent] along each axis into
  // the index range [0, kSamplesPerAxis - 1] used by the strided backend.
  // Values outside the range are clamped by the CPU-side Covfie wrapper used
  // in MagneticFields.hh; the GPU texture backend also clamps accesses.
  const auto translation = covfie::algebra::affine<3>::translation(-minExtent, -minExtent, -minExtent);
  const auto scale       = static_cast<float>(kSamplesPerAxis - 1) / (maxExtent - minExtent);
  const auto scaling     = covfie::algebra::affine<3>::scaling(scale, scale, scale);

  return ConstantCovfieField(covfie::make_parameter_pack(ConstantCovfieBackend::configuration_t(scaling * translation),
                                                         ConstantCovfieBackend::backend_t::configuration_t{},
                                                         std::move(inner.backend())));
}

/// @brief 2x2x2 covfie map with the same field value at every sample, so constant everywhere.
inline ConstantCovfieField MakeConstantCovfieField(const std::array<float, 3> &fieldValue, float minExtent = -10000.f,
                                                   float maxExtent = 10000.f)
{
  // Every sample has the same value, so interpolation anywhere represents a constant field
  return MakeSampledCovfieField([&](std::size_t, std::size_t, std::size_t) { return fieldValue; }, minExtent,
                                maxExtent);
}

/// @brief Write a map with Bz rising along x: 0 at x = -10000, bz at x = 10000.
/// @param output Stream receiving the covfie dump
/// @param bz Field value at the upper x edge
inline void WriteXGradientCovfieField(std::ostream &output, float bz)
{
  auto field = MakeSampledCovfieField(
      [&](std::size_t x, std::size_t, std::size_t) { return std::array<float, 3>{{0.f, 0.f, bz * x}}; });
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
