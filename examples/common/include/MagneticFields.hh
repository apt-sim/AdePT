// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "G4MagneticField.hh"
#include "G4ThreeVector.hh"

#include <memory>
#include <utility>
#include <variant>

#ifdef ADEPT_USE_EXT_BFIELD

#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/backend/transformer/clamp.hpp>
#include <covfie/core/field.hpp>

#include <fstream>

using cpu_field_t = covfie::field<covfie::backend::affine<covfie::backend::linear<
    covfie::backend::strided<covfie::vector::size3, covfie::backend::array<covfie::vector::float3>>>>>;

// Clamp the continuous grid coordinate to [0, n-1] before interpolating, as the CUDA texture of the GPU field does; the
// inner clamp keeps the upper interpolation neighbour inside the grid
using cpu_strided_t = covfie::backend::strided<covfie::vector::size3, covfie::backend::array<covfie::vector::float3>>;
using cpu_coord_clamp_t   = covfie::backend::clamp<covfie::backend::linear<covfie::backend::clamp<cpu_strided_t>>>;
using cpu_field_clamped_t = covfie::field<covfie::backend::affine<cpu_coord_clamp_t>>;

/// @brief Build the CPU field from the IO field read from a covfie file, clamped like the GPU texture.
/// @param ioField Field as read from the file (affine, linear, strided array)
/// @return Field that takes the edge value outside the map
inline cpu_field_clamped_t MakeClampedField(const cpu_field_t &ioField)
{
  const auto &grid = ioField.backend().get_backend().get_backend();
  const auto n     = grid.get_configuration();
  typename cpu_coord_clamp_t::configuration_t range{
      {0.f, 0.f, 0.f}, {static_cast<float>(n[0] - 1), static_cast<float>(n[1] - 1), static_cast<float>(n[2] - 1)}};
  return cpu_field_clamped_t(covfie::make_parameter_pack(ioField.backend().get_configuration(), std::move(range),
                                                         std::monostate{}, cpu_strided_t::owning_data_t(grid)));
}

#endif

class G4GenericMessenger;

class IField {
public:
  virtual ~IField()                                                         = default;
  virtual void GetFieldValue(const G4double point[4], double *bField) const = 0;
};

class UniformField : public IField {
public:
  UniformField(const G4ThreeVector &field) : fField(field) {}

  void GetFieldValue(const G4double[4], double *bField) const override
  {
    bField[0] = fField.x();
    bField[1] = fField.y();
    bField[2] = fField.z();
  }

private:
  G4ThreeVector fField;
};

class CovfieField : public IField {
public:
  CovfieField(const std::string &filename)
  {
    if (filename != "") {
#ifdef ADEPT_USE_EXT_BFIELD
      // Initialize field map using Covfie library
      std::ifstream ifs(filename, std::ifstream::binary);
      if (!ifs.good()) throw std::runtime_error("Failed to open field file: " + filename);

      // Covfie requires that the B field type of the writer is the same as the B field type of the reader
      // For the GPU B field in `GeneralMagneticField.cuh` no clamping of the field is possible, as this is done
      // automatically by the texture memory. Therefore, unclamped B fields must be generated as the IO B field, and
      // the CPU field clamps the same way as the texture
      cpu_field_t IoField = cpu_field_t(ifs);
      fCpuField           = MakeClampedField(IoField);
      fFieldView          = std::make_unique<cpu_field_clamped_t::view_t>(fCpuField);
#endif
    }
  }

  void GetFieldValue(const G4double point[4], double *bField) const override
  {
#ifdef ADEPT_USE_EXT_BFIELD
    auto field_value = fFieldView->at(point[0], point[1], point[2]);
    bField[0]        = field_value[0];
    bField[1]        = field_value[1];
    bField[2]        = field_value[2];
#endif
  }

private:
#ifdef ADEPT_USE_EXT_BFIELD
  cpu_field_clamped_t fCpuField;
  std::unique_ptr<cpu_field_clamped_t::view_t> fFieldView;
#endif
};

class MagneticField : public G4MagneticField {
public:
  MagneticField(std::unique_ptr<IField> field) : fField(std::move(field)) {}
  ~MagneticField() override = default;

  void GetFieldValue(const G4double point[4], double *bField) const override { fField->GetFieldValue(point, bField); }

private:
  std::unique_ptr<IField> fField;
};
