// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0
//
// Author:  S. Diederichs,   11 Nov 2024

#pragma once

#include <AdePT/AdePTBuildFlags.hh>
#include <AdePT/transport/support/Portability.hh>

#include <VecGeom/base/Vector3D.h>

#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <variant>

#ifdef ADEPT_USE_EXT_BFIELD

#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/clamp.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#if defined(__HIP__)
#include <covfie/hip/backend/primitive/hip_device_array.hpp>
#else
#include <covfie/cuda/backend/primitive/cuda_device_array.hpp>
#include <covfie/cuda/error_check.hpp>

#include <covfie/cuda/backend/primitive/cuda_texture.hpp>
#endif

using cpu_field_t = covfie::field<covfie::backend::affine<covfie::backend::linear<
    covfie::backend::strided<covfie::vector::size3, covfie::backend::array<covfie::vector::float3>>>>>;

// use normal GPU memory
// using cuda_field_t = covfie::field<covfie::backend::affine<covfie::backend::linear<
//     covfie::backend::strided<covfie::vector::size3, covfie::backend::cuda_device_array<covfie::vector::float3>>>>>;

#if defined(__HIP__)
// covfie has no HIP texture backend: trilinear interpolation over normal GPU memory. As the CUDA texture does, the grid
// coordinate is clamped to [0, n-1] before interpolating; the inner clamp keeps the upper neighbour inside the grid
template <class Array>
using hip_clamped_t    = covfie::backend::affine<covfie::backend::clamp<
       covfie::backend::linear<covfie::backend::clamp<covfie::backend::strided<covfie::vector::size3, Array>>>>>;
using cuda_field_t     = covfie::field<hip_clamped_t<covfie::backend::hip_device_array<covfie::vector::float3>>>;
using hip_host_field_t = covfie::field<hip_clamped_t<covfie::backend::array<covfie::vector::float3>>>;
#else
// use GPU texture memory
using cuda_field_t = covfie::field<
    covfie::backend::affine<covfie::backend::cuda_texture<covfie::vector::float3, covfie::vector::float3>>>;
#endif

using field_view_t = typename cuda_field_t::view_t;
#endif

class GeneralMagneticField {
public:
  GeneralMagneticField() = default;

  GeneralMagneticField(const GeneralMagneticField &)            = delete;
  GeneralMagneticField &operator=(const GeneralMagneticField &) = delete;

  bool InitializeFromFile(const std::string &filePath)
  {
    std::ifstream ifs(filePath, std::ifstream::binary);
#ifdef ADEPT_USE_EXT_BFIELD
    if (!ifs.good()) {
      std::cerr << "Failed to open input file " << filePath << "!" << std::endl;
      return false;
    }

    cpu_field_t cpuField(ifs);
    ifs.close();

#if defined(__HIP__)
    // device field in normal GPU memory
    const auto &grid = cpuField.backend().get_backend().get_backend();
    const auto n     = grid.get_configuration();
    typename hip_host_field_t::backend_t::backend_t::configuration_t range{
        {0.f, 0.f, 0.f}, {static_cast<float>(n[0] - 1), static_cast<float>(n[1] - 1), static_cast<float>(n[2] - 1)}};
    fFieldMap = std::make_unique<cuda_field_t>(
        hip_host_field_t(covfie::make_parameter_pack(cpuField.backend().get_configuration(), std::move(range),
                                                     std::monostate{}, typename std::decay_t<decltype(grid)>(grid))));
#else
    // create device field map in texture memory
    fFieldMap = std::make_unique<cuda_field_t>(covfie::make_parameter_pack(
        cpuField.backend().get_configuration(), cpuField.backend().get_backend().get_backend()));
#endif

    // alternative: create device field in global GPU memory
    // fFieldMap = std::make_unique<cuda_field_t>(cpuField);

    // Create the field view for the device data
    field_view_t fieldView(*fFieldMap);
    // Allocate device memory for the field view
    ADEPT_DEVICE_API_CALL(Malloc(&fFieldView, sizeof(field_view_t)));
    // Copy the field view to device
    ADEPT_DEVICE_API_CALL(
        Memcpy(fFieldView, &fieldView, sizeof(field_view_t), ADEPT_DEVICE_API_SYMBOL(MemcpyHostToDevice)));
    return true;
#endif
    return false;
  }

#ifdef ADEPT_USE_EXT_BFIELD
  __host__ __device__ field_view_t *GetFieldView() const { return fFieldView; }
#endif

  template <typename Real_t>
  __device__ vecgeom::Vector3D<Real_t> Evaluate(const vecgeom::Vector3D<Real_t> &pos) const
  {
#ifdef ADEPT_USE_EXT_BFIELD
    auto field_value = fFieldView->at(pos.x(), pos.y(), pos.z());
    return vecgeom::Vector3D<Real_t>(field_value[0], field_value[1], field_value[2]);
#else
    return vecgeom::Vector3D<Real_t>(0, 0, 0);
#endif
  }

  template <typename Real_t>
  __device__ vecgeom::Vector3D<Real_t> Evaluate(const Real_t pos_x, const Real_t pos_y, const Real_t pos_z) const
  {
#ifdef ADEPT_USE_EXT_BFIELD
    auto field_value = fFieldView->at(pos_x, pos_y, pos_z);
    return vecgeom::Vector3D<Real_t>(field_value[0], field_value[1], field_value[2]);
#else
    return vecgeom::Vector3D<Real_t>(0, 0, 0);
#endif
  }

  ~GeneralMagneticField()
  {
#ifdef ADEPT_USE_EXT_BFIELD
    try {
      ADEPT_DEVICE_API_CALL(Free(fFieldView));
    } catch (const std::exception &e) {
      std::cerr << "\033[31m" << "GPUstate::~GPUstate : Error during device API call" << "\033[0m" << std::endl;
      std::cerr << "\033[31m" << e.what() << "\033[0m" << std::endl;
    }
#endif
  }

private:
#ifdef ADEPT_USE_EXT_BFIELD
  std::unique_ptr<cuda_field_t> fFieldMap; // Field data
  field_view_t *fFieldView = nullptr;      // Device-stored field view, needed to access the data
#endif
};
