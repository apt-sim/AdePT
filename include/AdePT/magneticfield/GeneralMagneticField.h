// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0
//
// Author:  S. Diederichs,   11 Nov 2024

#ifndef GeneralMagneticField_H__
#define GeneralMagneticField_H__

#include <VecGeom/base/Vector3D.h>

#ifdef ADEPT_USE_EXT_BFIELD

#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/cuda/backend/primitive/cuda_device_array.hpp>
#include <covfie/cuda/error_check.hpp>

using cpu_field_t = covfie::field<covfie::backend::affine<covfie::backend::linear<
    covfie::backend::strided<covfie::vector::size3, covfie::backend::array<covfie::vector::float3>>>>>;

using cuda_field_t = covfie::field<covfie::backend::affine<covfie::backend::linear<
    covfie::backend::strided<covfie::vector::size3, covfie::backend::cuda_device_array<covfie::vector::float3>>>>>;
using field_view_t = typename cuda_field_t::view_t;
#endif

class GeneralMagneticField {
public:
  GeneralMagneticField() = default;

  bool InitializeFromFile(const std::string &filePath)
  {
#ifdef ADEPT_USE_EXT_BFIELD
    std::ifstream ifs(filePath, std::ifstream::binary);
    if (!ifs.good()) {
      std::cerr << "Failed to open input file " << filePath << "!" << std::endl;
      return false;
    }

    cpu_field_t cpuField(ifs);
    ifs.close();

    // create device field map
    fFieldMap = std::make_unique<cuda_field_t>(cpuField);

    // Create the field view for the device data
    field_view_t fieldView(*fFieldMap);

    // Allocate device memory for the field view
    cudaMalloc(&fFieldView, sizeof(field_view_t));
    cudaMemcpy(fFieldView, &fieldView, sizeof(field_view_t), cudaMemcpyHostToDevice);
    return true;
#endif
    return false;
  }

  __host__ __device__ field_view_t *GetFieldView() const { return fFieldView; }

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
    if (fFieldView) {
      cudaFree(fFieldView);
    }
#endif
  }

private:
#ifdef ADEPT_USE_EXT_BFIELD
  std::unique_ptr<cuda_field_t> fFieldMap; // Device-stored field data
  field_view_t *fFieldView = nullptr;      // Device-stored field view, needed to access the data
#endif
};

#endif // GeneralMagneticField_H__
