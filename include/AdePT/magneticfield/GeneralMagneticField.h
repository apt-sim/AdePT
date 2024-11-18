// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0
//
// Author:  S. Diederichs,   11 Nov 2024

#ifndef GeneralMagneticField_H__
#define GeneralMagneticField_H__

#include <VecGeom/base/Vector3D.h>

#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/cuda/backend/primitive/cuda_device_array.hpp>
#include <covfie/cuda/error_check.hpp>


#ifdef ADEPT_USE_EXT_BFIELD

using cpu_field_t = covfie::field<
    covfie::backend::affine<covfie::backend::linear<covfie::backend::strided<
        covfie::vector::size3,
        covfie::backend::array<covfie::vector::float3>>>>>;

using cuda_field_t = covfie::field<
    covfie::backend::affine<covfie::backend::linear<covfie::backend::strided<
        covfie::vector::size3,
        covfie::backend::cuda_device_array<covfie::vector::float3>>>>>;

#endif 

class GeneralMagneticField // : public VVectorField
{
public:

  // /** @brief Constructor providing the covfie file to be read in (cartesian) */
  // __host__ __device__ GeneralMagneticField(const vecgeom::Vector3D<float> &fieldVector) : fFieldMap(fieldVector)
  // {
  // }

  GeneralMagneticField() = default;


  bool InitializeFromFile(const std::string &filePath) {

#ifdef ADEPT_USE_EXT_BFIELD

    std::cout << "Starting read of input file..." << std::endl;

    std::ifstream ifs(filePath, std::ifstream::binary);
    if (!ifs.good()) {
      std::cerr << "Failed to open input file " << filePath << "!" << std::endl;
      return false;
    }

    // Create the CPU field from the file
    cpu_field_t cpuField(ifs);
    ifs.close();

    // Now, create the cuda_field_t on the GPU using the CPU field
    std::cout << "Casting magnetic field into CUDA array..." << std::endl;
    fFieldMap = std::make_unique<cuda_field_t>(cpuField);
    std::cout << "Done casting magnetic field into CUDA array!" << std::endl;
#endif
    return true;
  }

#ifdef ADEPT_USE_EXT_BFIELD
  typename cuda_field_t::view_t* GetGlobalView() {
    if (fFieldMap) {
      // Create a view from the field map
      typename cuda_field_t::view_t fieldView(*fFieldMap);

      // Allocate device memory for the view pointer
      typename cuda_field_t::view_t* d_fieldView;
      cudaMalloc(&d_fieldView, sizeof(fieldView));
      cudaMemcpy(d_fieldView, &fieldView, sizeof(fieldView), cudaMemcpyHostToDevice);

      return d_fieldView;
    }
    std::cerr << "Error: fFieldMap is not initialized.\n";
    return nullptr;
  }

  /** @brief Destructor */
  __host__ __device__ ~GeneralMagneticField() {}

  /** @brief Copy constructor */
  // __host__ __device__ GeneralMagneticField(const GeneralMagneticField &p) : fFieldMap(p.fFieldMap) {}

  /** Assignment operator */
  __host__ __device__ GeneralMagneticField &operator=(const GeneralMagneticField &p);

  template <typename Real_t>
  __device__ vecgeom::Vector3D<Real_t> Evaluate(const vecgeom::Vector3D<Real_t> &pos) const
  {
    // // Create a view from the field
    typename cuda_field_t::view_t view(*fFieldMap);

    // // Use the view to get the field value at the specified position
    auto field_value = view.at(pos.x(), pos.y(), pos.z());

    return vecgeom::Vector3D<Real_t>(field_value[0], field_value[1], field_value[2]);
  }
#endif
private:

#ifdef ADEPT_USE_EXT_BFIELD
  std::unique_ptr<cuda_field_t> fFieldMap; 
#endif
};


#endif // GeneralMagneticField_H__
