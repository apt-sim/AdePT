// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0
//
// Author:  J. Apostolakis,   16 Nov 2021

#ifndef UniformMagneticField_H__
#define UniformMagneticField_H__

// #include <VecCore/Math.h>
#include <VecGeom/base/Vector3D.h>

class UniformMagneticField // : public VVectorField
{
public:
  // static constexpr int gNumFieldComponents  = 3;
  // static constexpr bool gFieldChangesEnergy = false;

  /** @brief Constructor providing the constant field value (cartesian) */
  __host__ __device__ UniformMagneticField(const vecgeom::Vector3D<float> &fieldVector) : fFieldComponents(fieldVector)
  {
  }
  //:  VVectorField(gNumFieldComponents, gFieldChangesEnergy),

  /** @brief Constructor providing the constant field value (spherical) */
  __host__ __device__ UniformMagneticField(char, double vField, double vTheta, double vPhi);

  /** @brief Destructor */
  __host__ __device__ ~UniformMagneticField() {}

  /** @brief Copy constructor */
  __host__ __device__ UniformMagneticField(const UniformMagneticField &p) : fFieldComponents(p.fFieldComponents) {}
  // : VVectorField(gNumFieldComponents, gFieldChangesEnergy),

  /** Assignment operator */
  __host__ __device__ UniformMagneticField &operator=(const UniformMagneticField &p);

  /** @brief Templated field interface */
  template <typename Real_t>
  __host__ __device__ void Evaluate(const vecgeom::Vector3D<Real_t> & /*position*/,
                                    vecgeom::Vector3D<Real_t> &fieldValue) const
  {
    fieldValue.Set(Real_t(fFieldComponents.x()), Real_t(fFieldComponents.y()), Real_t(fFieldComponents.z()));
  }

  /** @brief Templated field interface */
  template <typename Real_t>
  __host__ __device__ auto Evaluate(Real_t x, Real_t y, Real_t z) const -> vecgeom::Vector3D<Real_t>
  {
    return {Real_t(fFieldComponents.x()), Real_t(fFieldComponents.y()), Real_t(fFieldComponents.z())};
  }

  /** @brief Templated field interface */
  template <typename Real_t>
  __host__ __device__ auto Evaluate(const vecgeom::Vector3D<Real_t> & /*position*/) const -> vecgeom::Vector3D<Real_t>
  {
    return {Real_t(fFieldComponents.x()), Real_t(fFieldComponents.y()), Real_t(fFieldComponents.z())};
  }

  /** @brief Interface to evaluate field at location */
  __host__ __device__ void GetFieldValue(const vecgeom::Vector3D<float> &position, vecgeom::Vector3D<float> &fieldValue)
  {
    Evaluate<float>(position, fieldValue);
  }

private:
  vecgeom::Vector3D<float> fFieldComponents;
};

#endif
