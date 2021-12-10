// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/*
 * Created on: January 5, 2021
 *
 *      Author: J. Apostolakis
 *
 *  Simple class to contain the value of uniform field
 *   and provide it in device (GPU) and host consistently
 *
 *  'Agnostic' version -- simple type, plain data,
 *                        it does not know where it resides (host or device)
 *                        no explicit transport of data to device
 */

#ifndef UNIFORMMAGFIELD_H
#define UNIFORMMAGFIELD_H

#include <VecGeom/base/Math.h>

class uniformMagField {
  using Precision = vecgeom::Precision;

public:
  __host__ __device__ uniformMagField(Precision inpFieldValue[3]);

  inline __host__ __device__ void ObtainField(Precision fieldValue[3]);

  inline __host__ __device__ void EvaluateField(const Precision pos[3], Precision fieldVal[3])
  {
    ObtainField(fieldVal);
  }

  inline __host__ __device__ void SetValue(Precision inpFieldValue[3]); // Only during setup!

private:
  Precision Bx, By, Bz;
};

inline __host__ __device__ uniformMagField::uniformMagField(Precision inpFieldValue[3])
{
  SetValue(inpFieldValue);
}

inline __host__ __device__ void uniformMagField::SetValue(Precision inpFieldValue[3])
{
  Bx = inpFieldValue[0];
  By = inpFieldValue[1];
  Bz = inpFieldValue[2];
}

inline __host__ __device__ void uniformMagField::ObtainField(Precision fieldValue[3])
{
  fieldValue[0] = Bx;
  fieldValue[1] = By;
  fieldValue[2] = Bz;
}

#endif
