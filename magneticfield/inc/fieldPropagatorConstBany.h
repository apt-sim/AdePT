// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author: J. Apostolakis  Nov/Dec 2020

#ifndef FIELD_PROPAGATOR_CONST_BANY_H
#define FIELD_PROPAGATOR_CONST_BANY_H

#include <VecGeom/base/Vector3D.h>

#include "ConstBzFieldStepper.h"
#include "ConstFieldHelixStepper.h"

class fieldPropagatorConstBany {
  using Precision = vecgeom::Precision;

public:
  inline __host__ __device__
  void stepInField(ConstFieldHelixStepper &helixAnyB, double kinE, double mass, int charge,
                   Precision step, vecgeom::Vector3D<vecgeom::Precision> &position,
                   vecgeom::Vector3D<vecgeom::Precision> &direction);
};

// ----------------------------------------------------------------------------

inline __host__ __device__ void fieldPropagatorConstBany::stepInField(ConstFieldHelixStepper &helixAnyB, double kinE,
                                                                      double mass, int charge, Precision step,
                                                                      vecgeom::Vector3D<vecgeom::Precision> &position,
                                                                      vecgeom::Vector3D<vecgeom::Precision> &direction)
{
  using Precision = vecgeom::Precision;
  if (charge != 0) {
    Precision momentumMag = sqrt(kinE * (kinE + 2.0 * mass));

    // For now all particles ( e-, e+, gamma ) can be propagated using this
    //   for gammas  charge = 0 works, and ensures that it goes straight.

    vecgeom::Vector3D<Precision> endPosition  = position;
    vecgeom::Vector3D<Precision> endDirection = direction;
    helixAnyB.DoStep<Precision,int>(position, direction, charge, momentumMag, step, endPosition, endDirection);
    position  = endPosition;
    direction = endDirection;
  } else {
    // Also move gammas - for now ..
    position = position + step * direction;
  }
}

#endif
