// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author: J. Apostolakis  Nov/Dec 2020

#ifndef FIELD_PROPAGATOR_CONST_BANY_H
#define FIELD_PROPAGATOR_CONST_BANY_H

#include <VecGeom/base/Vector3D.h>

#include "ConstBzFieldStepper.h"
#include "ConstFieldHelixStepper.h"

class fieldPropagatorConstBany {
public:
  __host__ __device__ void stepInField(ConstFieldHelixStepper &helixAnyB, double kinE, double mass, int charge,
                                       double step, vecgeom::Vector3D<double> &position,
                                       vecgeom::Vector3D<double> &direction);
};

// ----------------------------------------------------------------------------

__host__ __device__ void fieldPropagatorConstBany::stepInField(ConstFieldHelixStepper &helixAnyB, double kinE,
                                                               double mass, int charge, double step,
                                                               vecgeom::Vector3D<double> &position,
                                                               vecgeom::Vector3D<double> &direction)
{
  if (charge != 0) {
    double momentumMag = sqrt(kinE * (kinE + 2.0 * mass));

    // For now all particles ( e-, e+, gamma ) can be propagated using this
    //   for gammas  charge = 0 works, and ensures that it goes straight.

    vecgeom::Vector3D<double> endPosition  = position;
    vecgeom::Vector3D<double> endDirection = direction;
    helixAnyB.DoStep(position, direction, charge, momentumMag, step, endPosition, endDirection);
    position  = endPosition;
    direction = endDirection;
  } else {
    // Also move gammas - for now ..
    position = position + step * direction;
  }
}

#endif
