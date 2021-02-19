// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author: J. Apostolakis  Nov/Dec 2020

#ifndef FIELD_PROPAGATOR_CONST_BANY_H
#define FIELD_PROPAGATOR_CONST_BANY_H

#include <VecGeom/base/Vector3D.h>
#include <CopCore/PhysicalConstants.h>

#include "ConstBzFieldStepper.h"
#include "ConstFieldHelixStepper.h"

using copcore::units::kElectronMassC2;

class fieldPropagatorConstBany {
public:
  __host__ __device__ void stepInField(track &aTrack, ConstFieldHelixStepper &helixAnyB,
                                       vecgeom::Vector3D<double> &endPosition, vecgeom::Vector3D<double> &endDirection);
};

// ----------------------------------------------------------------------------

__host__ __device__ void fieldPropagatorConstBany::stepInField(track &aTrack, ConstFieldHelixStepper &helixAnyB,
                                                               vecgeom::Vector3D<double> &endPosition,
                                                               vecgeom::Vector3D<double> &endDirection)
{
  int charge  = aTrack.charge();
  double step = aTrack.interaction_length; // was float

  if (charge != 0) {
    double kinE        = aTrack.energy;
    double momentumMag = sqrt(kinE * (kinE + 2.0 * kElectronMassC2));
    // aTrack.mass() -- when extending with other charged particles

    // For now all particles ( e-, e+, gamma ) can be propagated using this
    //   for gammas  charge = 0 works, and ensures that it goes straight.

    helixAnyB.DoStep(aTrack.pos, aTrack.dir, (double)charge, momentumMag, step, endPosition, endDirection);
  } else {
    // Also move gammas - for now ..
    endPosition  = aTrack.pos + step * aTrack.dir;
    endDirection = aTrack.dir;
  }
}

#endif

// EvaluateField( pclPosition3d, fieldVector );    // initial field value

// vecgeom::Vector3D<floatE_t> momentum = momentumMag * aTrack.dir;
