// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author: J. Apostolakis  Nov/Decxo 2020

#include <VecGeom/base/Vector3D.h>
#include <CopCore/PhysicalConstants.h>

#include "ConstBzFieldStepper.h"
#include "ConstFieldStepper.h"

using copcore::units::kElectronMassC2;

__host__ __device__
void fieldPropagatorConstBz(track &    aTrack,
                            float      BzValue,
                            vecgeom::Vector3D<double> & endPosition,
                            vecgeom::Vector3D<double> & endDirection )
{
  double    step= aTrack.interaction_length;
  int     charge= aTrack.charge();
  
  if ( charge != 0.0 ) {
     double kinE = aTrack.energy;
     double momentumMag = sqrt( kinE * ( kinE + 2.0 * kElectronMassC2) );
     // aTrack.mass() -- when extending with other charged particles 
     
     ConstBzFieldStepper  helixBz(BzValue);
     
     // For now all particles ( e-, e+, gamma ) can be propagated using this
     //   for gammas  charge = 0 works, and ensures that it goes straight.
     
     helixBz.DoStep( aTrack.pos, aTrack.dir, charge, momentumMag, step,
                     endPosition, endDirection);
  } else {
     // Also move gammas - for now ..
     endPosition  = aTrack.pos + step * aTrack.dir;
     endDirection = aTrack.dir;
  }
}

#ifdef B_ANY_VALUE
__host__ __device__
void fieldPropagatorConstBgeneral(track &    aTrack,
                           // const vecgeom::Vector3D<double> magFieldVec,
                           ConstFieldHelixStepper  helixAnyB,

                           vecgeom::Vector3D<double> & endPosition,
                           vecgeom::Vector3D<double> & endDirection )
{
  double    step= aTrack.interaction_length;
  int     charge= aTrack.charge();
  
  if ( charge != 0.0 ) {
     double kinE = aTrack.energy;
     double momentumMag = sqrt( kinE * ( kinE + 2.0 * kElectronMassC2) );
     // aTrack.mass() -- when extending with other charged particles 
     
     // ConstFieldHelixStepper  helixAnyB(magFieldVec);
     
     // For now all particles ( e-, e+, gamma ) can be propagated using this
     //   for gammas  charge = 0 works, and ensures that it goes straight.
     
     helixAnyB.DoStep( aTrack.pos, aTrack.dir, charge, momentumMag, step,
                       endPosition, endDirection);
  } else {
     // Also move gammas - for now ..
     endPosition  = aTrack.pos + step * aTrack.dir;
     endDirection = aTrack.dir;
  }
}
#endif


// For RK methods

// template<unsigned int N>
// struct FieldPropagationBuffer
// {
//  int      charge[N];
//  floatX_t position[3][N];
//  floatE_t momentum[3][N];
//   int      index[N];
//  bool     active[N];
// };


// EvaluateField( pclPosition3d, fieldVector );    // initial field value

// vecgeom::Vector3D<floatE_t> momentum = momentumMag * aTrack.dir;
