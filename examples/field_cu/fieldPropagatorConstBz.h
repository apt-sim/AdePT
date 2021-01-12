// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author: J. Apostolakis  Nov/Decxo 2020

#include <VecGeom/base/Vector3D.h>
#include <CopCore/PhysicalConstants.h>

#include <AdePT/BlockData.h>

#include "ConstBzFieldStepper.h"


class fieldPropagatorConstBz
{
  public: 
    __host__ __device__
    void stepInField(track &    aTrack,
                     float      BzValue,
                     vecgeom::Vector3D<double> & endPosition,
                     vecgeom::Vector3D<double> & endDirection );

};

__global__
void moveInField(adept::BlockData<track> * trackBlock,
                 fieldPropagatorConstBz&   fieldProp,
                 float                     BzValue );


__host__ __device__
void fieldPropagatorConstBz::stepInField(track &    aTrack,
                                         float      BzValue,
                                         vecgeom::Vector3D<double> & endPosition,
                                         vecgeom::Vector3D<double> & endDirection )
{
  using copcore::units::kElectronMassC2;
   
  double    step= aTrack.interaction_length;
  int     charge= aTrack.charge();
  
  if ( charge != 0.0 ) {
     double kinE = aTrack.energy;
     double momentumMag = sqrt( kinE * ( kinE + 2.0 * kElectronMassC2) );
     // aTrack.mass() -- when extending with other charged particles 
     
     // For now all particles ( e-, e+, gamma ) can be propagated using this
     //   for gammas  charge = 0 works, and ensures that it goes straight.
     ConstBzFieldStepper  helixBz(BzValue);

     helixBz.DoStep( aTrack.pos, aTrack.dir, charge, momentumMag, step,
                     endPosition, endDirection);
  } else {
     // Also move gammas - for now ..
     endPosition  = aTrack.pos + step * aTrack.dir;
     endDirection = aTrack.dir;
  }
}

// -------------------------------------------------------------------------------

// V1 -- field along Z axis
__global__ void 
moveInField(adept::BlockData<track> *trackBlock,
            fieldPropagatorConstBz&    ,    // Just to choose this type of propagation
            float Bz )                      // Field strength carried there (for now)
{
  vecgeom::Vector3D<double> endPosition;
  vecgeom::Vector3D<double> endDirection;

  int maxIndex = trackBlock->GetNused() + trackBlock->GetNholes();   
  
  // Non-block version:
  //   int pclIdx = blockIdx.x * blockDim.x + threadIdx.x;
  fieldPropagatorConstBz fieldProp;
  for (int pclIdx  = blockIdx.x * blockDim.x + threadIdx.x;  pclIdx < maxIndex;
           pclIdx += blockDim.x * gridDim.x)
  {
     track& aTrack= (*trackBlock)[pclIdx];

     // check if you are not outside the used block
     if (pclIdx >= maxIndex || aTrack.status == dead) continue;

     // fieldPropagatorConstBz::
     fieldProp.stepInField(aTrack, Bz, endPosition, endDirection);

     // Update position, direction     
     aTrack.pos = endPosition;  
     aTrack.dir = endDirection;
  }
}

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
