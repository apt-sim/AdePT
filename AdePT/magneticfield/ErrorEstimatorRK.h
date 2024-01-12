// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0
//
//  ErrorEstimatorRK: Simple class to Estimate the overall relative error
//                     from the RK method's error per (integration) variable
//
//  Created on: November 15, 2021
//      Author: J. Apostolakis

#ifndef ErrorEstimatorRK_h
#define ErrorEstimatorRK_h

class ErrorEstimatorRK
{
  public:
    __host__ __device__   
    ErrorEstimatorRK( float  eps_rel_max,
                      float  minimumStep,
                      int    noComponents = 6
       ) :
        fEpsRelMax (eps_rel_max ),
        fInvEpsilonRelSq( 1.0 / (eps_rel_max * eps_rel_max) ),
        fMinimumStep( minimumStep )    // , fNoComponents( 6 )
        
    {} 
   
    template <class Real_t>
    __host__ __device__    
      Real_t EstimateSquareError( const Real_t   yError[],
                                  const Real_t & hStep,
                                  // const Real_t yValue[fNoComponents],
                                  const Real_t & magMomentumSq //   Initial momentum square (used for rel. error)
         ) const;
    //  Returns the Maximum Square Error: the maximum between 
    //    - position relative error square (magnitude^2): (momentum error vec)^2 / (initial momentum)^2
    //    - momentum relative error square (magnitude^2): (position error vec)^2 / (step length)^2
    //
    //  Last argument enables the use of initial momentum square in calculating the relative error

    __host__ __device__     
    float GetMaxRelativeError() const { return fEpsRelMax; }

  public:
    static constexpr const double tinyValue = 1.0e-30; // Just to ensure there is no division by zero
    
  private:
    const float  fEpsRelMax;
    const float  fInvEpsilonRelSq; // = 1.0 / (eps_rel_max * eps_rel_max);
    const float  fMinimumStep;
    // constexpr int    fNoComponents = 6;
    
};

template <class Real_t>
__host__ __device__
Real_t ErrorEstimatorRK::EstimateSquareError(
             const Real_t   yEstError[],  // [fNoComponents]
             const Real_t & hStep,
             const Real_t & magInitMomentumSq // (Initial) momentum square (used for rel. error)
      ) const
{
   Real_t invMagMomentumSq = 1.0 / (magInitMomentumSq + tinyValue);
   Real_t epsPosition;
   Real_t errpos_sq, errmom_sq;
   
   epsPosition = fEpsRelMax * vecCore::math::Max(hStep, Real_t(fMinimumStep));
   // Note: it uses the remaining step 'hStep'
   //       Could change it to use full step size ==> move it outside loop !! 2017.11.10 JA

   Real_t invEpsPositionSq = 1.0 / (epsPosition * epsPosition);

   // Evaluate accuracy
   errpos_sq = yEstError[0] * yEstError[0] + yEstError[1] * yEstError[1] + yEstError[2] * yEstError[2];
   errpos_sq *= invEpsPositionSq; // Scale relative to required tolerance
   // Accuracy for momentum
   
   Real_t sumerr_sq = yEstError[3] * yEstError[3] + yEstError[4] * yEstError[4] + yEstError[5] * yEstError[5];
   errmom_sq = fInvEpsilonRelSq * invMagMomentumSq * sumerr_sq ;
   
   return vecCore::math::Max(errpos_sq, errmom_sq); // Maximum Square Error
}
#endif
