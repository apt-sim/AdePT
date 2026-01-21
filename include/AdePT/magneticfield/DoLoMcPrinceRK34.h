// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0
//
// Author:  J. Apostolakis,  10 Jan 2024
//
// Implementation of the Dormand Lockyer McGorrigan Prince non-FSAL method for AdePT.
//
// Notes: 
//  - Current version is restricted to Magnetic fields (see EvaluateDerivatives.)
//  - It provides the next value of dy/ds in 'next_dydx'
//  - It uses a large number of registers and/or stack locations - 6 derivatives + In + Out + Err
//
//  Dormand-Lockyer-McGorrigan-Prince-6-3-4 non-FSAL method
//  ( 6 stage, 3rd & 4th order embedded RK method )

// Based on G4DoLoMcPrinceRK34, 
// Created by Somnath Banerjee, Google Summer of Code 2015, 7 July 2015
// Supervision: John Apostolakis, CERN
// --------------------------------------------------------------------

#pragma once

#include "RkIntegrationDriver.h"

template <class Equation_t, class T_Field, unsigned int Nvar, typename Real_t>
class DoLoMcPrinceRK34 // : public VScalarIntegrationStepper
{
  public:
    static constexpr unsigned int kMethodOrder = 4;

    static __host__ __device__
    void EvaluateDerivatives(const T_Field & field, const Real_t y[], int charge, Real_t dydx[]) ;

    static __host__ __device__   
    void StepWithErrorEstimate( const T_Field & field,
                                const Real_t * yIn, 
                                const Real_t * dydx,   
                                int    charge,
                                Real_t   Step,
                                Real_t   yOut[Nvar],     // Output:  y values at end,
                                Real_t   yerr[Nvar],     //          estimated errors, 
                                Real_t * next_dydx  );     //          next value of dydx
};

template <class Equation_t, class T_Field, unsigned int Nvar, typename Real_t>
__host__ __device__ void
DoLoMcPrinceRK34<Equation_t, T_Field, Nvar, Real_t>::
  EvaluateDerivatives( const T_Field& field, const Real_t yIn[], int charge, Real_t dy_ds[] )
{
  Equation_t::EvaluateDerivatives( /* const T_Field& */ field, yIn, charge, dy_ds );
}

template <class Equation_t, class T_Field, unsigned int Nvar, typename Real_t>
inline
__host__ __device__ void
DoLoMcPrinceRK34<Equation_t, T_Field, Nvar, Real_t>::
    StepWithErrorEstimate( const T_Field & field,
                           const Real_t  * yIn, 
                           const Real_t  * dydx,   
                           int      charge,
                           Real_t   Step,
                           Real_t   yOut[Nvar], 
                           Real_t   yErr[Nvar],
                           Real_t * next_dydx )
// yIn and yOut MUST NOT be aliases for same array   
{
  // The constants from the butcher tableu
  //
  static constexpr Real_t 
       b21 = 7.0/27.0 ,  
       b31 = 7.0/72.0 ,      b32 = 7.0/24.0 ,
       b41 = 3043.0/3528.0 , b42 = -3757.0/1176. ,  b43 = 1445.0/441.0,
       b51 = 17617.0/11662 , b52 = -4023.0/686.0 ,  b53 = 9372.0/1715. ,  b54 = -66.0/595.0 ,
       b61 = 29.0/238.0 ,    b62 = 0.0 ,            b63 = 216.0/385.0  ,  b64 = 54.0/85.0 ,  b65 = -7.0/22.0 ,

       dc1 = 363.0/2975.0 - b61 ,
       dc2 = 0.0 - b62 ,
       dc3 = 981.0/1750.0 - b63,
       dc4 = 2709.0/4250.0 - b64 ,
       dc5 = -3.0/10.0 - b65 ,
       dc6 = -1.0/50.0 ;        // end of declaration

  assert( yIn != yOut);

  // EvaluateDerivatives( field, yIn, charge,  dydx) ;      // 1st Step

  Real_t ak2[Nvar];
  {
    Real_t yTemp2[Nvar];
    for (unsigned int i = 0; i < Nvar; i++) {
      yTemp2[i] = yIn[i] + b21 * Step * dydx[i];
    }
    EvaluateDerivatives( field, yTemp2, charge, ak2); // 2nd Step
  }
  
  Real_t ak3[Nvar];
  {
    Real_t yTemp3[Nvar];
    for (unsigned int i = 0; i < Nvar; i++) {
      yTemp3[i] = yIn[i] + Step * (b31 * dydx[i] + b32 * ak2[i]);
    }
    EvaluateDerivatives( field, yTemp3, charge, ak3); // 3rd Step
  }
  
  Real_t ak4[Nvar];
  {
    Real_t yTemp4[Nvar];
    for (unsigned int i = 0; i < Nvar; i++) {
      yTemp4[i] = yIn[i] + Step * (b41 * dydx[i] + b42 * ak2[i] + b43 * ak3[i]);
    }
    EvaluateDerivatives( field, yTemp4, charge, ak4); // 4th Step
  }

  Real_t ak5[Nvar];
  {
    Real_t yTemp5[Nvar];
    for (unsigned int i = 0; i < Nvar; i++) {
      yTemp5[i] = yIn[i] + Step * (b51 * dydx[i] + b52 * ak2[i] + b53 * ak3[i] + b54 * ak4[i]);
    }
    EvaluateDerivatives( field, yTemp5, charge, ak5); // 5th Step
  }
  
  Real_t ak6[Nvar];
  for (unsigned int i = 0; i < Nvar; i++) {
      yOut[i] =
         yIn[i] + Step * (b61 * dydx[i] + b62 * ak2[i] + b63 * ak3[i] + b64 * ak4[i] + b65 * ak5[i]);
  }
  EvaluateDerivatives( field, yOut, charge, ak6); //   6th evaluation -- only for Derivative ... ?

  for (unsigned int i = 0; i < Nvar; i++)
  {      
    yErr[i] = Step*(dc1*dydx[i] + dc2*ak2[i] + dc3*ak3[i] + dc4*ak4[i]
                     + dc5*ak5[i] + dc6*ak6[i] ) ;
  }
   
  return ;
}

