// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0
//
// Author:  J. Apostolakis,  10 Jan 2024
//
//  Bogacki-Shampine - 4 - 3(2) non-FSAL implementation 
//
//  An implementation of the embedded RK method from the paper 
//  [1] P. Bogacki and L. F. Shampine,
//     "A 3(2) pair of Runge - Kutta formulas"
//     Appl. Math. Lett., vol. 2, no. 4, pp. 321-325, Jan. 1989.
//
// Non-FSAL implementation
// Based on G4BogackiShampine23
// Created: Somnath Banerjee, Google Summer of Code 2015, 20 May 2015
// Supervision: John Apostolakis, CERN
// --------------------------------------------------------------------

#pragma once

template <class Equation_t, class T_Field, unsigned int Nvar, typename Real_t>
class BogackiShampine23 // : public VScalarIntegrationStepper
{
   // using ThreeVector = vecgeom::Vector3D<Real_t>;

public:
  static constexpr unsigned int kMethodOrder = 3;

  static __host__ __device__
  void EvaluateDerivatives(const T_Field & field, const Real_t y[], int charge, Real_t dydx[]) ;

  static __host__ __device__   
  void StepWithErrorEstimate( const T_Field & field,
                                     const Real_t * yIn, 
                                     const Real_t * dydx,   
                                     int    charge,
                                     Real_t   Step,
                                     Real_t * yOut,        // Output:  y values at end,
                                     Real_t * yerr,        //          estimated errors, 
                                     Real_t * next_dydx);  //          next value of dydx
};

template <class Equation_t, class T_Field, unsigned int Nvar, typename Real_t>
__host__ __device__ void
BogackiShampine23<Equation_t, T_Field, Nvar, Real_t>::
  EvaluateDerivatives( const T_Field& field, const Real_t yIn[], int charge, Real_t dy_ds[] )
{
  Equation_t::EvaluateDerivatives( /* const T_Field& */ field, yIn, charge, dy_ds );
}

// The Bogacki shampine method has the following Butcher's tableau
//
// 0  |
// 1/2|1/2
// 3/4|0        3/4
// 1  |2/9      1/3     4/9
// -------------------
//    |2/9      1/3     4/9    0
//    |7/24 1/4 1/3 1/8

template <class Equation_t, class T_Field, unsigned int Nvar, typename Real_t>
inline
__host__ __device__ void
BogackiShampine23<Equation_t, T_Field, Nvar, Real_t>::
    StepWithErrorEstimate( const T_Field & field,
                           const Real_t  * yIn, 
                           const Real_t  * dydx,   
                           int      charge,
                           Real_t   Step,
                           Real_t * yOut, 
                           Real_t * yError,
                           Real_t * next_dydx )
// yIn and yOut MUST NOT be aliases for same array   
{
  assert( yIn != yOut );

  static constexpr Real_t
                 b21 = 0.5 ,
                 b31 = 0., b32 = 3.0 / 4.0,
                 b41 = 2.0 / 9.0, b42 = 1.0 / 3.0, b43 = 4.0 / 9.0;

  static constexpr Real_t dc1 = b41 - 7.0 / 24.0,  dc2 = b42 - 1.0 / 4.0,
                          dc3 = b43 - 1.0 / 3.0,   dc4 = - 1.0 / 8.0;
 
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
  
  for(unsigned int i=0;i< Nvar;i++)
  {
     yOut[i] = yIn[i] + Step*(b41*dydx[i]   + b42*ak2[i] + b43*ak3[i] );
  }
  EvaluateDerivatives( field, yOut, charge, next_dydx); // 3rd Step

  for(unsigned int i=0;i< Nvar;i++)
  {
    yError[i] = Step * (dc1 * dydx[i] + dc2 * ak2[i] + 
                         dc3 * ak3[i] + dc4 * next_dydx[i]);
  }
}