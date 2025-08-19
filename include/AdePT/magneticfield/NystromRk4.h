// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0
//
// Author:  J. Apostolakis,  11 Dec 2023
//
// Implementation of the Nystrom Rk4 Runge-Kutta integrator for AdePT.
// Derived from the Geant4 G4NystromRK4
// 
// Notes: 
//  - Current version is restricted to Magnetic fields (see EvaluateDerivatives.)
//  - It provides the next value of dy/ds in 'next_dydx'
//  - It uses a large number of registers and/or stack locations - 7 derivatives + In + Out + Err

template <class Equation_t, class T_Field, unsigned int Nvar, typename Real_t>
class NystromRK4 // : public VScalarIntegrationStepper
{
   // using ThreeVector = vecgeom::Vector3D<Real_t>;

public:
  static constexpr unsigned int kMethodOrder = 4;
  // inline NystromRK4(Equation_t *EqRhs, bool verbose = false);
  // NystromRK4(const NystromRK4 &) = delete;
  // ~NystromRK4() {}

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
NystromRK4<Equation_t, T_Field, Nvar, Real_t>::
  EvaluateDerivatives( const T_Field& field, const Real_t yIn[], int charge, Real_t dy_ds[] )
{
/* #ifdef VERBOSE_RHS
    using geant::units::tesla;
    std::cout << "NystromRK4::EvaluateDerivatives called with q= " << charge
              << " at Position = " << yIn[0] << " y= " << yIn[1] << " z= " << yIn[2]
              << " with Momentum = " << yIn[3] << " y= " << yIn[4] << " z= " << yIn[5] << " ";
#endif */
   
  // Vector3D<Real_t> Bfield;
  // Equation_t::EvaluateDerivativesReturnB( field, yIn, charge, dy_ds, Bfield );

  Equation_t::EvaluateDerivatives( /* const T_Field& */ field, yIn, charge, dy_ds );

 /*********
  using copcore::units::tesla;
  using std::setw;
  constexpr int prec= 5;
  constexpr int nf= prec+5;
  int old_prec = std::cout.precision(prec);
  std::cout << " DoPri5: Evaluate Derivatives - using B-field,  Bx= " << Bfield.x() / tesla << " By= " << Bfield.y() / tesla << " Bz= " << Bfield.z() / tesla << " ";
  std::cout << " gives Derivs dy_ds= :  " 
            << " x = " << setw(nf) << dy_ds[0] << " y = " << setw(nf) << dy_ds[1] << " z = " << setw(nf) << dy_ds[2]
            << " px= " << setw(nf) << dy_ds[3] << " py= " << setw(nf) << dy_ds[4] << " pz= " << setw(nf) << dy_ds[5]
            << std::endl;
  std::cout.precision(old_prec);
  ********/
}

template <class Equation_t, class T_Field, unsigned int Nvar, typename Real_t>
inline
__host__ __device__ void
NystromRK4<Equation_t, T_Field, Nvar, Real_t>::
    StepWithErrorEstimate( const T_Field & field,
                           const Real_t P[],    // yIn
                           const Real_t dPdS[], // dydx 
                           int      charge,
                           Real_t   Step,
                           Real_t * Po,  // yOut 
                           Real_t * Err, // yErr
                           Real_t * next_dydx )
// yIn and yOut MUST NOT be aliases for same array   
{
  assert( yIn != yOut );
  const Real_t perMillion = 1.0e-6;
  Real_t R[4] = {   P[0],   P[1] ,    P[2],  P[7] };   // x, y, z, t
  Real_t A[3] = {dPdS[0], dPdS[1], dPdS[2]};

  // m_iPoint[0]=R[0]; m_iPoint[1]=R[1]; m_iPoint[2]=R[2];

  constexpr Real_t one_sixth= 1./6.;
  const Real_t S  =     Step   ;
  const Real_t S5 =  .5*Step   ;
  const Real_t S4 = .25*Step   ;
  const Real_t S6 =     Step * one_sixth;   // Step / 6.;
  
  // Ensure that the location and cached field value are correct
  EvaluateDerivatives( field,  R );

  // Ensure that the momentum is set correctly.

  // - Quick check momentum magnitude (squared) against previous value
  Real_t newmom2 = (P[3]*P[3]+P[4]*P[4]+P[5]*P[5]); 
  Real_t oldmom2 = m_mom * m_mom;
  if( std::fabs(newmom2 - oldmom2) > perMillion * oldmom2 )
  {
     m_mom   = std::sqrt(newmom2) ;
     m_imom  = 1./m_mom;
     m_cof   = m_fEq->FCof()*m_imom;
  }

#ifdef  G4DEBUG_FIELD
  CheckCachedMomemtum( P, m_mom );
  CheckFieldPosition( P, m_fldPosition );
#endif
  
  // Point 1
  //
  Real_t K1[3] = { m_imom*dPdS[3], m_imom*dPdS[4], m_imom*dPdS[5] };
  
  // Point2
  //
  Real_t p[4] = { R[0]+S5*(A[0]+S4*K1[0]),
		              R[1]+S5*(A[1]+S4*K1[1]),
		              R[2]+S5*(A[2]+S4*K1[2]),
		              P[7]                   }; 
  EvaluateDerivatives( field, p);

  Real_t A2[3] = {A[0]+S5*K1[0], A[1]+S5*K1[1], A[2]+S5*K1[2]};
  Real_t K2[3] = {(A2[1]*m_lastField[2]-A2[2]*m_lastField[1])*m_cof,
		    (A2[2]*m_lastField[0]-A2[0]*m_lastField[2])*m_cof,
		    (A2[0]*m_lastField[1]-A2[1]*m_lastField[0])*m_cof};
 
  m_mPoint[0]=p[0]; m_mPoint[1]=p[1]; m_mPoint[2]=p[2];

  // Point 3 with the same magnetic field
  //
  Real_t A3[3] = {A[0]+S5*K2[0],A[1]+S5*K2[1],A[2]+S5*K2[2]};
  Real_t K3[3] = {(A3[1]*m_lastField[2]-A3[2]*m_lastField[1])*m_cof,
		    (A3[2]*m_lastField[0]-A3[0]*m_lastField[2])*m_cof,
		    (A3[0]*m_lastField[1]-A3[1]*m_lastField[0])*m_cof};
  
  // Point 4
  //
  p[0] = R[0]+S*(A[0]+S5*K3[0]);
  p[1] = R[1]+S*(A[1]+S5*K3[1]);
  p[2] = R[2]+S*(A[2]+S5*K3[2]);             

  EvaluateDerivatives( field, p);
  
  Real_t A4[3] = {A[0]+S*K3[0],A[1]+S*K3[1],A[2]+S*K3[2]};
  Real_t K4[3] = {(A4[1]*m_lastField[2]-A4[2]*m_lastField[1])*m_cof,
		    (A4[2]*m_lastField[0]-A4[0]*m_lastField[2])*m_cof,
		    (A4[0]*m_lastField[1]-A4[1]*m_lastField[0])*m_cof};
  
  // New position
  //
  Po[0] = P[0]+S*(A[0]+S6*(K1[0]+K2[0]+K3[0]));
  Po[1] = P[1]+S*(A[1]+S6*(K1[1]+K2[1]+K3[1]));
  Po[2] = P[2]+S*(A[2]+S6*(K1[2]+K2[2]+K3[2]));

  m_fPoint[0]=Po[0]; m_fPoint[1]=Po[1]; m_fPoint[2]=Po[2];

  // New direction
  //
  Po[3] = A[0]+S6*(K1[0]+K4[0]+2.*(K2[0]+K3[0]));
  Po[4] = A[1]+S6*(K1[1]+K4[1]+2.*(K2[1]+K3[1]));
  Po[5] = A[2]+S6*(K1[2]+K4[2]+2.*(K2[2]+K3[2]));

  // Errors
  //
  Err[3] = S*std::fabs(K1[0]-K2[0]-K3[0]+K4[0]);
  Err[4] = S*std::fabs(K1[1]-K2[1]-K3[1]+K4[1]);
  Err[5] = S*std::fabs(K1[2]-K2[2]-K3[2]+K4[2]);
  Err[0] = S*Err[3]                       ;
  Err[1] = S*Err[4]                       ;
  Err[2] = S*Err[5]                       ;
  Err[3]*= m_mom                          ;
  Err[4]*= m_mom                          ;
  Err[5]*= m_mom                          ;

  // Normalize momentum
  //
  Real_t normF = m_mom/std::sqrt(Po[3]*Po[3]+Po[4]*Po[4]+Po[5]*Po[5]);
  Po [3]*=normF; Po[4]*=normF; Po[5]*=normF; 

  // Pass Energy, time unchanged -- time is not integrated !!
  // Po[6]=P[6]; Po[7]=P[7];
#if ENABLE_CHORD_DIST
  for (unsigned int i = 0; i < Nvar; i++) {
    // Store Input and Final values, for possible use in calculating chord
    fLastInitialVector[i] = yIn[i];
    fLastFinalVector[i]   = yOut[i];
    fInitialDyDx[i]       = dydx[i];   // At initial point 
  }
#endif
  // fLastStepLength = Step;
  
  // std::cout << " Exiting StepWithErrorEstimate of scalar " << std::endl;

  return;
}

/**********************************************************************
#if ENABLE_CHORD_DIST
template <class Equation_t, unsigned int Nvar>
inline Real_t NystromRK4<Equation_t, Nvar>::DistChord() const
{
  // Coefficients were taken from Some Practical Runge-Kutta Formulas by Lawrence F. Shampine, page 149, c*
  static constexpr Real_t hf1 = 6025192743.0 / 30085553152.0,
      hf2 = 0.0,
      hf3 = 51252292925.0 / 65400821598.0,
      hf4 = - 2691868925.0 / 45128329728.0,
      hf5 = 187940372067.0 / 1594534317056.0,
      hf6 = - 1776094331.0 / 19743644256.0,
      hf7 = 11237099.0 / 235043384.0;

  Real_t midVector[3];

  for(int i = 0; i < 3; ++i) {
     midVector[i] = fLastInitialVector[i] + 0.5 * fLastStepLength * 
          (hf1 * fInitialDyDx[i] + hf2 * ak2[i] + hf3 * ak3[i] + 
           hf4 * ak4[i] + hf5 * ak5[i] + hf6 * ak6[i] + hf7 * next_dydx[i]);
  }
  Real_t  distChord;
  ThreeVector initialPoint, finalPoint, midPoint;

  initialPoint = ThreeVector(fLastInitialVector[0], fLastInitialVector[1], fLastInitialVector[2]);
  finalPoint   = ThreeVector(fLastFinalVector[0], fLastFinalVector[1], fLastFinalVector[2]);
  midPoint     = ThreeVector(midVector[0], midVector[1], midVector[2]);

  // Use stored values of Initial and Endpoint + new Midpoint to evaluate
  //  distance of Chord
  distChord  = GULineSection::Distline(midPoint, initialPoint, finalPoint);
  
  return distChord;
}
#endif
**********************************************************************/

// template <class Equation_t, unsigned int Nvar>
// inline void NystromRK4<Equation_t, Nvar>::PrintField(const char *label, const Real_t y[Nvar],
//                                                            const vecgeom::Vector3D<Real_t> &Bfield) const


// template <class Equation_t, unsigned int Nvar>
// inline void NystromRK4<Equation_t, Nvar>::PrintDyDx(const char *label, const Real_t dydx[Nvar],
//                                                           const Real_t y[Nvar]) const
