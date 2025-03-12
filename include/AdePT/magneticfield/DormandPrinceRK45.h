// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0
//
// Author:  J. Apostolakis,  15 Nov 2021
//
// Implementation of the Dormand Price 5(4) 7-stage Runge-Kutta integrator for AdePT.
//
// Notes:
//  - Current version is restricted to Magnetic fields (see EvaluateDerivatives.)
//  - It provides the next value of dy/ds in 'next_dydx'
//  - It uses a large number of registers and/or stack locations - 7 derivatives + In + Out + Err

template <class Equation_t, class T_Field, unsigned int Nvar, typename Real_t>
class DormandPrinceRK45 // : public VScalarIntegrationStepper
{
  // using ThreeVector = vecgeom::Vector3D<Real_t>;

public:
  static constexpr unsigned int kMethodOrder = 4;
  // inline DormandPrinceRK45(Equation_t *EqRhs, bool verbose = false);
  // DormandPrinceRK45(const DormandPrinceRK45 &) = delete;
  // ~DormandPrinceRK45() {}

  static __host__ __device__ void EvaluateDerivatives(const T_Field &field, const Real_t y[], int charge,
                                                      Real_t dydx[]);

  static __host__ __device__ void StepWithErrorEstimate(const T_Field &field, const Real_t *yIn, const Real_t *dydx,
                                                        int charge, Real_t Step,
                                                        Real_t *yOut,       // Output:  y values at end,
                                                        Real_t *yerr,       //          estimated errors,
                                                        Real_t *next_dydx); //          next value of dydx
};

template <class Equation_t, class T_Field, unsigned int Nvar, typename Real_t>
__host__ __device__ void DormandPrinceRK45<Equation_t, T_Field, Nvar, Real_t>::EvaluateDerivatives(const T_Field &field,
                                                                                                   const Real_t yIn[],
                                                                                                   int charge,
                                                                                                   Real_t dy_ds[])
{
  /* #ifdef VERBOSE_RHS
      using geant::units::tesla;
      std::cout << "DormandPrinceRK45::EvaluateDerivatives called with q= " << charge
                << " at Position = " << yIn[0] << " y= " << yIn[1] << " z= " << yIn[2]
                << " with Momentum = " << yIn[3] << " y= " << yIn[4] << " z= " << yIn[5] << " ";
  #endif */

  // Vector3D<Real_t> Bfield;
  // Equation_t::EvaluateDerivativesReturnB( field, yIn, charge, dy_ds, Bfield );

  Equation_t::EvaluateDerivatives(/* const T_Field& */ field, yIn, charge, dy_ds);

  /*********
   using copcore::units::tesla;
   using std::setw;
   constexpr int prec= 5;
   constexpr int nf= prec+5;
   int old_prec = std::cout.precision(prec);
   std::cout << " DoPri5: Evaluate Derivatives - using B-field,  Bx= " << Bfield.x() / tesla << " By= " << Bfield.y() /
   tesla << " Bz= " << Bfield.z() / tesla << " "; std::cout << " gives Derivs dy_ds= :  "
             << " x = " << setw(nf) << dy_ds[0] << " y = " << setw(nf) << dy_ds[1] << " z = " << setw(nf) << dy_ds[2]
             << " px= " << setw(nf) << dy_ds[3] << " py= " << setw(nf) << dy_ds[4] << " pz= " << setw(nf) << dy_ds[5]
             << std::endl;
   std::cout.precision(old_prec);
   ********/
}

template <class Equation_t, class T_Field, unsigned int Nvar, typename Real_t>
inline __host__ __device__ void DormandPrinceRK45<Equation_t, T_Field, Nvar, Real_t>::StepWithErrorEstimate(
    const T_Field &field, const Real_t *yIn, const Real_t *dydx, int charge, Real_t Step, Real_t *yOut, Real_t *yErr,
    Real_t *next_dydx)
// yIn and yOut MUST NOT be aliases for same array
{
  assert(yIn != yOut);

  static constexpr Real_t b21 = 0.2,

                          b31 = 3.0 / 40.0, b32 = 9.0 / 40.0,

                          b41 = 44.0 / 45.0, b42 = -56.0 / 15.0, b43 = 32.0 / 9.0,

                          b51 = 19372.0 / 6561.0, b52 = -25360.0 / 2187.0, b53 = 64448.0 / 6561.0, b54 = -212.0 / 729.0,

                          b61 = 9017.0 / 3168.0, b62 = -355.0 / 33.0, b63 = 46732.0 / 5247.0, b64 = 49.0 / 176.0,
                          b65 = -5103.0 / 18656.0,

                          b71 = 35.0 / 384.0, b72 = 0., b73 = 500.0 / 1113.0, b74 = 125.0 / 192.0,
                          b75 = -2187.0 / 6784.0, b76 = 11.0 / 84.0;

  static constexpr Real_t dc1 = -(b71 - 5179.0 / 57600.0), dc2 = -(b72 - .0), dc3 = -(b73 - 7571.0 / 16695.0),
                          dc4 = -(b74 - 393.0 / 640.0), dc5 = -(b75 + 92097.0 / 339200.0),
                          dc6 = -(b76 - 187.0 / 2100.0), dc7 = -(-1.0 / 40.0);

  // Initialise time to t0, needed when it is not updated by the integration.
  //       [ Note: Only for time dependent fields (usually electric)
  //                 is it neccessary to integrate the time.]
  // yOut[7] = yTemp[7]   = yIn[7];

  // EvaluateDerivatives( field, yIn, charge,  dydx) ;      // 1st Step

  Real_t ak2[Nvar], yTemp[Nvar];
  {
    for (unsigned int i = 0; i < Nvar; i++) {
      yTemp[i] = yIn[i] + b21 * Step * dydx[i];
    }
    EvaluateDerivatives(field, yTemp, charge, ak2); // 2nd Step
  }

  Real_t ak3[Nvar];
  {
    for (unsigned int i = 0; i < Nvar; i++) {
      yTemp[i] = yIn[i] + Step * (b31 * dydx[i] + b32 * ak2[i]);
    }
    EvaluateDerivatives(field, yTemp, charge, ak3); // 3rd Step
  }

  Real_t ak4[Nvar];
  {
    for (unsigned int i = 0; i < Nvar; i++) {
      yTemp[i] = yIn[i] + Step * (b41 * dydx[i] + b42 * ak2[i] + b43 * ak3[i]);
    }
    EvaluateDerivatives(field, yTemp, charge, ak4); // 4th Step
  }

  Real_t ak5[Nvar];
  {
    for (unsigned int i = 0; i < Nvar; i++) {
      yTemp[i] = yIn[i] + Step * (b51 * dydx[i] + b52 * ak2[i] + b53 * ak3[i] + b54 * ak4[i]);
    }
    EvaluateDerivatives(field, yTemp, charge, ak5); // 5th Step
  }

  Real_t ak6[Nvar];
  {
    for (unsigned int i = 0; i < Nvar; i++) {
      yTemp[i] = yIn[i] + Step * (b61 * dydx[i] + b62 * ak2[i] + b63 * ak3[i] + b64 * ak4[i] + b65 * ak5[i]);
    }
    EvaluateDerivatives(field, yTemp, charge, ak6); // 6th Step
  }

  // Real_t ak7[Nvar];  // -> Replaced by next_dydx
  for (unsigned int i = 0; i < Nvar; i++) {
    yOut[i] =
        yIn[i] + Step * (b71 * dydx[i] + b72 * ak2[i] + b73 * ak3[i] + b74 * ak4[i] + b75 * ak5[i] + b76 * ak6[i]);
  }
  EvaluateDerivatives(field, yOut, charge, next_dydx); // 7th and Final stage

  for (unsigned int i = 0; i < Nvar; i++) {
    // Estimate error as difference between 4th and 5th order methods
    //
    yErr[i] = Step * (dc1 * dydx[i] + dc2 * ak2[i] + dc3 * ak3[i] + dc4 * ak4[i] + dc5 * ak5[i] + dc6 * ak6[i] +
                      dc7 * next_dydx[i]);
    // std::cout<< "----In Stepper, yerrr is: "<<yErr[i]<<std::endl;
  }
#if ENABLE_CHORD_DIST
  for (unsigned int i = 0; i < Nvar; i++) {
    // Store Input and Final values, for possible use in calculating chord
    fLastInitialVector[i] = yIn[i];
    fLastFinalVector[i]   = yOut[i];
    fInitialDyDx[i]       = dydx[i]; // At initial point
  }
#endif
  // fLastStepLength = Step;

  // std::cout << " Exiting StepWithErrorEstimate of scalar " << std::endl;

  return;
}

/**********************************************************************
#if ENABLE_CHORD_DIST
template <class Equation_t, unsigned int Nvar>
inline Real_t DormandPrinceRK45<Equation_t, Nvar>::DistChord() const
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
// inline void DormandPrinceRK45<Equation_t, Nvar>::PrintField(const char *label, const Real_t y[Nvar],
//                                                            const vecgeom::Vector3D<Real_t> &Bfield) const

// template <class Equation_t, unsigned int Nvar>
// inline void DormandPrinceRK45<Equation_t, Nvar>::PrintDyDx(const char *label, const Real_t dydx[Nvar],
//                                                           const Real_t y[Nvar]) const
