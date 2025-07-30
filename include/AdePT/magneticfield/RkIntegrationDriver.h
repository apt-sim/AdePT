// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

/*
 * Created on: November 11-15, 2021
 *
 *      Author: J. Apostolakis
 */

#ifndef RKINTEGRATION_DRIVER_H_
#define RKINTEGRATION_DRIVER_H_

#include "ErrorEstimatorRK.h"

#include <VecCore/VecMath.h>
#include <VecGeom/base/Vector3D.h>

/**
 * A simple stepper treating the propagation of particles in a constant magnetic field
 *   ( not just along the z-axis-- for that there is ConstBzFieldHelixStepper )
 */

template <class Stepper_t, typename Real_t, typename Int_t, class Equation_t, class MagField_t>

class RkIntegrationDriver {

public:
  // No constructors!
  // ----------------
  // RkIntegrationDriver() = delete;
  // RkIntegrationDriver( const RkIntegrationDriver& ) = delete;
  // ~RkIntegrationDriver() = delete;

  // template<typename Real_t, typename Vector3D>
  // Real_t GetCurvature(Vector3D const & dir, float const charge, float const momentum) const;

  // Propagate the track along the numerical solution of the ODE by the requested length
  //   Inputs: current position, current direction, some particle properties, max # of steps
  //   Output: new position, new direction of particle, number of integration steps (current integration)
  //
  //
  //   Note: care needed to 'tune' max number of steps - compromise between getting work done & divergence
  //
  // Versions:
  //   1. Vector3D version
  // template <class Stepper_t, class Equation_t, class MagField_t>
  template <int Verbose = 1>
  static inline __host__ __device__ bool Advance(vecgeom::Vector3D<double> &position,
                                                 vecgeom::Vector3D<double> &momentumVec, Int_t const &charge,
                                                 // Real_t const &momentum,
                                                 Real_t const &step, MagField_t const &magField, Real_t dydx_next[],
                                                 Real_t &hgood, // dy_ds[Nvar] at final point (return only !! )
                                                 unsigned int maxTrials = 5, int cordIters = 0);

  // Invariants
  // ----------
  static constexpr int Nvar                   = 6;     // For now, adequate to integrate over x, y, z, px, py, pz
  static constexpr Real_t fEpsilonRelativeMax = 0.001; // For now .. to be a parameter
  // Note: In Geant4, there is an absolute error tolerance via the deltaOneStep variable. The relative error tolerance
  // per step is given by eps = deltaOneStep / stepLength. To ensure that the eps is reasonable, G4 sets a minimal and a
  // maximal relative error tolerance. Here in AdePT, we don't use an absolute error tolerance but only define the
  // maximum relative error tolerance that is then used all the time.
  static constexpr Real_t fMinimumStep = 0.01 * copcore::units::millimeter;
  static constexpr Real_t kSmall       = 1.0e-30; //  amount to add to vector magnitude to avoid div by zero

  // Auxiliary methods
  // -----------------

  static inline __host__ __device__ void PrintStep(vecgeom::Vector3D<Real_t> const &startPosition,
                                                   vecgeom::Vector3D<Real_t> const &startDirection, Int_t const &charge,
                                                   Real_t const &momentum, Real_t const &step,
                                                   vecgeom::Vector3D<Real_t> &endPosition,
                                                   vecgeom::Vector3D<Real_t> &endDirection);

  static inline __host__ __device__ bool IntegrateStep(const Real_t yStart[], const Real_t dydx[], int charge,
                                                       Real_t &xCurrent, // InOut
                                                       Real_t htry, const MagField_t &magField,
                                                       Real_t yEnd[],      // Out - values
                                                       Real_t next_dydx[], //     - next derivative
                                                       Real_t &hnext);

protected:
  static inline __host__ __device__ bool CheckModulus(Real_t &newdirX_v, Real_t &newdirY_v, Real_t &newdirZ_v);

}; // end class declaration

// ------------------------------------------------------------------------------------------------

#include <iostream>
#include <iomanip>

template <class Stepper_t, typename Real_t, typename Int_t, class Equation_t, class MagField_t>
template <int Verbose>
inline __host__ __device__ bool RkIntegrationDriver<Stepper_t, Real_t, Int_t, Equation_t, MagField_t>::Advance(
    vecgeom::Vector3D<double> &position,    //   In/Out
    vecgeom::Vector3D<double> &momentumVec, //   In/Out
    Int_t const &chargeInt, Real_t const &length, MagField_t const &magField, Real_t dydx_next[Nvar],
    Real_t &hgood,                        // dy_ds[] at final point (return only !! ), last good step
    unsigned int maxTrials, int cordIters // max allowed trials
)
{
  using vecgeom::Vector3D;

  Real_t yStart[Nvar] = {(Real_t)position[0],    (Real_t)position[1],    (Real_t)position[2],
                         (Real_t)momentumVec[0], (Real_t)momentumVec[1], (Real_t)momentumVec[2]};
  Real_t dydx[Nvar];
  Real_t yEnd[Nvar];

  Equation_t::EvaluateDerivatives(magField, yStart, chargeInt, dydx);

  Real_t stepAdvance = 0.0;
  // For first cord integration try full cordlength otherwise use last step from previous cord integration
  Real_t htry = cordIters == 0 ? length : vecCore::Min(hgood, length);

  bool done    = false;
  int numSteps = 0;

  bool allFailed = true;  // Have all steps until now failed
  bool goodStep  = false; // last step was good
  do {
    Real_t hnext;

    goodStep = IntegrateStep(yStart, dydx, chargeInt, stepAdvance, htry, magField, yEnd, dydx_next, hnext);

    Real_t hdid = goodStep ? htry : 0.0;
    allFailed   = allFailed && !goodStep;

    done  = (stepAdvance >= length);
    hgood = vecCore::Max(hnext, Real_t(fMinimumStep));

#ifdef RK_VERBOSE
    Real_t htryOld = htry;
#endif
    htry = hgood;
    if (goodStep && !done) {
#pragma unroll
      for (int i = 0; i < Nvar; i++) {
        yStart[i] = yEnd[i];
        dydx[i]   = dydx_next[i]; // Using FSAL property !
      }
      htry = vecCore::Min(hgood, length - stepAdvance);
    }
#ifdef RK_VERBOSE
    if (Verbose > 1) {
      printf(" n = %4d  stepAdvance = %10.5f ", numSteps, x);
      printf(" h:   suggested (input try) = %10.7f good? %1s next= %10.7f \n", // did= %10.7f  good= %10.7f
             htryOld, /* did, */ (goodStep ? "Y" : "N"), hnext /* , good */);
    }
#endif

    ++numSteps;

  } while (!done && numSteps < maxTrials);

  // printf("stepAdvance %.10f length %.10f htry %.10f done %d goodStep %d numSteps %d cordIters %d\n", stepAdvance,
  // length, htry, done, goodStep, numSteps, cordIters );

  //  In case of failed last step, we must not use it's 'yEnd' values !!
  if (goodStep) {
    // Real_t invM = 1.0 / (momentum+kSmall);
    position.Set(yEnd[0], yEnd[1], yEnd[2]);
    momentumVec.Set(yEnd[3], yEnd[4], yEnd[5]);
    // endDirection.Set( invM*yEnd[3], invM*yEnd[4], invM*yEnd[5] );
  } else {
    if (!allFailed) { // numSteps == maxTrials ){
      position.Set(yStart[0], yStart[1], yStart[2]);
      momentumVec.Set(yStart[3], yStart[4], yStart[5]);
    } else {
      printf("WARNING: B field integration has failed, this track seems stuck!\n");
    }
  }

  return done;
}

// ----------------------------------------------------------------------------------------

template <class Stepper_t, typename Real_t, typename Int_t, class Equation_t, class MagField_t>
inline __host__ __device__ bool RkIntegrationDriver<Stepper_t, Real_t, Int_t, Equation_t, MagField_t>::IntegrateStep(
    const Real_t yStart[], const Real_t dydx[], int charge,
    Real_t &xCurrent, // InOut
    Real_t htry, const MagField_t &magField,
    // Real_t eps_rel_max,
    Real_t yEnd[],      // Out - values
    Real_t next_dydx[], //     - next derivative
    Real_t &hnext)
{
  constexpr Real_t safetyFactor      = 0.9;
  constexpr Real_t shrinkPower       = -1.0 / Real_t(Stepper_t::kMethodOrder);
  constexpr Real_t growPower         = -1.0 / (Real_t(Stepper_t::kMethodOrder + 1));
  constexpr Real_t max_step_increase = 10.0; // Step size must not grow   more than 10x
  constexpr Real_t max_step_decrease = 0.1;  // Step size must not shrink more than 10x
  // const     Real_t errcon = std::pow( max_step_increase / safetyFactor , 1.0/growPower );

  Real_t yErr[Nvar];

  // static constexpr iMom0 = 3;  // Momentum starts at [3]
  // = Mag2 ( Vector3D<Real_v>( yStart[iMom0], yStart[imom0+1], yStart[iMom0+2]) );
  Real_t magMomentumSq = yStart[3] * yStart[3] + yStart[4] * yStart[4] + yStart[5] * yStart[5];

  Stepper_t::StepWithErrorEstimate(magField, yStart, dydx, charge, htry,
                                   yEnd,       // Output:  y values at end,
                                   yErr,       //          estimated errors,
                                   next_dydx); //          next value of dydx

  ErrorEstimatorRK errorEstimator(fEpsilonRelativeMax);
  Real_t errmax_sq = errorEstimator.EstimateSquareError(yErr, htry, magMomentumSq);
  // printf("magMomentumSq %.15f, yerr: %.15f %.15f %.15f %.15f %.15f %.15f\n", magMomentumSq, yErr[0], yErr[1],
  // yErr[2], yErr[3], yErr[4], yErr[5] ); printf("errmax_sq %f for htry %f\n", errmax_sq, htry);
  bool goodStep = errmax_sq <= 1.0;
  if (goodStep) {
    xCurrent += htry;
    // Store next_dydx ...
#if 1
    // New code - all threads run the same code, even power ...
    hnext = htry * vecCore::Min(safetyFactor * vecCore::Pow(errmax_sq, Real_t(0.5) * growPower), max_step_increase);
#else
    // Compute size of next Step
    hnext = (errmax_sq > errcon * errcon) ? h * safetyFactor * std::pow(errmax_sq, Real_t(0.5) * growPower)
                                          : hnext = max_step_increase * h;
#endif
  } else {

    // if we did a minimal step we force the step to be accepted
    if (htry <= fMinimumStep) {
      goodStep = true;
      xCurrent += htry;
      hnext = 2 * fMinimumStep;
    } else {
      // Step failed; compute the size of retrial Step.
      Real_t htemp = safetyFactor * htry * std::pow(errmax_sq, 0.5 * shrinkPower);
      hnext        = vecCore::Max(htemp, max_step_decrease * htry);
      // no more than than a factor of 10 smaller

      xCurrent = 0.0;

      if (xCurrent + hnext == xCurrent) {
        // Serious Problem --- under FLOW !!!   Report it ??????????????????????????
        printf("This should never happen, under flow in next step\n");
        hnext = 2.0 * vecCore::math::Max(htemp, htry);
      }
    }
  }
  return goodStep;
}

// ----------------------------------------------------------------------------------------

template <class Stepper_t, typename Real_t, typename Int_t, class Equation_t, class MagField_t>
bool RkIntegrationDriver<Stepper_t, Real_t, Int_t, Equation_t, MagField_t>::CheckModulus(Real_t &newdirX,
                                                                                         Real_t &newdirY,
                                                                                         Real_t &newdirZ)
{
  constexpr float perMillion = 1.0e-6;

  Real_t modulusDir = newdirX * newdirX + newdirY * newdirY + newdirZ * newdirZ;
  typename vecCore::Mask<Real_t> goodDir;
  goodDir = vecCore::Abs(modulusDir - Real_t(1.0)) < perMillion;

  bool allGood = vecCore::MaskFull(goodDir);
  assert(allGood && "Not all Directions are nearly 1");

  return allGood;
}

// ----------------------------------------------------------------------------------------

template <class Stepper_t, typename Real_t, typename Int_t, class Equation_t, class MagField_t>
inline __host__ __device__ void RkIntegrationDriver<Stepper_t, Real_t, Int_t, Equation_t, MagField_t>::PrintStep(
    vecgeom::Vector3D<Real_t> const &startPosition, vecgeom::Vector3D<Real_t> const &startDirection,
    Int_t const &charge, Real_t const &momentum, Real_t const &step, vecgeom::Vector3D<Real_t> &endPosition,
    vecgeom::Vector3D<Real_t> &endDirection)
{
  // Debug printing of input & output
  printf(" RKStepper::PrintStep \n");
  const int vectorSize = vecCore::VectorSize<Real_t>();
  Real_t x0, y0, z0, dirX0, dirY0, dirZ0;
  Real_t x, y, z, dx, dy, dz;
  x0    = startPosition.x();
  y0    = startPosition.y();
  z0    = startPosition.z();
  dirX0 = startDirection.x();
  dirY0 = startDirection.y();
  dirZ0 = startDirection.z();
  x     = endPosition.x();
  y     = endPosition.y();
  z     = endPosition.z();
  dx    = endDirection.x();
  dy    = endDirection.y();
  dz    = endDirection.z();
  for (int i = 0; i < vectorSize; i++) {
    printf("Start> Lane= %1d Pos= %8.5f %8.5f %8.5f  Dir= %8.5f %8.5f %8.5f ", i, x0, y0, z0, dirX0, dirY0, dirZ0);
    printf(" s= %10.6f ", step);     // / units::mm );
    printf(" q= %d ", charge);       // in e+ units ?
    printf(" p= %10.6f ", momentum); // / units::GeV );
    // printf(" ang= %7.5f ", angle );
    printf(" End> Pos= %9.6f %9.6f %9.6f  Mom= %9.6f %9.6f %9.6f\n", x, y, z, dx, dy, dz);
  }
}

#endif /* RKINTEGRATION_DRIVER_H_ */
