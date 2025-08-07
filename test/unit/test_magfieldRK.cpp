// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0
//
// Author:  J. Apostolakis, 17 Nov 2021
//

// #include <VecCore/Math.h>

#define __device__
#define __host__

#include <VecGeom/base/Vector3D.h>

#include <AdePT/magneticfield/MagneticFieldEquation.h>

#include <AdePT/magneticfield/UniformMagneticField.h>

#include <AdePT/magneticfield/PrintFieldVectors.h>

using Real_t = float;
// using Real_t = double;
using Field_t            = UniformMagneticField;
using MagFieldEquation_t = MagneticFieldEquation<Field_t>;

template <typename T>
using Vector3D = vecgeom::Vector3D<T>;

#include <iostream>

using std::cerr;
using std::cout;
using std::endl;
using std::setw;

using namespace vecCore::math;

// MagFieldEquation_t *CreateUniformFieldAndEquation(Vector3D<float> const &fieldValue);
// MagFieldEquation_t *CreateFieldAndEquation(const char *filename);

// CreateUniformField();

template <typename Real_t, typename Field_t> // , typename Equation_t>
bool TestEquation(Field_t const &magField);

constexpr unsigned int gNposmom = 6; // Position 3-vec + Momentum 3-vec

int gVerbose = 1;

template <typename Real_t, typename Field_t> // , typename Equation_t>
bool TestEquation(Field_t const &magField)
{
  using Equation_t = MagneticFieldEquation<Field_t>;

  const Real_t perMillion = Real_t(1e-6);
  bool hasError           = false; // Return value
  using Bool_v            = vecCore::Mask<Real_t>;

  //  1. Initialize meaningful state
  //
  Vector3D<Real_t> PositionVec(1., 2., 3.); // initial
  Vector3D<Real_t> MomentumVec(0., 0.1, 1.);
  Vector3D<Real_t> FieldVec(0., 0., 1.); // Magnetic field value (constant)

  // double PositionTime[4] = { PositionVec.x(), PositionVec.y(), PositionVec.z(), 0.0};

  //  2. Initialise object for Equation (of motion)
  //
  Real_t PositionMomentum[gNposmom] = {PositionVec[0], PositionVec[1], PositionVec[2],
                                       MomentumVec[0], MomentumVec[1], MomentumVec[2]};
  Real_t dydx[gNposmom];
  int charge = -1;

  //  3. Evaluate RHS / ie Derivatives of Equation of motion
  //

  //  Option 3a)  use a field value
  Equation_t::EvaluateDerivatives(PositionMomentum, FieldVec, charge, dydx);
  // ****************************

  //  Option 3b)  use a magnetic field class
  // Equation_t::EvaluateDerivatives(PositionMomentum, charge, magField, dydx);
  //-- - Better form of method:
  //-- Equation_t::EvaluateDerivatives(magField, PositionMomentum, charge, dydx);

  // ****************************

  Real_t FieldArr[3] = {FieldVec[0], FieldVec[1], FieldVec[2]};
  std::cout << " Values obtained: " << std::endl;
  PrintFieldVectors::PrintSixvecAndDyDx(PositionMomentum, charge, FieldArr, dydx);

  //  4. Test output values
  //
  Vector3D<Real_t> dPos_ds(dydx[0], dydx[1], dydx[2]);
  Real_t mag     = dPos_ds.Mag();
  Real_t magDiff = fabs(mag - 1.0);
  assert(magDiff < 1.0e-6);
  if (magDiff > 2.0e-7) {
    std::cerr << "WARNING>  dPos_ps (direction) is not a unit vector.  | Mag - 1 | = " << magDiff << std::endl;
  }

  Vector3D<Real_t> ForceVec(dydx[3], dydx[4], dydx[5]);

  // Check result
  Real_t MdotF = MomentumVec.Dot(ForceVec);
  Real_t BdotF = FieldVec.Dot(ForceVec);

  Real_t momentumMag = MomentumVec.Mag();
  Real_t fieldMag    = FieldVec.Mag();
  Real_t sineAngle   = FieldVec.Cross(MomentumVec).Mag() / (momentumMag * fieldMag);

  Real_t ForceMag = ForceVec.Mag();
  const Real_t c  = Real_t(copcore::units::kCLight);

  // Tolerance of difference in values (used below)
  Real_t tolerance = perMillion;

  if (gVerbose) {
    std::cout << " Output:  " << std::endl;
  }
  Bool_v error = (Abs(ForceMag - c * Abs(charge) * fieldMag * sineAngle)) > (tolerance * ForceMag);
  if (!vecCore::MaskEmpty(error)) {
    cerr << "ERROR: Force magnitude is not equal to   c * |charge| * |field| * sin( p, B )." << endl;
    cerr << "       Force magnitude = " << ForceMag << endl;
    cerr << "         other side =    " << c * Abs(charge) * fieldMag * sineAngle;
    cerr << " charge = " << charge << " field-Mag= " << fieldMag << std::endl;
    cerr << "       Force = " << ForceVec[0] << " " << ForceVec[1] << " " << ForceVec[2] << " " << endl;
    hasError = true;
  }

  error = (ForceMag == momentumMag * fieldMag);
  assert(vecCore::MaskEmpty(error)); // Must add coefficient !!

  error = Abs(MdotF) > tolerance * MomentumVec.Mag() * ForceVec.Mag();
  if (!vecCore::MaskEmpty(error)) {
    cerr << "ERROR: Force due to magnetic field is not perpendicular to momentum!!" << endl;
    hasError = true;
  } else if (gVerbose) {
    cout << " Success:  Good (near zero) dot product momentum . force " << endl;
  }

  error = Abs(BdotF) > tolerance * FieldVec.Mag() * ForceVec.Mag();
  if (!vecCore::MaskEmpty(error)) {
    cerr << "ERROR: Force due to magnetic field is not perpendicular to B field!" << std::endl;
    cerr << " Vectors:  BField   Force " << std::endl;
    for (int i = 0; i < 3; i++)
      cerr << "   [" << i << "] " << FieldVec[i] << " " << ForceVec[i] << std::endl;

    hasError = true;
  } else if (gVerbose) {
    cout << " Success:  Good (near zero) dot product magnetic-field . force " << std::endl;
  }

  return !hasError;
}

//   NEW Development Thurs 18 December 2021
//   ---------------
#include <AdePT/magneticfield/DormandPrinceRK45.h>
#include <AdePT/copcore/PhysicalConstants.h>

template <typename Real_t, typename Field_t> // , typename Equation_t>
bool TestStepper(Field_t const &magField)
//   -----------
{
  constexpr unsigned int Nvar = 6; // Number of integration variables
  using Equation_t            = MagneticFieldEquation<Field_t>;
  using Stepper_t             = DormandPrinceRK45<Equation_t, Field_t, Nvar, Real_t>;

  Real_t yPosMom[Nvar] = {0, 1, 2,                          //  PositionVec[0], PositionVec[1], PositionVec[2],
                          0, 0, 3.0 * copcore::units::GeV}; // MomentumVec[0], MomentumVec[1],  MomentumVec[2] } ;

  Real_t dy_ds[Nvar];

  Real_t yOut[Nvar];      // Output:  y values at end,
  Real_t yerr[Nvar];      //          estimated errors,
  Real_t next_dyds[Nvar]; //          next value of dydx

  int charge  = -1;
  Real_t step = 0.1;

  // Evaluate dy_ds
  // Equation_t::EvaluateDerivatives(magField, yPosMom, charge, dy_ds);
  Vector3D<float> magFieldVec(1, 2, 3);
  Equation_t::EvaluateDerivativesReturnB(magField, yPosMom, charge, dy_ds, magFieldVec);
  std::cout << " magField = " << magFieldVec[0] << " , " << magFieldVec[1] << " , " << magFieldVec[2] << std::endl;

  Stepper_t::StepWithErrorEstimate(magField, yPosMom, dy_ds, charge, step, yOut, yerr, next_dyds);

  float magFieldOut[3] = {magFieldVec[0], magFieldVec[1], magFieldVec[2]};

  PrintFieldVectors::PrintSixvecAndDyDx(yPosMom, charge, magFieldOut, dy_ds);

  return true;
}

#include <AdePT/magneticfield/RkIntegrationDriver.h>

//   NEW Development Friday 19 November 2021
//   ---------------

const unsigned int trialsPerCall = 6; //  Try 100 later
const unsigned int MaxTrials     = 30;
template <typename Real_t, typename Field_t> // , typename Equation_t>
bool TestDriverAdvance(Field_t const &magField, Real_t hLength = 300)
//   -----------
{
  constexpr unsigned int Nvar = 6; // Number of integration variables
  using Equation_t            = MagneticFieldEquation<Field_t>;
  using Stepper_t             = DormandPrinceRK45<Equation_t, Field_t, Nvar, Real_t>;
  using Driver_t              = RkIntegrationDriver<Stepper_t, Real_t, int, Equation_t, Field_t>;

  Vector3D<double> Position  = {0.0, 0.001, 0.002}; //  PositionVec[0], PositionVec[1], PositionVec[2],
  Vector3D<double> Direction = {0.0, 1.0, 0.0};
  double momentumMag         = 3.0 * copcore::units::GeV;

  Vector3D<double> Momentum = momentumMag * Direction;

  Real_t dy_ds[Nvar];

  int charge = -1;

  Real_t yStart[Nvar] = {(Real_t)Position[0], (Real_t)Position[1], (Real_t)Position[2],
                         (Real_t)Momentum[0], (Real_t)Momentum[1], (Real_t)Momentum[2]};
  // std::cout << "yStart: " << std::endl;
  // PrintFieldVectors::PrintSixvec( yStart );

  Vector3D<float> magFieldStart(0, 0, 0);
  Equation_t::EvaluateDerivativesReturnB(magField, yStart, charge, dy_ds, magFieldStart);
  // std::cout << " magField = " << magFieldStart[0] << " , " << magFieldStart[1] << " , " << magFieldStart[2] <<
  // std::endl;

  // Do a couple of steps
  unsigned int totalTrials = 0;

  // Test the simple Advance method
  bool verbose       = false;
  double sumAdvanced = 0.0;

  bool done   = false;
  Real_t hTry = hLength; // suggested 'good' length per integration step
  Real_t dydx_end[Nvar];

  std::cout << " t:   " << setw(3) << totalTrials << " s= " << setw(9) << sumAdvanced << " ";
  Real_t magFieldStartArr[3] = {magFieldStart[0], magFieldStart[1], magFieldStart[2]};
  PrintFieldVectors::PrintLineSixvecDyDx(yStart, charge, magFieldStartArr, dy_ds);

  Vector3D<double> momentumVec = momentumMag * Direction;
  Real_t lastGoodStep          = 0.;

  do {
    Real_t hAdvanced = 0.0;

    done = Driver_t::Advance(Position, momentumVec, charge, hLength, magField, dydx_end, lastGoodStep, MaxTrials);
    // Runge-Kutta single call
    ++totalTrials;
    std::cout << "Advanced returned:  done= " << (done ? "Yes" : " No") << " hAdvanced = " << hAdvanced
              << " hNext = " << hTry << std::endl;

    Real_t yPosMom[Nvar] = {(Real_t)Position[0],    (Real_t)Position[1],    (Real_t)Position[2],
                            (Real_t)momentumVec[0], (Real_t)momentumVec[1], (Real_t)momentumVec[2]};
    sumAdvanced += hAdvanced;

    if (verbose)
      std::cout << "- Trials:   max/call = " << trialsPerCall << "  total done = " << totalTrials
                << ((totalTrials == trialsPerCall) ? " MAXimum reached !! " : " ") << std::endl;
    else
      std::cout << " t: " << setw(4) << totalTrials << " s= " << setw(9) << sumAdvanced << " ";

    Vector3D<float> magFieldEnd;
    Equation_t::EvaluateDerivativesReturnB(magField, yPosMom, charge, dy_ds, magFieldEnd);
    Real_t magFieldEndArr[3] = {magFieldEnd[0], magFieldEnd[1], magFieldEnd[2]};

    //--- PrintFieldVectors::PrintSixvecAndDyDx(
    PrintFieldVectors::PrintLineSixvecDyDx(yPosMom, charge, magFieldEndArr, // dydx_end );
                                           dy_ds);

  } while (!done && totalTrials < MaxTrials);

  Vector3D<double> magFieldEnd;
  magField.Evaluate(Position, magFieldEnd);

  std::cout << " Mag field at end = " << magFieldEnd[0] << " , " << magFieldEnd[1] << "  , " << magFieldEnd[2]
            << std::endl;

  // Test the 'full' DoStep methodx

  // Finish the integration ...

  return true;
}

// ------------------------------------------------------------------------------------------------

#include <AdePT/magneticfield/ConstFieldHelixStepper.h>

//   NEW test Code Friday 26 November 2021
//   -------------

template <typename Real_t, typename Field_t> // , typename Equation_t>
bool CheckDriverVsHelix(Field_t const &magField, const Real_t stepLength = 300.0)
//   -----------
{
  constexpr unsigned int Nvar = 6; // Number of integration variables
  using Equation_t            = MagneticFieldEquation<Field_t>;
  using Stepper_t             = DormandPrinceRK45<Equation_t, Field_t, Nvar, Real_t>;
  using Driver_t              = RkIntegrationDriver<Stepper_t, Real_t, int, Equation_t, Field_t>;

  Vector3D<Real_t> const startPosition  = {0, 0.001, 0.002}; //  PositionVec[0], PositionVec[1], PositionVec[2],
  Vector3D<Real_t> const startDirection = {0, 1.0, 0.0};
  const Real_t momentumMag              = 30.0 * copcore::units::MeV;

  Vector3D<Real_t> const startMomentum = momentumMag * startDirection;

  Real_t dy_ds[Nvar];
  int charge = -1;

  Vector3D<float> magFieldStart(1, 2, 3);
  Real_t yStart[Nvar] = {startPosition[0], startPosition[1], startPosition[2],
                         startMomentum[0], startMomentum[1], startMomentum[2]};
  Equation_t::EvaluateDerivativesReturnB(magField, yStart, charge, dy_ds, magFieldStart);
  // std::cout << " magField = " << magFieldStart[0] << " , " << magFieldStart[1] << " , " << magFieldStart[2] <<
  // std::endl;

  // For use by Runge Kutta
  Real_t hLength             = stepLength;
  Vector3D<double> Position  = startPosition;
  Vector3D<double> Direction = startDirection;

  // -- Test the simple Advance method
  unsigned int totalTrials = 0;
  bool verbose             = false;
  Real_t dydx_end[Nvar];

  if (verbose) {
    // std::cout << "yStart: " << std::endl;
    // PrintFieldVectors::PrintSixvec( yStart );
    std::cout << " t:   " << setw(3) << totalTrials << " s= " << setw(9) << 0.0 << " ";
    Real_t magFieldStartArr[3] = {magFieldStart[0], magFieldStart[1], magFieldStart[2]};
    PrintFieldVectors::PrintLineSixvecDyDx(yStart, charge, magFieldStartArr, dy_ds);
  }

  bool unfinished    = true;
  Real_t sumAdvanced = 0; //  length integrated

  constexpr int maxTrials = 500;

  Vector3D<double> momentumVec = momentumMag * Direction;
  Real_t lastGoodStep          = 0.;

  do {
    Real_t hAdvanced = 0; //  length integrated

    bool done = Driver_t::Advance(Position, momentumVec, charge, hLength, magField, dydx_end, lastGoodStep, MaxTrials);
    //   Runge-Kutta single call ( number of steps <= trialsPerCall )
    ++totalTrials;

    hLength -= hAdvanced;
    unfinished = !done; // (hLength > 0.0);

    sumAdvanced += hAdvanced; // Gravy ..

  } while (unfinished && (totalTrials < maxTrials));
  // Ensure that the 'lane' integration is done .... or at least we tried hard

  // Cast the solution into simple variables - for comparison
  Vector3D<Real_t> endPositionRK  = Position;
  Vector3D<Real_t> endDirectionRK = (1.0 / momentumMag) * momentumVec;
  Real_t magDirectionRK           = endDirectionRK.Mag();

  // For use by Helix method ...
  Vector3D<Real_t> endPositionHx, endDirectionHx;

  ConstFieldHelixStepper helixBvec(magFieldStart);
  //********************

  // Helix solution --- for comparison
#if 1
  // fieldPropagatorConstBany fieldPropagatorBany;
  helixBvec.DoStep(startPosition, startDirection, charge, momentumMag, stepLength,
                   /**************/ endPositionHx, endDirectionHx);
#else
  helixBvec.DoStep<Real_t, int>(startPosition, startDirection, charge, momentumMag, stepLength, endPositionHx,
                                endDirectionHx);
#endif

  // Compare the results HERE
  Vector3D<Real_t> shiftVec = (endPositionRK - endPositionHx);
  Vector3D<Real_t> deltaDir = (endDirectionRK - endDirectionHx);

  if (verbose) {
    std::cout << "- Trials:  max/call= " << trialsPerCall << "  total done = " << totalTrials
              << ((totalTrials == trialsPerCall) ? " MAXimum reached !!" : " ") << std::endl;
  } else {
    std::cout << " t: " << setw(4) << totalTrials << " s= " << setw(7) << sumAdvanced << " "
              << " l= " << setw(7) << stepLength << " "
              << " d||*1e6 = " << setw(10) << 1.e6 * (magDirectionRK - 1.0) << " | ";
  }
  Real_t yPosMom[Nvar] = {(Real_t)endPositionHx[0], (Real_t)endPositionHx[1], (Real_t)endPositionHx[2],
                          (Real_t)momentumVec[0],   (Real_t)momentumVec[1],   (Real_t)momentumVec[2]};
  Vector3D<float> magFieldEnd;
  Equation_t::EvaluateDerivativesReturnB(magField, yPosMom, charge, dy_ds, magFieldEnd);
  Real_t magFieldEndArr[3] = {magFieldEnd[0], magFieldEnd[1], magFieldEnd[2]};

  constexpr Real_t magFactorPos = 1.0 / Driver_t:: // RkIntegrationDriver<>
                                  fEpsilonRelativeMax;
  constexpr Real_t magFactorDir = magFactorPos; // 1.0e+6;

  Real_t mulFactorPos      = magFactorPos / stepLength;
  Real_t deltaPosMom[Nvar] = {mulFactorPos * shiftVec[0], mulFactorPos * shiftVec[1], mulFactorPos * shiftVec[2],
                              magFactorDir * deltaDir[0], magFactorDir * deltaDir[1], magFactorDir * deltaDir[2]};

  //--- PrintFieldVectors::PrintSixvecAndDyDx(
  PrintFieldVectors::PrintLineSixvecDyDx(yPosMom, charge, magFieldEndArr, deltaPosMom);

  // magField.Evaluate( endPositionHx, magFieldEnd );
  // std::cout << " Mag field at end = " << magFieldEnd[0] << " , " << magFieldEnd[1]
  //           << "  , " <<  magFieldEnd[2] << std::endl;

  // Test the 'full' DoStep methodx

  // Finish the integration ...

  return true;
}

//--------------------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
  using copcore::units::tesla;
  Vector3D<float> fieldValueZ(0.0, 0.0, 1.0 * tesla);
  bool allStepGood;

  UniformMagneticField *pConstantBfield = new UniformMagneticField(fieldValueZ);
  UniformMagneticField Bx(Vector3D<float>(10.0 * tesla, 0.0, 0.0));
  UniformMagneticField By(Vector3D<float>(0.0, 10.0 * tesla, 0.0));
  UniformMagneticField Bz(Vector3D<float>(0.0, 0.0, 10.0 * tesla));

  cout << " -- Testing mag-field equation with float.   " << endl;
  cout << " ---------------------------------------------------------------------" << endl;
  // NOTE: No return value of testEquation is queried or used!
  TestEquation<float, UniformMagneticField>(*pConstantBfield);
  TestEquation<float, UniformMagneticField>(Bx);
  TestEquation<float, UniformMagneticField>(By);
  TestEquation<float, UniformMagneticField>(Bz);
  cout << endl;

  cout << " ---------------------------------------------------------------------" << endl;
  cout << " -- Testing Dormand-Prince stepper with magfield and equation (float).   " << endl;
  cout << " ---------------------------------------------------------------------" << endl;
  bool okStep1Float = TestStepper<float, UniformMagneticField>(*pConstantBfield);
  allStepGood       = okStep1Float;
#ifdef MORE
  bool okStep2Float = TestStepper<float, UniformMagneticField>(Bx);
  bool okStep3Float = TestStepper<float, UniformMagneticField>(By);
  bool okStep4Float = TestStepper<float, UniformMagneticField>(Bz);
  allStepGood &&    = okStep2Float && okStep3Float && okStep4Float;
#endif
  cout << endl;

  // float lenInc= 250;
  cout << " -------------------------------------------------------------------------------" << endl;
  cout << " -- Testing Integration driver 'Advance' with Dormand-Prince stepper (float). --" << endl;
  cout << " -------------------------------------------------------------------------------" << endl;

  using copcore::units::millimeter;
  bool okDoPri1Float = TestDriverAdvance<float, UniformMagneticField>(*pConstantBfield, 1000.0 * millimeter);

  bool allDoPriGood = okDoPri1Float;
  for (float len = 250; len < 5000; len *= 2.0) {
    cout << " -- Bx field -- \n";
    bool okDoPri2Float = TestDriverAdvance<float, UniformMagneticField>(Bx, len);

    cout << " -- By field -- \n";
    bool okDoPri3Float = TestDriverAdvance<float, UniformMagneticField>(By, 2.0 * len);

    cout << " -- Bz field -- \n";
    bool okDoPri4Float = TestDriverAdvance<float, UniformMagneticField>(Bz, 3.0 * len);

    allDoPriGood |= okDoPri1Float && okDoPri2Float && okDoPri3Float && okDoPri4Float;
  }

  cout << " -------------------------------------------------------------------------------" << endl;
  cout << " -- Testing Integration driver 'Advance' vs Helix stepper (float).            --" << endl;
  cout << " -------------------------------------------------------------------------------" << endl;
  bool allChecksGood = true;

  for (float len = 10; len < 1000000; len *= 2.0) {
    bool okCheck1 = CheckDriverVsHelix(*pConstantBfield, len);
    allChecksGood = allChecksGood && okCheck1;
  }
  cout << " Checks vs Helix result = " << (allChecksGood ? " OK ! " : " Errors Found ") << "\n";

  for (double len = 10; len < 1000000; len *= 2.0) {
    bool okCheck1 = CheckDriverVsHelix<double, UniformMagneticField>(*pConstantBfield, len);
    allChecksGood = allChecksGood && okCheck1;
  }

  return !(allStepGood && allDoPriGood && allChecksGood);
}
