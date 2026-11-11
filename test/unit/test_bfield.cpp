// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

// Host-side GTest coverage for the AdePT magnetic-field transport pieces. The
// suite deliberately avoids launching CUDA kernels or requiring geometry setup:
// tiny fake fields, steppers, drivers, and navigators isolate contracts that
// were previously only covered indirectly by integration tests.
//
// Covered behavior:
// - UniformMagneticField storage/evaluation for float and double position types.
// - MagneticFieldEquation invariants: unit direction derivative, Lorentz force
//   perpendicularity, charge sign handling, and zero-field behavior.
// - fieldPropagatorRungeKutta::ComputeSafeLength: the drift from the tangent
//   after one safe length equals deltaChord on the exact helix for any angle
//   between momentum and field, and straight tracks get a safe length far
//   beyond any step.
// - RkIntegrationDriver adaptive bookkeeping, including rejected substeps after
//   already accepted progress.
// - Real Dormand-Prince integration in float and double, checked against the
//   closed-form constant-field helix.
// Navigation handoff behavior for fieldPropagatorRungeKutta lives in
// test_bfield_navigation.cpp.

// The production headers annotate many small functions for CUDA. Defining the
// qualifiers away lets this test compile the templated logic as ordinary host
// C++, which keeps the checks fast and runnable through GTest.
#define __device__
#define __host__

#include <AdePT/transport/support/PhysicalConstants.h>
#include <AdePT/transport/magneticfield/ConstBzFieldStepper.h>
#include <AdePT/transport/magneticfield/ConstFieldHelixStepper.h>
#include <AdePT/transport/magneticfield/fieldConstants.h>
#include <AdePT/transport/magneticfield/fieldPropagatorRungeKutta.h>
#include <AdePT/transport/magneticfield/DormandPrinceRK45.h>
#include <AdePT/transport/magneticfield/MagneticFieldEquation.h>
#include <AdePT/transport/magneticfield/OffsetMagneticField.h>
#include <AdePT/transport/magneticfield/RkIntegrationDriver.h>
#include <AdePT/transport/magneticfield/UniformMagneticField.cuh>
#include <AdePT/transport/tracks/SafetyCache.cuh>
#include <VecGeom/base/Vector3D.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr unsigned int kNvar = 6;

// Shared assertions and small adapters used by several independent test groups.

template <typename T>
void ExpectVectorNear(const vecgeom::Vector3D<T> &actual, const vecgeom::Vector3D<T> &expected, double tolerance)
{
  const double delta = (actual - expected).Length();
  EXPECT_LE(delta, tolerance) << "actual={" << actual[0] << ", " << actual[1] << ", " << actual[2] << "} expected={"
                              << expected[0] << ", " << expected[1] << ", " << expected[2] << "}";
}

template <typename T>
vecgeom::Vector3D<T> MomentumDerivative(const T dydx[kNvar])
{
  return {dydx[3], dydx[4], dydx[5]};
}

// Minimal equation/stepper doubles for RK driver bookkeeping tests. They make
// the integrated state linear in the path length, so any lost or duplicated
// progress shows up directly as an x-position error.

struct DummyField {};
struct UnusedRkDriver {};
struct UnusedNavigator {};

struct LinearEquation {
  static void EvaluateDerivatives(const DummyField &, const double[], int, double dydx[kNvar])
  {
    dydx[0] = 1.0;
    dydx[1] = 0.0;
    dydx[2] = 0.0;
    dydx[3] = 0.0;
    dydx[4] = 0.0;
    dydx[5] = 0.0;
  }
};

// Forces every attempted RK substep to fail. This directly checks that a
// rejected trial does not reset the cumulative progress already owned by the
// caller.
struct RejectingStepper {
  static constexpr unsigned int kMethodOrder = 4;

  static void StepWithErrorEstimate(const DummyField &, const double yStart[kNvar], const double dydx[kNvar], int,
                                    double htry, double yEnd[kNvar], double yErr[kNvar], double nextDydx[kNvar])
  {
    for (unsigned int i = 0; i < kNvar; ++i) {
      yEnd[i]     = yStart[i] + htry * dydx[i];
      yErr[i]     = 0.0;
      nextDydx[i] = dydx[i];
    }
    yErr[0] = 10.0 * htry;
  }
};

// Accepts one substep, rejects the next, then accepts again. This reproduces the
// important failure shape for the old bug: an accepted yStart already moved
// forward, followed by a rejected substep that must not make Advance forget that
// accepted distance.
struct OneRejectedSubstepStepper {
  static constexpr unsigned int kMethodOrder = 4;
  inline static int calls                    = 0;

  static void Reset() { calls = 0; }

  static void StepWithErrorEstimate(const DummyField &, const double yStart[kNvar], const double dydx[kNvar], int,
                                    double htry, double yEnd[kNvar], double yErr[kNvar], double nextDydx[kNvar])
  {
    ++calls;
    for (unsigned int i = 0; i < kNvar; ++i) {
      yEnd[i]     = yStart[i] + htry * dydx[i];
      yErr[i]     = 1.0e-12 * htry;
      nextDydx[i] = dydx[i];
    }
    if (calls == 2) yErr[0] = 10.0 * htry;
  }
};

// Shared data for real RK-vs-analytic-helix tests. The tolerances are split by
// precision because UniformMagneticField stores the field as float while the RK
// driver itself may run in either float or double.

struct RkCase {
  std::string name;
  vecgeom::Vector3D<float> field;
  vecgeom::Vector3D<double> direction;
  int charge;
  double momentum;
  double step;
  double doublePositionTolerance;
  double doubleDirectionTolerance;
  double floatPositionTolerance;
  double floatDirectionTolerance;
};

// Compare the actual Dormand-Prince field integration against the exact
// constant-field helix solution for a set of charges, field orientations,
// momenta, and step lengths.
template <typename Real_t>
void CheckUniformFieldAgainstHelix(const RkCase &testCase, double positionTolerance, double directionTolerance)
{
  using Field    = UniformMagneticField;
  using Equation = MagneticFieldEquation<Field>;
  using Stepper  = DormandPrinceRK45<Equation, Field, kNvar, Real_t>;
  using Driver   = RkIntegrationDriver<Stepper, Real_t, int, Equation, Field>;

  const Field field(testCase.field);
  const vecgeom::Vector3D<double> startPosition{0.25, -0.5, 1.0};
  const vecgeom::Vector3D<double> startDirection = testCase.direction.Unit();
  vecgeom::Vector3D<double> position             = startPosition;
  vecgeom::Vector3D<double> momentum             = testCase.momentum * startDirection;
  Real_t dydxEnd[kNvar]                          = {};
  Real_t lastGoodStep                            = 0.0;

  const Real_t advanced =
      Driver::Advance(position, momentum, testCase.charge, Real_t(testCase.step), field, dydxEnd, lastGoodStep, 500);
  constexpr double stepTolerance = std::is_same_v<Real_t, float> ? 1.0e-5 : 1.0e-12;
  EXPECT_NEAR(advanced, Real_t(testCase.step), stepTolerance);

  vecgeom::Vector3D<double> expectedPosition;
  vecgeom::Vector3D<double> expectedDirection;
  ConstFieldHelixStepper(
      vecgeom::Vector3D<double>{double(testCase.field[0]), double(testCase.field[1]), double(testCase.field[2])})
      .DoStep(startPosition, startDirection, testCase.charge, testCase.momentum, testCase.step, expectedPosition,
              expectedDirection);

  const vecgeom::Vector3D<double> direction = momentum.Unit();
  ExpectVectorNear(position, expectedPosition, positionTolerance);
  ExpectVectorNear(direction, expectedDirection.Unit(), directionTolerance);
  EXPECT_NEAR(direction.Length(), 1.0, directionTolerance);
}

// Cases are intentionally modest but not all axis-aligned: they cover both
// charge signs, multiple field directions, and enough path length to expose a
// meaningful curvature error without making float tolerances too loose.
std::array<RkCase, 5> UniformFieldCases()
{
  using copcore::units::MeV;
  using copcore::units::tesla;

  return {{{"bz_electron_short",
            {0.0f, 0.0f, float(1.0 * tesla)},
            {0.1, 1.0, 0.2},
            -1,
            250.0 * MeV,
            50.0,
            1.0e-6,
            1.0e-8,
            1.0e-3,
            5.0e-5},
           {"bz_positron_short",
            {0.0f, 0.0f, float(1.0 * tesla)},
            {0.1, 1.0, 0.2},
            1,
            250.0 * MeV,
            50.0,
            1.0e-6,
            1.0e-8,
            1.0e-3,
            5.0e-5},
           {"tilted_field_high_momentum",
            {float(0.8 * tesla), float(-0.2 * tesla), float(0.4 * tesla)},
            {0.3, 0.7, 0.6},
            1,
            1000.0 * MeV,
            250.0,
            5.0e-6,
            5.0e-8,
            2.0e-3,
            5.0e-5},
           {"x_field",
            {float(1.5 * tesla), 0.0f, 0.0f},
            {0.2, 0.6, 0.75},
            -1,
            750.0 * MeV,
            120.0,
            2.0e-6,
            2.0e-8,
            1.0e-3,
            5.0e-5},
           {"y_field",
            {0.0f, float(-0.7 * tesla), 0.0f},
            {0.9, 0.2, 0.3},
            1,
            400.0 * MeV,
            80.0,
            2.0e-6,
            2.0e-8,
            1.0e-3,
            5.0e-5}}};
}

} // namespace

// Field storage and equation tests. These are cheap invariants for the lowest
// layers of the magnetic-field stack.

// UniformMagneticField should be a position-independent field lookup. This
// catches accidental dependence on the position type or lossy return conversion
// when callers use either float or double coordinates.
TEST(UniformMagneticField, EvaluatesStoredFloatFieldForFloatAndDoublePositions)
{
  const vecgeom::Vector3D<float> storedField{1.25f, -2.5f, 0.75f};
  const UniformMagneticField field(storedField);

  const auto floatValue = field.Evaluate(vecgeom::Vector3D<float>{10.0f, 20.0f, -30.0f});
  EXPECT_FLOAT_EQ(floatValue[0], storedField[0]);
  EXPECT_FLOAT_EQ(floatValue[1], storedField[1]);
  EXPECT_FLOAT_EQ(floatValue[2], storedField[2]);

  const auto doubleValue = field.Evaluate(vecgeom::Vector3D<double>{-1.0, 2.0, 3.0});
  EXPECT_DOUBLE_EQ(doubleValue[0], double(storedField[0]));
  EXPECT_DOUBLE_EQ(doubleValue[1], double(storedField[1]));
  EXPECT_DOUBLE_EQ(doubleValue[2], double(storedField[2]));
}

// The equation of motion must use the unit momentum direction for dx/ds and
// produce a Lorentz kick perpendicular to both momentum and magnetic field.
TEST(MagneticFieldEquation, DerivativeDirectionIsUnitAndForceIsPerpendicular)
{
  using Equation = MagneticFieldEquation<UniformMagneticField>;

  const vecgeom::Vector3D<double> momentum{3.0, 4.0, 12.0};
  const vecgeom::Vector3D<double> bField{0.25, -0.5, 1.5};
  const double momentumMag = momentum.Length();
  const double y[kNvar]    = {1.0, -2.0, 3.0, momentum[0], momentum[1], momentum[2]};
  double dydx[kNvar]       = {};

  Equation::EvaluateDerivatives(y, bField, -1, dydx);

  EXPECT_NEAR(dydx[0], momentum[0] / momentumMag, 1.0e-14);
  EXPECT_NEAR(dydx[1], momentum[1] / momentumMag, 1.0e-14);
  EXPECT_NEAR(dydx[2], momentum[2] / momentumMag, 1.0e-14);
  const vecgeom::Vector3D<double> positionDerivative{dydx[0], dydx[1], dydx[2]};
  EXPECT_NEAR(positionDerivative.Length(), 1.0, 1.0e-14);

  const auto force              = MomentumDerivative(dydx);
  const double sineAngle        = bField.Cross(momentum).Length() / (momentumMag * bField.Length());
  const double expectedForceMag = copcore::units::kCLight * bField.Length() * sineAngle;
  EXPECT_NEAR(force.Length(), expectedForceMag, 1.0e-12 * expectedForceMag);
  EXPECT_NEAR(momentum.Dot(force), 0.0, 1.0e-12 * momentum.Length() * force.Length());
  EXPECT_NEAR(bField.Dot(force), 0.0, 1.0e-12 * bField.Length() * force.Length());
}

// Flipping the charge should only flip dp/ds. The spatial derivative is the
// track direction and must not depend on the charge sign.
TEST(MagneticFieldEquation, ChargeSignOnlyFlipsMomentumDerivative)
{
  using Equation = MagneticFieldEquation<UniformMagneticField>;

  const vecgeom::Vector3D<double> bField{0.0, 0.0, 1.0};
  const double y[kNvar]  = {0.0, 0.0, 0.0, 1.0, 2.0, 3.0};
  double negative[kNvar] = {};
  double positive[kNvar] = {};

  Equation::EvaluateDerivatives(y, bField, -1, negative);
  Equation::EvaluateDerivatives(y, bField, 1, positive);

  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(negative[i], positive[i]);
    EXPECT_DOUBLE_EQ(negative[i + 3], -positive[i + 3]);
  }
}

// With B = 0 the momentum derivative must vanish exactly. This guards against
// stale field values or cross-product sign mistakes leaking into zero-field mode.
TEST(MagneticFieldEquation, ZeroFieldKeepsMomentumConstant)
{
  using Equation = MagneticFieldEquation<UniformMagneticField>;

  const double y[kNvar]  = {0.0, 0.0, 0.0, 1.0, -2.0, 3.0};
  const double bField[3] = {0.0, 0.0, 0.0};
  double dydx[kNvar]     = {};

  Equation::EvaluateDerivativesGivenB(y, bField, -1, dydx);
  EXPECT_DOUBLE_EQ(dydx[3], 0.0);
  EXPECT_DOUBLE_EQ(dydx[4], 0.0);
  EXPECT_DOUBLE_EQ(dydx[5], 0.0);
}

// The constant-Bz stepper is a specialized analytic integrator for uniform
// fields along z. It should agree with the general constant-field helix stepper
// for the same field while keeping the simpler fast path available.
TEST(ConstBzFieldStepper, MatchesGeneralHelixStepperForZField)
{
  using copcore::units::MeV;
  using copcore::units::tesla;

  const float bz = float(1.25 * tesla);
  const ConstBzFieldStepper zStepper(bz);
  const ConstFieldHelixStepper generalStepper(vecgeom::Vector3D<double>{0.0, 0.0, double(bz)});
  const vecgeom::Vector3D<double> startPosition{0.25, -0.5, 1.0};
  const double momentum = 350.0 * MeV;
  const double step     = 75.0;

  for (const int charge : {-1, 1}) {
    for (const vecgeom::Vector3D<double> startDirection :
         {vecgeom::Vector3D<double>{0.9, 0.1, 0.4}.Unit(), vecgeom::Vector3D<double>{-0.2, 0.8, 0.3}.Unit()}) {
      SCOPED_TRACE(testing::Message() << "charge=" << charge << " direction={" << startDirection[0] << ", "
                                      << startDirection[1] << ", " << startDirection[2] << "}");

      vecgeom::Vector3D<double> zPosition;
      vecgeom::Vector3D<double> zDirection;
      vecgeom::Vector3D<double> generalPosition;
      vecgeom::Vector3D<double> generalDirection;

      zStepper.DoStep(startPosition, startDirection, charge, momentum, step, zPosition, zDirection);
      generalStepper.DoStep(startPosition, startDirection, charge, momentum, step, generalPosition, generalDirection);

      ExpectVectorNear(zPosition, generalPosition, 1.0e-12);
      ExpectVectorNear(zDirection, generalDirection, 1.0e-14);
    }
  }
}

// ComputeSafeLength returns the arc length after which the true trajectory has
// drifted by fieldConstants::deltaChord away from the straight line along the
// starting direction. The drift follows the curvature of the 3D trajectory,
// which depends on the angle between momentum and field. The check measures the
// drift on the exact constant-field helix, for several angles, momenta, charges
// and field directions. It is two-sided: a longer safe length lets the
// trajectory drift beyond deltaChord, and a shorter one costs extra steps, each
// of which counts towards the looper limit.
template <typename Real_t>
void CheckSafeLengthDriftEqualsDeltaChord()
{
  using copcore::units::MeV;
  using copcore::units::tesla;
  using Propagator = fieldPropagatorRungeKutta<UniformMagneticField, UnusedRkDriver, Real_t, UnusedNavigator>;

  const vecgeom::Vector3D<double> startPosition{0.25, -0.5, 1.0};
  const double deltaChord = fieldConstants::deltaChord;

  for (vecgeom::Vector3D<Real_t> field :
       {vecgeom::Vector3D<Real_t>{0.0, 0.0, Real_t(3.8 * tesla)},
        vecgeom::Vector3D<Real_t>{Real_t(0.8 * tesla), Real_t(-0.2 * tesla), Real_t(0.4 * tesla)}}) {
    const vecgeom::Vector3D<double> fieldVec{double(field[0]), double(field[1]), double(field[2])};
    const vecgeom::Vector3D<double> fieldDir = fieldVec.Unit();
    // Any unit vector perpendicular to the field, to set the angle between momentum and field.
    const vecgeom::Vector3D<double> perpDir = fieldDir.Cross(vecgeom::Vector3D<double>{1.0, 1.0, 0.0}).Unit();

    for (const double pitchDeg : {90.0, 60.0, 30.0, 10.0, 3.0}) {
      const double pitch                        = pitchDeg * M_PI / 180.0;
      const vecgeom::Vector3D<double> direction = std::cos(pitch) * fieldDir + std::sin(pitch) * perpDir;

      for (const double momentumMag : {50.0 * MeV, 2000.0 * MeV}) {
        for (const int charge : {-1, 1}) {
          SCOPED_TRACE(testing::Message() << "field={" << fieldVec[0] << ", " << fieldVec[1] << ", " << fieldVec[2]
                                          << "} pitch=" << pitchDeg << "deg p=" << momentumMag << " q=" << charge);

          vecgeom::Vector3D<double> momentum = momentumMag * direction;
          const double safeLength            = Propagator::ComputeSafeLength(momentum, field, charge);

          vecgeom::Vector3D<double> endPosition;
          vecgeom::Vector3D<double> endDirection;
          ConstFieldHelixStepper(fieldVec).DoStep(startPosition, direction, charge, momentumMag, safeLength,
                                                  endPosition, endDirection);

          const vecgeom::Vector3D<double> displacement = endPosition - startPosition;
          const double drift                           = (displacement - displacement.Dot(direction) * direction).Mag();

          // The safe length uses the leading-order drift, curvature * s^2 / 2. The higher-order
          // terms stay below 1% for these cases.
          EXPECT_NEAR(drift, deltaChord, 0.02 * deltaChord);
        }
      }
    }
  }
}

TEST(FieldPropagatorRungeKutta, SafeLengthDriftEqualsDeltaChordInDoublePrecision)
{
  CheckSafeLengthDriftEqualsDeltaChord<double>();
}

TEST(FieldPropagatorRungeKutta, SafeLengthDriftEqualsDeltaChordInFloatPrecision)
{
  CheckSafeLengthDriftEqualsDeltaChord<float>();
}

// A track moving along the field line, or through a region without field, goes
// straight. Its safe length is then far beyond any step, so geometry and physics
// alone set the step length. This matters on a boundary, where the geometric
// safety is zero and the safe length is the only other lower bound on the step.
template <typename Real_t>
void CheckSafeLengthDoesNotLimitUnbentTracks()
{
  using copcore::units::MeV;
  using copcore::units::tesla;
  using Propagator = fieldPropagatorRungeKutta<UniformMagneticField, UnusedRkDriver, Real_t, UnusedNavigator>;

  constexpr double kNoLimit = 1.0e6; // mm, far larger than any detector

  vecgeom::Vector3D<Real_t> fieldAlongZ{0.0, 0.0, Real_t(3.8 * tesla)};
  vecgeom::Vector3D<double> momentumAlongZ{0.0, 0.0, 300.0 * MeV};
  EXPECT_GT(Propagator::ComputeSafeLength(momentumAlongZ, fieldAlongZ, -1), kNoLimit);

  // Antiparallel to a tilted field, aligned with the field values as stored in
  // Real_t. The curvature is then pure rounding, which must stay negligible.
  vecgeom::Vector3D<Real_t> tiltedField{Real_t(0.8 * tesla), Real_t(-0.2 * tesla), Real_t(0.4 * tesla)};
  vecgeom::Vector3D<double> momentumAgainstField =
      -300.0 * MeV *
      vecgeom::Vector3D<double>{double(tiltedField[0]), double(tiltedField[1]), double(tiltedField[2])}.Unit();
  EXPECT_GT(Propagator::ComputeSafeLength(momentumAgainstField, tiltedField, 1), kNoLimit);

  vecgeom::Vector3D<Real_t> noField{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> momentum{100.0 * MeV, 200.0 * MeV, 50.0 * MeV};
  EXPECT_GT(Propagator::ComputeSafeLength(momentum, noField, -1), kNoLimit);
}

TEST(FieldPropagatorRungeKutta, SafeLengthDoesNotLimitUnbentTracksInDoublePrecision)
{
  CheckSafeLengthDoesNotLimitUnbentTracks<double>();
}

TEST(FieldPropagatorRungeKutta, SafeLengthDoesNotLimitUnbentTracksInFloatPrecision)
{
  CheckSafeLengthDoesNotLimitUnbentTracks<float>();
}

// RK driver tests. The first two are targeted regression tests for rejected
// substep progress accounting; the later ones exercise the real stepper against
// simple physics expectations in both precisions.

// A rejected substep must not alter progress already owned by the caller. This
// protects the retry path where the driver rejects before accepting new distance.
TEST(RkIntegrationDriver, RejectedStepKeepsCumulativeProgress)
{
  using Driver = RkIntegrationDriver<RejectingStepper, double, int, LinearEquation, DummyField>;

  const double yStart[kNvar] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  const double dydx[kNvar]   = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double yEnd[kNvar]         = {};
  double nextDydx[kNvar]     = {};
  double hnext               = 0.0;
  double progress            = 7.5;

  const bool goodStep = Driver::IntegrateStep(yStart, dydx, 1, progress, 1.0, DummyField{}, yEnd, nextDydx, hnext);

  EXPECT_FALSE(goodStep);
  EXPECT_DOUBLE_EQ(progress, 7.5);
}

// If the trial budget runs out after some accepted progress, Advance must report
// only that accepted distance instead of pretending the requested step completed.
TEST(RkIntegrationDriver, ReportsAcceptedProgressWhenTrialBudgetIsExhausted)
{
  using Driver = RkIntegrationDriver<OneRejectedSubstepStepper, double, int, LinearEquation, DummyField>;

  OneRejectedSubstepStepper::Reset();
  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> momentum{1.0, 0.0, 0.0};
  double dydxEnd[kNvar] = {};
  double lastGoodStep   = 4.0;
  const double advanced = Driver::Advance(position, momentum, 1, 10.0, DummyField{}, dydxEnd, lastGoodStep, 2, 1);

  EXPECT_LT(advanced, 10.0);
  EXPECT_DOUBLE_EQ(advanced, 4.0);
  EXPECT_NEAR(position[0], 4.0, 1.0e-12);
  EXPECT_NEAR(position[1], 0.0, 1.0e-12);
  EXPECT_NEAR(position[2], 0.0, 1.0e-12);
}

// A rejected substep in the middle of the integration should not make the next
// accepted trial overshoot the requested path length.
TEST(RkIntegrationDriver, AdvanceDoesNotOvershootAfterRejectedSubstep)
{
  using Driver = RkIntegrationDriver<OneRejectedSubstepStepper, double, int, LinearEquation, DummyField>;

  OneRejectedSubstepStepper::Reset();
  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> momentum{1.0, 0.0, 0.0};
  double dydxEnd[kNvar] = {};
  double lastGoodStep   = 4.0;

  const double advanced = Driver::Advance(position, momentum, 1, 10.0, DummyField{}, dydxEnd, lastGoodStep, 20, 1);
  EXPECT_DOUBLE_EQ(advanced, 10.0);
  EXPECT_NEAR(position[0], 10.0, 1.0e-12);
  EXPECT_NEAR(position[1], 0.0, 1.0e-12);
  EXPECT_NEAR(position[2], 0.0, 1.0e-12);
}

// Zero field should reduce the RK integration to a straight line. Running this
// for both Real_t values catches precision-specific template mistakes without a
// geometry dependency.
template <typename Real_t>
void CheckZeroFieldStraightLine()
{
  using Field    = UniformMagneticField;
  using Equation = MagneticFieldEquation<Field>;
  using Stepper  = DormandPrinceRK45<Equation, Field, kNvar, Real_t>;
  using Driver   = RkIntegrationDriver<Stepper, Real_t, int, Equation, Field>;

  const Field field({0.0f, 0.0f, 0.0f});
  const vecgeom::Vector3D<double> startPosition{-1.0, 0.25, 2.0};
  const vecgeom::Vector3D<double> startDirection = vecgeom::Vector3D<double>{0.2, -0.4, 0.9}.Unit();
  const double momentumMag                       = 42.0 * copcore::units::MeV;
  vecgeom::Vector3D<double> position             = startPosition;
  vecgeom::Vector3D<double> momentum             = momentumMag * startDirection;
  Real_t dydxEnd[kNvar]                          = {};
  Real_t lastGoodStep                            = 0.0;

  const Real_t advanced = Driver::Advance(position, momentum, -1, Real_t(125.0), field, dydxEnd, lastGoodStep, 100);
  constexpr double stepTolerance = std::is_same_v<Real_t, float> ? 1.0e-5 : 1.0e-12;
  EXPECT_NEAR(advanced, Real_t(125.0), stepTolerance);
  ExpectVectorNear(position, startPosition + 125.0 * startDirection, std::is_same_v<Real_t, float> ? 2.0e-5 : 1.0e-12);
  ExpectVectorNear(momentum.Unit(), startDirection, std::is_same_v<Real_t, float> ? 2.0e-7 : 1.0e-14);
}

// Double-precision RK integration should collapse to straight-line transport
// when the magnetic field is zero.
TEST(RkIntegrationDriver, ZeroFieldIsStraightLineInDoublePrecision)
{
  CheckZeroFieldStraightLine<double>();
}

// Float-precision RK integration should obey the same zero-field straight-line
// contract, within the looser float tolerance.
TEST(RkIntegrationDriver, ZeroFieldIsStraightLineInFloatPrecision)
{
  CheckZeroFieldStraightLine<float>();
}

// In a constant field the double-precision RK result should match the analytic
// helix. This catches unit, charge-sign, and field-orientation regressions.
TEST(RkIntegrationDriver, UniformFieldMatchesHelixInDoublePrecision)
{
  for (const auto &testCase : UniformFieldCases()) {
    SCOPED_TRACE(testCase.name);
    CheckUniformFieldAgainstHelix<double>(testCase, testCase.doublePositionTolerance,
                                          testCase.doubleDirectionTolerance);
  }
}

// The same helix comparison is run for the float RK path, where the tolerance
// reflects the float field storage and integration arithmetic.
TEST(RkIntegrationDriver, UniformFieldMatchesHelixInFloatPrecision)
{
  for (const auto &testCase : UniformFieldCases()) {
    SCOPED_TRACE(testCase.name);
    CheckUniformFieldAgainstHelix<float>(testCase, testCase.floatPositionTolerance, testCase.floatDirectionTolerance);
  }
}

// Run the production mixed-precision scheme (float offset from a double-precision
// start point, field seen through OffsetMagneticField) against the exact helix
// and return the final position error.
template <typename Real_t>
double AnchoredHelixPositionError(const RkCase &testCase, const vecgeom::Vector3D<double> &startPosition,
                                  double arcLength)
{
  using Field         = UniformMagneticField;
  using AnchoredField = OffsetMagneticField<Field>;
  using Equation      = MagneticFieldEquation<AnchoredField>;
  using Stepper       = DormandPrinceRK45<Equation, AnchoredField, kNvar, Real_t>;
  using Driver        = RkIntegrationDriver<Stepper, Real_t, int, Equation, AnchoredField>;

  const Field field(testCase.field);
  const AnchoredField anchoredField(field, startPosition);
  const vecgeom::Vector3D<double> startDirection = testCase.direction.Unit();
  vecgeom::Vector3D<double> displacement{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> momentum = testCase.momentum * startDirection;
  Real_t dydxEnd[kNvar]              = {};
  Real_t lastGoodStep                = 0.0;

  const Real_t advanced = Driver::Advance(displacement, momentum, testCase.charge, Real_t(arcLength), anchoredField,
                                          dydxEnd, lastGoodStep, 500);
  EXPECT_NEAR(advanced, Real_t(arcLength), (std::is_same_v<Real_t, float> ? 1.0e-5 : 1.0e-12) * arcLength);

  vecgeom::Vector3D<double> expectedPosition;
  vecgeom::Vector3D<double> expectedDirection;
  ConstFieldHelixStepper(
      vecgeom::Vector3D<double>{double(testCase.field[0]), double(testCase.field[1]), double(testCase.field[2])})
      .DoStep(startPosition, startDirection, testCase.charge, testCase.momentum, arcLength, expectedPosition,
              expectedDirection);

  return (startPosition + displacement - expectedPosition).Length();
}

// The float accuracy must not depend on where in the detector the chord starts.
// Without the offset scheme, the float resolution at large coordinates
// (~1e-4 mm at r = 1 m) dominates the error and this test fails at the large radius.
TEST(RkIntegrationDriver, AnchoredFloatAccuracyIsIndependentOfStartRadius)
{
  const vecgeom::Vector3D<double> nearOrigin{0.25, -0.5, 1.0};
  const vecgeom::Vector3D<double> largeRadius{2078.46, -2078.46, 2078.46}; // r ~ 3.6 m

  for (const auto &testCase : UniformFieldCases()) {
    SCOPED_TRACE(testCase.name);
    const double errorNearOrigin  = AnchoredHelixPositionError<float>(testCase, nearOrigin, testCase.step);
    const double errorLargeRadius = AnchoredHelixPositionError<float>(testCase, largeRadius, testCase.step);
    EXPECT_LE(errorNearOrigin, testCase.floatPositionTolerance);
    EXPECT_LE(errorLargeRadius, testCase.floatPositionTolerance);
  }
}

// Very short steps far from the origin are what produced stuck tracks before
// this fix: the step was smaller than float can represent at such coordinates,
// so the bookkeeping advanced but the position did not follow. The step must
// move the track by the requested distance.
TEST(RkIntegrationDriver, AnchoredFloatResolvesMicronArcsAtLargeRadius)
{
  const vecgeom::Vector3D<double> largeRadius{2078.46, -2078.46, 2078.46}; // r ~ 3.6 m
  const auto testCase = UniformFieldCases()[2];                            // tilted field, 1 GeV

  // 1e-7 mm matches the anti-stuck push (100 * vecgeom::kTolerance) in the
  // electron kernels, which was lost in float at this radius before the fix.
  for (const double arc : {1.0e-3, 1.0e-4, 1.0e-7}) {
    SCOPED_TRACE(arc);
    using Field         = UniformMagneticField;
    using AnchoredField = OffsetMagneticField<Field>;
    using Equation      = MagneticFieldEquation<AnchoredField>;
    using Stepper       = DormandPrinceRK45<Equation, AnchoredField, kNvar, float>;
    using Driver        = RkIntegrationDriver<Stepper, float, int, Equation, AnchoredField>;

    const Field field(testCase.field);
    const AnchoredField anchoredField(field, largeRadius);
    vecgeom::Vector3D<double> displacement{0.0, 0.0, 0.0};
    vecgeom::Vector3D<double> momentum = testCase.momentum * testCase.direction.Unit();
    float dydxEnd[kNvar]               = {};
    float lastGoodStep                 = 0.0;

    const float advanced =
        Driver::Advance(displacement, momentum, testCase.charge, float(arc), anchoredField, dydxEnd, lastGoodStep, 500);

    EXPECT_NEAR(advanced, arc, 1.0e-5 * arc);
    EXPECT_NEAR(displacement.Length(), arc, 1.0e-2 * arc);
  }
}

namespace {

// Constant field that records every position it is asked to evaluate. Used to
// check that OffsetMagneticField passes the true absolute coordinates, in
// double, to the underlying field.
struct PositionRecordingField {
  vecgeom::Vector3D<float> fieldValue;
  mutable std::vector<vecgeom::Vector3D<double>> queries;

  explicit PositionRecordingField(const vecgeom::Vector3D<float> &value) : fieldValue(value) {}

  template <typename Real_t>
  vecgeom::Vector3D<Real_t> Evaluate(Real_t x, Real_t y, Real_t z) const
  {
    queries.emplace_back(double(x), double(y), double(z));
    return {Real_t(fieldValue[0]), Real_t(fieldValue[1]), Real_t(fieldValue[2])};
  }

  template <typename Real_t>
  vecgeom::Vector3D<Real_t> Evaluate(const vecgeom::Vector3D<Real_t> &position) const
  {
    return Evaluate(position[0], position[1], position[2]);
  }
};

// Field whose z component grows linearly with the absolute z coordinate:
// B_z(z) = slope * z. If the lookup loses the absolute position, the field
// strength is badly wrong and the trajectory visibly different.
struct LinearBzGradientField {
  double slopePerMm; // field (internal units) per mm of z

  explicit LinearBzGradientField(double slope) : slopePerMm(slope) {}

  template <typename Real_t>
  vecgeom::Vector3D<Real_t> Evaluate(Real_t /*x*/, Real_t /*y*/, Real_t z) const
  {
    return {Real_t(0), Real_t(0), Real_t(slopePerMm * double(z))};
  }

  template <typename Real_t>
  vecgeom::Vector3D<Real_t> Evaluate(const vecgeom::Vector3D<Real_t> &position) const
  {
    return Evaluate(position[0], position[1], position[2]);
  }
};

} // namespace

// The wrapped lookup must receive the absolute coordinates, reconstructed in
// double. We check the recorded query points directly, for a step so small
// that float cannot represent it next to the anchor coordinates: every query
// must lie on the short arc at the anchor (not near the origin, as it would
// if the anchor were not added), and the queries must still differ from each
// other (adding the anchor in float would round them all to the same value).
TEST(OffsetMagneticField, RestoresAbsoluteCoordinatesInDoubleForSubUlpArcs)
{
  using copcore::units::tesla;
  using Field         = PositionRecordingField;
  using AnchoredField = OffsetMagneticField<Field>;
  using Equation      = MagneticFieldEquation<AnchoredField>;
  using Stepper       = DormandPrinceRK45<Equation, AnchoredField, kNvar, float>;
  using Driver        = RkIntegrationDriver<Stepper, float, int, Equation, AnchoredField>;

  const vecgeom::Vector3D<double> anchor{2078.46, -2078.46, 2078.46}; // r ~ 3.6 m, float ULP ~ 2.4e-4 mm
  const double arc = 1.0e-4;                                          // below one float ULP of the anchor coordinates

  const Field field(vecgeom::Vector3D<float>{0.0f, 0.0f, float(2.0 * tesla)});
  const AnchoredField anchoredField(field, anchor);
  vecgeom::Vector3D<double> displacement{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> momentum = 100.0 * copcore::units::MeV * vecgeom::Vector3D<double>{0.3, -0.9, 0.31}.Unit();
  float dydxEnd[kNvar]               = {};
  float lastGoodStep                 = 0.0;

  const float advanced =
      Driver::Advance(displacement, momentum, -1, float(arc), anchoredField, dydxEnd, lastGoodStep, 500);
  EXPECT_NEAR(advanced, arc, 1.0e-5 * arc);

  ASSERT_GE(field.queries.size(), 7u); // at least the 7 Dormand-Prince stages
  double maxSpread = 0.0;
  for (const auto &query : field.queries) {
    // Without the anchor, the queries would end up ~3.6 m away, near the origin.
    EXPECT_LE((query - anchor).Length(), 2.0 * arc)
        << "query not at absolute coordinates: {" << query[0] << ", " << query[1] << ", " << query[2] << "}";
    maxSpread = std::max(maxSpread, (query - field.queries.front()).Length());
  }
  // The queried points must spread out along the arc. If the anchor were added
  // in float, they would all round to the same value and the spread would be zero.
  EXPECT_GE(maxSpread, 0.5 * arc);
}

// Same check at the physics level: with a field that depends strongly on the
// absolute z coordinate, the float integration far from the origin must give
// the same trajectory as the double integration. If the absolute coordinate
// were lost, the field would be nearly zero and the end point off by orders
// of magnitude.
TEST(OffsetMagneticField, AnchoredFloatMatchesDoubleInGradientFieldAtLargeOrigin)
{
  using copcore::units::MeV;
  using copcore::units::tesla;
  using Field = LinearBzGradientField;

  // B_z = 1 T per 1000 mm of z: ~2 T at the start position, ~0 near the origin.
  const Field field(1.0 * tesla / 1000.0);
  const vecgeom::Vector3D<double> start{1500.0, -800.0, 2000.0};
  const vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0}; // bending stays in xy, so B_z stays ~2 T
  const double momentumMag = 100.0 * MeV;
  const double arc         = 20.0;

  // Double-precision reference on absolute coordinates.
  vecgeom::Vector3D<double> positionD = start;
  vecgeom::Vector3D<double> momentumD = momentumMag * direction;
  {
    using Equation        = MagneticFieldEquation<Field>;
    using Stepper         = DormandPrinceRK45<Equation, Field, kNvar, double>;
    using Driver          = RkIntegrationDriver<Stepper, double, int, Equation, Field>;
    double dydxEnd[kNvar] = {};
    double lastGoodStep   = 0.0;
    const double advanced = Driver::Advance(positionD, momentumD, -1, arc, field, dydxEnd, lastGoodStep, 500);
    ASSERT_NEAR(advanced, arc, 1.0e-12 * arc);
  }
  // The reference must actually bend; otherwise this test could not detect a
  // lost absolute coordinate.
  const double sagitta = (positionD - (start + arc * direction)).Length();
  ASSERT_GT(sagitta, 0.5);

  // Float integration with the offset scheme, as used in production.
  using AnchoredField = OffsetMagneticField<Field>;
  using EquationF     = MagneticFieldEquation<AnchoredField>;
  using StepperF      = DormandPrinceRK45<EquationF, AnchoredField, kNvar, float>;
  using DriverF       = RkIntegrationDriver<StepperF, float, int, EquationF, AnchoredField>;

  const AnchoredField anchoredField(field, start);
  vecgeom::Vector3D<double> displacement{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> momentumF = momentumMag * direction;
  float dydxEndF[kNvar]               = {};
  float lastGoodStepF                 = 0.0;
  const float advancedF =
      DriverF::Advance(displacement, momentumF, -1, float(arc), anchoredField, dydxEndF, lastGoodStepF, 500);
  EXPECT_NEAR(advancedF, arc, 1.0e-5 * arc);

  ExpectVectorNear(start + displacement, positionD, 2.0e-3);
}

// Safety is allowed to be conservative, but it must never be overestimated by
// losing a small displacement at large global coordinates. This catches the case
// where the safety origin is stored in float and a sub-ULP move disappears.
TEST(SafetyCache, PreservesSmallDisplacementsAtLargeGlobalCoordinates)
{
  SafetyCache safetyCache;
  const vecgeom::Vector3D<double> origin{4601.5, 0.0, 0.0};
  const vecgeom::Vector3D<double> position{4601.5002, 0.0, 0.0};

  safetyCache.Refresh(origin, 3.0e-4);

  const double expectedSafety = 3.0e-4 - (position - origin).Length();
  EXPECT_NEAR(safetyCache.SafetyAt(position), expectedSafety, 1.0e-15);
}
