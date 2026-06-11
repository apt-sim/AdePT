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
// - RkIntegrationDriver adaptive bookkeeping, including rejected substeps after
//   already accepted progress.
// - Real Dormand-Prince integration in float and double, checked against the
//   closed-form constant-field helix.
// - fieldPropagatorRungeKutta handoff behavior between RK advancement, safety,
//   chord navigation, and failed-advance handling.

// The production headers annotate many small functions for CUDA. Defining the
// qualifiers away lets this test compile the templated logic as ordinary host
// C++, which keeps the checks fast and runnable through GTest.
#define __device__
#define __host__

#include <AdePT/transport/support/PhysicalConstants.h>
#include <AdePT/transport/magneticfield/ConstFieldHelixStepper.h>
#include <AdePT/transport/magneticfield/DormandPrinceRK45.h>
#include <AdePT/transport/magneticfield/MagneticFieldEquation.h>
#include <AdePT/transport/magneticfield/RkIntegrationDriver.h>
#include <AdePT/transport/magneticfield/UniformMagneticField.cuh>
#include <AdePT/transport/magneticfield/fieldPropagatorRungeKutta.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavigationState.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>
#include <type_traits>

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

// Minimal propagator/navigator doubles for fieldPropagatorRungeKutta tests.
// These isolate the contract between the RK driver and VecGeom navigation
// without depending on a real geometry.

// Simulates the all-trials-failed path in RkIntegrationDriver::Advance. It
// leaves the endpoint unchanged, which is the case that used to risk normalizing
// a zero-length chord.
struct FailingPropagatorDriver {
  static double Advance(vecgeom::Vector3D<double> &, vecgeom::Vector3D<double> &, int, double, const DummyField &,
                        double[kNvar], double &, unsigned int, int)
  {
    return 0.0;
  }
};

// Deterministic successful RK substitute: advance exactly along the incoming
// momentum direction and report the full trial step as the last good step.
struct StraightLinePropagatorDriver {
  static double Advance(vecgeom::Vector3D<double> &position, vecgeom::Vector3D<double> &momentum, int, double step,
                        const DummyField &, double[kNvar], double &lastGoodStep, unsigned int, int)
  {
    position += step * momentum.Unit();
    lastGoodStep = step;
    return step;
  }
};

// Reports that the requested RK advance did not fully finish, while still
// returning a nonzero accepted endpoint. RkIntegrationDriver can do this after
// exhausting its trial budget with partial accepted progress, and the propagator
// must bookkeep only that accepted distance.
struct PartialThenCompletePropagatorDriver {
  inline static int calls = 0;

  static void Reset() { calls = 0; }

  static double Advance(vecgeom::Vector3D<double> &position, vecgeom::Vector3D<double> &momentum, int, double step,
                        const DummyField &, double[kNvar], double &lastGoodStep, unsigned int, int)
  {
    ++calls;
    const double acceptedStep = (calls == 1) ? 2.0 : step;
    position += acceptedStep * momentum.Unit();
    lastGoodStep = acceptedStep;
    return acceptedStep;
  }
};

// Navigator used when a test expects the propagator to return before geometry
// stepping. Any ComputeStepAndNextVolume call is therefore a test failure.
struct ThrowingNavigator {
  static constexpr double kBoundaryPush = 0.0;
  inline static int stepCalls           = 0;
  inline static int safetyCalls         = 0;

  static void Reset()
  {
    stepCalls   = 0;
    safetyCalls = 0;
  }

  static double ComputeSafety(const vecgeom::Vector3D<double> &, const vecgeom::NavigationState &, double)
  {
    ++safetyCalls;
    return 0.0;
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &, const vecgeom::Vector3D<double> &, double,
                                         const vecgeom::NavigationState &, vecgeom::NavigationState &, double)
  {
    ++stepCalls;
    ADD_FAILURE() << "navigation should not be called";
    return 0.0;
  }
};

// Fake navigator with a boundary at one quarter of the proposed chord. This
// verifies that fieldPropagatorRungeKutta shortens the accepted chord according
// to the navigator result and marks the boundary state.
struct BoundaryNavigator {
  static constexpr double kBoundaryPush = 0.0;
  inline static int stepCalls           = 0;

  static void Reset() { stepCalls = 0; }

  static double ComputeSafety(const vecgeom::Vector3D<double> &, const vecgeom::NavigationState &, double)
  {
    return 0.0;
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &, const vecgeom::Vector3D<double> &,
                                         double step, const vecgeom::NavigationState &,
                                         vecgeom::NavigationState &nextState, double)
  {
    ++stepCalls;
    nextState.SetBoundaryState(true);
    return 0.25 * step;
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

// RK driver tests. The first two are targeted regression tests for rejected
// substep progress accounting; the later ones exercise the real stepper against
// simple physics expectations in both precisions.

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

TEST(RkIntegrationDriver, ZeroFieldIsStraightLineInDoublePrecision)
{
  CheckZeroFieldStraightLine<double>();
}

TEST(RkIntegrationDriver, ZeroFieldIsStraightLineInFloatPrecision)
{
  CheckZeroFieldStraightLine<float>();
}

TEST(RkIntegrationDriver, UniformFieldMatchesHelixInDoublePrecision)
{
  for (const auto &testCase : UniformFieldCases()) {
    SCOPED_TRACE(testCase.name);
    CheckUniformFieldAgainstHelix<double>(testCase, testCase.doublePositionTolerance,
                                          testCase.doubleDirectionTolerance);
  }
}

TEST(RkIntegrationDriver, UniformFieldMatchesHelixInFloatPrecision)
{
  for (const auto &testCase : UniformFieldCases()) {
    SCOPED_TRACE(testCase.name);
    CheckUniformFieldAgainstHelix<float>(testCase, testCase.floatPositionTolerance, testCase.floatDirectionTolerance);
  }
}

// Full fieldPropagatorRungeKutta contract tests. These keep the RK driver fake
// and focus on the surrounding propagation logic: failed RK advance, safety
// shortcut, and navigator-limited chord acceptance.

TEST(FieldPropagatorRungeKutta, FailedAdvanceStopsBeforeZeroChordNormalization)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, FailingPropagatorDriver, double, ThrowingNavigator>;

  ThrowingNavigator::Reset();
  vecgeom::Vector3D<double> position{1.0, 2.0, 3.0};
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  vecgeom::NavigationState currentState;
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = true;
  bool zeroFirstStep   = false;
  int itersDone        = 0;

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 5.0, 5.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated, 0.0,
                                                            10, itersDone, 0, zeroFirstStep, false);

  EXPECT_DOUBLE_EQ(moved, 0.0);
  EXPECT_FALSE(propagated);
  EXPECT_EQ(ThrowingNavigator::stepCalls, 0);
  ExpectVectorNear(position, vecgeom::Vector3D<double>{1.0, 2.0, 3.0}, 0.0);
  ExpectVectorNear(direction, vecgeom::Vector3D<double>{1.0, 0.0, 0.0}, 0.0);
}

TEST(FieldPropagatorRungeKutta, UsesAcceptedArcLengthWhenDriverReportsIncomplete)
{
  using Propagator =
      fieldPropagatorRungeKutta<DummyField, PartialThenCompletePropagatorDriver, double, ThrowingNavigator>;

  PartialThenCompletePropagatorDriver::Reset();
  ThrowingNavigator::Reset();
  vecgeom::Vector3D<double> position{1.0, 2.0, 3.0};
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  vecgeom::NavigationState currentState;
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  bool zeroFirstStep   = false;
  int itersDone        = 0;

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 5.0, 5.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated, 100.0,
                                                            10, itersDone, 0, zeroFirstStep, false);

  EXPECT_DOUBLE_EQ(moved, 5.0);
  EXPECT_TRUE(propagated);
  EXPECT_EQ(PartialThenCompletePropagatorDriver::calls, 2);
  EXPECT_EQ(ThrowingNavigator::stepCalls, 0);
  ExpectVectorNear(position, vecgeom::Vector3D<double>{6.0, 2.0, 3.0}, 1.0e-14);
  ExpectVectorNear(direction, vecgeom::Vector3D<double>{1.0, 0.0, 0.0}, 0.0);
}

TEST(FieldPropagatorRungeKutta, UsesSafetyToAcceptFullChordWithoutNavigation)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, StraightLinePropagatorDriver, double, ThrowingNavigator>;

  ThrowingNavigator::Reset();
  vecgeom::Vector3D<double> position{1.0, 2.0, 3.0};
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  vecgeom::NavigationState currentState;
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  bool zeroFirstStep   = false;
  int itersDone        = 0;

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 5.0, 5.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated, 100.0,
                                                            10, itersDone, 0, zeroFirstStep, false);

  EXPECT_DOUBLE_EQ(moved, 5.0);
  EXPECT_TRUE(propagated);
  EXPECT_EQ(ThrowingNavigator::stepCalls, 0);
  ExpectVectorNear(position, vecgeom::Vector3D<double>{6.0, 2.0, 3.0}, 1.0e-14);
  ExpectVectorNear(direction, vecgeom::Vector3D<double>{1.0, 0.0, 0.0}, 0.0);
}

TEST(FieldPropagatorRungeKutta, StopsAtNavigatorBoundaryAlongChord)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, StraightLinePropagatorDriver, double, BoundaryNavigator>;

  BoundaryNavigator::Reset();
  vecgeom::Vector3D<double> position{1.0, 2.0, 3.0};
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  vecgeom::NavigationState currentState;
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  bool zeroFirstStep   = false;
  int itersDone        = 0;

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 8.0, 8.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated, 0.0,
                                                            10, itersDone, 0, zeroFirstStep, false);

  EXPECT_DOUBLE_EQ(moved, 2.0);
  EXPECT_TRUE(propagated);
  EXPECT_TRUE(nextState.IsOnBoundary());
  EXPECT_EQ(BoundaryNavigator::stepCalls, 1);
  ExpectVectorNear(position, vecgeom::Vector3D<double>{3.0, 2.0, 3.0}, 0.0);
  ExpectVectorNear(direction, vecgeom::Vector3D<double>{1.0, 0.0, 0.0}, 0.0);
}
