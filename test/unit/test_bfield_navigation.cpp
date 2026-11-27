// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

// Host-side GTest coverage for the fieldPropagatorRungeKutta handoff between
// RK advancement, geometric safety, and VecGeom navigation. The tests use fake
// drivers and navigators so they can pin boundary-navigation contracts without
// requiring a real geometry or launching CUDA kernels.

// The production headers annotate many small functions for CUDA. Defining the
// qualifiers away lets this test compile the templated logic as ordinary host
// C++, which keeps the checks fast and runnable through GTest.
#define __device__
#define __host__

#include <AdePT/transport/support/PhysicalConstants.h>
#include <AdePT/transport/magneticfield/DormandPrinceRK45.h>
#include <AdePT/transport/magneticfield/MagneticFieldEquation.h>
#include <AdePT/transport/magneticfield/OffsetMagneticField.h>
#include <AdePT/transport/magneticfield/RkIntegrationDriver.h>
#include <AdePT/transport/magneticfield/UniformMagneticField.cuh>
#include <AdePT/transport/magneticfield/fieldPropagatorRungeKutta.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavigationState.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <vector>

namespace {

constexpr unsigned int kNvar              = 6;
constexpr unsigned long kParentNavIndex   = 1;
constexpr unsigned long kDaughterNavIndex = 2;
constexpr unsigned long kOtherNavIndex    = 3;

struct DummyField {};

template <typename T>
void ExpectVectorNear(const vecgeom::Vector3D<T> &actual, const vecgeom::Vector3D<T> &expected, double tolerance)
{
  const double delta = (actual - expected).Length();
  EXPECT_LE(delta, tolerance) << "actual={" << actual[0] << ", " << actual[1] << ", " << actual[2] << "} expected={"
                              << expected[0] << ", " << expected[1] << ", " << expected[2] << "}";
}

vecgeom::NavigationState MakeState(unsigned long navIndex, bool onBoundary)
{
  vecgeom::NavigationState state;
  state.SetNavIndex(navIndex);
  state.SetBoundaryState(onBoundary);
  return state;
}

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

// Exact circular motion keeps this carry-over regression independent of RK
// tolerances while recording the arc lengths requested by the propagator.
struct RecordingCircularPropagatorDriver {
  static constexpr double kRadius = 8.0;
  inline static std::array<double, 16> requestedSteps{};
  inline static unsigned int calls = 0;

  static void Reset()
  {
    calls = 0;
    requestedSteps.fill(0.0);
  }

  static double Advance(vecgeom::Vector3D<double> &position, vecgeom::Vector3D<double> &momentum, int, double step,
                        const DummyField &, double[kNvar], double &lastGoodStep, unsigned int, int)
  {
    if (calls < requestedSteps.size()) requestedSteps[calls] = step;
    ++calls;
    const double angle = step / kRadius;
    const auto tangent = momentum.Unit();
    const vecgeom::Vector3D<double> normal{-tangent[1], tangent[0], 0.0};
    position += kRadius * (std::sin(angle) * tangent + (1.0 - std::cos(angle)) * normal);
    momentum     = momentum.Mag() * (std::cos(angle) * tangent + std::sin(angle) * normal);
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

// Produces a chord whose direction differs from the physical direction kept in
// the momentum vector. This models the boundary-ambiguity shape where the RK
// chord can touch one side of a tolerance surface while the transported
// direction points somewhere else.
struct BentChordPropagatorDriver {
  static double Advance(vecgeom::Vector3D<double> &position, vecgeom::Vector3D<double> &, int, double step,
                        const DummyField &, double[kNvar], double &lastGoodStep, unsigned int, int)
  {
    position += step * vecgeom::Vector3D<double>{1.0, 0.0, -1.0e-5}.Unit();
    lastGoodStep = step;
    return step;
  }
};

// Produces a first RK chord shorter than the ambiguity band. This protects the
// classifier from accepting a physical-direction crossing that lies beyond the
// endpoint actually integrated by the RK driver.
struct TinyBentChordPropagatorDriver {
  static constexpr double kAcceptedStep = 4.0e-8;

  static double Advance(vecgeom::Vector3D<double> &position, vecgeom::Vector3D<double> &, int, double step,
                        const DummyField &, double[kNvar], double &lastGoodStep, unsigned int, int)
  {
    const double acceptedStep = (step < kAcceptedStep) ? step : kAcceptedStep;
    position += acceptedStep * vecgeom::Vector3D<double>{1.0, 0.0, -1.0e-5}.Unit();
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
                                         const vecgeom::NavigationState &, vecgeom::NavigationState &)
  {
    ++stepCalls;
    ADD_FAILURE() << "navigation should not be called";
    return 0.0;
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &position,
                                         const vecgeom::Vector3D<double> &direction, double step,
                                         const vecgeom::NavigationState &currentState,
                                         vecgeom::NavigationState &nextState, long &hitSurfaceIndex)
  {
    hitSurfaceIndex = -1;
    return ComputeStepAndNextVolume(position, direction, step, currentState, nextState);
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
                                         vecgeom::NavigationState &nextState)
  {
    ++stepCalls;
    nextState.SetBoundaryState(true);
    return 0.25 * step;
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &position,
                                         const vecgeom::Vector3D<double> &direction, double step,
                                         const vecgeom::NavigationState &currentState,
                                         vecgeom::NavigationState &nextState, long &hitSurfaceIndex)
  {
    hitSurfaceIndex = 0;
    return ComputeStepAndNextVolume(position, direction, step, currentState, nextState);
  }
};

// Navigator stub with a configurable finite safety and an optional boundary at
// a fixed distance on a chosen navigation call. Counts safety refreshes and
// navigation calls so tests can assert how many chords were integrated, and
// records the chords passed to navigation.
struct FiniteSafetyNavigator {
  static constexpr double kBoundaryPush        = 1.0e-8;
  static constexpr int kMaxRecordedChords      = 32;
  inline static int safetyCalls                = 0;
  inline static int stepCalls                  = 0;
  inline static double safety                  = 0.0;
  inline static double boundaryDistance        = -1.0; // < 0: never report a boundary
  inline static int boundaryOnCall             = -1;   // 1-based navigation call that reports it
  inline static unsigned long boundaryNavIndex = kDaughterNavIndex;
  inline static std::array<vecgeom::Vector3D<double>, kMaxRecordedChords> chordStarts{};
  inline static std::array<vecgeom::Vector3D<double>, kMaxRecordedChords> chordDirections{};
  inline static std::array<double, kMaxRecordedChords> chordLengths{};

  static void Reset(double safetyValue, double boundaryDist, int boundaryCall)
  {
    safetyCalls      = 0;
    stepCalls        = 0;
    safety           = safetyValue;
    boundaryDistance = boundaryDist;
    boundaryOnCall   = boundaryCall;
  }

  static double ComputeSafety(const vecgeom::Vector3D<double> &, const vecgeom::NavigationState &, double)
  {
    ++safetyCalls;
    return safety;
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &position,
                                         const vecgeom::Vector3D<double> &direction, double step,
                                         const vecgeom::NavigationState &currentState,
                                         vecgeom::NavigationState &nextState)
  {
    if (stepCalls < kMaxRecordedChords) {
      chordStarts[stepCalls]     = position;
      chordDirections[stepCalls] = direction;
      chordLengths[stepCalls]    = step;
    }
    ++stepCalls;
    if (stepCalls == boundaryOnCall && boundaryDistance >= 0.0 && boundaryDistance < step) {
      nextState = MakeState(boundaryNavIndex, true);
      return boundaryDistance;
    }
    currentState.CopyTo(&nextState);
    nextState.SetBoundaryState(false);
    return step + 1.0; // no boundary before the proposed step
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &position,
                                         const vecgeom::Vector3D<double> &direction, double step,
                                         const vecgeom::NavigationState &currentState,
                                         vecgeom::NavigationState &nextState, long &hitSurfaceIndex)
  {
    hitSurfaceIndex = 0;
    return ComputeStepAndNextVolume(position, direction, step, currentState, nextState);
  }
};

struct NavigationResponse {
  double move;
  unsigned long navIndex;
  bool onBoundary;
};

// Scripted navigator for boundary-ambiguity algorithm tests. A negative move,
// or a positive move beyond the requested maximum step, is interpreted as "no
// boundary before the proposed step". This models the VecGeom max-step contract
// used by the physical-direction probe.
struct ScriptedNavigator {
  static constexpr double kBoundaryPush                                     = 1.0e-8;
  static constexpr int kMaxScriptedCalls                                    = 256;
  inline static std::array<NavigationResponse, kMaxScriptedCalls> responses = {};
  inline static std::array<double, kMaxScriptedCalls> requestedSteps        = {};
  inline static int responseCount                                           = 0;
  inline static int stepCalls                                               = 0;

  static void Reset(std::initializer_list<NavigationResponse> script)
  {
    responseCount = 0;
    stepCalls     = 0;
    requestedSteps.fill(0.0);
    for (const NavigationResponse &response : script) {
      responses[responseCount++] = response;
    }
  }

  static double ComputeSafety(const vecgeom::Vector3D<double> &, const vecgeom::NavigationState &, double)
  {
    return 0.0;
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &, const vecgeom::Vector3D<double> &,
                                         double step, const vecgeom::NavigationState &currentState,
                                         vecgeom::NavigationState &nextState)
  {
    const int responseIndex = (stepCalls < responseCount) ? stepCalls : responseCount - 1;
    if (stepCalls < kMaxScriptedCalls) requestedSteps[stepCalls] = step;
    ++stepCalls;
    const NavigationResponse &response = responses[responseIndex];
    currentState.CopyTo(&nextState);
    if (response.move >= 0.0 && response.move <= step) {
      nextState.SetNavIndex(response.navIndex);
      nextState.SetBoundaryState(response.onBoundary);
      return response.move;
    }
    nextState.SetBoundaryState(false);
    return step;
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &position,
                                         const vecgeom::Vector3D<double> &direction, double step,
                                         const vecgeom::NavigationState &currentState,
                                         vecgeom::NavigationState &nextState, long &hitSurfaceIndex)
  {
    hitSurfaceIndex = stepCalls;
    return ComputeStepAndNextVolume(position, direction, step, currentState, nextState);
  }
};

} // namespace

// Existing fieldPropagatorRungeKutta navigation contracts. These keep the RK
// driver fake and focus on failed RK advance, safety shortcuts, and
// navigator-limited chord acceptance.

// If the RK driver makes no accepted progress, the propagator must stop before
// normalizing a zero-length chord or asking geometry to navigate a bogus step.
TEST(FieldPropagatorRungeKuttaNavigation, FailedAdvanceStopsBeforeZeroChordNormalization)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, FailingPropagatorDriver, double, ThrowingNavigator>;

  ThrowingNavigator::Reset();
  vecgeom::Vector3D<double> position{1.0, 2.0, 3.0};
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  vecgeom::NavigationState currentState;
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = true;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 5.0, 5.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 0.0);
  EXPECT_FALSE(propagated);
  EXPECT_EQ(ThrowingNavigator::stepCalls, 0);
  ExpectVectorNear(position, vecgeom::Vector3D<double>{1.0, 2.0, 3.0}, 0.0);
  ExpectVectorNear(direction, vecgeom::Vector3D<double>{1.0, 0.0, 0.0}, 0.0);
}

// The RK driver may return partial accepted progress after exhausting its trial
// budget. The propagator must bookkeep that accepted arc length and continue.
TEST(FieldPropagatorRungeKuttaNavigation, UsesAcceptedArcLengthWhenDriverReportsIncomplete)
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
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 100.0);

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 5.0, 5.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 5.0);
  EXPECT_TRUE(propagated);
  EXPECT_EQ(PartialThenCompletePropagatorDriver::calls, 2);
  EXPECT_EQ(ThrowingNavigator::stepCalls, 0);
  ExpectVectorNear(position, vecgeom::Vector3D<double>{6.0, 2.0, 3.0}, 1.0e-14);
  ExpectVectorNear(direction, vecgeom::Vector3D<double>{1.0, 0.0, 0.0}, 0.0);
}

// A large cached geometric safety should allow the whole chord to be accepted
// without calling VecGeom navigation. This protects the fast safety shortcut.
TEST(FieldPropagatorRungeKuttaNavigation, UsesSafetyToAcceptFullChordWithoutNavigation)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, StraightLinePropagatorDriver, double, ThrowingNavigator>;

  ThrowingNavigator::Reset();
  vecgeom::Vector3D<double> position{1.0, 2.0, 3.0};
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  vecgeom::NavigationState currentState;
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 100.0);

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 5.0, 5.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 5.0);
  EXPECT_TRUE(propagated);
  EXPECT_EQ(ThrowingNavigator::stepCalls, 0);
  ExpectVectorNear(position, vecgeom::Vector3D<double>{6.0, 2.0, 3.0}, 1.0e-14);
  ExpectVectorNear(direction, vecgeom::Vector3D<double>{1.0, 0.0, 0.0}, 0.0);
}

// Geometry safety may enlarge one arc without enlarging the curvature
// allowance for subsequent chords. The first 10 mm arc has a 9.362 mm chord,
// wholly inside the initial safety sphere. That safety is exhausted later.
// Reusing its 10 mm arc length would incorrectly request the remaining 5 mm
// in one go; the original 2 mm allowance must instead be restored.
TEST(FieldPropagatorRungeKuttaNavigation, DoesNotCarrySafetyEnlargedArcIntoNextChord)
{
  using Driver     = RecordingCircularPropagatorDriver;
  using Propagator = fieldPropagatorRungeKutta<DummyField, Driver, double, FiniteSafetyNavigator>;
  Driver::Reset();
  FiniteSafetyNavigator::Reset(/*safetyValue=*/0.0, /*boundaryDist=*/-1.0, /*boundaryCall=*/-1);

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  const auto currentState = MakeState(kParentNavIndex, false);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 10.0);

  constexpr double safeLength  = 2.0;
  constexpr double physicsStep = 15.0;
  const double moved           = Propagator::ComputeStepAndNextVolume(
      DummyField{}, 10.0, 1.0, 1, physicsStep, safeLength, position, direction, currentState, nextState,
      hitSurfaceIndex, propagated, safetyCache, 20, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, physicsStep);
  EXPECT_TRUE(propagated);
  EXPECT_FALSE(nextState.IsOnBoundary());
  EXPECT_TRUE(nextState.HasSamePathAsOther(currentState));
  EXPECT_DOUBLE_EQ(Driver::requestedSteps[0], 10.0);
  EXPECT_DOUBLE_EQ(Driver::requestedSteps[1], safeLength);
  EXPECT_EQ(Driver::calls, 4u);
  EXPECT_DOUBLE_EQ(Driver::requestedSteps[2], safeLength);
  EXPECT_DOUBLE_EQ(Driver::requestedSteps[3], 1.0);
  EXPECT_EQ(FiniteSafetyNavigator::stepCalls, 3);

  const double angle = physicsStep / Driver::kRadius;
  ExpectVectorNear(
      position,
      vecgeom::Vector3D<double>{Driver::kRadius * std::sin(angle), Driver::kRadius * (1.0 - std::cos(angle)), 0.0},
      1.0e-12);
  ExpectVectorNear(direction, vecgeom::Vector3D<double>{std::cos(angle), std::sin(angle), 0.0}, 1.0e-12);
}

// The cached safety has to be reduced to the current chord start, not to the
// proposed endpoint. Otherwise a safe chord can be rejected and sent to geometry.
TEST(FieldPropagatorRungeKuttaNavigation, UsesCurrentSafetyBeforeTestingCandidateChord)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, StraightLinePropagatorDriver, double, ThrowingNavigator>;

  ThrowingNavigator::Reset();
  vecgeom::Vector3D<double> position{1.0, 2.0, 3.0};
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  vecgeom::NavigationState currentState;
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 6.0);

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 5.0, 5.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 5.0);
  EXPECT_TRUE(propagated);
  EXPECT_EQ(ThrowingNavigator::stepCalls, 0);
  ExpectVectorNear(position, vecgeom::Vector3D<double>{6.0, 2.0, 3.0}, 1.0e-14);
  ExpectVectorNear(direction, vecgeom::Vector3D<double>{1.0, 0.0, 0.0}, 0.0);
}

// The stored cache value belongs to its origin, not necessarily to the current
// position. A cache from an older point must be reduced before accepting a chord.
TEST(FieldPropagatorRungeKuttaNavigation, ReducesSafetyFromCacheOriginBeforeTestingChord)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, StraightLinePropagatorDriver, double, BoundaryNavigator>;

  BoundaryNavigator::Reset();
  vecgeom::Vector3D<double> position{1.0, 2.0, 3.0};
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  vecgeom::NavigationState currentState;
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(vecgeom::Vector3D<double>{0.0, 2.0, 3.0}, 5.5);

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 5.0, 5.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 1.25);
  EXPECT_TRUE(propagated);
  EXPECT_TRUE(nextState.IsOnBoundary());
  EXPECT_EQ(BoundaryNavigator::stepCalls, 1);
  ExpectVectorNear(position, vecgeom::Vector3D<double>{2.25, 2.0, 3.0}, 0.0);
  ExpectVectorNear(direction, vecgeom::Vector3D<double>{1.0, 0.0, 0.0}, 0.0);
}

// When the safety shortcut is not enough, the propagator must defer to the
// navigator and stop at the boundary distance reported along the chord.
TEST(FieldPropagatorRungeKuttaNavigation, StopsAtNavigatorBoundaryAlongChord)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, StraightLinePropagatorDriver, double, BoundaryNavigator>;

  BoundaryNavigator::Reset();
  vecgeom::Vector3D<double> position{1.0, 2.0, 3.0};
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  vecgeom::NavigationState currentState;
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 8.0, 8.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 2.0);
  EXPECT_TRUE(propagated);
  EXPECT_TRUE(nextState.IsOnBoundary());
  EXPECT_EQ(BoundaryNavigator::stepCalls, 1);
  ExpectVectorNear(position, vecgeom::Vector3D<double>{3.0, 2.0, 3.0}, 0.0);
  ExpectVectorNear(direction, vecgeom::Vector3D<double>{1.0, 0.0, 0.0}, 0.0);
}

// Boundary-ambiguity tests for the Geant4-inspired direction-resolved
// classifier in the RK field propagator. These use a bent RK chord so the
// chord navigator and physical-direction probe can deliberately disagree.

// Chord artifact case: the chord clips a daughter boundary inside the ambiguity
// band, but the physical-direction probe finds no boundary before the ambiguity
// limit. The propagator must reduce the arc and stay in the current volume.
TEST(FieldPropagatorRungeKuttaNavigation, ReducesTinyChordArtifactWhenPhysicalDirectionStaysInCurrentVolume)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, BentChordPropagatorDriver, double, ScriptedNavigator>;

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{0.0, 0.0, 1.0};
  const vecgeom::NavigationState currentState = MakeState(kParentNavIndex, true);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  ScriptedNavigator::Reset({
      {2.5e-8, kDaughterNavIndex, true},
      {-1.0, kParentNavIndex, false},
      {-1.0, kParentNavIndex, false},
  });

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 4.0, 1.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 100, itersDone, 0, false);

  EXPECT_GT(moved, 2.5e-8);
  EXPECT_TRUE(propagated);
  EXPECT_EQ(nextState.GetNavIndex(), currentState.GetNavIndex());
  EXPECT_FALSE(nextState.IsOnBoundary());
  EXPECT_DOUBLE_EQ(ScriptedNavigator::requestedSteps[1], 10.0 * ScriptedNavigator::kBoundaryPush);
  EXPECT_GE(ScriptedNavigator::stepCalls, 2);
}

// The ambiguity classifier is only for tracks already located on a boundary.
// Away from a boundary, even a push-scale chord crossing is a normal navigator
// result and must not trigger the physical-direction probe.
TEST(FieldPropagatorRungeKuttaNavigation, DoesNotProbePhysicalDirectionWhenStartIsNotOnBoundary)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, BentChordPropagatorDriver, double, ScriptedNavigator>;

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{0.0, 0.0, 1.0};
  const vecgeom::NavigationState currentState  = MakeState(kParentNavIndex, false);
  const vecgeom::NavigationState daughterState = MakeState(kDaughterNavIndex, true);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  ScriptedNavigator::Reset({
      {5.0e-9, kDaughterNavIndex, true},
      {-1.0, kParentNavIndex, false},
  });

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 4.0, 1.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 5.0e-9);
  EXPECT_TRUE(propagated);
  EXPECT_TRUE(nextState.HasSamePathAsOther(daughterState));
  EXPECT_TRUE(nextState.IsOnBoundary());
  EXPECT_EQ(ScriptedNavigator::stepCalls, 1);
}

// The special classifier is restricted to the tiny ambiguity band. A chord hit
// just outside 10*kBoundaryPush is already a finite boundary crossing and should
// be accepted without any physical-direction probe.
TEST(FieldPropagatorRungeKuttaNavigation, DoesNotProbePhysicalDirectionOutsideTinyBoundaryBand)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, BentChordPropagatorDriver, double, ScriptedNavigator>;

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{0.0, 0.0, 1.0};
  const vecgeom::NavigationState currentState  = MakeState(kParentNavIndex, true);
  const vecgeom::NavigationState daughterState = MakeState(kDaughterNavIndex, true);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  ScriptedNavigator::Reset({
      {1.1e-7, kDaughterNavIndex, true},
      {-1.0, kParentNavIndex, false},
  });

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 4.0, 1.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 1.1e-7);
  EXPECT_TRUE(propagated);
  EXPECT_TRUE(nextState.HasSamePathAsOther(daughterState));
  EXPECT_TRUE(nextState.IsOnBoundary());
  EXPECT_GT(moved, 10.0 * ScriptedNavigator::kBoundaryPush);
  EXPECT_EQ(ScriptedNavigator::stepCalls, 1);
}

// Immediate physical return/backscatter: both the chord and physical direction
// find the previous volume at push scale. This remains a zero-length relocation
// so the navigator can re-establish the boundary state without fabricated motion.
TEST(FieldPropagatorRungeKuttaNavigation, AcceptsImmediatePhysicalReturnCrossing)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, BentChordPropagatorDriver, double, ScriptedNavigator>;

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{0.0, 0.0, 1.0};
  const vecgeom::NavigationState currentState = MakeState(kDaughterNavIndex, true);
  const vecgeom::NavigationState parentState  = MakeState(kParentNavIndex, true);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  ScriptedNavigator::Reset({
      {5.0e-9, kParentNavIndex, true},
      {5.0e-9, kParentNavIndex, true},
  });

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 4.0, 1.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 0.0);
  EXPECT_FALSE(propagated);
  EXPECT_TRUE(nextState.HasSamePathAsOther(parentState));
  EXPECT_TRUE(nextState.IsOnBoundary());
  ExpectVectorNear(position, vecgeom::Vector3D<double>{0.0, 0.0, 0.0}, 0.0);
  EXPECT_EQ(ScriptedNavigator::stepCalls, 2);
}

// Review regression: the chord clips the tolerance surface at zero, but the
// physical direction exits at a real finite distance inside the ambiguity band.
// The finite direction crossing must win over the chord's zero-distance result.
TEST(FieldPropagatorRungeKuttaNavigation, UsesFiniteDirectionCrossingWhenChordHitIsPushScale)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, BentChordPropagatorDriver, double, ScriptedNavigator>;

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{0.0, 0.0, 1.0};
  const vecgeom::NavigationState currentState = MakeState(kDaughterNavIndex, true);
  const vecgeom::NavigationState parentState  = MakeState(kParentNavIndex, true);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  ScriptedNavigator::Reset({
      {0.0, kParentNavIndex, true},
      {5.0e-8, kParentNavIndex, true},
  });

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 4.0, 1.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 5.0e-8);
  EXPECT_TRUE(propagated);
  EXPECT_TRUE(nextState.HasSamePathAsOther(parentState));
  EXPECT_TRUE(nextState.IsOnBoundary());
  EXPECT_GT(moved, ScriptedNavigator::kBoundaryPush);
  EXPECT_LT(moved, 10.0 * ScriptedNavigator::kBoundaryPush);
  EXPECT_DOUBLE_EQ(ScriptedNavigator::requestedSteps[1], 10.0 * ScriptedNavigator::kBoundaryPush);
  EXPECT_EQ(ScriptedNavigator::stepCalls, 2);
}

// Position consistency regression: when a finite crossing is accepted from the
// physical-direction probe, the stored position must lie on that physical ray,
// not on the different RK chord ray that produced the original push-scale hit.
TEST(FieldPropagatorRungeKuttaNavigation, PlacesDirectionResolvedFiniteCrossingOnPhysicalRay)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, BentChordPropagatorDriver, double, ScriptedNavigator>;

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{0.0, 0.0, 1.0};
  const vecgeom::NavigationState currentState = MakeState(kDaughterNavIndex, true);
  const vecgeom::NavigationState parentState  = MakeState(kParentNavIndex, true);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  ScriptedNavigator::Reset({
      {0.0, kParentNavIndex, true},
      {5.0e-8, kParentNavIndex, true},
  });

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 4.0, 1.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 5.0e-8);
  EXPECT_TRUE(propagated);
  EXPECT_TRUE(nextState.HasSamePathAsOther(parentState));
  EXPECT_TRUE(nextState.IsOnBoundary());
  ExpectVectorNear(position, vecgeom::Vector3D<double>{0.0, 0.0, 5.0e-8}, 0.0);
  EXPECT_EQ(ScriptedNavigator::stepCalls, 2);
}

// Short integrated chord: the physical-direction probe must not be allowed to
// accept a crossing farther away than the endpoint produced by the current RK
// advance, even if that crossing is still inside the ambiguity band.
TEST(FieldPropagatorRungeKuttaNavigation, DoesNotAcceptDirectionCrossingBeyondIntegratedChord)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, TinyBentChordPropagatorDriver, double, ScriptedNavigator>;

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{0.0, 0.0, 1.0};
  const vecgeom::NavigationState currentState = MakeState(kDaughterNavIndex, true);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  ScriptedNavigator::Reset({
      {0.0, kParentNavIndex, true},
      {5.0e-8, kParentNavIndex, true},
  });

  const double moved =
      Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 4.0, 1.0, position, direction, currentState,
                                           nextState, hitSurfaceIndex, propagated, safetyCache, 1, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 0.0);
  EXPECT_FALSE(propagated);
  EXPECT_EQ(nextState.GetNavIndex(), currentState.GetNavIndex());
  EXPECT_FALSE(nextState.IsOnBoundary());
  EXPECT_DOUBLE_EQ(ScriptedNavigator::requestedSteps[0], TinyBentChordPropagatorDriver::kAcceptedStep);
  EXPECT_DOUBLE_EQ(ScriptedNavigator::requestedSteps[1], TinyBentChordPropagatorDriver::kAcceptedStep);
  EXPECT_EQ(ScriptedNavigator::stepCalls, 2);
}

// Finite tiny crossing: when the boundary is inside the ambiguity band but
// farther than the push distance, preserve the navigator distance instead of
// rounding it to zero or to the full ambiguity limit.
TEST(FieldPropagatorRungeKuttaNavigation, KeepsFiniteTinyDirectionCrossingAtNavigatorDistance)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, BentChordPropagatorDriver, double, ScriptedNavigator>;

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{0.0, 0.0, 1.0};
  const vecgeom::NavigationState currentState = MakeState(kDaughterNavIndex, true);
  const vecgeom::NavigationState parentState  = MakeState(kParentNavIndex, true);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  ScriptedNavigator::Reset({
      {5.0e-8, kParentNavIndex, true},
      {5.0e-8, kParentNavIndex, true},
  });

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 4.0, 1.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_DOUBLE_EQ(moved, 5.0e-8);
  EXPECT_TRUE(propagated);
  EXPECT_TRUE(nextState.HasSamePathAsOther(parentState));
  EXPECT_TRUE(nextState.IsOnBoundary());
  EXPECT_LT(moved, 10.0 * ScriptedNavigator::kBoundaryPush);
  EXPECT_EQ(ScriptedNavigator::stepCalls, 2);
}

// End-to-end mixed-precision test with the real float RK driver: at large
// coordinates (r ~ 5 m) a very short physics step must move the track by
// exactly the length the propagator reports. Before this fix the float state
// carried the absolute position, and such steps are smaller than float can
// represent there: the propagator reported an advance but the position did
// not follow, producing stuck tracks.
TEST(FieldPropagatorRungeKuttaNavigation, MixedPrecisionMicronStepMovesTrackAtLargeRadius)
{
  using copcore::units::MeV;
  using copcore::units::tesla;

  using Field         = UniformMagneticField;
  using AnchoredField = OffsetMagneticField<Field>;
  using Equation      = MagneticFieldEquation<AnchoredField>;
  using Stepper       = DormandPrinceRK45<Equation, AnchoredField, kNvar, float>;
  using Driver        = RkIntegrationDriver<Stepper, float, int, Equation, AnchoredField>;
  using Propagator    = fieldPropagatorRungeKutta<Field, Driver, float, ThrowingNavigator>;

  const Field field(vecgeom::Vector3D<float>{0.0f, 0.0f, float(2.0 * tesla)});

  // 1e-7 mm matches the anti-stuck push (100 * vecgeom::kTolerance) in the
  // electron kernels, which was lost in float at this radius before the fix.
  for (const double physicsStep : {1.0e-3, 1.0e-4, 1.0e-7}) {
    SCOPED_TRACE(physicsStep);
    const vecgeom::Vector3D<double> start{3000.0, 3000.0, 3000.0};
    vecgeom::Vector3D<double> position = start;
    vecgeom::Vector3D<double> direction{0.3, -0.9, 0.3086067};
    direction = direction.Unit();

    const vecgeom::NavigationState currentState = MakeState(kParentNavIndex, false);
    vecgeom::NavigationState nextState;
    long hitSurfaceIndex = -1;
    bool propagated      = false;
    int itersDone        = 0;
    SafetyCache safetyCache;
    safetyCache.Refresh(position, 1.0e3); // far from any boundary: navigation must not be consulted

    const double moved = Propagator::ComputeStepAndNextVolume(
        field, 100.0 * MeV, copcore::units::kElectronMassC2, -1, physicsStep, /*safeLength=*/10.0, position, direction,
        currentState, nextState, hitSurfaceIndex, propagated, safetyCache, 10, itersDone, 0, false);

    EXPECT_TRUE(propagated);
    EXPECT_NEAR(moved, physicsStep, 1.0e-5 * physicsStep);
    EXPECT_NEAR((position - start).Length(), moved, 1.0e-2 * physicsStep);
  }
}

// Multi-chord mixed-precision propagation at detector-scale coordinates: the
// physics step is split into several chords, so the anchor moves with the
// track between Advance calls. The accumulated step must match the requested
// physics step and the actual displacement.
TEST(FieldPropagatorRungeKuttaNavigation, MixedPrecisionMultiChordAdvancesWithMovingAnchor)
{
  using copcore::units::MeV;
  using copcore::units::tesla;

  using Field         = UniformMagneticField;
  using AnchoredField = OffsetMagneticField<Field>;
  using Equation      = MagneticFieldEquation<AnchoredField>;
  using Stepper       = DormandPrinceRK45<Equation, AnchoredField, kNvar, float>;
  using Driver        = RkIntegrationDriver<Stepper, float, int, Equation, AnchoredField>;
  using Propagator    = fieldPropagatorRungeKutta<Field, Driver, float, FiniteSafetyNavigator>;

  const Field field(vecgeom::Vector3D<float>{0.0f, 0.0f, float(2.0 * tesla)});

  const vecgeom::Vector3D<double> start{3000.0, 3000.0, 3000.0};
  vecgeom::Vector3D<double> position = start;
  vecgeom::Vector3D<double> direction{0.3, -0.9, 0.3086067};
  direction = direction.Unit();

  const vecgeom::NavigationState currentState = MakeState(kParentNavIndex, false);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0); // no cached safety: chords are limited by safeLength

  // Safety refreshes return just above the chord length, so every chord after
  // the first is validated against the refreshed safety and no boundary is hit.
  FiniteSafetyNavigator::Reset(/*safetyValue=*/2.5, /*boundaryDist=*/-1.0, /*boundaryCall=*/-1);

  const double physicsStep = 10.0;
  const double moved       = Propagator::ComputeStepAndNextVolume(
      field, 100.0 * MeV, copcore::units::kElectronMassC2, -1, physicsStep, /*safeLength=*/2.0, position, direction,
      currentState, nextState, hitSurfaceIndex, propagated, safetyCache, 10, itersDone, 0, false);

  EXPECT_TRUE(propagated);
  EXPECT_NEAR(moved, physicsStep, 1.0e-4 * physicsStep);
  // Several 2 mm chords were integrated, each starting from the advanced position.
  EXPECT_GE(FiniteSafetyNavigator::safetyCalls, 3);
  // With R ~ 167 mm the 10 mm arc stays close to its chord.
  const double displacement = (position - start).Length();
  EXPECT_LE(displacement, moved + 1.0e-6);
  EXPECT_GE(displacement, 0.97 * moved);
}

// Mixed-precision boundary crossing on a later chord: the first chord advances
// fully (moving the anchor), the second chord intersects a boundary partway.
// The accepted step must be the first chord plus the navigator-reported
// distance, with the end state flagged as on-boundary.
TEST(FieldPropagatorRungeKuttaNavigation, MixedPrecisionChordBoundaryCrossingAtLargeRadius)
{
  using copcore::units::MeV;
  using copcore::units::tesla;

  using Field         = UniformMagneticField;
  using AnchoredField = OffsetMagneticField<Field>;
  using Equation      = MagneticFieldEquation<AnchoredField>;
  using Stepper       = DormandPrinceRK45<Equation, AnchoredField, kNvar, float>;
  using Driver        = RkIntegrationDriver<Stepper, float, int, Equation, AnchoredField>;
  using Propagator    = fieldPropagatorRungeKutta<Field, Driver, float, FiniteSafetyNavigator>;

  const Field field(vecgeom::Vector3D<float>{0.0f, 0.0f, float(2.0 * tesla)});

  const vecgeom::Vector3D<double> start{3000.0, 3000.0, 3000.0};
  vecgeom::Vector3D<double> position = start;
  vecgeom::Vector3D<double> direction{0.3, -0.9, 0.3086067};
  direction = direction.Unit();

  const vecgeom::NavigationState currentState = MakeState(kParentNavIndex, false);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  // Small refreshed safety forces navigation on every chord; the second
  // navigation call reports a boundary 0.1 mm into the chord. There, the 2 mm
  // chord lies less than deltaIntersection from the arc, so the crossing needs
  // no refinement.
  FiniteSafetyNavigator::Reset(/*safetyValue=*/0.5, /*boundaryDist=*/0.1, /*boundaryCall=*/2);

  const double physicsStep = 6.0;
  const double moved       = Propagator::ComputeStepAndNextVolume(
      field, 100.0 * MeV, copcore::units::kElectronMassC2, -1, physicsStep, /*safeLength=*/2.0, position, direction,
      currentState, nextState, hitSurfaceIndex, propagated, safetyCache, 10, itersDone, 0, false);

  EXPECT_TRUE(propagated);
  EXPECT_TRUE(nextState.IsOnBoundary());
  EXPECT_EQ(FiniteSafetyNavigator::stepCalls, 2);
  // First chord: full 2 mm arc; second chord: 0.1 mm to the boundary.
  EXPECT_NEAR(moved, 2.1, 2.0e-3);
  const double displacement = (position - start).Length();
  EXPECT_NEAR(displacement, moved, 2.0e-3);
}

// Persistent same-state ambiguity: if both chord and physical direction remain
// in the same volume at push scale, this is not a backscatter/return crossing.
// Leave it unresolved so the reduced-step retry path handles it.
TEST(FieldPropagatorRungeKuttaNavigation, KeepsPersistentTinySameStateSeparateFromBackscatter)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, BentChordPropagatorDriver, double, ScriptedNavigator>;

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{0.0, 0.0, 1.0};
  const vecgeom::NavigationState currentState = MakeState(kOtherNavIndex, true);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = true;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  ScriptedNavigator::Reset({
      {5.0e-9, kOtherNavIndex, true},
      {5.0e-9, kOtherNavIndex, true},
  });

  const double moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, 4.0, 1.0, position, direction,
                                                            currentState, nextState, hitSurfaceIndex, propagated,
                                                            safetyCache, 10, itersDone, 0, false);

  EXPECT_EQ(moved, 0.0);
  EXPECT_FALSE(propagated);
  EXPECT_TRUE(nextState.HasSamePathAsOther(currentState));
  EXPECT_GE(ScriptedNavigator::stepCalls, 2);
}

// Chord check outside the geometry safety. There, navigation uses the chord in
// place of the arc, so, as in Geant4's G4ChordFinder, a chord is accepted while
// its distance from the arc is below deltaChord, and an arc whose chord is
// farther away is retried shorter. On the circle of the recording driver the
// propagator's distance estimate is exact.

namespace {

constexpr double kCircleRadius = RecordingCircularPropagatorDriver::kRadius;

double CircleChordDistanceForArc(double arc)
{
  return kCircleRadius * (1.0 - std::cos(0.5 * arc / kCircleRadius));
}

double CircleChordDistanceForChord(double chord)
{
  return kCircleRadius - std::sqrt(kCircleRadius * kCircleRadius - 0.25 * chord * chord);
}

double CircleArcForChordDistance(double distance)
{
  return 2.0 * kCircleRadius * std::acos(1.0 - distance / kCircleRadius);
}

struct PropagationResult {
  double moved    = 0.0;
  bool propagated = false;
};

// Propagates along the circle of RecordingCircularPropagatorDriver, starting at the origin with the given cached
// safety. Later safety refreshes return refreshedSafety.
PropagationResult PropagateOnCircle(double physicsStep, double safeLength, double cachedSafety, double refreshedSafety)
{
  using Propagator =
      fieldPropagatorRungeKutta<DummyField, RecordingCircularPropagatorDriver, double, FiniteSafetyNavigator>;
  RecordingCircularPropagatorDriver::Reset();
  FiniteSafetyNavigator::Reset(refreshedSafety, /*boundaryDist=*/-1.0, /*boundaryCall=*/-1);

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  const auto currentState = MakeState(kParentNavIndex, false);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, cachedSafety);

  PropagationResult result;
  result.moved = Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, physicsStep, safeLength, position,
                                                      direction, currentState, nextState, hitSurfaceIndex,
                                                      result.propagated, safetyCache, 10, itersDone, 0, false);
  return result;
}

} // namespace

// Geant4 accepts any chord closer than deltaChord to its arc. So does the
// propagator: such an arc is navigated as requested, without a retry.
TEST(FieldPropagatorRungeKuttaNavigation, AcceptsChordCloserThanDeltaChordToItsArc)
{
  const double arc = CircleArcForChordDistance(0.9 * fieldConstants::deltaChord);

  const auto result = PropagateOnCircle(/*physicsStep=*/arc, /*safeLength=*/arc, /*cachedSafety=*/0.0,
                                        /*refreshedSafety=*/0.0);

  EXPECT_TRUE(result.propagated);
  EXPECT_DOUBLE_EQ(result.moved, arc);
  EXPECT_EQ(RecordingCircularPropagatorDriver::calls, 1u);
  EXPECT_EQ(FiniteSafetyNavigator::stepCalls, 1);
}

// An arc whose chord is farther than deltaChord from it is retried from the
// same start, as in Geant4: the distance grows with the square of the arc
// length, so the retry requests kChordRetryFraction * arc * sqrt(deltaChord /
// distance). Only accepted chords reach navigation.
TEST(FieldPropagatorRungeKuttaNavigation, RetriesArcWhoseChordIsFartherThanDeltaChord)
{
  using Driver     = RecordingCircularPropagatorDriver;
  using Propagator = fieldPropagatorRungeKutta<DummyField, Driver, double, FiniteSafetyNavigator>;
  const double arc = CircleArcForChordDistance(1.5 * fieldConstants::deltaChord);

  const auto result = PropagateOnCircle(/*physicsStep=*/arc, /*safeLength=*/arc, /*cachedSafety=*/0.0,
                                        /*refreshedSafety=*/0.0);

  EXPECT_TRUE(result.propagated);
  EXPECT_NEAR(result.moved, arc, 1.0e-12);
  EXPECT_DOUBLE_EQ(Driver::requestedSteps[0], arc);
  EXPECT_NEAR(Driver::requestedSteps[1], Propagator::kChordRetryFraction * arc * std::sqrt(1.0 / 1.5), 1.0e-12);
  EXPECT_LT(CircleChordDistanceForArc(Driver::requestedSteps[1]), fieldConstants::deltaChord);
  // The retried arc and the rest of the step.
  EXPECT_EQ(Driver::calls, 3u);
  ASSERT_EQ(FiniteSafetyNavigator::stepCalls, 2);
  for (int i = 0; i < FiniteSafetyNavigator::stepCalls; ++i) {
    EXPECT_LT(CircleChordDistanceForChord(FiniteSafetyNavigator::chordLengths[i]), fieldConstants::deltaChord);
  }
}

// Like the first arc, a retry may extend to the geometry safety: an arc within
// the safety needs neither a chord check nor navigation. Here the 6 mm arc
// leaves the 4.5 mm safety and its chord is 0.56 mm from it. The shortened arc
// would be 3.94 mm, so the retry takes the 4.5 mm safety instead.
TEST(FieldPropagatorRungeKuttaNavigation, RetriesTooCurvedArcAtLongerGeometrySafety)
{
  using Driver = RecordingCircularPropagatorDriver;

  const auto result = PropagateOnCircle(/*physicsStep=*/6.0, /*safeLength=*/6.0, /*cachedSafety=*/4.5,
                                        /*refreshedSafety=*/0.0);

  EXPECT_TRUE(result.propagated);
  EXPECT_NEAR(result.moved, 6.0, 1.0e-12);
  EXPECT_EQ(Driver::calls, 3u);
  EXPECT_DOUBLE_EQ(Driver::requestedSteps[0], 6.0);
  EXPECT_DOUBLE_EQ(Driver::requestedSteps[1], 4.5);
  EXPECT_DOUBLE_EQ(Driver::requestedSteps[2], 1.5);
  // Only the last arc, beyond the safety, is navigated.
  EXPECT_EQ(FiniteSafetyNavigator::stepCalls, 1);
}

// The safety is computed at most once per chord start. A retry starts from the
// same point, so it reuses the refreshed value. Here the first arc stays within
// the initial 10 mm safety; the second needs a refresh and one retry, and the
// third starts from a new point.
TEST(FieldPropagatorRungeKuttaNavigation, RefreshesSafetyOncePerChordStart)
{
  using Driver = RecordingCircularPropagatorDriver;

  const auto result = PropagateOnCircle(/*physicsStep=*/16.0, /*safeLength=*/6.0, /*cachedSafety=*/10.0,
                                        /*refreshedSafety=*/0.0);

  EXPECT_TRUE(result.propagated);
  EXPECT_NEAR(result.moved, 16.0, 1.0e-12);
  ASSERT_EQ(Driver::calls, 4u);
  EXPECT_DOUBLE_EQ(Driver::requestedSteps[0], 10.0);
  EXPECT_DOUBLE_EQ(Driver::requestedSteps[1], 6.0);
  EXPECT_LT(Driver::requestedSteps[2], 6.0);
  EXPECT_EQ(FiniteSafetyNavigator::stepCalls, 2);
  EXPECT_EQ(FiniteSafetyNavigator::safetyCalls, 2);
}

namespace {

// Bz rises linearly along x from 0.1 T at x = 0 to 4 T at x = 100 mm and stays constant beyond.
struct RisingField {
  static constexpr double kStartField = 0.1 * copcore::units::tesla;
  static constexpr double kEndField   = 4.0 * copcore::units::tesla;
  static constexpr double kRampLength = 100.0 * copcore::units::millimeter;

  template <typename Real_t>
  vecgeom::Vector3D<Real_t> Evaluate(Real_t x, Real_t, Real_t) const
  {
    const double fraction = std::clamp(double(x) / kRampLength, 0.0, 1.0);
    return {Real_t(0.), Real_t(0.), Real_t(kStartField + fraction * (kEndField - kStartField))};
  }

  template <typename Real_t>
  vecgeom::Vector3D<Real_t> Evaluate(const vecgeom::Vector3D<Real_t> &position) const
  {
    return Evaluate(position[0], position[1], position[2]);
  }
};

} // namespace

// Where the chord check matters: the arc length is predicted from the field at
// the start of the step, and a track that moves into a stronger field bends
// more than predicted. Here a 50 MeV electron starts in 0.1 T and enters a
// field that rises to 4 T within 100 mm. Without the check, the predicted arc
// is navigated as one chord that lies about 0.4 mm from the trajectory. With
// it, every navigated chord lies within deltaChord of a densely integrated
// reference trajectory.
TEST(FieldPropagatorRungeKuttaNavigation, ChordsOfTrackEnteringStrongerFieldStayWithinDeltaChord)
{
  using copcore::units::MeV;
  using Equation   = MagneticFieldEquation<RisingField>;
  using Stepper    = DormandPrinceRK45<Equation, RisingField, kNvar, double>;
  using Driver     = RkIntegrationDriver<Stepper, double, int, Equation, RisingField>;
  using Propagator = fieldPropagatorRungeKutta<RisingField, Driver, double, FiniteSafetyNavigator>;

  const RisingField field;
  constexpr int charge                           = -1;
  const double mass                              = copcore::units::kElectronMassC2;
  const double momentumMag                       = 50.0 * MeV;
  const double kinE                              = std::sqrt(momentumMag * momentumMag + mass * mass) - mass;
  const vecgeom::Vector3D<double> start          = {0.0, 0.0, 0.0};
  const vecgeom::Vector3D<double> startDirection = {1.0, 0.0, 0.0};

  vecgeom::Vector3D<double> momentum   = momentumMag * startDirection;
  vecgeom::Vector3D<double> startField = field.Evaluate(start);
  const double safeLength              = Propagator::ComputeSafeLength(momentum, startField, charge);
  const double physicsStep             = safeLength;

  FiniteSafetyNavigator::Reset(/*safetyValue=*/0.0, /*boundaryDist=*/-1.0, /*boundaryCall=*/-1);
  vecgeom::Vector3D<double> position  = start;
  vecgeom::Vector3D<double> direction = startDirection;
  const auto currentState             = MakeState(kParentNavIndex, false);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  const double moved = Propagator::ComputeStepAndNextVolume(
      field, kinE, mass, charge, physicsStep, safeLength, position, direction, currentState, nextState, hitSurfaceIndex,
      propagated, safetyCache, 10, itersDone, 0, false);

  EXPECT_TRUE(propagated);
  EXPECT_NEAR(moved, physicsStep, 1.0e-9 * physicsStep);
  ASSERT_LE(FiniteSafetyNavigator::stepCalls, FiniteSafetyNavigator::kMaxRecordedChords);

  // Reference trajectory from the same start, integrated in 0.05 mm pieces.
  std::vector<vecgeom::Vector3D<double>> reference{start};
  vecgeom::Vector3D<double> referencePosition = start;
  vecgeom::Vector3D<double> referenceMomentum = momentumMag * startDirection;
  double dydx[kNvar];
  double lastGoodStep = 0.0;
  int driverIters     = 0;
  for (double done = 0.0; done < physicsStep;) {
    const double piece = std::min(0.05, physicsStep - done);
    done += Driver::Advance(referencePosition, referenceMomentum, charge, piece, field, dydx, lastGoodStep, 30,
                            driverIters);
    reference.push_back(referencePosition);
  }

  // Each chord covers the reference points that project onto it.
  for (int i = 0; i < FiniteSafetyNavigator::stepCalls; ++i) {
    SCOPED_TRACE(i);
    const auto &chordStart     = FiniteSafetyNavigator::chordStarts[i];
    const auto &chordDirection = FiniteSafetyNavigator::chordDirections[i];
    const double chordLength   = FiniteSafetyNavigator::chordLengths[i];
    double distance            = 0.0;
    for (const auto &point : reference) {
      const vecgeom::Vector3D<double> offset = point - chordStart;
      const double along                     = offset.Dot(chordDirection);
      if (along < 0.0 || along > chordLength) continue;
      distance = std::max(distance, (offset - along * chordDirection).Length());
    }
    EXPECT_LT(distance, fieldConstants::deltaChord);
  }
}

namespace {

// Navigator for the half-space n.r < d, whose boundary is the plane n.r = d. Steps are exact, so the crossing found
// on a chord is the chord's true intersection with the plane. The safety is zero, so every chord is navigated.
struct PlaneNavigator {
  static constexpr double kBoundaryPush = 1.0e-8;
  inline static vecgeom::Vector3D<double> normal{1.0, 0.0, 0.0};
  inline static double offset = 0.0;
  inline static int stepCalls = 0;

  static void Reset(const vecgeom::Vector3D<double> &n, double d)
  {
    normal    = n;
    offset    = d;
    stepCalls = 0;
  }

  static double ComputeSafety(const vecgeom::Vector3D<double> &, const vecgeom::NavigationState &, double)
  {
    return 0.0;
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &position,
                                         const vecgeom::Vector3D<double> &direction, double step,
                                         const vecgeom::NavigationState &currentState,
                                         vecgeom::NavigationState &nextState)
  {
    ++stepCalls;
    const double approach = normal.Dot(direction);
    if (approach > 0.0) {
      const double distance = (offset - normal.Dot(position)) / approach;
      if (distance >= 0.0 && distance < step) {
        nextState = MakeState(kDaughterNavIndex, true);
        return distance;
      }
    }
    currentState.CopyTo(&nextState);
    nextState.SetBoundaryState(false);
    return step + 1.0;
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &position,
                                         const vecgeom::Vector3D<double> &direction, double step,
                                         const vecgeom::NavigationState &currentState,
                                         vecgeom::NavigationState &nextState, long &hitSurfaceIndex)
  {
    hitSurfaceIndex = 0;
    return ComputeStepAndNextVolume(position, direction, step, currentState, nextState);
  }
};

struct PlaneCrossingResult {
  double moved    = 0.0;
  bool propagated = false;
  vecgeom::Vector3D<double> position;
  vecgeom::Vector3D<double> direction;
  vecgeom::NavigationState nextState;
};

// Propagates from the origin along +x on the circle of RecordingCircularPropagatorDriver (radius 8 mm, bending
// towards +y) towards the plane n.r = d.
PlaneCrossingResult CrossPlaneOnCircle(const vecgeom::Vector3D<double> &n, double d, double safeLength,
                                       int maxIterations, double physicsStep = 10.0)
{
  using Propagator = fieldPropagatorRungeKutta<DummyField, RecordingCircularPropagatorDriver, double, PlaneNavigator>;
  RecordingCircularPropagatorDriver::Reset();
  PlaneNavigator::Reset(n, d);

  PlaneCrossingResult result;
  result.position         = {0.0, 0.0, 0.0};
  result.direction        = {1.0, 0.0, 0.0};
  const auto currentState = MakeState(kParentNavIndex, false);
  long hitSurfaceIndex    = -1;
  int itersDone           = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(result.position, 0.0);
  result.moved = Propagator::ComputeStepAndNextVolume(
      DummyField{}, 10.0, 1.0, 1, physicsStep, safeLength, result.position, result.direction, currentState,
      result.nextState, hitSurfaceIndex, result.propagated, safetyCache, maxIterations, itersDone, 0, false);
  return result;
}

PlaneCrossingResult CrossPlaneOnCircle(double planeX, double safeLength, int maxIterations)
{
  return CrossPlaneOnCircle({1.0, 0.0, 0.0}, planeX, safeLength, maxIterations);
}

// Point of the circle of RecordingCircularPropagatorDriver, from the origin along +x, at the turning angle phi.
vecgeom::Vector3D<double> CirclePoint(double phi)
{
  return {kCircleRadius * std::sin(phi), kCircleRadius * (1.0 - std::cos(phi)), 0.0};
}

// First point where that circle crosses the plane n.r = d, found by bisection on the turning angle.
vecgeom::Vector3D<double> CircleCrossing(const vecgeom::Vector3D<double> &n, double d)
{
  double lo = 0.0, hi = 0.0;
  while (n.Dot(CirclePoint(hi)) < d)
    hi += 1.0e-3;
  lo = hi - 1.0e-3;
  for (int i = 0; i < 100; ++i) {
    const double mid                        = 0.5 * (lo + hi);
    (n.Dot(CirclePoint(mid)) < d ? lo : hi) = mid;
  }
  return CirclePoint(0.5 * (lo + hi));
}

vecgeom::Vector3D<double> CircleCrossing(double planeX)
{
  return CircleCrossing({1.0, 0.0, 0.0}, planeX);
}

// Distance of a point from that circle.
double DistanceFromCircle(const vecgeom::Vector3D<double> &point)
{
  return std::abs((point - vecgeom::Vector3D<double>{0.0, kCircleRadius, 0.0}).Length() - kCircleRadius);
}

// Plane normal to the track at the arc length s along that circle, with 2 mm chords and a 20 mm physics step. The
// chords end at multiples of 2 mm, so the plane is met by the chord over [floor(s / 2) * 2, +2 mm]. A plane at
// 1.5 mm into a chord is met 3/4 along it, about 0.05 mm from the track; the refined arc then ends about 2 um short
// of the plane (as the arc up to the chord's crossing fraction is shorter than the arc up to the plane), so the
// crossing needs a further chord after the refined one.
PlaneCrossingResult CrossNormalPlaneOnCircle(double s)
{
  const double phi = s / kCircleRadius;
  const vecgeom::Vector3D<double> n{std::cos(phi), std::sin(phi), 0.0};
  return CrossPlaneOnCircle(n, n.Dot(CirclePoint(phi)), /*safeLength=*/2.0, /*maxIterations=*/10,
                            /*physicsStep=*/20.0);
}

} // namespace

// The plane at 17.5 mm is met on the ninth of ten passes. A refinement there would leave only the tenth pass for the
// refined arc, which ends short of the plane, and the call would end without a boundary, short of it. A refinement
// therefore needs two passes left: the crossing is accepted unrefined, on the plane, as without refinement.
TEST(FieldPropagatorRungeKuttaNavigation, AcceptsUnrefinedCrossingWhenRefinedArcCouldNotReachTheSurface)
{
  constexpr double s = 17.5;
  const vecgeom::Vector3D<double> n{std::cos(s / kCircleRadius), std::sin(s / kCircleRadius), 0.0};

  const auto result = CrossNormalPlaneOnCircle(s);

  EXPECT_TRUE(result.propagated);
  EXPECT_TRUE(result.nextState.IsOnBoundary());
  EXPECT_EQ(result.nextState.GetNavIndex(), kDaughterNavIndex);
  EXPECT_EQ(RecordingCircularPropagatorDriver::calls, 9u);
  EXPECT_EQ(PlaneNavigator::stepCalls, 9);
  EXPECT_NEAR(n.Dot(result.position), n.Dot(CirclePoint(s / kCircleRadius)), 1.0e-10);
  const double offTrack = DistanceFromCircle(result.position);
  EXPECT_GT(offTrack, fieldConstants::deltaIntersection);
  EXPECT_LT(offTrack, fieldConstants::deltaChord);
}

// The same plane one chord earlier, at 15.5 mm, is met on the eighth pass. The refinement runs on the ninth pass, its
// arc ends short of the plane, and the tenth pass meets the plane near the start of its chord, within
// deltaIntersection of the track.
TEST(FieldPropagatorRungeKuttaNavigation, RefinedArcEndingShortOfTheSurfaceReachesItWithTheReservedPass)
{
  constexpr double s = 15.5;
  const vecgeom::Vector3D<double> n{std::cos(s / kCircleRadius), std::sin(s / kCircleRadius), 0.0};

  const auto result = CrossNormalPlaneOnCircle(s);

  EXPECT_TRUE(result.propagated);
  EXPECT_TRUE(result.nextState.IsOnBoundary());
  EXPECT_EQ(RecordingCircularPropagatorDriver::calls, 10u);
  EXPECT_EQ(PlaneNavigator::stepCalls, 10);
  EXPECT_NEAR(n.Dot(result.position), n.Dot(CirclePoint(s / kCircleRadius)), 1.0e-10);
  EXPECT_LE(DistanceFromCircle(result.position), fieldConstants::deltaIntersection);
}

// The first chord, 2 mm long, meets the plane at three quarters of its length, where it lies about 0.05 mm from the
// arc. The crossing is refined: the arc is integrated again up to the crossing, until the crossing found on the chord
// is within deltaIntersection of the track. The track ends on the plane at the circle's crossing point, with the
// circle's direction there.
TEST(FieldPropagatorRungeKuttaNavigation, RefinesBoundaryCrossingToDeltaIntersection)
{
  constexpr double planeX     = 1.5;
  constexpr double safeLength = 2.0;

  const auto result = CrossPlaneOnCircle(planeX, safeLength, /*maxIterations=*/10);

  const auto crossing     = CircleCrossing(planeX);
  const double crossAngle = std::asin(planeX / kCircleRadius);
  EXPECT_TRUE(result.propagated);
  EXPECT_TRUE(result.nextState.IsOnBoundary());
  EXPECT_EQ(result.nextState.GetNavIndex(), kDaughterNavIndex);
  EXPECT_DOUBLE_EQ(RecordingCircularPropagatorDriver::requestedSteps[0], safeLength);
  EXPECT_GT(RecordingCircularPropagatorDriver::calls, 1u);
  EXPECT_LE(RecordingCircularPropagatorDriver::calls, 4u);
  EXPECT_NEAR(result.position[0], planeX, 1.0e-12);
  ExpectVectorNear(result.position, crossing, fieldConstants::deltaIntersection);
  ExpectVectorNear(result.direction, vecgeom::Vector3D<double>{std::cos(crossAngle), std::sin(crossAngle), 0.0},
                   1.0e-3);
  // The refined arcs start at the origin, and the step is counted along the final chord up to the crossing: the
  // chord of the circle up to its crossing point, which is shorter than the arc by about arc^3 / (24 R^2).
  EXPECT_NEAR(result.moved, 2.0 * kCircleRadius * std::sin(0.5 * crossAngle), fieldConstants::deltaIntersection);
}

// Near the start of a chord, the chord stays within deltaIntersection of the arc: the crossing is accepted as found,
// with one integration and one navigation.
TEST(FieldPropagatorRungeKuttaNavigation, AcceptsCrossingWithinDeltaIntersectionWithoutRefinement)
{
  constexpr double planeX = 1.0e-3;

  const auto result = CrossPlaneOnCircle(planeX, /*safeLength=*/2.0, /*maxIterations=*/10);

  EXPECT_TRUE(result.propagated);
  EXPECT_TRUE(result.nextState.IsOnBoundary());
  EXPECT_EQ(RecordingCircularPropagatorDriver::calls, 1u);
  EXPECT_EQ(PlaneNavigator::stepCalls, 1);
  ExpectVectorNear(result.position, CircleCrossing(planeX), fieldConstants::deltaIntersection);
}

// A refinement needs two more passes: the refined arc, and the chord after it in case the refined arc ends short of
// the surface. With a limit of two passes, the crossing found on the first chord is accepted as before: on the plane,
// but about 0.05 mm from the track.
TEST(FieldPropagatorRungeKuttaNavigation, AcceptsUnrefinedCrossingAtIterationLimit)
{
  constexpr double planeX = 1.5;

  const auto result = CrossPlaneOnCircle(planeX, /*safeLength=*/2.0, /*maxIterations=*/2);

  EXPECT_TRUE(result.propagated);
  EXPECT_TRUE(result.nextState.IsOnBoundary());
  EXPECT_EQ(RecordingCircularPropagatorDriver::calls, 1u);
  EXPECT_NEAR(result.position[0], planeX, 1.0e-12);
  const double offTrack = (result.position - CircleCrossing(planeX)).Length();
  EXPECT_GT(offTrack, 0.02);
  EXPECT_LT(offTrack, fieldConstants::deltaChord);
}

namespace {

// Crosses planes along the first two chords (2 mm each), whose normals make the angle (radians) with the start
// direction, and returns the largest distance of a crossing from the track. Every crossing ends on its plane in the
// same call, also where a refined arc ends short of the plane and the step continues with a new chord.
double CrossPlanesAlongFirstChords(double angle, int &crossings)
{
  double worst = 0.0;
  const vecgeom::Vector3D<double> n{std::cos(angle), std::sin(angle), 0.0};
  const vecgeom::Vector3D<double> center{0.0, kCircleRadius, 0.0};
  for (double along = 0.05; along < 3.9; along += 0.05) {
    // the plane through the point of the track at the arc length 'along' from the start
    const double phi = along / kCircleRadius;
    const double d   = n.Dot(CirclePoint(phi));
    // A plane that the track meets nearly tangentially can be crossed and recrossed between the ends of one chord,
    // which then misses it, as in Geant4: deltaChord bounds that, not the refinement.
    if (n.Dot(vecgeom::Vector3D<double>{std::cos(phi), std::sin(phi), 0.0}) < 0.25) continue;
    SCOPED_TRACE(::testing::Message() << "angle " << angle << " along " << along);
    const auto result = CrossPlaneOnCircle(n, d, /*safeLength=*/2.0, /*maxIterations=*/10);
    ++crossings;
    EXPECT_TRUE(result.propagated);
    EXPECT_TRUE(result.nextState.IsOnBoundary());
    EXPECT_NEAR(n.Dot(result.position), d, 1.0e-12);
    worst = std::max(worst, std::abs((result.position - center).Length() - kCircleRadius));
  }
  return worst;
}

} // namespace

// Planes whose normal is at up to 34 degrees to the start direction: every crossing ends on the plane, within
// deltaIntersection of the track, in one call.
TEST(FieldPropagatorRungeKuttaNavigation, RefinedCrossingsAreWithinDeltaIntersectionAlongTheChord)
{
  int crossings = 0;
  for (double angle : {-0.6, -0.3, 0.0, 0.3, 0.6}) {
    SCOPED_TRACE(angle);
    EXPECT_LE(CrossPlanesAlongFirstChords(angle, crossings), fieldConstants::deltaIntersection);
  }
  EXPECT_GT(crossings, 300);
}

// Near grazing incidence, each refinement moves the crossing found on the chord only a little, and
// kMaxCrossingRefinements or the iteration limit can stop the refinement first. The crossing is then accepted on the
// plane, but closer to the track than the unrefined chord crossing.
TEST(FieldPropagatorRungeKuttaNavigation, GrazingCrossingsStayOnThePlaneNearTheTrack)
{
  int crossings = 0;
  for (double angle : {-1.2, -0.9, 0.9, 1.2}) {
    SCOPED_TRACE(angle);
    EXPECT_LE(CrossPlanesAlongFirstChords(angle, crossings), 0.25 * fieldConstants::deltaChord);
  }
}

// Reports a boundary at the middle of every chord.
struct MidChordNavigator {
  static constexpr double kBoundaryPush = 0.0;
  inline static int stepCalls           = 0;

  static double ComputeSafety(const vecgeom::Vector3D<double> &, const vecgeom::NavigationState &, double)
  {
    return 0.0;
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &, const vecgeom::Vector3D<double> &,
                                         double step, const vecgeom::NavigationState &,
                                         vecgeom::NavigationState &nextState)
  {
    ++stepCalls;
    nextState = MakeState(kDaughterNavIndex, true);
    return 0.5 * step;
  }

  static double ComputeStepAndNextVolume(const vecgeom::Vector3D<double> &position,
                                         const vecgeom::Vector3D<double> &direction, double step,
                                         const vecgeom::NavigationState &currentState,
                                         vecgeom::NavigationState &nextState, long &hitSurfaceIndex)
  {
    hitSurfaceIndex = 0;
    return ComputeStepAndNextVolume(position, direction, step, currentState, nextState);
  }
};

// A boundary reported at the middle of every chord, whatever its length, stays far from the arc: halving the arc
// only quarters the distance. The propagator refines at most kMaxCrossingRefinements times and then accepts the
// crossing on the chord.
TEST(FieldPropagatorRungeKuttaNavigation, LimitsCrossingRefinementsPerCall)
{
  using Driver     = RecordingCircularPropagatorDriver;
  using Propagator = fieldPropagatorRungeKutta<DummyField, Driver, double, MidChordNavigator>;
  Driver::Reset();
  MidChordNavigator::stepCalls = 0;

  vecgeom::Vector3D<double> position{0.0, 0.0, 0.0};
  vecgeom::Vector3D<double> direction{1.0, 0.0, 0.0};
  const auto currentState = MakeState(kParentNavIndex, false);
  vecgeom::NavigationState nextState;
  long hitSurfaceIndex = -1;
  bool propagated      = false;
  int itersDone        = 0;
  SafetyCache safetyCache;
  safetyCache.Refresh(position, 0.0);

  Propagator::ComputeStepAndNextVolume(DummyField{}, 10.0, 1.0, 1, /*physicsStep=*/10.0, /*safeLength=*/2.0, position,
                                       direction, currentState, nextState, hitSurfaceIndex, propagated, safetyCache, 10,
                                       itersDone, 0, false);

  EXPECT_TRUE(propagated);
  EXPECT_TRUE(nextState.IsOnBoundary());
  EXPECT_EQ(Driver::calls, 1u + Propagator::kMaxCrossingRefinements);
  EXPECT_EQ(MidChordNavigator::stepCalls, 1 + Propagator::kMaxCrossingRefinements);
  // Each refinement integrates up to the crossing: half of the previous arc.
  for (int i = 1; i <= Propagator::kMaxCrossingRefinements; ++i) {
    EXPECT_NEAR(Driver::requestedSteps[i], 0.5 * Driver::requestedSteps[i - 1], 1.0e-12);
  }
}
