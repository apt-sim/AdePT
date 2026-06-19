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
#include <AdePT/transport/magneticfield/fieldPropagatorRungeKutta.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavigationState.h>

#include <gtest/gtest.h>

#include <array>
#include <initializer_list>

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
