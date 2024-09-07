// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author: J. Apostolakis  Nov/Dec 2020

#pragma once

#include <VecGeom/base/Vector3D.h>
#include <AdePT/copcore/PhysicalConstants.h>

#include <AdePT/base/BlockData.h>
#include <AdePT/navigation/AdePTNavigator.h>

#include <AdePT/magneticfield/ConstBzFieldStepper.h>

// Data structures for statistics of propagation chords

class fieldPropagatorConstBz {
  using Precision = vecgeom::Precision;
  using Vector3D  = vecgeom::Vector3D<vecgeom::Precision>;

public:
  __host__ __device__ fieldPropagatorConstBz(Precision Bz) { BzValue = Bz; }
  __host__ __device__ ~fieldPropagatorConstBz() {}

  __host__ __device__ void stepInField(double kinE, double mass, int charge, Precision step, Vector3D &position,
                                       Vector3D &direction);

  __host__ __device__ Precision ComputeSafeLength(Precision momentumMag, int charge, const Vector3D &direction);

  template <class Navigator = AdePTNavigator>
  __host__ __device__ Precision ComputeStepAndNextVolume(double kinE, double mass, int charge, Precision physicsStep,
                                                         Vector3D &position, Vector3D &direction,
                                                         vecgeom::NavigationState const &current_state,
                                                         vecgeom::NavigationState &new_state, long &hitsurf_index,
                                                         bool &propagated, const Precision safety = 0.0,
                                                         const int max_iteration = 100);

private:
  Precision BzValue;
};

// -----------------------------------------------------------------------------

__host__ __device__ void fieldPropagatorConstBz::stepInField(double kinE, double mass, int charge, Precision step,
                                                             vecgeom::Vector3D<vecgeom::Precision> &position,
                                                             vecgeom::Vector3D<vecgeom::Precision> &direction)
{
  if (charge != 0) {
    Precision momentumMag = sqrt(kinE * (kinE + 2.0 * mass));

    // For now all particles ( e-, e+, gamma ) can be propagated using this
    //   for gammas  charge = 0 works, and ensures that it goes straight.
    ConstBzFieldStepper helixBz(BzValue);

    Vector3D endPosition  = position;
    Vector3D endDirection = direction;
    helixBz.DoStep<Vector3D, Precision, int>(position, direction, charge, momentumMag, step, endPosition, endDirection);
    position  = endPosition;
    direction = endDirection;
  } else {
    // Also move gammas - for now ..
    position = position + step * direction;
  }
}

__host__ __device__ Precision fieldPropagatorConstBz::ComputeSafeLength(Precision momentumMag, int charge,
                                                                        const Vector3D &direction)
{
  // Maximum allowed error made by approximating step along helix with step along straight line
  constexpr Precision gEpsilonDeflect = 1.E-2 * copcore::units::cm;

  // Direction projection in plane perpendicular to field vector
  Precision dirxy = sqrt((1 - direction[2]) * (1 + direction[2]));

  Precision bend = std::fabs(fieldConstants::kB2C * charge * BzValue) / momentumMag;

  // R = helix radius, curv = 1./R = curvature in plane perpendicular to the field
  // Precision curv = bend / (dirxy + 1.e-30);

  // Distance along the track direction to reach the maximum allowed error
  return sqrt(2 * gEpsilonDeflect / (bend * dirxy + 1.e-30));
}

// Determine the step along curved trajectory for charged particles in a field.
//  ( Same name as as navigator method. )
template <class Navigator>
__host__ __device__ Precision fieldPropagatorConstBz::ComputeStepAndNextVolume(
    double kinE, double mass, int charge, Precision physicsStep, vecgeom::Vector3D<vecgeom::Precision> &position,
    vecgeom::Vector3D<vecgeom::Precision> &direction, vecgeom::NavigationState const &current_state,
    vecgeom::NavigationState &next_state, long &hitsurf_index, bool &propagated, const vecgeom::Precision safetyIn,
    const int max_iterations)
{
  using Precision = vecgeom::Precision;
#ifdef VECGEOM_FLOAT_PRECISION
  const Precision kPush = 10 * vecgeom::kTolerance;
#else
  const Precision kPush = 0;
#endif

  Precision momentumMag = sqrt(kinE * (kinE + 2 * mass));

  // Distance along the track direction to reach the maximum allowed error
  const Precision safeLength = ComputeSafeLength(momentumMag, charge, direction);

  ConstBzFieldStepper helixBz(BzValue);

  Precision stepDone           = 0;
  Precision remains            = physicsStep;
  const Precision epsilon_step = 1.0e-7 * physicsStep; // Ignore remainder if < e_s * PhysicsStep
  int chordIters               = 0;

  bool continueIteration = false;

  Precision safety      = safetyIn;
  Vector3D safetyOrigin = position;
  // Prepare next_state in case we skip navigation inside the safety sphere.
  current_state.CopyTo(&next_state);
  next_state.SetBoundaryState(false);

  Precision maxNextSafeMove = safeLength;

  bool lastWasZero = false;
  //  Locate the intersection of the curved trajectory and the boundaries of the current
  //    volume (including daughters).
  do {
    Vector3D endPosition  = position;
    Vector3D endDirection = direction;
    Precision safeMove    = min(remains, maxNextSafeMove);

    helixBz.DoStep<Vector3D, Precision, int>(position, direction, charge, momentumMag, safeMove, endPosition,
                                             endDirection);

    Vector3D chordVec  = endPosition - position;
    Precision chordLen = chordVec.Length();
    Vector3D chordDir  = (1 / chordLen) * chordVec;

    Precision currentSafety = safety - (position - safetyOrigin).Length();
    Precision move;
    if (currentSafety > chordLen) {
      move = chordLen;
    } else {
      Precision newSafety = 0;
      if (stepDone > 0) {
        newSafety = Navigator::ComputeSafety(position, current_state);
      }
      if (newSafety > chordLen) {
        move         = chordLen;
        safetyOrigin = position;
        safety       = newSafety;
      } else {
#ifdef ADEPT_USE_SURF
        move = Navigator::ComputeStepAndNextVolume(position, chordDir, chordLen, current_state, next_state,
                                                   hitsurf_index, kPush);
#else
        move = Navigator::ComputeStepAndNextVolume(position, chordDir, chordLen, current_state, next_state, kPush);
#endif
      }
    }

    static constexpr Precision ReduceFactor = 0.1;
    static constexpr int ReduceIters        = 6;

    if (lastWasZero && chordIters >= ReduceIters) {
      lastWasZero = false;
    }

    if (move == chordLen) {
      position  = endPosition;
      direction = endDirection;
      move      = safeMove;
      // We want to try the maximum step in the next iteration.
      maxNextSafeMove   = safeLength;
      continueIteration = true;
    } else if (move <= kPush + Navigator::kBoundaryPush && stepDone == 0) {
      // FIXME: Even for zero steps, the Navigator will return kPush + possibly
      // Navigator::kBoundaryPush instead of a real 0.
      move        = 0;
      lastWasZero = true;

      // Reduce the step attempted in the next iteration to navigate around
      // boundaries where the chord step may end in a volume we just left.
      maxNextSafeMove   = ReduceFactor * safeMove;
      continueIteration = chordIters < ReduceIters;

      if (!continueIteration) {
        // Let's move to the other side of this boundary -- this side we cannot progress !!
        move = Navigator::kBoundaryPush;
        // printf("fieldProp-ConstBz: pushing by %10.4g \n ", move );
      }
    } else {
      // Accept the intersection point on the surface.  This means that
      //   the point at the boundary will be on the 'straight'-line chord,
      //   not the curved trajectory.
      // ( This involves a bias -- relevant for muons in trackers.
      //   Currently it's controlled/limited by the acceptable step size ie. 'safeLength' )
      position = position + move * chordDir;

      // Primitive approximation of end direction and move to the crossing point ...
      Precision fraction = chordLen > 0 ? move / chordLen : 0;
      direction          = direction * (1.0 - fraction) + endDirection * fraction;
      direction          = direction.Unit();
      // safeMove is how much the track would have been moved if not hitting the boundary
      // We approximate the actual reduction along the curved trajectory to be the same
      // as the reduction of the full chord due to the boundary crossing.
      move              = fraction * safeMove;
      continueIteration = false;
    }
    stepDone += move;
    remains -= move;
    chordIters++;

  } while (continueIteration && (remains > epsilon_step) && (chordIters < max_iterations));

  propagated = (chordIters < max_iterations);
  return stepDone;
}
