// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author: J. Apostolakis  Nov/Dec 2020

#pragma once

#include <VecGeom/base/Vector3D.h>
#include <AdePT/copcore/PhysicalConstants.h>

#include <AdePT/base/BlockData.h>
#include <AdePT/navigation/AdePTNavigator.h>

#include "fieldConstants.h"

#include <AdePT/magneticfield/ConstBzFieldStepper.h>

// Data structures for statistics of propagation chords

class fieldPropagatorConstBz {
  using Vector3D = vecgeom::Vector3D<double>;

public:
  __host__ __device__ fieldPropagatorConstBz(double Bz) { BzValue = Bz; }
  __host__ __device__ ~fieldPropagatorConstBz() {}

  __host__ __device__ double ComputeSafeLength(double momentumMag, int charge, const Vector3D &direction);

  template <class Navigator = AdePTNavigator>
  __host__ __device__ double ComputeStepAndNextVolume(double kinE, double mass, int charge, double physicsStep,
                                                      Vector3D &position, Vector3D &direction,
                                                      vecgeom::NavigationState const &current_state,
                                                      vecgeom::NavigationState &new_state, long &hitsurf_index,
                                                      bool &propagated, const double safety = 0.0,
                                                      const int max_iteration = 100);

private:
  double BzValue;
};

// -----------------------------------------------------------------------------

__host__ __device__ double fieldPropagatorConstBz::ComputeSafeLength(double momentumMag, int charge,
                                                                     const Vector3D &direction)
{
  // Direction projection in plane perpendicular to field vector
  double dirxy = sqrt((1 - direction[2]) * (1 + direction[2]));

  double bend = std::fabs(fieldConstants::kB2C * charge * BzValue) / momentumMag;

  // R = helix radius, curv = 1./R = curvature in plane perpendicular to the field
  // double curv = bend / (dirxy + 1.e-30);

  // Distance along the track direction to reach the maximum allowed error
  return sqrt(2 * fieldConstants::deltaChord / (bend * dirxy + 1.e-30));
}

// Determine the step along curved trajectory for charged particles in a field.
//  ( Same name as as navigator method. )
template <class Navigator>
__host__ __device__ double fieldPropagatorConstBz::ComputeStepAndNextVolume(
    double kinE, double mass, int charge, double physicsStep, vecgeom::Vector3D<double> &position,
    vecgeom::Vector3D<double> &direction, vecgeom::NavigationState const &current_state,
    vecgeom::NavigationState &next_state, long &hitsurf_index, bool &propagated, const double safetyIn,
    const int max_iterations)
{
  const double kPush = 0;

  double momentumMag = sqrt(kinE * (kinE + 2 * mass));

  // Distance along the track direction to reach the maximum allowed error
  const double safeLength = ComputeSafeLength(momentumMag, charge, direction);

  ConstBzFieldStepper helixBz(BzValue);

  double stepDone           = 0;
  double remains            = physicsStep;
  const double epsilon_step = 1.0e-7 * physicsStep; // Ignore remainder if < e_s * PhysicsStep
  int chordIters            = 0;

  bool continueIteration = false;

  double safety         = safetyIn;
  Vector3D safetyOrigin = position;
  // Prepare next_state in case we skip navigation inside the safety sphere.
  current_state.CopyTo(&next_state);
  next_state.SetBoundaryState(false);

  double maxNextSafeMove = safeLength;

  bool lastWasZero = false;
  //  Locate the intersection of the curved trajectory and the boundaries of the current
  //    volume (including daughters).
  do {
    Vector3D endPosition  = position;
    Vector3D endDirection = direction;
    double safeMove       = min(remains, maxNextSafeMove);

    helixBz.DoStep<Vector3D, double, int>(position, direction, charge, momentumMag, safeMove, endPosition,
                                          endDirection);

    Vector3D chordVec = endPosition - position;
    double chordLen   = chordVec.Length();
    Vector3D chordDir = (1 / chordLen) * chordVec;

    double currentSafety = safety - (position - safetyOrigin).Length();
    double move;
    if (currentSafety > chordLen) {
      move = chordLen;
    } else {
      double newSafety = 0;
      if (stepDone > 0) {
        // Use maximum accuracy only if safety is smaller than physicalStepLength
        newSafety = Navigator::ComputeSafety(position, current_state, remains);
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

    static constexpr double ReduceFactor = 0.1;
    static constexpr int ReduceIters     = 6;

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
      double fraction = chordLen > 0 ? move / chordLen : 0;
      direction       = direction * (1.0 - fraction) + endDirection * fraction;
      direction       = direction.Unit();
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
