// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author: J. Apostolakis  Nov/Dec 2020

#pragma once

#include <VecGeom/base/Vector3D.h>
#include <CopCore/PhysicalConstants.h>

#include <AdePT/BlockData.h>
#include <AdePT/LoopNavigator.h>

#include "ConstBzFieldStepper.h"

// Data structures for statistics of propagation chords

class fieldPropagatorConstBz {
  using Precision = vecgeom::Precision;
public:
  __host__ __device__ fieldPropagatorConstBz(float Bz) { BzValue = Bz; }
  __host__ __device__ ~fieldPropagatorConstBz() {}

  __host__ __device__ void stepInField(double kinE, double mass, int charge, double step,
                                       vecgeom::Vector3D<Precision> &position, vecgeom::Vector3D<Precision> &direction);

  template <bool Relocate = true, class Navigator = LoopNavigator>
  __host__ __device__ double ComputeStepAndPropagatedState(double kinE, double mass, int charge, double physicsStep,
                                                           vecgeom::Vector3D<Precision> &position,
                                                           vecgeom::Vector3D<Precision> &direction,
                                                           vecgeom::NavStateIndex const &current_state,
                                                           vecgeom::NavStateIndex &new_state);

private:
  float BzValue;
};

// -----------------------------------------------------------------------------

__host__ __device__ void fieldPropagatorConstBz::stepInField(double kinE, double mass, int charge, double step,
                                                             vecgeom::Vector3D<vecgeom::Precision> &position,
                                                             vecgeom::Vector3D<vecgeom::Precision> &direction)
{
  using Precision = vecgeom::Precision;
  if (charge != 0) {
    double momentumMag = sqrt(kinE * (kinE + 2.0 * mass));

    // For now all particles ( e-, e+, gamma ) can be propagated using this
    //   for gammas  charge = 0 works, and ensures that it goes straight.
    ConstBzFieldStepper helixBz(BzValue);

    vecgeom::Vector3D<Precision> endPosition  = position;
    vecgeom::Vector3D<Precision> endDirection = direction;
    helixBz.DoStep<vecgeom::Vector3D<Precision>,Precision,int>(position, direction, charge, momentumMag, step, endPosition, endDirection);
    position  = endPosition;
    direction = endDirection;
  } else {
    // Also move gammas - for now ..
    position = position + step * direction;
  }
}

// Determine the step along curved trajectory for charged particles in a field.
//  ( Same name as as navigator method. )
template <bool Relocate, class Navigator>
__host__ __device__ double fieldPropagatorConstBz::ComputeStepAndPropagatedState(
    double kinE, double mass, int charge, double physicsStep, vecgeom::Vector3D<vecgeom::Precision> &position,
    vecgeom::Vector3D<vecgeom::Precision> &direction, vecgeom::NavStateIndex const &current_state,
    vecgeom::NavStateIndex &next_state)
{
  using Precision = vecgeom::Precision;
  double momentumMag = sqrt(kinE * (kinE + 2.0 * mass));
  double momentumXYMag =
      momentumMag * sqrt((1. - direction[2]) * (1. + direction[2])); // only XY component matters for the curvature

  double curv = std::fabs(ConstBzFieldStepper::kB2C * charge * BzValue) / (momentumXYMag + 1.0e-30); // norm for step

  constexpr double gEpsilonDeflect = 1.E-2 * copcore::units::cm;

  // acceptable lateral error from field ~ related to delta_chord sagital distance

  // constexpr double invEpsD= 1.0 / gEpsilonDeflect;

  double safeLength =
      sqrt(2 * gEpsilonDeflect / curv); // max length along curve for deflectionn
                                        // = sqrt( 2.0 / ( invEpsD * curv) ); // Candidate for fast inv-sqrt

  ConstBzFieldStepper helixBz(BzValue);

  double stepDone = 0.0;
  double remains  = physicsStep;

  const double epsilon_step = 1.0e-7 * physicsStep; // Ignore remainder if < e_s * PhysicsStep

  if (charge == 0) {
    if (Relocate) {
      stepDone = Navigator::ComputeStepAndPropagatedState(position, direction, remains, current_state, next_state);
    } else {
      stepDone = Navigator::ComputeStepAndNextVolume(position, direction, remains, current_state, next_state);
    }
    position += stepDone * direction;
  } else {
    bool fullChord = false;

    //  Locate the intersection of the curved trajectory and the boundaries of the current
    //    volume (including daughters).
    //  Most electron tracks are short, limited by physics interactions -- the expected
    //    average value of iterations is small.
    //    ( Measuring iterations to confirm the maximum. )
    constexpr int maxChordIters = 10;
    int chordIters              = 0;
    do {
      vecgeom::Vector3D<Precision> endPosition  = position;
      vecgeom::Vector3D<Precision> endDirection = direction;
      double safeMove                        = min(remains, safeLength);

      // fieldPropagatorConstBz( aTrack, BzValue, endPosition, endDirection ); -- Doesn't work
      helixBz.DoStep<vecgeom::Vector3D<Precision>,Precision,int>(position, direction, charge, momentumMag, safeMove, endPosition, endDirection);

      vecgeom::Vector3D<Precision> chordVec = endPosition - position;
      double chordLen                    = chordVec.Length();
      vecgeom::Vector3D<Precision> chordDir = (1.0 / chordLen) * chordVec;

      double move;
      if (Relocate) {
        move = Navigator::ComputeStepAndPropagatedState(position, chordDir, chordLen, current_state, next_state);
      } else {
        move = Navigator::ComputeStepAndNextVolume(position, chordDir, chordLen, current_state, next_state);
      }

      fullChord = (move == chordLen);
      if (fullChord) {
        position  = endPosition;
        direction = endDirection;
        move      = safeMove;
      } else {
        // Accept the intersection point on the surface.  This means that
        //   the point at the boundary will be on the 'straight'-line chord,
        //   not the curved trajectory.
        // ( This involves a bias -- relevant for muons in trackers.
        //   Currently it's controlled/limited by the acceptable step size ie. 'safeLength' )
        position = position + move * chordDir;

        // Primitive approximation of end direction ...
        double fraction = chordLen > 0 ? move / chordLen : 0.0;
        direction       = direction * (1.0 - fraction) + endDirection * fraction;
        direction       = direction.Unit();
      }
      stepDone += move;
      remains -= move;
      chordIters++;

    } while ((!next_state.IsOnBoundary()) && fullChord && (remains > epsilon_step) && (chordIters < maxChordIters));
  }

  return stepDone;
}
