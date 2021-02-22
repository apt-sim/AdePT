// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author: J. Apostolakis  Nov/Dec 2020

#pragma once

#include <VecGeom/base/Vector3D.h>
#include <CopCore/PhysicalConstants.h>

#include <AdePT/BlockData.h>
#include <AdePT/LoopNavigator.h>

#include "track.h"
#include "ConstBzFieldStepper.h"

// Data structures for statistics of propagation chords

class fieldPropagatorConstBz {
public:
  __host__ __device__ fieldPropagatorConstBz(float Bz) { BzValue = Bz; }
  __host__ __device__ ~fieldPropagatorConstBz() {}

  __host__ __device__ void stepInField(track &aTrack, vecgeom::Vector3D<double> &endPosition,
                                       vecgeom::Vector3D<double> &endDirection);

  template <bool Relocate = true>
  __device__ //  __host__
      double
      ComputeStepAndPropagatedState(track &aTrack, float physicsStep);

private:
  float BzValue;
};

constexpr double kPushField = 1.e-8;

// -----------------------------------------------------------------------------

__host__ __device__ void fieldPropagatorConstBz::stepInField(track &aTrack,
                                                             // float      BzValue,
                                                             vecgeom::Vector3D<double> &endPosition,
                                                             vecgeom::Vector3D<double> &endDirection)
{
  using copcore::units::kElectronMassC2;

  double step = aTrack.interaction_length;
  int charge  = aTrack.charge();

  if (charge != 0.0) {
    double kinE        = aTrack.energy;
    double momentumMag = sqrt(kinE * (kinE + 2.0 * kElectronMassC2));
    // aTrack.mass() -- when extending with other charged particles

    // For now all particles ( e-, e+, gamma ) can be propagated using this
    //   for gammas  charge = 0 works, and ensures that it goes straight.
    ConstBzFieldStepper helixBz(BzValue);

    helixBz.DoStep(aTrack.pos, aTrack.dir, charge, momentumMag, step, endPosition, endDirection);
  } else {
    // Also move gammas - for now ..
    endPosition  = aTrack.pos + step * aTrack.dir;
    endDirection = aTrack.dir;
  }
}

// Determine the step along curved trajectory for charged particles in a field.
//  ( Same name as as navigator method. )
template <bool Relocate>
__device__ // __host__
    double
    fieldPropagatorConstBz::ComputeStepAndPropagatedState(track &aTrack, float physicsStep)
{
  if (aTrack.status != alive) return 0.0;

  double kinE        = aTrack.energy;
  double momentumMag = sqrt(kinE * (kinE + 2.0 * copcore::units::kElectronMassC2));

  double charge = aTrack.charge();
  double curv   = std::fabs(ConstBzFieldStepper::kB2C * charge * BzValue) / (momentumMag + 1.0e-30); // norm for step

  constexpr double gEpsilonDeflect = 1.E-2 * copcore::units::cm;
  // acceptable lateral error from field ~ related to delta_chord sagital distance

  // constexpr double invEpsD= 1.0 / gEpsilonDeflect;

  double safeLength =
      2. * sqrt(gEpsilonDeflect / curv); // max length along curve for deflectionn
                                         // = 2. * sqrt( 1.0 / ( invEpsD * curv) ); // Candidate for fast inv-sqrt

  vecgeom::Vector3D<double> position  = aTrack.pos;
  vecgeom::Vector3D<double> direction = aTrack.dir;

  ConstBzFieldStepper helixBz(BzValue);

  float stepDone = 0.0;
  double remains = physicsStep;

  constexpr double epsilon_step = 1.0e-7; // Ignore remainder if < e_s * PhysicsStep

  if (aTrack.charge() == 0.0) {
    if (Relocate) {
      stepDone = LoopNavigator::ComputeStepAndPropagatedState(position, direction, physicsStep, aTrack.current_state,
                                                              aTrack.next_state);
    } else {
      stepDone = LoopNavigator::ComputeStepAndNextVolume(position, direction, physicsStep, aTrack.current_state,
                                                         aTrack.next_state);
    }
    position += (stepDone + kPushField) * direction;
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
      vecgeom::Vector3D<double> endPosition  = position;
      vecgeom::Vector3D<double> endDirection = direction;
      double safeMove                        = min(remains, safeLength);

      // fieldPropagatorConstBz( aTrack, BzValue, endPosition, endDirection ); -- Doesn't work
      helixBz.DoStep(position, direction, charge, momentumMag, safeMove, endPosition, endDirection);

      vecgeom::Vector3D<double> chordVec = endPosition - aTrack.pos;
      double chordLen                    = chordVec.Length();
      vecgeom::Vector3D<double> chordDir = (1.0 / chordLen) * chordVec;

      double move;
      if (Relocate) {
        move = LoopNavigator::ComputeStepAndPropagatedState(position, chordDir, chordLen, aTrack.current_state,
                                                            aTrack.next_state);
      } else {
        move = LoopNavigator::ComputeStepAndNextVolume(position, chordDir, chordLen, aTrack.current_state,
                                                       aTrack.next_state);
      }

      fullChord = (move == chordLen);
      if (fullChord) {
        position  = endPosition;
        direction = endDirection;
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

    } while ((!aTrack.next_state.IsOnBoundary()) && fullChord && (remains > epsilon_step * physicsStep) &&
             (chordIters < maxChordIters));
  }
  // stepDone= physicsStep - remains;

  aTrack.pos = position;
  aTrack.dir = direction;

  return stepDone; // physicsStep-remains;
}
