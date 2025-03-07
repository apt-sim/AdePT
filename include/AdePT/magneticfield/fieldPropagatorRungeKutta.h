// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author: J. Apostolakis  15 Nov 2021

#ifndef FIELD_PROPAGATOR_RUNGEKUTTA_H
#define FIELD_PROPAGATOR_RUNGEKUTTA_H

#include <VecGeom/base/Vector3D.h>

#include "fieldConstants.h" // For kB2C factor with units

#include "UniformMagneticField.h"
#include <AdePT/magneticfield/RkIntegrationDriver.h>

#include <VecGeom/navigation/NavigationState.h>

template <class Field_t, class RkDriver_t, typename Real_t, class Navigator>
class fieldPropagatorRungeKutta {
public:
  static inline __host__ __device__ __host__ __device__ Real_t ComputeStepAndNextVolume(
      Field_t const &magneticField, double kinE, double mass, int charge, double physicsStep,
      vecgeom::Vector3D<Real_t> &position, vecgeom::Vector3D<Real_t> &direction,
      vecgeom::NavigationState const &current_state, vecgeom::NavigationState &next_state, long &hitsurf_index,
      bool &propagated, const Real_t &safetyIn, const int max_iterations, int &iterDone, int threadId);
  // Move the track,
  //   updating 'position', 'direction', the next state and returning the length moved.

  // Calculate safety
  static inline __host__ __device__ Real_t ComputeSafeLength(vecgeom::Vector3D<Real_t> &momentumVec,
                                                             vecgeom::Vector3D<Real_t> &BfieldVec, int charge);

protected:
  static constexpr unsigned int fMaxTrials = 100;
  static constexpr unsigned int Nvar       = 6; // For position (3) and momentum (3) -- invariant

#ifdef VECGEOM_FLOAT_PRECISION
  static constexpr Real_t kPush = 10 * vecgeom::kTolerance;
#else
  static constexpr Real_t kPush = 0.;
#endif

  // Cannot change the energy (or momentum magnitude) -- currently usable only for pure magnetic fields
};

// ----------------------------------------------------------------------------

template <class Field_t, class RkDriver_t, typename Real_t, class Navigator_t>
inline __host__ __device__ Real_t
fieldPropagatorRungeKutta<Field_t, RkDriver_t, Real_t, Navigator_t>::ComputeSafeLength(
    vecgeom::Vector3D<Real_t> &momentumVec, vecgeom::Vector3D<Real_t> &BfieldVec, int charge)
{
  Real_t bmag2                      = BfieldVec.Mag2();
  Real_t ratioOverFld               = (bmag2 > 0) ? momentumVec.Dot(BfieldVec) / bmag2 : 0.0;
  vecgeom::Vector3D<Real_t> PtransB = momentumVec - ratioOverFld * BfieldVec;

  Real_t bmag = sqrt(bmag2);

  // Real_t curv = fabs(Track::kB2C * charge * bmag / ( PtransB.Mag() + tiny));

  // Calculate inverse curvature instead - save a division
  Real_t inv_curv = fabs(PtransB.Mag() / (fieldConstants::kB2C * Real_t(charge) * bmag + 1.0e-30));
  // acceptable lateral error from field ~ related to delta_chord sagital distance
  return sqrt(Real_t(2.0) * fieldConstants::gEpsilonDeflect *
              inv_curv); // max length along curve for deflectionn
                         // = sqrt( 2.0 / ( invEpsD * curv) ); // Candidate for fast inv-sqrt
}

// Determine the step along curved trajectory for charged particles in a field.
//  ( Same name as the navigator method. )

template <class Field_t, class RkDriver_t, typename Real_t, class Navigator_t>
inline __host__ __device__ Real_t
fieldPropagatorRungeKutta<Field_t, RkDriver_t, Real_t, Navigator_t>::ComputeStepAndNextVolume(
    Field_t const &magField, double kinE, double mass, int charge, double physicsStep,
    vecgeom::Vector3D<Real_t> &position, vecgeom::Vector3D<Real_t> &direction,
    vecgeom::NavigationState const &current_state, vecgeom::NavigationState &next_state, long &hitsurf_index,
    bool &propagated, const Real_t &safetyIn, //  eventually In/Out ?
    const int max_iterations, int &itersDone  //  useful for now - to monitor and report -- unclear if needed later
    ,
    int indx)
{
  // using copcore::units::MeV;

  const Real_t momentumMag              = sqrt(kinE * (kinE + 2.0 * mass));
  vecgeom::Vector3D<Real_t> momentumVec = momentumMag * direction;

  vecgeom::Vector3D<Real_t> B0fieldVec = magField.Evaluate(position); // Field value at starting point

  const Real_t safeLength = ComputeSafeLength /*<Real_t>*/ (momentumVec, B0fieldVec, charge);

  Precision maxNextSafeMove = safeLength; // It can be reduced if, at the start, a boundary is encountered

  Real_t stepDone           = 0.0;
  Real_t remains            = physicsStep;
  const Real_t tiniest_step = 1.0e-7 * physicsStep; // Ignore remainder if < e_s * PhysicsStep
  int chordIters            = 0;

  constexpr bool inZeroFieldRegion =
      false; // This could be a per-region flag ... - better depend on template parameter?
  bool found_end = false;

  if (inZeroFieldRegion) {
#ifdef ADEPT_USE_SURF
    stepDone = Navigator_t::ComputeStepAndNextVolume(position, direction, remains, current_state, next_state,
                                                     hitsurf_index, kPush);
#else
    stepDone = Navigator_t::ComputeStepAndNextVolume(position, direction, remains, current_state, next_state, kPush);
#endif
    position += stepDone * direction;
  } else {
    bool continueIteration       = false;
    bool fullChord               = false;
    const Real_t inv_momentumMag = 1.0 / momentumMag;

    Precision safety                       = safetyIn;
    vecgeom::Vector3D<Real_t> safetyOrigin = position;
    // Prepare next_state in case we skip navigation inside the safety sphere.
    current_state.CopyTo(&next_state);
    next_state.SetBoundaryState(false);

    bool lastWasZero = false; // Debug only ?  JA 2022.09.05

    //  Locate the intersection of the curved trajectory and the boundaries of the current
    //    volume (including daughters).
    do {
      static constexpr Precision ReduceFactor = 0.1;
      static constexpr int ReduceIters        = 6;

      vecgeom::Vector3D<Real_t> endPosition    = position;
      vecgeom::Vector3D<Real_t> endMomentumVec = momentumVec;                   // momentumMag * direction;
      const Real_t safeArc                     = min(remains, maxNextSafeMove); // safeLength);

      Real_t dydx_end[Nvar]; // not used at the moment, but could be used for FSAL between cord integrations
      bool done =
          RkDriver_t::Advance(endPosition, endMomentumVec, charge, safeArc, magField, dydx_end, /*max_trials=*/30);

      //-----------------
      vecgeom::Vector3D<Real_t> chordDir     = endPosition - position; // not yet normalized!
      Real_t chordLen                        = chordDir.Length();
      vecgeom::Vector3D<Real_t> endDirection = inv_momentumMag * endMomentumVec;
      chordDir *= (1.0 / chordLen); // Now the normalized direction of the chord!

      Precision currentSafety = safety - (position - safetyOrigin).Length();
      Precision move;
      if (currentSafety > chordLen) {
        move = chordLen;
      } else {
        Precision newSafety = 0;
        if (stepDone > 0) {
#ifdef ADEPT_USE_SURF
          // Use maximum accuracy only if safety is smaller than physicalStepLength
          newSafety = Navigator_t::ComputeSafety(position, current_state, physicsStep);
#else
          newSafety = Navigator_t::ComputeSafety(position, current_state);
#endif
        }
        if (newSafety > chordLen) {
          move         = chordLen;
          safetyOrigin = position;
          safety       = newSafety;
        } else {
#ifdef ADEPT_USE_SURF
          move = Navigator_t::ComputeStepAndNextVolume(position, chordDir, chordLen, current_state, next_state,
                                                       hitsurf_index, kPush);
#else
          move = Navigator_t::ComputeStepAndNextVolume(position, chordDir, chordLen, current_state, next_state, kPush);
#endif
        }
      }

      // Real_t curvedStep;

      lastWasZero = lastWasZero && !(chordIters >= ReduceIters);
      // if (lastWasZero && chordIters >= ReduceIters) {
      //   lastWasZero = false;
      // }

      fullChord = (move == chordLen); // linearStep
      if (fullChord) {
        position    = endPosition;
        momentumVec = endMomentumVec;

        direction = endDirection;
        move      = safeArc; // curvedStep

        maxNextSafeMove   = safeArc; // Reset it, once a step succeeds!!
        continueIteration = true;
      } else if (move <= kPush + Navigator_t::kBoundaryPush && stepDone == 0) { // linearStep
        // Cope with a track at a boundary that wants to bend back into the previous
        //   volume in the first step (by reducing the attempted distance.)
        // FIXME: Even for zero steps, the Navigator will return kPush + possibly
        // Navigator::kBoundaryPush instead of a real 0.
        move        = 0; // curvedStep
        lastWasZero = true;

        // Reduce the step attempted in the next iteration to navigate around
        // boundaries where the chord step may end in a volume we just left.
        maxNextSafeMove   = ReduceFactor * safeArc;
        continueIteration = chordIters < ReduceIters;

        if (!continueIteration) {
          // Let's move to the other side of this boundary -- this side we cannot progress !!
          move = Navigator_t::kBoundaryPush; // curvedStep
        }
      } else {
        assert(next_state.IsOnBoundary());
        // assert( linearStep == chordLen );

        // USE the intersection point on the chord & surface as the 'solution', ie. instead
        //     of the (potential) true point on the intersection of the curve and the boundary.
        // ( This involves a bias -- typically important only for muons in trackers.
        //   Currently it's controlled/limited by the acceptable step size ie. 'safeLength' )
        Real_t fraction = vecCore::Max(move / chordLen, 0.); // linearStep
        move            = fraction * safeArc;                // curvedStep
#ifndef ENDPOINT_ON_CURVE
        // Primitive approximation of end direction and linearStep to the crossing point ...
        position    = position + move * chordDir; // linearStep
        direction   = direction * (1.0 - fraction) + endDirection * fraction;
        direction   = direction.Unit();
        momentumVec = momentumMag * direction;
        // safeArc is how much the track would have been moved if not hitting the boundary
        // We approximate the actual reduction along the curved trajectory to be the same
        // as the reduction of the full chord due to the boundary crossing.
#else
        // Alternative approximation of end position & direction -- calling RK again
        //  Better accuracy (e.g. for comparing with Helix) -- but the point will not be on the surface !!
        // IntegrateTrackToEnd(magField, position, momentumVec, charge, move, indx); // curvedStep
        Real_t dydx_end[Nvar]; // not used at the moment, but could be used for FSAL between cord integrations
        bool done = RkDriver_t::Advance(position, momentumVec, charge, move, magField, dydx_end, /*max_trials=*/30);

        direction = inv_momentumMag * momentumVec; // momentumVec.Unit();
#endif
        continueIteration = false;
      }

      stepDone += move; // curvedStep
      remains -= move;  // curvedStep
      chordIters++;

      found_end = ((move > 0) && next_state.IsOnBoundary()) // curvedStep Fix 2022.09.05 JA
                  || (remains <= tiniest_step);

    } while (!found_end && continueIteration && (chordIters < max_iterations));
  }

  propagated = found_end;
  itersDone += chordIters;
  //  = (chordIters < max_iterations);  // ---> Misses success on the last step!
  return stepDone;
}

#endif
