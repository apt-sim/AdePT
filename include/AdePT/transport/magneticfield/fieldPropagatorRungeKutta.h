// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author: J. Apostolakis  15 Nov 2021

#pragma once

#include <VecGeom/base/Vector3D.h>

#include "fieldConstants.h" // For kB2C factor with units

#include <AdePT/transport/magneticfield/RkIntegrationDriver.h>
#include <AdePT/transport/tracks/SafetyCache.cuh>

#include <VecGeom/navigation/NavigationState.h>

/// @brief Runge-Kutta propagator in magnetic field
/// @tparam Field_t Field type
/// @tparam RkDriver_t Driver type
/// @tparam Real_t Precision type
/// @tparam Navigator Navigator type
template <class Field_t, class RkDriver_t, typename Real_t, class Navigator>
class fieldPropagatorRungeKutta {
public:
  /// @brief Propagating method attempting to push a particle with the step proposed by physics
  /// @param magneticField Field map
  /// @param kinE Kinetc energy
  /// @param mass Particle mass
  /// @param charge Particle charge
  /// @param physicsStep Step to propagate with
  /// @param safeLength B-field safety depending only on track curvature
  /// @param position Particle position
  /// @param direction Particle direction
  /// @param[in] current_state Current geometry state
  /// @param[out] next_state Geometry state after propagation
  /// @param[out] hitsurf_index Index of the hit surface (surface model only)
  /// @param[out] propagated Checks if the step was fully propagated
  /// @param safetyCache Cached geometric isotropic safety
  /// @param max_iterations Maximum allowed iterations
  /// @param[out] iterDone Number of iterations performed
  /// @param threadId Thread id
  /// @param verbose Verbosity
  /// @return Length of the step made
  static inline __host__ __device__ double ComputeStepAndNextVolume(
      Field_t const &magneticField, double kinE, double mass, int charge, double physicsStep, double safeLength,
      vecgeom::Vector3D<double> &position, vecgeom::Vector3D<double> &direction,
      vecgeom::NavigationState const &current_state, vecgeom::NavigationState &next_state, long &hitsurf_index,
      bool &propagated, SafetyCache &safetyCache, const int max_iterations, int &iterDone, int threadId,
      bool verbose = false);
  // Move the track,
  //   updating 'position', 'direction', the next state and returning the length moved.

  // Calculate safety
  static inline __host__ __device__ Real_t ComputeSafeLength(vecgeom::Vector3D<double> &momentumVec,
                                                             vecgeom::Vector3D<Real_t> &BfieldVec, int charge);

protected:
  static constexpr unsigned int fMaxTrials = 100;
  static constexpr unsigned int Nvar       = 6; // For position (3) and momentum (3) -- invariant
  // Cannot change the energy (or momentum magnitude) -- currently usable only for pure magnetic fields
};

// ----------------------------------------------------------------------------

template <class Field_t, class RkDriver_t, typename Real_t, class Navigator_t>
inline __host__ __device__ Real_t
fieldPropagatorRungeKutta<Field_t, RkDriver_t, Real_t, Navigator_t>::ComputeSafeLength(
    vecgeom::Vector3D<double> &momentumVec, vecgeom::Vector3D<Real_t> &BfieldVec, int charge)
{
  Real_t bmag2                      = BfieldVec.Mag2();
  Real_t ratioOverFld               = (bmag2 > 0) ? momentumVec.Dot(BfieldVec) / bmag2 : 0.0;
  vecgeom::Vector3D<Real_t> PtransB = momentumVec - ratioOverFld * BfieldVec;

  Real_t bmag = sqrt(bmag2);

  // Real_t curv = fabs(Track::kB2C * charge * bmag / ( PtransB.Mag() + tiny));

  // Calculate inverse curvature instead - save a division
  Real_t inv_curv = fabs(PtransB.Mag() / (fieldConstants::kB2C * Real_t(charge) * bmag + 1.0e-30));
  // acceptable lateral error from field
  return sqrt(Real_t(2.0) * fieldConstants::deltaChord *
              inv_curv); // max length along curve for deflection
                         // = sqrt( 2.0 / ( invEpsD * curv) ); // Candidate for fast inv-sqrt
}

// Determine the step along curved trajectory for charged particles in a field.
//  ( Same name as the navigator method. )

template <class Field_t, class RkDriver_t, typename Real_t, class Navigator_t>
inline __host__ __device__ double fieldPropagatorRungeKutta<Field_t, RkDriver_t, Real_t, Navigator_t>::
    ComputeStepAndNextVolume(Field_t const &magField, double kinE, double mass, int charge, double physicsStep,
                             double safeLength, vecgeom::Vector3D<double> &position,
                             vecgeom::Vector3D<double> &direction, vecgeom::NavigationState const &current_state,
                             vecgeom::NavigationState &next_state, long &hitsurf_index, bool &propagated,
                             SafetyCache &safetyCache, const int max_iterations,
                             int &itersDone, //  useful for now - to monitor and report -- unclear if needed later
                             int index, bool verbose)
{
  double stepDone = 0.0;         ///< step already done
  Real_t remains  = physicsStep; ///< remainder of the step to be done
  constexpr bool inZeroFieldRegion =
      false; // This could be a per-region flag ... - better depend on template parameter?
  if (inZeroFieldRegion) {
#ifdef ADEPT_USE_SURF
    stepDone =
        Navigator_t::ComputeStepAndNextVolume(position, direction, remains, current_state, next_state, hitsurf_index);
#else
    stepDone = Navigator_t::ComputeStepAndNextVolume(position, direction, remains, current_state, next_state);
#endif
    position += stepDone * direction;
    return stepDone;
  }

  // Maximum integration trials
  constexpr int kMaxTrials = 30;
  // Limit for ignoring remainder depending on physics step
  const Real_t tiniest_step = 1.0e-7 * physicsStep;

  const double momentumMag              = sqrt(kinE * (kinE + 2.0 * mass));
  vecgeom::Vector3D<double> momentumVec = momentumMag * direction;

  // The allowed safe move is normally determined by the bending accuracy.
  // It can be enlarged by the geometric safety at the current position.
  Real_t maxNextSafeMove = vecCore::Max(Real_t(safeLength), Real_t(safetyCache.SafetyAt(position)));
  int chordIters         = 0;   ///< number of iterations for this integration
  Real_t last_good_step  = 0.0; ///< to be reused for next cord iteration
  bool found_end         = false;
  bool continueIteration = false;
  bool fullChord         = false;
  // bool lastWasZero             = false; // Debug only ?  JA 2022.09.05
  const Real_t inv_momentumMag = 1.0 / momentumMag;

  // Prepare next_state in case we skip navigation inside the safety sphere.
  current_state.CopyTo(&next_state);
  next_state.SetBoundaryState(false);
#if ADEPT_DEBUG_TRACK > 0
  if (verbose) {
    printf("| fieldPropagatorRK start pos {%.19f, %.19f, %.19f} dir {%.19f, %.19f, %.19f} physicsStep %g safety %g "
           "safeLength %g",
           position[0], position[1], position[2], direction[0], direction[1], direction[2], physicsStep,
           safetyCache.SafetyAt(position), safeLength);
    current_state.Print();
  }
#endif

  //  Locate the intersection of the curved trajectory and the boundaries of the current
  //    volume (including daughters).
  do {
    static constexpr Real_t ReduceFactor       = 0.1; ///< Factor to reduce the first step in case of crossing
    static constexpr int ReduceIters           = 6;   ///< Number of reduced step trials to move away from the boundary
    static constexpr Real_t kBoundaryAmbiguity = Real_t(10.0) * Navigator_t::kBoundaryPush;

    // Position and momentum at the end of the current arc
    vecgeom::Vector3D<double> endPosition    = position;
    vecgeom::Vector3D<double> endMomentumVec = momentumVec;

    // Reduce the cached safety to the current chord start before testing the
    // proposed chord. The position is updated only after the chord is accepted.
    Real_t safetyAtChordStart = safetyCache.SafetyAt(position);

    // Note: safeArc is not limited by geometry, so after pushing we need to validate that we have not crossed
    const Real_t safeArc = vecCore::Min(remains, maxNextSafeMove);

    Real_t dydx_end[Nvar]; // not reused at the moment, but could be used for FSAL between cord integrations
                           // Integrate the step.
    const Real_t arcAdvanced = RkDriver_t::Advance(endPosition, endMomentumVec, charge, safeArc, magField, dydx_end,
                                                   last_good_step, kMaxTrials, chordIters);

    //----------------- Get chord
    vecgeom::Vector3D<double> chordDir = endPosition - position; // not yet normalized!
    double chordLen                    = chordDir.Length();
    // Advance can exhaust its trials after partial accepted progress. Use the
    // accepted arc length for bookkeeping, and only reject endpoints that made
    // no progress or cannot form a finite chord.
    if (!(arcAdvanced > 0.0) || !(chordLen > 0.0)) {
      propagated = false;
      break;
    }
    chordDir *= (1.0 / chordLen); // Now the normalized direction of the chord!
    vecgeom::Vector3D<double> endDirection = inv_momentumMag * endMomentumVec;
    // Normalize direction, which is NOT after calling Advance
    endDirection.Normalize();

#if ADEPT_DEBUG_TRACK > 0
    if (verbose) {
      if (chordIters > 0)
        printf("| field_point: pos {%.19f, %.19f, %.19f} dir {%.19f, %.19f, %.19f}\n", position[0], position[1],
               position[2], direction[0], direction[1], direction[2]);
      printf("| Advance #%d: safeArc %g | chordLen %g | reducedSafety %g ", chordIters, safeArc, chordLen,
             safetyAtChordStart);
    }
#endif
    if (safetyAtChordStart <= chordLen && stepDone > 0) {
      // The reduced cached safety is not enough for this chord, so refresh the
      // cache at the current chord start before falling back to navigation.
      safetyAtChordStart = safetyCache.Refresh(position, Navigator_t::ComputeSafety(position, current_state, remains));
#if ADEPT_DEBUG_TRACK > 0
      if (verbose) printf("| refreshedSafety %g  ", safetyAtChordStart);
#endif
    }

    double move;
    if (safetyAtChordStart > chordLen) {
      // The move is safe
      move = chordLen;
    } else {
      // The move is not safe.
      // We need to check if the arc actually crosses any boundary along the chord and within chordLen
#if ADEPT_DEBUG_TRACK > 0
      if (verbose)
        printf("\n| +++  ComputeStepAndNextVolume pos {%.17f, %.17f, %.17f} chordDir {%.17f, %.17f, %.17f} "
               "chordLen %g\n",
               position[0], position[1], position[2], chordDir[0], chordDir[1], chordDir[2], chordLen);
#endif

#ifdef ADEPT_USE_SURF
      move =
          Navigator_t::ComputeStepAndNextVolume(position, chordDir, chordLen, current_state, next_state, hitsurf_index);
#else
      move = Navigator_t::ComputeStepAndNextVolume(position, chordDir, chordLen, current_state, next_state);
#endif
    }

    // lastWasZero &= chordIters < ReduceIters;

    fullChord = (move >= chordLen);
    if (fullChord) {
      // No boundary hit along the chord -> update
      // Note: the arc may have actually hit a corner, but this is accepted within the sagita error
      // which is accounted for by chordLen
      position    = endPosition;
      momentumVec = endMomentumVec;

      direction = endDirection;
      move      = arcAdvanced; // curvedStep
#if ADEPT_DEBUG_TRACK > 0
      if (verbose) printf("| full chord advance %g ", arcAdvanced);
#endif

      maxNextSafeMove   = vecCore::Max(arcAdvanced, Real_t(safetyCache.SafetyAt(position))); // Reset after success.
      continueIteration = true;
    } else if (stepDone == 0 && current_state.IsOnBoundary() && next_state.IsOnBoundary() &&
               move <= kBoundaryAmbiguity) {
      // A first chord step that immediately reports a boundary while already on
      // a boundary is ambiguous: the chord may touch a tolerance surface even if
      // the physical direction stays in the current volume. Probe once along the
      // physical direction before accepting the chord result.
      // The probe is also bounded by this RK chord so any accepted
      // distance stays inside the integrated endpoint used for the
      // chord interpolation below.
      const Real_t directionProbeLimit =
          vecCore::Min(Real_t(chordLen), vecCore::Min(Real_t(remains), kBoundaryAmbiguity));

      vecgeom::NavigationState directionState;
      current_state.CopyTo(&directionState);
      directionState.SetBoundaryState(false);

#ifdef ADEPT_USE_SURF
      long directionHitsurfIndex = hitsurf_index;
      const double directionMove = Navigator_t::ComputeStepAndNextVolume(
          position, direction, directionProbeLimit, current_state, directionState, directionHitsurfIndex);
#else
      const double directionMove = Navigator_t::ComputeStepAndNextVolume(position, direction, directionProbeLimit,
                                                                         current_state, directionState);
#endif
      const bool directionHit          = directionState.IsOnBoundary();
      const bool directionTiny         = directionHit && directionMove <= kBoundaryAmbiguity;
      const bool directionStateChanged = !directionState.HasSamePathAsOther(current_state);

      if (!directionTiny) {
        // Chord-only boundary artifact: do not accept the chord state. Retry the
        // field integration with a shorter arc so the next chord is less likely
        // to clip the nearby surface.
        current_state.CopyTo(&next_state);
        next_state.SetBoundaryState(false);
        move              = 0.;
        maxNextSafeMove   = ReduceFactor * arcAdvanced;
        continueIteration = chordIters < ReduceIters;
#if ADEPT_DEBUG_TRACK > 0
        if (verbose) printf("| TINY CHORD ARTIFACT reducedAdvance %g continue %d", maxNextSafeMove, continueIteration);
#endif
      } else if (directionMove <= Navigator_t::kBoundaryPush) {
        // Both chord and physical direction report a boundary at the push scale.
        // Treat a clean state change as a zero-length relocation/backscatter,
        // matching the navigator contract without fabricating travelled length.
        if (directionStateChanged) {
          directionState.CopyTo(&next_state);
#ifdef ADEPT_USE_SURF
          hitsurf_index = directionHitsurfIndex;
#endif
#if ADEPT_DEBUG_TRACK > 0
          if (verbose) {
            printf("| DIRECTION-RESOLVED ZERO-LENGTH BOUNDARY RELOCATION hitting ");
            next_state.Print();
          }
#endif
        } else {
          current_state.CopyTo(&next_state);
          next_state.SetBoundaryState(false);
          maxNextSafeMove = ReduceFactor * arcAdvanced;
#if ADEPT_DEBUG_TRACK > 0
          if (verbose) printf("| UNRESOLVED TINY BOUNDARY AMBIGUITY");
#endif
        }
        move              = 0.;
        continueIteration = !directionStateChanged && chordIters < ReduceIters;
      } else {
        // The boundary is inside the ambiguity band, but it is still farther than
        // the navigator push. If the chord itself only clipped the tolerance
        // surface at push scale, preserve the physical-direction crossing that
        // resolved the ambiguity instead.
        if (move <= Navigator_t::kBoundaryPush) {
          directionState.CopyTo(&next_state);
#ifdef ADEPT_USE_SURF
          hitsurf_index = directionHitsurfIndex;
#endif
          move = directionMove;
        }
        double fraction = vecCore::Max(move / chordLen, 0.);
        position += move * chordDir;
        direction   = direction * (1.0 - fraction) + endDirection * fraction;
        direction   = direction.Unit();
        momentumVec = momentumMag * direction;
#if ADEPT_DEBUG_TRACK > 0
        if (verbose) {
          printf("| finite tiny boundary crossing %g hitting ", move);
          next_state.Print();
        }
#endif
        continueIteration = false;
      }
    } else {
      // A boundary is on the way at non-zero distance
      assert(next_state.IsOnBoundary());
      // assert( linearStep == chordLen );

      // USE the intersection point on the chord & surface as the 'solution', ie. instead
      //     of the (potential) true point on the intersection of the curve and the boundary.
      // ( This involves a bias -- typically important only for muons in trackers.
      //   Currently it's controlled/limited by the acceptable step size ie. 'safeLength' )
      double fraction = vecCore::Max(move / chordLen, 0.); // linearStep

      // Primitive approximation of end direction and linearStep to the crossing point ...
      position += move * chordDir; // linearStep
      direction   = direction * (1.0 - fraction) + endDirection * fraction;
      direction   = direction.Unit();
      momentumVec = momentumMag * direction;
      // safeArc is how much the track would have been moved if not hitting the boundary
      // We approximate the actual reduction along the curved trajectory to be the same
      // as the reduction of the full chord due to the boundary crossing.
#if ADEPT_DEBUG_TRACK > 0
      if (verbose) {
        printf("| linear step to crossing point %g hitting ", move);
        next_state.Print();
      }
#endif
      continueIteration = false;
    }

    stepDone += move; // curvedStep
    remains -= move;  // curvedStep
    chordIters++;
    found_end = ((move > 0) && next_state.IsOnBoundary()) // curvedStep Fix 2022.09.05 JA
                || (remains <= tiniest_step);
#if ADEPT_DEBUG_TRACK > 0
    if (verbose)
      printf("| stepDone %g remains %g foundEnd = %d chordIters %d point {%.8f, %.8f, %.8f} dir {%.8f, %.8f, "
             "%.8f}\n",
             stepDone, remains, found_end, chordIters, position[0], position[1], position[2], direction[0],
             direction[1], direction[2]);
#endif

  } while (!found_end && continueIteration && (chordIters < max_iterations));

  propagated = found_end;
  itersDone += chordIters;

  return stepDone;
}
