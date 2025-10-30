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
  /// @param current_state[in] Current geometry state
  /// @param next_state[out] Geometry state after propagation
  /// @param hitsurf_index[out] Index of the hit surface (surface model only)
  /// @param propagated[out] Checks if the step was fully propagated
  /// @param safetyIn Geometric isotropic safety
  /// @param max_iterations Maximum allowed iterations
  /// @param iterDone[out] Number of iterations performed
  /// @param threadId Thread id
  /// @param zero_first_step Detected zero first step
  /// @param verbose Verbosity
  /// @return Length of the step made
  static inline __host__ __device__ double ComputeStepAndNextVolume(
      Field_t const &magneticField, double kinE, double mass, int charge, double physicsStep, double safeLength,
      vecgeom::Vector3D<double> &position, vecgeom::Vector3D<double> &direction,
      vecgeom::NavigationState const &current_state, vecgeom::NavigationState &next_state, long &hitsurf_index,
      bool &propagated, const Real_t &safetyIn, const int max_iterations, int &iterDone, int threadId,
      bool &zero_first_step, bool verbose = false);
  // Move the track,
  //   updating 'position', 'direction', the next state and returning the length moved.

  // Calculate safety
  static inline __host__ __device__ Real_t ComputeSafeLength(vecgeom::Vector3D<double> &momentumVec,
                                                             vecgeom::Vector3D<Real_t> &BfieldVec, int charge);

protected:
  static constexpr unsigned int fMaxTrials = 100;
  static constexpr unsigned int Nvar       = 6; // For position (3) and momentum (3) -- invariant
  static constexpr Real_t kPush            = 0.;
  static constexpr Real_t kDistCheckPush   = kPush + Navigator::kBoundaryPush;
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
                             const Real_t &safetyIn, //  eventually In/Out ?
                             const int max_iterations,
                             int &itersDone, //  useful for now - to monitor and report -- unclear if needed later
                             int indx, bool &zero_first_step, bool verbose)
{
  double stepDone = 0.0;         ///< step already done
  Real_t remains  = physicsStep; ///< remainder of the step to be done
  zero_first_step = false;
  constexpr bool inZeroFieldRegion =
      false; // This could be a per-region flag ... - better depend on template parameter?
  if (inZeroFieldRegion) {
#ifdef ADEPT_USE_SURF
    stepDone = Navigator_t::ComputeStepAndNextVolume(position, direction, remains, current_state, next_state,
                                                     hitsurf_index, kDistCheckPush);
#else
    stepDone =
        Navigator_t::ComputeStepAndNextVolume(position, direction, remains, current_state, next_state, kDistCheckPush);
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

  // The allowed safe move is normally determined by the bending accuracy
  // This is reduced if starting from a boundary and crossing a boundary in the first step
  Real_t maxNextSafeMove = max(safeLength, safetyIn); // It can be reduced if, at the start, a boundary is encountered
  int chordIters         = 0;                         ///< number of iterations for this integration
  Real_t last_good_step  = 0.0;                       ///< to be re-used for next cord iteration
  bool found_end         = false;
  bool continueIteration = false;
  bool fullChord         = false;
  // bool lastWasZero             = false; // Debug only ?  JA 2022.09.05
  const Real_t inv_momentumMag = 1.0 / momentumMag;

  // Cache safety origin and value at the start point
  double safety                          = safetyIn;
  vecgeom::Vector3D<double> safetyOrigin = position;
  // Prepare next_state in case we skip navigation inside the safety sphere.
  current_state.CopyTo(&next_state);
  next_state.SetBoundaryState(false);
#if ADEPT_DEBUG_TRACK > 0
  if (verbose) {
    printf("| fieldPropagatorRK start pos {%.19f, %.19f, %.19f} dir {%.19f, %.19f, %.19f} physicsStep %g safety %g "
           "safeLength %g",
           position[0], position[1], position[2], direction[0], direction[1], direction[2], physicsStep, safety,
           safeLength);
    current_state.Print();
  }
#endif

  //  Locate the intersection of the curved trajectory and the boundaries of the current
  //    volume (including daughters).
  do {
    static constexpr Real_t ReduceFactor = 0.1; ///< Factor to reduce the first step in case of crossing
    static constexpr int ReduceIters     = 6;   ///< Number of reduced step trials to move away from the boundary

    // Position and momentum at the end of the current arc
    vecgeom::Vector3D<double> endPosition    = position;
    vecgeom::Vector3D<double> endMomentumVec = momentumVec;

    // Note: safeArc is not limited by geometry, so after pushing we need to validate that we have not crossed
    const Real_t safeArc = min(remains, maxNextSafeMove);

    Real_t dydx_end[Nvar]; // not re-used at the moment, but could be used for FSAL between cord integrations
                           // Integrate the step.
    /*bool done = */
    RkDriver_t::Advance(endPosition, endMomentumVec, charge, safeArc, magField, dydx_end, last_good_step, kMaxTrials,
                        chordIters);

    //----------------- Get chord
    vecgeom::Vector3D<double> chordDir = endPosition - position; // not yet normalized!
    double chordLen                    = chordDir.Length();
    chordDir *= (1.0 / chordLen); // Now the normalized direction of the chord!
    vecgeom::Vector3D<double> endDirection = inv_momentumMag * endMomentumVec;
    // Normalize direction, which is NOT after calling Advance
    endDirection.Normalize();

    // Subtract from the existing safety after the move
    Real_t currentSafety = safety - (endPosition - safetyOrigin).Length();
#if ADEPT_DEBUG_TRACK > 0
    if (verbose) {
      if (chordIters > 0)
        printf("| field_point: pos {%.19f, %.19f, %.19f} dir {%.19f, %.19f, %.19f}\n", position[0], position[1],
               position[2], direction[0], direction[1], direction[2]);
      printf("| Advance #%d: safeArc %g | chordLen %g | reducedSafety %g ", chordIters, safeArc, chordLen,
             currentSafety);
    }
#endif
    double move;
    if (currentSafety > chordLen) {
      // The move is still safe
      move = chordLen;
    } else {
      // Safety is violated by the move, set it to 0 and recompute it if not the first step
      double newSafety = 0;
      if (stepDone > 0) {
        // Use maximum accuracy only if safety is smaller than the step remainder
        newSafety = Navigator_t::ComputeSafety(position, current_state, remains);
#if ADEPT_DEBUG_TRACK > 0
        if (verbose) printf("| newSafety %g  ", newSafety);
#endif
      }
      if (newSafety > chordLen) {
        // The recomputed safety was actually larger than the chord -> safe step
        move = chordLen;
        // update safety with the computed one BEFORE the arc advance
        safetyOrigin = position;
        safety       = newSafety;
      } else {
        // We need to check if the arc actually crosses any boundary along the chord and withing chordLen
#if ADEPT_DEBUG_TRACK > 0
        if (verbose)
          printf("\n| +++  ComputeStepAndNextVolume pos {%.17f, %.17f, %.17f} chordDir {%.17f, %.17f, %.17f} "
                 "chordLen %g push %g\n",
                 position[0], position[1], position[2], chordDir[0], chordDir[1], chordDir[2], chordLen, kPush);
#endif

#ifdef ADEPT_USE_SURF
        move = Navigator_t::ComputeStepAndNextVolume(position, chordDir, chordLen, current_state, next_state,
                                                     hitsurf_index, kDistCheckPush);
#else
        move = Navigator_t::ComputeStepAndNextVolume(position, chordDir, chordLen, current_state, next_state,
                                                     kDistCheckPush);
#endif
      }
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
      move      = safeArc; // curvedStep
#if ADEPT_DEBUG_TRACK > 0
      if (verbose) printf("| full chord advance %g ", safeArc);
#endif

      maxNextSafeMove   = max(safeArc, safety); // Reset it, once a step succeeds!!
      continueIteration = true;
    } else if (stepDone == 0 && move <= kDistCheckPush) {
      // Cope with a track at a boundary that wants to bend back into the previous
      // volume in the first step (by reducing the attempted distance.)

      // Deal with back-scattered tracks that need to be relocated. Check distance along initial direction.
#ifdef ADEPT_USE_SURF
      move = Navigator_t::ComputeStepAndNextVolume(position, direction, remains, current_state, next_state,
                                                   hitsurf_index, kDistCheckPush);
#else
      move = Navigator_t::ComputeStepAndNextVolume(position, direction, remains, current_state, next_state,
                                                   kDistCheckPush);
#endif

      if (move <= kDistCheckPush) {
#if ADEPT_DEBUG_TRACK > 0
        if (verbose) {
          printf("| BACK-SCATTERING or WRONG RELOCATION detected hitting ");
          next_state.Print();
        }
#endif
        zero_first_step = true;
        return 0.;
      }

      // Reduce the step attempted in the next iteration to navigate around
      // boundaries where the chord step may end in a volume we just left.
      // lastWasZero = true;
      move              = 0.;
      maxNextSafeMove   = ReduceFactor * safeArc;
      continueIteration = chordIters < ReduceIters;

      if (!continueIteration) {
        // Let's move to the other side of this boundary -- this side we cannot progress !!
        move = Navigator_t::kBoundaryPush; // curvedStep
        position += move * chordDir;
      }
#if ADEPT_DEBUG_TRACK > 0
      if (verbose)
        printf("| FIRST STEP BENDING BACK %g  reducedAdvance %g continue %d", move, maxNextSafeMove, continueIteration);
#endif
    } else {
      // A boundary is on the way at non-zero distance
      assert(next_state.IsOnBoundary());
      // assert( linearStep == chordLen );

      // USE the intersection point on the chord & surface as the 'solution', ie. instead
      //     of the (potential) true point on the intersection of the curve and the boundary.
      // ( This involves a bias -- typically important only for muons in trackers.
      //   Currently it's controlled/limited by the acceptable step size ie. 'safeLength' )
      double fraction = vecCore::Max(move / chordLen, 0.); // linearStep
      // The actual distance to the boundary is along the arc, so changing it as below would
      // be appropriate, however this won't put the last step on the real boundary which is error-prone
      // move = fraction * safeArc; // curvedStep
#ifndef ENDPOINT_ON_CURVE
      // Primitive approximation of end direction and linearStep to the crossing point ...
      position += move * chordDir; // linearStep
      direction   = direction * (1.0 - fraction) + endDirection * fraction;
      direction   = direction.Unit();
      momentumVec = momentumMag * direction;
      // safeArc is how much the track would have been moved if not hitting the boundary
      // We approximate the actual reduction along the curved trajectory to be the same
      // as the reduction of the full chord due to the boundary crossing.
#else
      // Alternative approximation of end position & direction -- calling RK again
      //  Better accuracy (e.g. for comparing with Helix) but the point will not be on the surface !!
      // bool done =
      RkDriver_t::Advance(position, momentumVec, charge, move, magField, dydx_end, kMaxTrials);

      direction = inv_momentumMag * momentumVec; // requires re-normalization after Advance
      direction.Normalize();
#endif
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

#endif
