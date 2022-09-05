// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author: J. Apostolakis  15 Nov 2021

#ifndef FIELD_PROPAGATOR_RUNGEKUTTA_H
#define FIELD_PROPAGATOR_RUNGEKUTTA_H

#include <VecGeom/base/Vector3D.h>

#include "fieldConstants.h"         // For kB2C factor with units

#include "UniformMagneticField.h"
#include "RkIntegrationDriver.h"

template<class Field_t, class RkDriver_t, typename Real_t, class Navigator>
class fieldPropagatorRungeKutta {
public:
  static
  inline __host__ __device__
  void stepInField( Field_t const & magneticField,     
                    Real_t kinE,
                    Real_t mass,
                    int charge,
                    Real_t step,
                    vecgeom::Vector3D<Real_t> &position,
                    vecgeom::Vector3D<Real_t> &direction
                    , int id // For debugging
     );

  static   
  inline __host__ __device__
  __host__ __device__ Real_t ComputeStepAndNextVolume(
      Field_t const & magneticField,
      double kinE, double mass, int charge, double physicsStep,
      vecgeom::Vector3D<Real_t> &position,
      vecgeom::Vector3D<Real_t> &direction,
      vecgeom::NavStateIndex const &current_state,
      vecgeom::NavStateIndex &next_state,
      bool         &  propagated,
      const Real_t &  /*safety*/,
      const int max_iterations
      , int & iterDone
      , int   threadId
     );
   // Move the track,
   //   updating 'position', 'direction', the next state and returning the length moved.
protected:
   static inline __host__ __device__
   void IntegrateTrackToEnd( Field_t const & magField,  // RkDriver_t &integrationDriverRK,
                             vecgeom::Vector3D<Real_t> & position,  vecgeom::Vector3D<Real_t> & momentumVec,
                             int charge,  Real_t stepLength
                             , int  id               // Temporary - for debugging
                             , bool verbose = true   //    >>2
      );

   static inline __host__ __device__
   bool IntegrateTrackForProgress( Field_t const & magField,  // RkDriver_t &integrationDriverRK,
                                   vecgeom::Vector3D<Real_t> & position,
                                   vecgeom::Vector3D<Real_t> & momentumVec,
                                   int charge, 
                                   Real_t & stepLength          //   In/Out - In = requested; Out = last trial / next value ??
      // unsigned int & totalTrials
      );
   
  static constexpr unsigned int fMaxTrials= 100;
  static constexpr unsigned int Nvar = 6;       // For position (3) and momentum (3) -- invariant

#ifdef VECGEOM_FLOAT_PRECISION
  static constexpr Real_t kPush = 10 * vecgeom::kTolerance;
#else
  static constexpr Real_t kPush = 0.;
#endif
   
  // Cannot change the energy (or momentum magnitude) -- currently usable only for pure magnetic fields
};

#define VERBOSE_STEP_IN_THREAD 1    // 2022.09.05  14:00 -- look for failed RK integration hAdvanced = 0.0

// ----------------------------------------------------------------------------
template<class Field_t, class RkDriver_t, typename Real_t, class Navigator_t>
inline __host__ __device__  // __host__ __device_
void fieldPropagatorRungeKutta<Field_t, RkDriver_t, Real_t, Navigator_t>
   ::stepInField( Field_t const & magField,  // RkDriver_t &integrationDriverRK,
                  Real_t kinE,
                  Real_t mass,
                  int charge,
                  Real_t step,
                  vecgeom::Vector3D<Real_t> &position,
                  vecgeom::Vector3D<Real_t> &direction
                  , int id // For debugging
      )
{
  Real_t momentumMag = sqrt(kinE * (kinE + 2.0 * mass));
  Real_t invMomentumMag = 1.0 / momentumMag;
   
  // Only charged particles ( e-, e+, any .. ) can be propagated 
  vecgeom::Vector3D<Real_t> positionVec = position;
  vecgeom::Vector3D<Real_t> momentumVec = momentumMag * direction;
  IntegrateTrackToEnd( magField,
                       positionVec,
                       momentumVec,
                       charge,
                       step
                       , id, true  // For debugging
     );
  position  = positionVec;
  direction = invMomentumMag * momentumVec;
  // Deviation of magnitude of direction from unit indicates integration error
}

// ----------------------------------------------------------------------------

template<class Field_t, class RkDriver_t, typename Real_t, class Navigator_t>
inline __host__ __device__
void fieldPropagatorRungeKutta<Field_t, RkDriver_t, Real_t, Navigator_t>::IntegrateTrackToEnd(
      // RkDriver_t & integrationDriverRK,
      Field_t const & magField,
      vecgeom::Vector3D<Real_t> & position,
      vecgeom::Vector3D<Real_t> & momentumVec,
      int charge,
      Real_t stepLength
      , int  id
      , bool verbose
   )
{
  // Version 1.  Finish the integration of lanes ...
  // Future alternative (ToDo):  return unfinished intergration, in order to interleave loading of other 'lanes'

  const unsigned int trialsPerCall = vecCore::Min( 30U, fMaxTrials / 2) ;  // Parameter that can be tuned
  unsigned int totalTrials=0;
  static int callNum = 0; 
  callNum ++;

  Real_t  lenRemains = stepLength;

  Real_t  hTry = stepLength;    // suggested 'good' length per integration step
  bool unfinished = true;

  Real_t  totLen = 0.0;
  unsigned int loopCt=0;
  do {
    Real_t hAdvanced = 0;     //  length integrated this iteration (of do-while)
    Real_t  dydx_end[Nvar];
    
#if VERBOSE_STEP_IN_THREAD
    const vecgeom::Vector3D<Real_t> posBegin= position;    // For print
    const vecgeom::Vector3D<Real_t> momBegin= momentumVec; //   >>
#endif

    bool done=
       RkDriver_t::Advance( position, momentumVec, charge, lenRemains, magField, hTry, dydx_end,
                            hAdvanced, totalTrials,
                            // id,     // Temporary ?
                            trialsPerCall);
    //   Runge-Kutta single call ( number of steps <= trialsPerCall )

    lenRemains -= hAdvanced;
    unfinished = lenRemains > 0.0; /// Was = !done  ... for debugging ????

    totLen+= hAdvanced;
    loopCt++;

// #define  VERBOSE_STEP_IN_THREAD  1
#if VERBOSE_STEP_IN_THREAD
    if( hAdvanced == 0.0 ) {
       const vecgeom::Vector3D<Real_t> deltaPos= position - posBegin;
       const vecgeom::Vector3D<Real_t> deltaMomentum= momentumVec - momBegin;
       
       printf("-fpRK-loop %s, id %3d call %4d lpCt %2d sum-iters %3d  hdid= %9.5g totLen= %9.5g lenRemains= %9.5g "
              " ret= %1d #=  pos = %9.6g %9.6g %9.6g   momemtumV= %14.9g %14.9g %14.9g  hTry= %7.4g  remains= %7.4g "
              "  Delta-pos= %9.6g %9.6g %9.6g  (mag= %8.6g)  Delta-mom= %9.6g %9.6g %9.6g (mag= %8.6g) "
              " \n",
              "hAdvanced = ZERO", //    Reflect if( .. ) condition above !!
              id, callNum, loopCt, totalTrials, hAdvanced, totLen, lenRemains,
              done,
              position[0], position[1], position[2],
              momentumVec[0], momentumVec[1], momentumVec[2],
              hTry, lenRemains
              , deltaPos[0], deltaPos[1], deltaPos[2], deltaPos.Mag()
              , deltaMomentum[0], deltaMomentum[1], deltaMomentum[2], deltaMomentum.Mag()
          );
    }
#endif
    // sumAdvanced += hAdvanced;  // Gravy ..

  } while ( unfinished  && (totalTrials < fMaxTrials) );

  
  if( loopCt > 1 ) { printf( " fieldPropagatorRK: id %3d call %4d --- LoopCt reached %d ", id, callNum, loopCt );  }
     
}

#if 0
template<class Field_t, class RkDriver_t, typename Real_t, class Navigator_t>
inline __host__ __device__
bool fieldPropagatorRungeKutta<Field_t, RkDriver_t, Real_t, Navigator_t>::IntegrateOneStep(
      Field_t const & magField,
      vecgeom::Vector3D<Real_t> & position,
      vecgeom::Vector3D<Real_t> & momentumVec,
      int       charge,
      Real_t  & lenRemains,
      Real_t  & hTry,          //  Trial step size
      Real_t  & hAdvanced,     //  Length advanced
      int     & totalTrials
   )
//  Return whether full length was completed
{
  Real_t  hTry = stepLength;    // suggested 'good' length per integration step
  bool unfinished = true;

  hAdvanced = 0;           //  length integrated this iteration
  Real_t  dydx_end[Nvar];
    
  bool done=
       RkDriver_t::Advance( position, momentumVec, charge, lenRemains, magField, hTry, dydx_end,
                            hAdvanced, totalTrials, trialsPerCall);
    //   Runge-Kutta single call ( number of steps <= trialsPerCall )
    
  lenRemains -= hAdvanced;
  return done;
}
#endif

// ----------------------------------------------------------------------------

template<class Field_t, class RkDriver_t, typename Real_t, class Navigator_t>
inline __host__ __device__
bool fieldPropagatorRungeKutta<Field_t, RkDriver_t, Real_t, Navigator_t>::IntegrateTrackForProgress(
      Field_t const & magField, 
      vecgeom::Vector3D<Real_t> & position,
      vecgeom::Vector3D<Real_t> & momentumVec,
      int charge,
      Real_t & stepLength          //   In/Out - In = requested; Out = last trial / next value ??
      // , unsigned int & totalTrials
   )
{
  // Version 2.  Try to get some progress in the integration of this threads - but not more than trialsPerCall ...
  // Future alternative (ToDo):  return unfinished intergration, in order to interleave loading of other 'lanes'
  const unsigned int trialsPerCall = vecCore::Min( 6U, fMaxTrials / 2) ;  // Parameter that can be tuned
  
  Real_t  lenRemains = stepLength;

  Real_t  hTry = stepLength;    // suggested 'good' length per integration step
  bool    done= false;
  // bool unfinished = true;

  int     totalTrials= 0;
  Real_t  hAdvanced = 0;     //  length integrated this iteration (of do-while)  
  do {

    Real_t  dydx_end[Nvar];
    
    done= RkDriver_t::Advance( position, momentumVec, charge, lenRemains, magField, 
                               hTry, dydx_end, hAdvanced, totalTrials, trialsPerCall);
    //   Runge-Kutta one call for 1+ iterations ( number of steps <= trialsPerCall )
    
    stepLength -= hAdvanced;
    // unfinished = !done;  // (lenRemains > 0.0);

  } while ( hAdvanced == 0.0 && totalTrials < fMaxTrials );

  return done; 
}


/********************
ComputeIntersection
{
  done =     
     Driver_t::Advance( Position, momentumVec, charge, hLength,
                          magField, hTry, dydx_end,
                          hAdvanced, totalTrials, trialsPerCall);
    // Runge-Kutta single call
    
    Real_t yPosMom[Nvar] = { Position[0], Position[1], Position[2],
                             momentumVec[0], momentumVec[1], momentumVec[2] } ;    
    sumAdvanced += hAdvanced;
    
    Vector3D<float> magFieldEnd;
    Equation_t::EvaluateDerivativesReturnB(magField, yPosMom, charge, dy_ds, magFieldEnd);  
    Real_t magFieldEndArr[3] = { magFieldEnd[0],   magFieldEnd[1],   magFieldEnd[2] };

    //--- PrintFieldVectors::PrintSixvecAndDyDx(
    PrintFieldVectors::PrintLineSixvecDyDx( yPosMom, charge, magFieldEndArr, // dydx_end );
                                            dy_ds);
    
    hLength -= hAdvanced;
    done = (hLength <= 0.0);

  } while ( ! done );
   
}
**********/

template<typename Real_t>
inline __host__ __device__ Real_t
inverseCurvature(
    vecgeom::Vector3D<Real_t> & momentumVec,
    vecgeom::Vector3D<Real_t> & BfieldVec,
    int charge
   )
{
  Real_t bmag2 = BfieldVec.Mag2();
  Real_t ratioOverFld = (bmag2>0) ? momentumVec.Dot(BfieldVec) / bmag2 : 0.0;
  vecgeom::Vector3D<Real_t> PtransB = momentumVec - ratioOverFld * BfieldVec;
  
  Real_t bmag  = sqrt(bmag2);
  
  // Real_t curv = fabs(Track::kB2C * charge * bmag / ( PtransB.Mag() + tiny));

  // Calculate inverse curvature instead - save a division
  Real_t inv_curv = fabs( PtransB.Mag()
                          / ( fieldConstants::kB2C * Real_t(charge) * bmag + 1.0e-30)
     );
  return inv_curv;
}

// Determine the step along curved trajectory for charged particles in a field.
//  ( Same name as as navigator method. )

#define CHECK_EVERY_SUBSTEP  1

#ifdef CHECK_EVERY_SUBSTEP
//  Extra check at each integration that the result agrees with Helix/Bz
#include "ConstBzFieldStepper.h"
#include "ConstFieldHelixStepper.h"

#include "CompareResponses.h" 
#endif

template<class Field_t, class RkDriver_t, typename Real_t, class Navigator_t>
inline __host__ __device__ Real_t
fieldPropagatorRungeKutta<Field_t, RkDriver_t, Real_t, Navigator_t> ::ComputeStepAndNextVolume(
    Field_t const & magField,
    double kinE, double mass, int charge, double physicsStep,
    vecgeom::Vector3D<Real_t> & position,
    vecgeom::Vector3D<Real_t> & direction,
    vecgeom::NavStateIndex const & current_state,
    vecgeom::NavStateIndex       & next_state,
    bool         & propagated,
    const Real_t & /*safety*/,  //  eventually In/Out ?
    const int max_iterations
    , int & itersDone           //  useful for now - to monitor and report -- unclear if needed later
    , int   indx
   )
{
  using copcore::units::MeV;
  
  const Real_t momentumMag   = sqrt(kinE * (kinE + 2.0 * mass));
  vecgeom::Vector3D<Real_t> momentumVec = momentumMag * direction;
  /** Printf( " momentum Mag = %9.3g MeV/c , from kinE = %9.4g MeV , mass = %5.3g MeV - check E^2-p^2-m0^2= %7.3g MeV^2\n",
          momentumMag / MeV, kinE / MeV, mass / MeV,
          ( 2 * mass * kinE + kinE * kinE - momentumMag * momentumMag ) / (MeV * MeV) );  **/

  vecgeom::Vector3D<Real_t>  B0fieldVec = { 0.0, 0.0, 0.0 };  // Field value at starting point
  magField.Evaluate( position, B0fieldVec );
  
  Real_t inv_curv = inverseCurvature /*<Real_t>*/ ( momentumVec, B0fieldVec, charge );
  // printf( "   B-field = %9.3g (T)   momentum_P = %9.5g (MeV/c)  R = inv_curv = %9.5f (cm) \n",
  //         B0fieldVec.Mag(),   momentumVec.Mag() / copcore::units::MeV ,  inv_curv / copcore::units::millimeter );
  // constexpr Real_t invEpsD= 1.0 / gEpsilonDeflect;
  
  // acceptable lateral error from field ~ related to delta_chord sagital distance
  const Real_t safeLength =
     sqrt( Real_t(2.0) * fieldConstants::gEpsilonDeflect * inv_curv); // max length along curve for deflectionn
                                        // = sqrt( 2.0 / ( invEpsD * curv) ); // Candidate for fast inv-sqrt

  Precision maxNextSafeMove = safeLength; // It can be reduced if, at the start, a boundary is encountered
  
  Real_t stepDone           = 0.0;
  Real_t remains            = physicsStep;
  const Real_t tiniest_step = 1.0e-7 * physicsStep; // Ignore remainder if < e_s * PhysicsStep
  int chordIters            = 0;

  constexpr bool inZeroFieldRegion= false; // This could be a per-region flag ... - better depend on template parameter?
  bool found_end = false;

  if ( inZeroFieldRegion ) {
    stepDone = Navigator_t::ComputeStepAndNextVolume(position, direction, remains, current_state, next_state, kPush);
    position += stepDone * direction;
  } else {
    bool continueIteration = false;     
    bool fullChord = false;
    // vecgeom::Vector3D<Real_t> momentumVec = momentumMag * direction;
    const Real_t  inv_momentumMag = 1.0 / momentumMag;

    bool   lastWasZero = false;   // Debug only ?  JA 2022.09.05
    
    //  Locate the intersection of the curved trajectory and the boundaries of the current
    //    volume (including daughters).
    do {
      static constexpr Precision ReduceFactor = 0.5;
      static constexpr int       ReduceIters  = 5;

      vecgeom::Vector3D<Real_t> endPosition    = position;
      vecgeom::Vector3D<Real_t> endMomentumVec = momentumVec; // momentumMag * direction;
      const Real_t safeArc = min(remains, maxNextSafeMove); // safeLength);
      
      IntegrateTrackToEnd( magField, endPosition, endMomentumVec, charge, safeArc, indx);
      //-----------------
#ifdef CHECK_EVERY_SUBSTEP
      if( safeArc < 1.0e-03 * physicsStep && !lastWasZero ) {
         printf( "fpConstRK WARNING> very small safeMove = %10.5g  - vs physicsStep= %10.5g  \n", safeArc, physicsStep );
         printf("%4s oneStep-Check track (id= %3d)  e_kin= %8.4g stepLen= %12.9g (safeLen= %10.7g) chord-iter= %5d\n ",
                "Short", indx, kinE, safeArc, safeLength, chordIters);
      }
#endif
      vecgeom::Vector3D<Real_t> chordVec     = endPosition - position;
      Real_t chordDist = chordVec.Length();     
      vecgeom::Vector3D<Real_t> endDirection = inv_momentumMag * endMomentumVec;
      chordVec *= (1.0 / chordDist);  // Now the direction of the chord!

#ifdef CHECK_EVERY_SUBSTEP
      // Check vs Helix solution -- temporary 2022.06.27      
      vecgeom::Vector3D<Real_t> endPositionHelix  = position;
      vecgeom::Vector3D<Real_t> endDirectionHelix = direction; // momentumMag * direction;      

      ConstFieldHelixStepper   helix(B0fieldVec);
      // ConstBzFieldStepper helix(B0fieldVec[2]); // Bz component -- Assumes that Bx= By = 0 and Bz = const.
      // helix.DoStep<vecgeom::Vector3D<Real_t>, Real_t, int>(...
      helix.DoStep(position, direction, charge, momentumMag, safeArc, endPositionHelix, endDirectionHelix);

      constexpr Precision thesholdDiff=3.0e-5;
      bool badPosition = 
        CompareResponseVector3D( indx, position, endPositionHelix, endPosition, "Position-perStep", thesholdDiff );
      bool badDirection =
        CompareResponseVector3D( indx, direction, endDirectionHelix, endDirection, "Direction-perStep", thesholdDiff );

      if( badPosition || badDirection) {  
         printf("%4s oneStep-Check track (id= %3d)  e_kin= %8.4g stepLen= %12.9g chord-iter= %5d\n ",
             "Bad", indx, kinE, safeArc, chordIters);         
      }
#endif
      // Check Intersection
      //-- vecgeom::Vector3D<Real_t> ChordDir= (1.0/chordDist) * ChordVec;
      Real_t linearStep = Navigator_t::ComputeStepAndNextVolume(position, chordVec, chordDist, current_state, next_state, kPush);
      Real_t curvedStep;

      if( lastWasZero && chordIters >= ReduceIters ) {          
         // printf( "-fieldProp-RK: LastWasZero> stepDone= %10.5g  - vs chordDist= %10.5g  \n", linearStep, chordDist );
         lastWasZero = false;         
      }

      fullChord = (linearStep == chordDist);
      if (fullChord) {
        position    = endPosition;
        momentumVec = endMomentumVec;
        
        direction  = endDirection;
        curvedStep = safeArc;

        maxNextSafeMove   = safeArc;  // Reset it, once a step succeeds!!
        continueIteration = true;        
      } else if (linearStep <= kPush + Navigator_t::kBoundaryPush && stepDone == 0) {
        // Cope with a track at a boundary that wants to bend back into the previous
        //   volume in the first step (by reducing the attempted distance.)
        // FIXME: Even for zero steps, the Navigator will return kPush + possibly
        // Navigator::kBoundaryPush instead of a real 0.
        curvedStep = 0;
        lastWasZero = true;
        
        // Reduce the step attempted in the next iteration to navigate around
        // boundaries where the chord step may end in a volume we just left.
        maxNextSafeMove   = ReduceFactor * safeArc;
        continueIteration = chordIters < ReduceIters;

      } else {
        assert( next_state.IsOnBoundary() );
        // assert( linearStep == chordDist );
        
        // USE the intersection point on the chord & surface as the 'solution', ie. instead
        //     of the (potential) true point on the intersection of the curve and the boundary.
        // ( This involves a bias -- typically important only for muons in trackers.
        //   Currently it's controlled/limited by the acceptable step size ie. 'safeLength' )
        Real_t fraction = chordDist > 0 ? linearStep / chordDist : 0.0;
        curvedStep = fraction * safeArc;        
#ifndef ENDPOINT_ON_CURVE
        // Primitive approximation of end direction and linearStep to the crossing point ...        
        position = position + linearStep * chordVec;        
        direction       = direction * (1.0 - fraction) + endDirection * fraction;
        direction       = direction.Unit();
        momentumVec     = momentumMag * direction;
        // safeArc is how much the track would have been moved if not hitting the boundary
        // We approximate the actual reduction along the curved trajectory to be the same
        // as the reduction of the full chord due to the boundary crossing.
#else
        // Alternative approximation of end position & direction -- calling RK again
        //  Better accuracy (e.g. for comparing with Helix) -- but the point will not be on the surface !!
        IntegrateTrackToEnd( magField, position, momentumVec, charge, curvedStep, indx);
        direction = inv_momentumMag * momentumVec;   // momentumVec.Unit();
#endif
        continueIteration = false;      
      }

      stepDone += curvedStep;
      remains -= curvedStep;
      chordIters++;

      found_end = (  (curvedStep > 0) && next_state.IsOnBoundary() )      // Fix 2022.09.05 JA
               || (remains <= tiniest_step);

#ifdef CHECK_EVERY_SUBSTEP
      // if( ! badPosition && ! badDirection)

      if( stepDone < 1.0e-03 * physicsStep && !lastWasZero ) {
         printf("--fpRK Small step done - id %3d call= %-2d lpCt= %2d "  // "sum %3d "               //  5 int args
                " kinE= %8.6g  safeL= %-8.4g tried= %-8.4g kept= %8.4g totL= %8.5g remain= %6.3g "    //  4 float args
                "  endPos = %-9.6g %9.6g %9.6g  "
                "  endDir= %14.9g %14.9g %14.9g  "
                "  endMom= %14.9g %14.9g %14.9g \n",
                indx, itersDone+1, chordIters-1, // itersDone+chordIters, 
                kinE, safeLength, safeArc, curvedStep, stepDone, remains,
                endPosition[0], endPosition[1], endPosition[2],                
                endDirection[0], endDirection[1], endDirection[2],                 
                endMomentumVec[0], endMomentumVec[1], endMomentumVec[2] );
      }
#endif
    } while ( !found_end && continueIteration && (chordIters < max_iterations) );
  }

  propagated = found_end;
  itersDone += chordIters;
       //  = (chordIters < max_iterations);  // ---> Misses success on the last step!
  return stepDone;
}

#endif
