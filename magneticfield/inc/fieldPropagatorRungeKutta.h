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

    const vecgeom::Vector3D<Real_t> posBegin= position;    // For print
    const vecgeom::Vector3D<Real_t> momBegin= momentumVec; //   >>
    
    bool done=
       RkDriver_t::Advance( position, momentumVec, charge, lenRemains, magField, hTry, dydx_end,
                            hAdvanced, totalTrials,
                            // id,     // Temporary ?
                            trialsPerCall);
    //   Runge-Kutta single call ( number of steps <= trialsPerCall )
    
    lenRemains -= hAdvanced;
    unfinished = lenRemains > 0.0; /// Was = !done  ... for debugging ????

    totLen+= hAdvanced;
    
    // if( !done || (loopCt++>0) ){
    if( id == 1 ) {
       const vecgeom::Vector3D<Real_t> deltaPos= position - posBegin;
       const vecgeom::Vector3D<Real_t> deltaMomentum= momentumVec - momBegin;
       
       printf(" id %3d call %4d lpCt %2d sum-iters %3d  hdid= %9.5g " //  totLen= %9.5g lenRemains= %9.5g "
              " ret= %1d #=  pos = %9.6g %9.6g %9.6g   momemtumV= %14.9g %14.9g %14.9g  hTry= %7.4g  remains= %7.4g "
              "  Delta-pos= %9.6g %9.6g %9.6g  (mag= %8.6g)  Delta-mom= %9.6g %9.6g %9.6g (mag= %8.6g) "
              " \n",
              id, callNum, loopCt, totalTrials, hAdvanced, // totLen, lenRemains,
              done,
              position[0], position[1], position[2],
              momentumVec[0], momentumVec[1], momentumVec[2],
              hTry, lenRemains
              , deltaPos[0], deltaPos[1], deltaPos[2], deltaPos.Mag()
              , deltaMomentum[0], deltaMomentum[1], deltaMomentum[2], deltaMomentum.Mag()
          );
    }
    // sumAdvanced += hAdvanced;  // Gravy ..

  } while ( unfinished  && (totalTrials < fMaxTrials) );
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
    
    std::cout << "Advanced returned:  done= " << ( done ? "Yes" : " No" )
              << " hAdvanced = " << hAdvanced 
              << " hNext = " << hTry
              << std::endl;
    
    Real_t yPosMom[Nvar] = { Position[0], Position[1], Position[2],
                             momentumVec[0], momentumVec[1], momentumVec[2] } ;    
    sumAdvanced += hAdvanced;
    
    if( verbose ) 
       std::cout << "- Trials:   max/call = " << trialsPerCall << "  total done = " << totalTrials
                 <<  ( (totalTrials == trialsPerCall) ?  " MAXimum reached !! " :  " " )
                 << std::endl;
    else
       std::cout << " t: " << setw(4) << totalTrials << " s= " << setw(9) << sumAdvanced << " ";

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

#define CHECK_STEP  1

#ifdef CHECK_STEP
//  Extra check at each integration that the result agrees with Helix/Bz
#include "ConstBzFieldStepper.h"

static __device__ __host__
bool  CompareResponseVector3D_perStep(
   int id,
   vecgeom::Vector3D<Precision> const & originalVec,
   vecgeom::Vector3D<Precision> const & baselineVec,
   vecgeom::Vector3D<Precision> const & resultVec,   // Output of new method
   const char                   * vecName,    
   Precision                      thresholdRel    // fraction difference allowed
   )
// Returns 'true' if values are 'bad'...

// Copy of method in Example15/electrons.cu  2022.06.27
   
{
   bool bad = false; // Good ..
   Precision magOrig= originalVec.Mag();
   vecgeom::Vector3D<Precision> moveBase = baselineVec-originalVec;
   vecgeom::Vector3D<Precision> moveRes  = resultVec-originalVec;
   Precision magMoveBase = moveBase.Mag();
   Precision magDiffRes  = moveRes.Mag();

   if ( std::fabs( magDiffRes / magMoveBase) - 1.0 > thresholdRel
        || 
        ( resultVec - baselineVec ).Mag() > thresholdRel * magMoveBase 
      ){
      // printf("Difference seen in vector %s : ", vecName );
      printf(" id %3d - Diff in %s: "
             " new-base= %14.9g %14.9g %14.9g (mag= %14.9g) "
             " mv_Res/mv_Base-1 = %7.3g | mv/base:  mag= %9.4g v3= %14.9f %14.9f %14.9f  || mv-new: mag= %9.4g | "
             " || origVec= %14.9f %14.9f %14.9f (mag=%14.9f) || base= %14.9f %14.9f %14.9f (mag=%9.4g) || mv/new:3= %14.9f %14.9f %14.9f (mag = %14.9g)\n",
             id, vecName,
             resultVec[0]-baselineVec[0], resultVec[1]-baselineVec[1], resultVec[2]-baselineVec[2], (resultVec-baselineVec).Mag(),             
             (moveRes.Mag() / moveBase.Mag() - 1.0),
             moveBase.Mag(), moveBase[0], moveBase[1], moveBase[2],
//      printf("   new-original: mag= %20.16g ,  new_vec= %14.9f , %14.9f , %14.9f \n",
             moveRes.Mag(), 
             originalVec[0], originalVec[1], originalVec[2], originalVec.Mag(),
             baselineVec[0], baselineVec[1], baselineVec[2], baselineVec.Mag(),     
             moveRes[0], moveRes[1], moveRes[2], moveRes.Mag() // );
         );      
      bad= true;
   }
   return bad;
};
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
  
#if 1
  Real_t inv_curv = inverseCurvature /*<Real_t>*/ ( momentumVec, B0fieldVec, charge );
#else
  Real_t bmag = B0fieldVec.Mag();
  // Real_t curv = std::fabs(fieldConstants::kB2C * charge * BzValue) / (momentumXYMag + 1.0e-30); // norm for step

  // constexpr Real_t toKilogauss = 1.0 / copcore::units::kilogauss; //  Is this needed ? JA 2021.12.08
  Real_t curvaturePlus =  
     fabs(fieldConstants::kB2C * Real_t(charge) * (bmag  /* * toKiloGauss*/ )) / (momentumMag + Real_t(1.0e-30));

  Real_t ratioOverFld = (bmag>0) ? momentumVec.Dot(B0fieldVec) / (bmag * bmag) : 0.0;
  vecgeom::Vector3D<Real_t> PtransB = momentumVec - ratioOverFld * B0fieldVec;

  // Calculate inverse curvature instead - save a division
  Real_t inv_curv = fabs(    PtransB.Mag() /
                           ( Track::kB2C * charge * bmag + tiny)
                        );
#endif
  // printf( "   B-field = %9.3g (T)   momentum_P = %9.5g (MeV/c)  R = inv_curv = %9.5f (cm) \n",
  //         B0fieldVec.Mag(),   momentumVec.Mag() / copcore::units::MeV ,  inv_curv / copcore::units::millimeter );
  // constexpr Real_t invEpsD= 1.0 / gEpsilonDeflect;
  
  // acceptable lateral error from field ~ related to delta_chord sagital distance
  Real_t safeLength =
     sqrt( Real_t(2.0) * fieldConstants::gEpsilonDeflect * inv_curv); // max length along curve for deflectionn
                                        // = sqrt( 2.0 / ( invEpsD * curv) ); // Candidate for fast inv-sqrt

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
    bool fullChord = false;
    // vecgeom::Vector3D<Real_t> momentumVec = momentumMag * direction;
    const Real_t  inv_momentumMag = 1.0 / momentumMag;
 
    //  Locate the intersection of the curved trajectory and the boundaries of the current
    //    volume (including daughters).
    do {
      vecgeom::Vector3D<Real_t> endPosition    = position;
      vecgeom::Vector3D<Real_t> endMomentumVec = momentumVec; // momentumMag * direction;
      const Real_t safeArc = min(remains, safeLength);
      
      IntegrateTrackToEnd( magField, endPosition, endMomentumVec, charge, safeArc, indx);
      //-----------------
      vecgeom::Vector3D<Real_t> chordVec     = endPosition - position;
      Real_t chordLen = chordVec.Length();      
      vecgeom::Vector3D<Real_t> endDirection = inv_momentumMag * endMomentumVec;      

      chordVec *= (1.0 / chordLen);  // It's not the direction

#ifdef CHECK_STEP
      // Check vs Helix solution -- temporary 2022.06.27      
      vecgeom::Vector3D<Real_t> endPositionHelix  = position;
      vecgeom::Vector3D<Real_t> endDirectionHelix = direction; // momentumMag * direction;      

      // ConstFieldStepper helixBz(B0fieldVec);
      ConstBzFieldStepper helixBz(B0fieldVec[2]); // Bz component -- Assumes that Bx= By = 0 and Bz = const. 
      helixBz.DoStep<vecgeom::Vector3D<Real_t>, Real_t, int>(position, direction, charge, momentumMag, safeArc,
                                                        endPositionHelix, endDirectionHelix);

      constexpr Precision thesholdDiff=3.0e-5;
      bool badPosition = 
        CompareResponseVector3D( indx, position, endPositionHelix, endPosition, "Position-perStep", thesholdDiff );
      bool badDirection =
        CompareResponseVector3D( indx, direction, endDirectionHelix, endMomentumVec.Unit(), "Direction-perStep", thesholdDiff );

      const char* Outcome[2]={ "Good", " Bad" };
      printf("%4s oneStep-Check track (id= %3d)  e_kin= %8.4g stepLen= %7.3g chord-iter= %5d\n ", Outcome[badPosition||badDirection],
               indx, kinE, safeArc, chordIters);
      if( badPosition || badDirection) {        
        // currentTrack.print(indx, /* verbose= */ true );
      }
#endif
      // Check Intersection
      
      Real_t linearStep = Navigator_t::ComputeStepAndNextVolume(position, chordVec, chordLen, current_state, next_state, kPush);
      Real_t curvedStep;

      fullChord = (linearStep == chordLen);
      if (fullChord) {
        position   = endPosition;
        direction  = endDirection;
        curvedStep = safeArc;
      } else {
        assert( next_state.IsOnBoundary() );
        // assert( linearStep == chordLen );
        
        // USE the intersection point on the chord & surface as the 'solution', ie. instead
        //     of the (potential) true point on the intersection of the curve and the boundary.
        // ( This involves a bias -- typically important only for muons in trackers.
        //   Currently it's controlled/limited by the acceptable step size ie. 'safeLength' )
        position = position + linearStep * chordVec;

        // Primitive approximation of end direction and linearStep to the crossing point ...
        Real_t fraction = chordLen > 0 ? linearStep / chordLen : 0.0;
        direction       = direction * (1.0 - fraction) + endDirection * fraction;
        direction       = direction.Unit();
        // safeArc is how much the track would have been moved if not hitting the boundary
        // We approximate the actual reduction along the curved trajectory to be the same
        // as the reduction of the full chord due to the boundary crossing.
        curvedStep = fraction * safeArc;
      }

      // if( idx == 1 ) {
      //    printf(" 2end: dir-dir0 = %8.5g %8.5g %8.5g\n", direction -   ); 
      // }
      
      stepDone += curvedStep;
      remains -= curvedStep;
      chordIters++;

      found_end = next_state.IsOnBoundary() || (remains <= tiniest_step);
      
    } while ( !found_end && (chordIters < max_iterations) );

    // } while ((!next_state.IsOnBoundary()) && fullChord && (remains > tiniest_step) && (chordIters < max_iterations));
  }

  propagated = found_end;
  itersDone = chordIters;
       //  = (chordIters < max_iterations);  // ---> Misses success on the last step!
  return stepDone;
}

#endif
