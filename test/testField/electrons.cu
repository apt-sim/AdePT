// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include "testField.cuh"

#include <AdePT/navigation/AdePTNavigator.h>

#include <AdePT/copcore/PhysicalConstants.h>

// Classes for Runge-Kutta integration
#include <AdePT/magneticfield/MagneticFieldEquation.h>
#include <AdePT/magneticfield/DormandPrinceRK45.h>
#include <AdePT/magneticfield/fieldPropagatorRungeKutta.h>
#include <AdePT/magneticfield/fieldPropagatorConstBz.h>

#include <G4HepEmElectronManager.hh>
#include <G4HepEmElectronTrack.hh>
#include <G4HepEmElectronInteractionBrem.hh>
#include <G4HepEmElectronInteractionIoni.hh>
#include <G4HepEmElectronInteractionUMSC.hh>
#include <G4HepEmPositronInteractionAnnihilation.hh>
// Pull in implementation.
#include <G4HepEmRunUtils.icc>
#include <G4HepEmInteractionUtils.icc>
#include <G4HepEmElectronManager.icc>
#include <G4HepEmElectronInteractionBrem.icc>
#include <G4HepEmElectronInteractionIoni.icc>
#include <G4HepEmElectronInteractionUMSC.icc>
#include <G4HepEmPositronInteractionAnnihilation.icc>

#ifdef CHECK_RESULTS
#include "CompareResponses.h"
#endif

__device__ float *gPtrBzFieldValue_dev = nullptr;

// Transfer pointer to memory address of BzFieldValue_dev to device ...
//
__global__ void SetBzFieldPtr(float *pBzFieldValue_dev)
{
  gPtrBzFieldValue_dev = pBzFieldValue_dev;
}

// Compute the physics and geometry step limit, transport the electrons while
// applying the continuous effects and maybe a discrete process that could
// generate secondaries.
template <bool IsElectron>
static __device__ __forceinline__ void TransportElectrons(Track *electrons, const adept::MParray *active,
                                                          Secondaries &secondaries, adept::MParray *activeQueue,
                                                          GlobalScoring *globalScoring,
                                                          ScoringPerVolume *scoringPerVolume)
{
#ifdef VECGEOM_FLOAT_PRECISION
  const Precision kPush = 10 * vecgeom::kTolerance;
#else
  const Precision kPush = 0.;
#endif
  constexpr int Charge  = IsElectron ? -1 : 1;
  constexpr double Mass = copcore::units::kElectronMassC2;

  constexpr int Nvar   = 6;
  using Field_t        = UniformMagneticField; // ToDO:  Change to non-uniform type !!
  using Equation_t     = MagneticFieldEquation<Field_t>;
  using Stepper_t      = DormandPrinceRK45<Equation_t, Field_t, Nvar, vecgeom::Precision>;
  using DoPri5Driver_t = RkIntegrationDriver<Stepper_t, vecgeom::Precision, int, Equation_t, Field_t>;

  Field_t magField(vecgeom::Vector3D<float>(0.0, 0.0, *gPtrBzFieldValue_dev));
  // 2.0*copcore::units::tesla) ); // -> Obtain it from object ?

#ifdef REPORT_OPTION
  static bool ReportOption   = true;
  static const char *RunType = "Runge-Kutta field propagation";
  if (ReportOption && blockIdx.x == 0 && threadIdx.x == 0) {
    printf("-- Run type: %s .\n\n", RunType);
    ReportOption = false;
  }
#endif

  // DoPri5Driver_t
  //  Static method fieldPropagatorRungeKutta<DoPri5Driver_t, vecgeom::Precision>
  //     no object fieldPropagatorRK()

  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    Track &currentTrack = electrons[slot];
    auto energy         = currentTrack.energy;
    auto pos            = currentTrack.pos;
    auto dir            = currentTrack.dir;
    auto navState       = currentTrack.navState;
#ifndef ADEPT_USE_SURF
    const int volumeID  = navState.Top()->GetLogicalVolume()->id();
#else
    const int volumeID  = navState.GetLogicalId();
#endif

    // the MCC vector is indexed by the logical volume id
    const int theMCIndex = MCIndex[volumeID];

    auto survive = [&] {
      currentTrack.energy   = energy;
      currentTrack.pos      = pos;
      currentTrack.dir      = dir;
      currentTrack.navState = navState;
      activeQueue->push_back(slot);
    };

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmElectronTrack elTrack;
    G4HepEmTrack *theTrack = elTrack.GetTrack();
    theTrack->SetEKin(energy);
    theTrack->SetMCIndex(theMCIndex);
    theTrack->SetOnBoundary(navState.IsOnBoundary());
    theTrack->SetCharge(Charge);
    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    mscData->fIsFirstStep        = currentTrack.initialRange < 0;
    mscData->fInitialRange       = currentTrack.initialRange;
    mscData->fDynamicRangeFactor = currentTrack.dynamicRangeFactor;
    mscData->fTlimitMin          = currentTrack.tlimitMin;

    // Initial value of magnetic field
    // Equation_t::EvaluateDerivativesReturnB(magField, pos,
    //                                       , momentum* dir,
    //                                       , charge, dy_ds, magFieldStart);

    // Prepare a branched RNG state while threads are synchronized. Even if not
    // used, this provides a fresh round of random numbers and reduces thread
    // divergence because the RNG state doesn't need to be advanced later.
    RanluxppDouble newRNG(currentTrack.rngState.BranchNoAdvance());

    // Compute safety, needed for MSC step limit.
    double safety = 0;
    if (!navState.IsOnBoundary()) {
      safety = AdePTNavigator::ComputeSafety(pos, navState);
    }
    theTrack->SetSafety(safety);

    G4HepEmRandomEngine rnge(&currentTrack.rngState);

    // Sample the `number-of-interaction-left` and put it into the track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft = currentTrack.numIALeft[ip];
      if (numIALeft <= 0) {
        numIALeft = -std::log(currentTrack.Uniform());
      }
      theTrack->SetNumIALeft(numIALeft, ip);
    }

    // Call G4HepEm to compute the physics step limit.
    G4HepEmElectronManager::HowFar(&g4HepEmData, &g4HepEmPars, &elTrack, &rnge);

    // Remember MSC values for the next step(s).
    currentTrack.initialRange       = mscData->fInitialRange;
    currentTrack.dynamicRangeFactor = mscData->fDynamicRangeFactor;
    currentTrack.tlimitMin          = mscData->fTlimitMin;

    // Get result into variables.
    double geometricalStepLengthFromPhysics = theTrack->GetGStepLength();
    // The phyiscal step length is the amount that the particle experiences
    // which might be longer than the geometrical step length due to MSC. As
    // long as we call PerformContinuous in the same kernel we don't need to
    // care, but we need to make this available when splitting the operations.
    // double physicalStepLength = elTrack.GetPStepLength();
    int winnerProcessIndex = theTrack->GetWinnerProcessIndex();
    // Leave the range and MFP inside the G4HepEmTrack. If we split kernels, we
    // also need to carry them over!

    // Check if there's a volume boundary in between.
    bool propagated = true;
    long hitsurf_index = -1;
    double geometryStepLength;
    vecgeom::NavigationState nextState;

    float BzFieldValue = *gPtrBzFieldValue_dev; // Use vecgeom::Precision ?
    if (BzFieldValue != 0.0) {
      UniformMagneticField magneticFieldB(vecgeom::Vector3D<float>(0.0, 0.0, BzFieldValue));
      // using fieldPropagatorRK = fieldPropagatorRungetKutta<RkDriver_t, Precision>;

      // Set up the Integration using Runge-Kutta DoPri5 method
      //
      constexpr unsigned int Nvar = 6; // Number of integration variables
      using Field_t               = UniformMagneticField;
      using Equation_t            = MagneticFieldEquation<Field_t>;
      using Stepper_t             = DormandPrinceRK45<Equation_t, Field_t, Nvar, Precision>;
      using RkDriver_t            = RkIntegrationDriver<Stepper_t, Precision, int, Equation_t, Field_t>;

      constexpr int max_iterations = 10;

#ifdef CHECK_RESULTS
      // Store starting values
      const vecgeom::Vector3D<Precision> startPosition  = pos;
      const vecgeom::Vector3D<Precision> startDirection = dir;

      // Start Baseline reply
      vecgeom::Vector3D<Precision> positionHx  = startPosition;
      vecgeom::Vector3D<Precision> directionHx = startDirection;
      vecgeom::NavigationState nextStateHx;
      bool propagatedHx;

      fieldPropagatorConstBz fieldPropagatorBz(BzFieldValue);
      Precision helixStepLength = fieldPropagatorBz.ComputeStepAndNextVolume<AdePTNavigator>(
          energy, Mass, Charge, geometricalStepLengthFromPhysics, positionHx, directionHx, navState, nextStateHx,
          hitsurf_index, propagatedHx, safety, max_iterations);
      // activeSize < 100 ? max_iterations : max_iters_tail );
      // End   Baseline reply
#endif
      int iterDone = -1;
      geometryStepLength =
          fieldPropagatorRungeKutta<Field_t, RkDriver_t, Precision, AdePTNavigator>::ComputeStepAndNextVolume(
              magneticFieldB, energy, Mass, Charge, geometricalStepLengthFromPhysics, pos, dir, navState, nextState,
              hitsurf_index, propagated, /*lengthDone,*/ safety,
              // activeSize < 100 ? max_iterations : max_iters_tail ), // Was
              max_iterations, iterDone, slot);
#ifdef CHECK_RESULTS
#define formatBool(b) ((b) ? "yes " : "no")
      constexpr Precision thresholdDiff = 3.0e-3;
      bool diffLength = false, badPosition = false, badDirection = false;
      vecgeom::NavigationState &currNavState = navState;
      bool sameLevel                         = nextState.GetLevel() == nextStateHx.GetLevel();
      bool sameIndex                         = nextState.GetNavIndex() == nextStateHx.GetNavIndex();

      if (std::fabs(helixStepLength - geometryStepLength) > 1.0e-4 * helixStepLength) {
        bool sameNextVol = (nextState.GetLevel() == nextStateHx.GetLevel()) &&
                           (nextStateHx.GetNavIndex() == nextStateHx.GetNavIndex()) &&
                           (nextState.IsOnBoundary() == nextStateHx.IsOnBoundary());
        printf(
            "\ns-len diff: id= %3d kinE= %12.7g phys-request= %11.6g  helix-did= %11.6g rk-did= %11.6g (l-diff= %7.4g)"
            "  -- NavStates (curr/next RK/next Hx) :  Levels %1d %1d %1d  NavIdx: %5u %5u %5u  OnBoundary: %3s %3s %3s "
            "Agree? %9s \n",
            slot, energy, geometricalStepLengthFromPhysics, helixStepLength, geometryStepLength,
            geometryStepLength - helixStepLength, navState.GetLevel(), nextState.GetLevel(), nextStateHx.GetLevel(),
            navState.GetNavIndex(), nextState.GetNavIndex(), nextStateHx.GetNavIndex(),
            // navState.IsOnBoundary()),  nextState.IsOnBoundary(),  nextStateHx.IsOnBoundary(),
            formatBool(navState.IsOnBoundary()), formatBool(nextState.IsOnBoundary()),
            formatBool(nextStateHx.IsOnBoundary()), (sameNextVol ? "-Same-" : "-NotSame-"));
        diffLength = true;
      } else {
        badPosition  = CompareResponseVector3D(slot, startPosition, positionHx, pos, "Position", thresholdDiff);
        badDirection = CompareResponseVector3D(slot, startDirection, directionHx, dir, "Direction", thresholdDiff);
      }
      const char *Outcome[2] = {"Good", " Bad"};
      bool problem           = diffLength || badPosition || badDirection;
      if (problem) {
        printf("%4s track (id= %3d)  e_kin= %8.4g stepReq= %9.5g (did: RK= %9.5g vs hlx= %9.5g , diff= %9.5g) iters= "
               "%5d\n ",
               Outcome[problem], // [diffLength||badPosition||badDirection],
               slot, energy, geometricalStepLengthFromPhysics, geometryStepLength, helixStepLength,
               geometryStepLength - helixStepLength, iterDone);
        currentTrack.print(slot, /* verbose= */ true);
      }
#endif
    } else {
#ifdef ADEPT_USE_SURF
      geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics,
                                                                    navState, nextState, hitsurf_index, kPush);
#else
      geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics,
                                                                    navState, nextState, kPush);
#endif
      pos += geometryStepLength * dir;
    }

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    navState.SetBoundaryState(nextState.IsOnBoundary());

    // Propagate information from geometrical step to MSC.
    theTrack->SetDirection(dir.x(), dir.y(), dir.z());
    theTrack->SetGStepLength(geometryStepLength);
    theTrack->SetOnBoundary(nextState.IsOnBoundary());

    // Apply continuous effects.
    bool stopped = G4HepEmElectronManager::PerformContinuous(&g4HepEmData, &g4HepEmPars, &elTrack, &rnge);

    // Collect the direction change and displacement by MSC.
    const double *direction = theTrack->GetDirection();
    dir.Set(direction[0], direction[1], direction[2]);
    if (!nextState.IsOnBoundary()) {
      const double *mscDisplacement = mscData->GetDisplacement();
      vecgeom::Vector3D<Precision> displacement(mscDisplacement[0], mscDisplacement[1], mscDisplacement[2]);
      const double dLength2            = displacement.Length2();
      constexpr double kGeomMinLength  = 5 * copcore::units::nm;          // 0.05 [nm]
      constexpr double kGeomMinLength2 = kGeomMinLength * kGeomMinLength; // (0.05 [nm])^2
      if (dLength2 > kGeomMinLength2) {
        const double dispR = std::sqrt(dLength2);
        // Estimate safety by subtracting the geometrical step length.
        safety -= geometryStepLength;
        constexpr double sFact = 0.99;
        double reducedSafety   = sFact * safety;

        // Apply displacement, depending on how close we are to a boundary.
        // 1a. Far away from geometry boundary:
        if (reducedSafety > 0.0 && dispR <= reducedSafety) {
          pos += displacement;
        } else {
          // Recompute safety.
          safety        = AdePTNavigator::ComputeSafety(pos, navState);
          reducedSafety = sFact * safety;

          // 1b. Far away from geometry boundary:
          if (reducedSafety > 0.0 && dispR <= reducedSafety) {
            pos += displacement;
            // 2. Push to boundary:
          } else if (reducedSafety > kGeomMinLength) {
            pos += displacement * (reducedSafety / dispR);
          }
          // 3. Very small safety: do nothing.
        }
      }
    }

    // Collect the charged step length (might be changed by MSC).
    atomicAdd(&globalScoring->chargedSteps, 1);
    atomicAdd(&scoringPerVolume->chargedTrackLength[volumeID], elTrack.GetPStepLength());

    // Collect the changes in energy and deposit.
    energy               = theTrack->GetEKin();
    double energyDeposit = theTrack->GetEnergyDeposit();
    atomicAdd(&globalScoring->energyDeposit, energyDeposit);
    atomicAdd(&scoringPerVolume->energyDeposit[volumeID], energyDeposit);

    // Save the `number-of-interaction-left` in our track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft           = theTrack->GetNumIALeft(ip);
      currentTrack.numIALeft[ip] = numIALeft;
    }

    if (stopped) {
      if (!IsElectron) {
        // Annihilate the stopped positron into two gammas heading to opposite
        // directions (isotropic).
        Track &gamma1 = secondaries.gammas.NextTrack();
        Track &gamma2 = secondaries.gammas.NextTrack();
        atomicAdd(&globalScoring->numGammas, 2);

        const double cost = 2 * currentTrack.Uniform() - 1;
        const double sint = sqrt(1 - cost * cost);
        const double phi  = k2Pi * currentTrack.Uniform();
        double sinPhi, cosPhi;
        sincos(phi, &sinPhi, &cosPhi);

        gamma1.InitAsSecondary(pos, navState);
        newRNG.Advance();
        gamma1.rngState = newRNG;
        gamma1.energy   = copcore::units::kElectronMassC2;
        gamma1.dir.Set(sint * cosPhi, sint * sinPhi, cost);

        gamma2.InitAsSecondary(pos, navState);
        // Reuse the RNG state of the dying track.
        gamma2.rngState = currentTrack.rngState;
        gamma2.energy   = copcore::units::kElectronMassC2;
        gamma2.dir      = -gamma1.dir;
      }
      // Particles are killed by not enqueuing them into the new activeQueue.
      continue;
    }

    if (nextState.IsOnBoundary()) {
      // For now, just count that we hit something.
      atomicAdd(&globalScoring->hits, 1);

      // Kill the particle if it left the world.
      if (!nextState.IsOutside()) {
#ifdef ADEPT_USE_SURF
        AdePTNavigator::RelocateToNextVolume(pos, dir, hitsurf_index, nextState);
#else
        AdePTNavigator::RelocateToNextVolume(pos, dir, nextState);
#endif
        // Move to the next boundary.
        navState = nextState;
        survive();
      }
      continue;
    } else if (!propagated) {
      // Did not yet reach the interaction point due to error in the magnetic
      // field propagation. Try again next time.
      survive();
      continue;
    } else if (winnerProcessIndex < 0) {
      // No discrete process, move on.
      survive();
      continue;
    }

    // Reset number of interaction left for the winner discrete process.
    // (Will be resampled in the next iteration.)
    currentTrack.numIALeft[winnerProcessIndex] = -1.0;

    // Check if a delta interaction happens instead of the real discrete process.
    if (G4HepEmElectronManager::CheckDelta(&g4HepEmData, theTrack, currentTrack.Uniform())) {
      // A delta interaction happened, move on.
      survive();
      continue;
    }

    // Perform the discrete interaction, make sure the branched RNG state is
    // ready to be used.
    newRNG.Advance();
    // Also advance the current RNG state to provide a fresh round of random
    // numbers after MSC used up a fair share for sampling the displacement.
    currentTrack.rngState.Advance();

    const double theElCut = g4HepEmData.fTheMatCutData->fMatCutData[theMCIndex].fSecElProdCutE;

    switch (winnerProcessIndex) {
    case 0: {
      // Invoke ionization (for e-/e+):
      double deltaEkin = (IsElectron) ? G4HepEmElectronInteractionIoni::SampleETransferMoller(theElCut, energy, &rnge)
                                      : G4HepEmElectronInteractionIoni::SampleETransferBhabha(theElCut, energy, &rnge);

      double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
      double dirSecondary[3];
      G4HepEmElectronInteractionIoni::SampleDirections(energy, deltaEkin, dirSecondary, dirPrimary, &rnge);

      Track &secondary = secondaries.electrons.NextTrack();
      atomicAdd(&globalScoring->numElectrons, 1);

      secondary.InitAsSecondary(pos, navState);
      secondary.rngState = newRNG;
      secondary.energy   = deltaEkin;
      secondary.dir.Set(dirSecondary[0], dirSecondary[1], dirSecondary[2]);

      energy -= deltaEkin;
      dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
      survive();
      break;
    }
    case 1: {
      // Invoke model for Bremsstrahlung: either SB- or Rel-Brem.
      double logEnergy = std::log(energy);
      double deltaEkin = energy < g4HepEmPars.fElectronBremModelLim
                             ? G4HepEmElectronInteractionBrem::SampleETransferSB(&g4HepEmData, energy, logEnergy,
                                                                                 theMCIndex, &rnge, IsElectron)
                             : G4HepEmElectronInteractionBrem::SampleETransferRB(&g4HepEmData, energy, logEnergy,
                                                                                 theMCIndex, &rnge, IsElectron);

      double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
      double dirSecondary[3];
      G4HepEmElectronInteractionBrem::SampleDirections(energy, deltaEkin, dirSecondary, dirPrimary, &rnge);

      Track &gamma = secondaries.gammas.NextTrack();
      atomicAdd(&globalScoring->numGammas, 1);

      gamma.InitAsSecondary(pos, navState);
      gamma.rngState = newRNG;
      gamma.energy   = deltaEkin;
      gamma.dir.Set(dirSecondary[0], dirSecondary[1], dirSecondary[2]);

      energy -= deltaEkin;
      dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
      survive();
      break;
    }
    case 2: {
      // Invoke annihilation (in-flight) for e+
      double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
      double theGamma1Ekin, theGamma2Ekin;
      double theGamma1Dir[3], theGamma2Dir[3];
      G4HepEmPositronInteractionAnnihilation::SampleEnergyAndDirectionsInFlight(
          energy, dirPrimary, &theGamma1Ekin, theGamma1Dir, &theGamma2Ekin, theGamma2Dir, &rnge);

      Track &gamma1 = secondaries.gammas.NextTrack();
      Track &gamma2 = secondaries.gammas.NextTrack();
      atomicAdd(&globalScoring->numGammas, 2);

      gamma1.InitAsSecondary(pos, navState);
      gamma1.rngState = newRNG;
      gamma1.energy   = theGamma1Ekin;
      gamma1.dir.Set(theGamma1Dir[0], theGamma1Dir[1], theGamma1Dir[2]);

      gamma2.InitAsSecondary(pos, navState);
      // Reuse the RNG state of the dying track.
      gamma2.rngState = currentTrack.rngState;
      gamma2.energy   = theGamma2Ekin;
      gamma2.dir.Set(theGamma2Dir[0], theGamma2Dir[1], theGamma2Dir[2]);

      // The current track is killed by not enqueuing into the next activeQueue.
      break;
    }
    }
  }
}

// Instantiate kernels for electrons and positrons.
__global__ void TransportElectrons(Track *electrons, const adept::MParray *active, Secondaries secondaries,
                                   adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                   ScoringPerVolume *scoringPerVolume)
{
  TransportElectrons</*IsElectron*/ true>(electrons, active, secondaries, activeQueue, globalScoring, scoringPerVolume);
}
__global__ void TransportPositrons(Track *positrons, const adept::MParray *active, Secondaries secondaries,
                                   adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                   ScoringPerVolume *scoringPerVolume)
{
  TransportElectrons</*IsElectron*/ false>(positrons, active, secondaries, activeQueue, globalScoring,
                                           scoringPerVolume);
}
