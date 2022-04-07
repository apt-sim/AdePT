// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include "TestEm3.cuh"

#include <AdePT/BVHNavigator.h>
#include <fieldPropagatorConstBz.h>

#include <CopCore/PhysicalConstants.h>

#define NOFLUCTUATION

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

// Compute the physics and geometry step limit, transport the electrons while
// applying the continuous effects and maybe a discrete process that could
// generate secondaries.
template <bool IsElectron>
static __device__ __forceinline__ void TransportElectrons(Track *electrons, const adept::MParray *active,
                                                          Secondaries &secondaries, adept::MParray *activeQueue,
                                                          GlobalScoring *globalScoring,
                                                          ScoringPerVolume *scoringPerVolume)
{
  constexpr int Charge  = IsElectron ? -1 : 1;
  constexpr double Mass = copcore::units::kElectronMassC2;
  fieldPropagatorConstBz fieldPropagatorBz(BzFieldValue);

  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    Track &currentTrack = electrons[slot];
    auto volume         = currentTrack.navState.Top();
    int volumeID        = volume->id();
    int theMCIndex      = MCIndex[volumeID];

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmElectronTrack elTrack;
    G4HepEmTrack *theTrack = elTrack.GetTrack();
    theTrack->SetEKin(currentTrack.energy);
    theTrack->SetMCIndex(theMCIndex);
    theTrack->SetOnBoundary(currentTrack.navState.IsOnBoundary());
    theTrack->SetCharge(Charge);
    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    mscData->fIsFirstStep        = currentTrack.initialRange < 0;
    mscData->fInitialRange       = currentTrack.initialRange;
    mscData->fDynamicRangeFactor = currentTrack.dynamicRangeFactor;
    mscData->fTlimitMin          = currentTrack.tlimitMin;

    // Prepare a branched RNG state while threads are synchronized. Even if not
    // used, this provides a fresh round of random numbers and reduces thread
    // divergence because the RNG state doesn't need to be advanced later.
    RanluxppDouble newRNG(currentTrack.rngState.BranchNoAdvance());

    // Compute safety, needed for MSC step limit.
    double safety = 0;
    if (!currentTrack.navState.IsOnBoundary()) {
      safety = BVHNavigator::ComputeSafety(currentTrack.pos, currentTrack.navState);
    }
    theTrack->SetSafety(safety);

    RanluxppDoubleEngine rnge(&currentTrack.rngState);

    // Sample the `number-of-interaction-left` and put it into the track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft = currentTrack.numIALeft[ip];
      if (numIALeft <= 0) {
        numIALeft                  = -std::log(currentTrack.Uniform());
        currentTrack.numIALeft[ip] = numIALeft;
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
    double geometryStepLength;
    vecgeom::NavStateIndex nextState;
    if (BzFieldValue != 0) {
      geometryStepLength = fieldPropagatorBz.ComputeStepAndNextVolume<BVHNavigator>(
          currentTrack.energy, Mass, Charge, geometricalStepLengthFromPhysics, currentTrack.pos, currentTrack.dir,
          currentTrack.navState, nextState, propagated, safety);
    } else {
      geometryStepLength = BVHNavigator::ComputeStepAndNextVolume(
          currentTrack.pos, currentTrack.dir, geometricalStepLengthFromPhysics, currentTrack.navState, nextState);
      currentTrack.pos += geometryStepLength * currentTrack.dir;
    }

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (currentTrack.navState = nextState only if relocated
    // in case of a boundary; see below)
    currentTrack.navState.SetBoundaryState(nextState.IsOnBoundary());

    // Propagate information from geometrical step to MSC.
    theTrack->SetDirection(currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z());
    theTrack->SetGStepLength(geometryStepLength);
    theTrack->SetOnBoundary(nextState.IsOnBoundary());

    // Apply continuous effects.
    bool stopped = G4HepEmElectronManager::PerformContinuous(&g4HepEmData, &g4HepEmPars, &elTrack, &rnge);

    // Collect the direction change and displacement by MSC.
    const double *direction = theTrack->GetDirection();
    currentTrack.dir.Set(direction[0], direction[1], direction[2]);
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
          currentTrack.pos += displacement;
        } else {
          // Recompute safety.
          safety        = BVHNavigator::ComputeSafety(currentTrack.pos, currentTrack.navState);
          reducedSafety = sFact * safety;

          // 1b. Far away from geometry boundary:
          if (reducedSafety > 0.0 && dispR <= reducedSafety) {
            currentTrack.pos += displacement;
            // 2. Push to boundary:
          } else if (reducedSafety > kGeomMinLength) {
            currentTrack.pos += displacement * (reducedSafety / dispR);
          }
          // 3. Very small safety: do nothing.
        }
      }
    }

    // Collect the charged step length (might be changed by MSC).
    atomicAdd(&globalScoring->chargedSteps, 1);
    atomicAdd(&scoringPerVolume->chargedTrackLength[volumeID], elTrack.GetPStepLength());

    // Collect the changes in energy and deposit.
    currentTrack.energy  = theTrack->GetEKin();
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

        gamma1.InitAsSecondary(/*parent=*/currentTrack);
        newRNG.Advance();
        gamma1.rngState = newRNG;
        gamma1.energy   = copcore::units::kElectronMassC2;
        gamma1.dir.Set(sint * cosPhi, sint * sinPhi, cost);

        gamma2.InitAsSecondary(/*parent=*/currentTrack);
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
      if (nextState.Top() != nullptr) {
        activeQueue->push_back(slot);
        BVHNavigator::RelocateToNextVolume(currentTrack.pos, currentTrack.dir, nextState);

        // Move to the next boundary.
        currentTrack.navState = nextState;
      }
      continue;
    } else if (!propagated) {
      // Did not yet reach the interaction point due to error in the magnetic
      // field propagation. Try again next time.
      activeQueue->push_back(slot);
      continue;
    } else if (winnerProcessIndex < 0) {
      // No discrete process, move on.
      activeQueue->push_back(slot);
      continue;
    }

    // Reset number of interaction left for the winner discrete process.
    // (Will be resampled in the next iteration.)
    currentTrack.numIALeft[winnerProcessIndex] = -1.0;

    // Check if a delta interaction happens instead of the real discrete process.
    if (G4HepEmElectronManager::CheckDelta(&g4HepEmData, theTrack, currentTrack.Uniform())) {
      // A delta interaction happened, move on.
      activeQueue->push_back(slot);
      continue;
    }

    // Perform the discrete interaction, make sure the branched RNG state is
    // ready to be used.
    newRNG.Advance();
    // Also advance the current RNG state to provide a fresh round of random
    // numbers after MSC used up a fair share for sampling the displacement.
    currentTrack.rngState.Advance();

    const double energy   = currentTrack.energy;
    const double theElCut = g4HepEmData.fTheMatCutData->fMatCutData[theMCIndex].fSecElProdCutE;

    switch (winnerProcessIndex) {
    case 0: {
      // Invoke ionization (for e-/e+):
      double deltaEkin = (IsElectron) ? G4HepEmElectronInteractionIoni::SampleETransferMoller(theElCut, energy, &rnge)
                                      : G4HepEmElectronInteractionIoni::SampleETransferBhabha(theElCut, energy, &rnge);

      double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirSecondary[3];
      G4HepEmElectronInteractionIoni::SampleDirections(energy, deltaEkin, dirSecondary, dirPrimary, &rnge);

      Track &secondary = secondaries.electrons.NextTrack();
      atomicAdd(&globalScoring->numElectrons, 1);

      secondary.InitAsSecondary(/*parent=*/currentTrack);
      secondary.rngState = newRNG;
      secondary.energy   = deltaEkin;
      secondary.dir.Set(dirSecondary[0], dirSecondary[1], dirSecondary[2]);

      currentTrack.energy = energy - deltaEkin;
      currentTrack.dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
      // The current track continues to live.
      activeQueue->push_back(slot);
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

      double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirSecondary[3];
      G4HepEmElectronInteractionBrem::SampleDirections(energy, deltaEkin, dirSecondary, dirPrimary, &rnge);

      Track &gamma = secondaries.gammas.NextTrack();
      atomicAdd(&globalScoring->numGammas, 1);

      gamma.InitAsSecondary(/*parent=*/currentTrack);
      gamma.rngState = newRNG;
      gamma.energy   = deltaEkin;
      gamma.dir.Set(dirSecondary[0], dirSecondary[1], dirSecondary[2]);

      currentTrack.energy = energy - deltaEkin;
      currentTrack.dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
      // The current track continues to live.
      activeQueue->push_back(slot);
      break;
    }
    case 2: {
      // Invoke annihilation (in-flight) for e+
      double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double theGamma1Ekin, theGamma2Ekin;
      double theGamma1Dir[3], theGamma2Dir[3];
      G4HepEmPositronInteractionAnnihilation::SampleEnergyAndDirectionsInFlight(
          energy, dirPrimary, &theGamma1Ekin, theGamma1Dir, &theGamma2Ekin, theGamma2Dir, &rnge);

      Track &gamma1 = secondaries.gammas.NextTrack();
      Track &gamma2 = secondaries.gammas.NextTrack();
      atomicAdd(&globalScoring->numGammas, 2);

      gamma1.InitAsSecondary(/*parent=*/currentTrack);
      gamma1.rngState = newRNG;
      gamma1.energy   = theGamma1Ekin;
      gamma1.dir.Set(theGamma1Dir[0], theGamma1Dir[1], theGamma1Dir[2]);

      gamma2.InitAsSecondary(/*parent=*/currentTrack);
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
