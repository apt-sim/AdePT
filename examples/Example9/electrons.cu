// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include "example9.cuh"

#include <fieldPropagatorConstBz.h>

#include <CopCore/PhysicalConstants.h>

#include <G4HepEmElectronManager.hh>
#include <G4HepEmElectronTrack.hh>
#include <G4HepEmElectronInteractionBrem.hh>
#include <G4HepEmElectronInteractionIoni.hh>
#include <G4HepEmPositronInteractionAnnihilation.hh>
// Pull in implementation.
#include <G4HepEmRunUtils.icc>
#include <G4HepEmInteractionUtils.icc>
#include <G4HepEmElectronManager.icc>
#include <G4HepEmElectronInteractionBrem.icc>
#include <G4HepEmElectronInteractionIoni.icc>
#include <G4HepEmPositronInteractionAnnihilation.icc>

__device__ struct G4HepEmElectronManager electronManager;

// Create a pair of e-/e+ from the intermediate gamma.
__host__ __device__ void PairProduce(Secondaries &secondaries, const Track &currentTrack, double energy,
                                     const double *dir)
{
  Track &electron = secondaries.electrons.NextTrack();
  Track &positron = secondaries.positrons.NextTrack();

  // TODO: Distribute energy and momentum.
  double remainingEnergy = energy - 2 * copcore::units::kElectronMassC2;

  electron.InitAsSecondary(/*parent=*/currentTrack);
  electron.energy = remainingEnergy / 2;
  electron.dir.Set(dir[0], dir[1], dir[2]);

  positron.InitAsSecondary(/*parent=*/currentTrack);
  positron.energy = remainingEnergy / 2;
  positron.dir.Set(dir[0], dir[1], dir[2]);
}

// Compute the physics and geometry step limit, transport the electrons while
// applying the continuous effects and maybe a discrete process that could
// generate secondaries.
template <bool IsElectron>
__global__ void TransportElectrons(Track *electrons, const adept::MParray *active, Secondaries secondaries,
                                   adept::MParray *activeQueue, adept::MParray *relocateQueue, GlobalScoring *scoring)
{
  constexpr int Charge  = IsElectron ? -1 : 1;
  constexpr double Mass = copcore::units::kElectronMassC2;
  fieldPropagatorConstBz fieldPropagatorBz(BzFieldValue);

  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    Track &currentTrack = electrons[slot];
    auto volume         = currentTrack.currentState.Top();
    if (volume == nullptr) {
      // The particle left the world, kill it by not enqueuing into activeQueue.
      continue;
    }

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmElectronTrack elTrack;
    G4HepEmTrack *theTrack = elTrack.GetTrack();
    theTrack->SetEKin(currentTrack.energy);
    // For now, just assume a single material.
    int theMCIndex = 1;
    theTrack->SetMCIndex(theMCIndex);
    theTrack->SetCharge(Charge);

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
    electronManager.HowFar(&g4HepEmData, &g4HepEmPars, &elTrack);

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
    double geometryStepLength = fieldPropagatorBz.ComputeStepAndPropagatedState</*Relocate=*/false>(
        currentTrack.energy, Mass, Charge, geometricalStepLengthFromPhysics, currentTrack.pos, currentTrack.dir,
        currentTrack.currentState, currentTrack.nextState);

    if (currentTrack.nextState.IsOnBoundary()) {
      theTrack->SetGStepLength(geometryStepLength);
      theTrack->SetOnBoundary(true);
    }

    // Apply continuous effects.
    bool stopped = electronManager.PerformContinuous(&g4HepEmData, &g4HepEmPars, &elTrack);
    // Collect the changes.
    currentTrack.energy = theTrack->GetEKin();
    atomicAdd(&scoring->energyDeposit, theTrack->GetEnergyDeposit());

    // Save the `number-of-interaction-left` in our track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft           = theTrack->GetNumIALeft(ip);
      currentTrack.numIALeft[ip] = numIALeft;
    }

    if (stopped) {
      if (!IsElectron) {
        // For a stopped positron, we should call annihilation but this produces
        // a gamma which we don't yet have processes for. Deposit the amount of
        // energy that the photon would have from the annihilation at rest with
        // an electron.
        atomicAdd(&scoring->energyDeposit, 2 * copcore::units::kElectronMassC2);
      }
      // Particles are killed by not enqueuing them into the new activeQueue.
      continue;
    }

    if (currentTrack.nextState.IsOnBoundary()) {
      // For now, just count that we hit something.
      atomicAdd(&scoring->hits, 1);

      activeQueue->push_back(slot);
      relocateQueue->push_back(slot);

      // Move to the next boundary.
      currentTrack.SwapStates();
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
    if (electronManager.CheckDelta(&g4HepEmData, theTrack, currentTrack.Uniform())) {
      // A delta interaction happened, move on.
      activeQueue->push_back(slot);
      continue;
    }

    // Perform the discrete interaction.
    RanluxppDoubleEngine rnge(&currentTrack.rngState);

    const double energy   = currentTrack.energy;
    const double theElCut = g4HepEmData.fTheMatCutData->fMatCutData[theMCIndex].fSecElProdCutE;

    switch (winnerProcessIndex) {
    case 0: {
      // Invoke ionization (for e-/e+):
      double deltaEkin = (IsElectron) ? SampleETransferMoller(theElCut, energy, &rnge)
                                      : SampleETransferBhabha(theElCut, energy, &rnge);

      double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirSecondary[3];
      SampleDirectionsIoni(energy, deltaEkin, dirSecondary, dirPrimary, &rnge);

      Track &secondary = secondaries.electrons.NextTrack();
      atomicAdd(&scoring->secondaries, 1);

      secondary.InitAsSecondary(/*parent=*/currentTrack);
      secondary.energy = deltaEkin;
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
                             ? SampleETransferBremSB(&g4HepEmData, energy, logEnergy, theMCIndex, &rnge, IsElectron)
                             : SampleETransferBremRB(&g4HepEmData, energy, logEnergy, theMCIndex, &rnge, IsElectron);

      double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirSecondary[3];
      SampleDirectionsBrem(energy, deltaEkin, dirSecondary, dirPrimary, &rnge);

      // We would need to create a gamma, but only do so if it has enough energy
      // to immediately pair-produce. Otherwise just deposit the energy locally.
      if (deltaEkin > 2 * copcore::units::kElectronMassC2) {
        PairProduce(secondaries, currentTrack, deltaEkin, dirSecondary);
        atomicAdd(&scoring->secondaries, 2);
      } else {
        atomicAdd(&scoring->energyDeposit, deltaEkin);
      }

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
      SampleEnergyAndDirectionsForAnnihilationInFlight(energy, dirPrimary, &theGamma1Ekin, theGamma1Dir, &theGamma2Ekin,
                                                       theGamma2Dir, &rnge);

      // For each of the two gammas, pair-produce if they have enough energy
      // or deposit the energy locally.
      if (theGamma1Ekin > 2 * copcore::units::kElectronMassC2) {
        PairProduce(secondaries, currentTrack, theGamma1Ekin, theGamma1Dir);
        atomicAdd(&scoring->secondaries, 2);
      } else {
        atomicAdd(&scoring->energyDeposit, theGamma1Ekin);
      }
      if (theGamma2Ekin > 2 * copcore::units::kElectronMassC2) {
        PairProduce(secondaries, currentTrack, theGamma2Ekin, theGamma2Dir);
        atomicAdd(&scoring->secondaries, 2);
      } else {
        atomicAdd(&scoring->energyDeposit, theGamma2Ekin);
      }

      // The current track is killed by not enqueuing into the next activeQueue.
      break;
    }
    }
  }
}

// Instantiate template for electrons and positrons.
template __global__ void TransportElectrons</*IsElectron*/ true>(Track *electrons, const adept::MParray *active,
                                                                 Secondaries secondaries, adept::MParray *activeQueue,
                                                                 adept::MParray *relocateQueue, GlobalScoring *scoring);
template __global__ void TransportElectrons</*IsElectron*/ false>(Track *electrons, const adept::MParray *active,
                                                                  Secondaries secondaries, adept::MParray *activeQueue,
                                                                  adept::MParray *relocateQueue,
                                                                  GlobalScoring *scoring);
