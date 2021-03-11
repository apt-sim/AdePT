// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include "example9.cuh"

#include <AdePT/LoopNavigator.h>
#include <CopCore/PhysicalConstants.h>

#include <G4HepEmGammaManager.hh>
#include <G4HepEmTrack.hh>
#include <G4HepEmGammaInteractionCompton.hh>
#include <G4HepEmGammaInteractionConversion.hh>
// Pull in implementation.
#include <G4HepEmGammaManager.icc>
#include <G4HepEmGammaInteractionCompton.icc>
#include <G4HepEmGammaInteractionConversion.icc>

__device__ struct G4HepEmGammaManager gammaManager;

constexpr double kPush = 1.e-8 * copcore::units::cm;

__global__ void TransportGammas(Track *gammas, const adept::MParray *active, Secondaries secondaries,
                                adept::MParray *activeQueue, adept::MParray *relocateQueue, GlobalScoring *scoring)
{
  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    Track &currentTrack = gammas[slot];
    auto volume         = currentTrack.currentState.Top();
    if (volume == nullptr) {
      // The particle left the world, kill it by not enqueuing into activeQueue.
      continue;
    }

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmTrack emTrack;
    emTrack.SetEKin(currentTrack.energy);
    // For now, just assume a single material.
    int theMCIndex = 1;
    emTrack.SetMCIndex(theMCIndex);

    // Sample the `number-of-interaction-left` and put it into the track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft = currentTrack.numIALeft[ip];
      if (numIALeft <= 0) {
        numIALeft                  = -std::log(currentTrack.Uniform());
        currentTrack.numIALeft[ip] = numIALeft;
      }
      emTrack.SetNumIALeft(numIALeft, ip);
    }

    // Call G4HepEm to compute the physics step limit.
    gammaManager.HowFar(&g4HepEmData, &g4HepEmPars, &emTrack);

    // Get result into variables.
    double geometricalStepLengthFromPhysics = emTrack.GetGStepLength();
    int winnerProcessIndex = emTrack.GetWinnerProcessIndex();
    // Leave the range and MFP inside the G4HepEmTrack. If we split kernels, we
    // also need to carry them over!

    // Check if there's a volume boundary in between.
    double geometryStepLength =
        LoopNavigator::ComputeStepAndNextVolume(currentTrack.pos, currentTrack.dir, geometricalStepLengthFromPhysics,
                                                currentTrack.currentState, currentTrack.nextState);
    currentTrack.pos += (geometryStepLength + kPush) * currentTrack.dir;

    if (currentTrack.nextState.IsOnBoundary()) {
      emTrack.SetGStepLength(geometryStepLength);
      emTrack.SetOnBoundary(true);
    }

    gammaManager.UpdateNumIALeft(&emTrack);

    // Save the `number-of-interaction-left` in our track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft           = emTrack.GetNumIALeft(ip);
      currentTrack.numIALeft[ip] = numIALeft;
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

    // Perform the discrete interaction.
    RanluxppDoubleEngine rnge(&currentTrack.rngState);

    const double energy   = currentTrack.energy;

    switch (winnerProcessIndex) {
    case 0: {
      // Invoke gamma conversion to e-/e+ pairs, if the energy is above the threshold.
      if (energy < 2 * copcore::units::kElectronMassC2) {
        activeQueue->push_back(slot);
        continue;
      }

      double logEnergy = std::log(energy);
      double elKinEnergy, posKinEnergy;
      SampleKinEnergies(&g4HepEmData, energy, logEnergy, theMCIndex, elKinEnergy, posKinEnergy, &rnge);

      double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirSecondaryEl[3], dirSecondaryPos[3];
      SampleDirections(dirPrimary, dirSecondaryEl, dirSecondaryPos, elKinEnergy, posKinEnergy, &rnge);

      Track &electron = secondaries.electrons.NextTrack();
      Track &positron = secondaries.positrons.NextTrack();
      atomicAdd(&scoring->secondaries, 2);

      electron.InitAsSecondary(/*parent=*/currentTrack);
      electron.energy = elKinEnergy;
      electron.dir.Set(dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]);

      positron.InitAsSecondary(/*parent=*/currentTrack);
      positron.energy = posKinEnergy;
      positron.dir.Set(dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]);

      // The current track is killed by not enqueuing into the next activeQueue.
      break;
    }
    case 1: {
      // Invoke Compton scattering of gamma.
      constexpr double LowEnergyThreshold = 100 * copcore::units::eV;
      if (energy < LowEnergyThreshold) {
        activeQueue->push_back(slot);
        continue;
      }
      const double origDirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirPrimary[3];
      const double newEnergyGamma = SamplePhotonEnergyAndDirection(energy, dirPrimary, origDirPrimary, &rnge);
      vecgeom::Vector3D<double> newDirGamma(dirPrimary[0], dirPrimary[1], dirPrimary[2]);

      const double energyEl = energy - newEnergyGamma;
      if (energyEl > LowEnergyThreshold) {
        // Create a secondary electron and sample/compute directions.
        Track &electron = secondaries.electrons.NextTrack();
        atomicAdd(&scoring->secondaries, 1);

        electron.InitAsSecondary(/*parent=*/currentTrack);
        electron.energy = energyEl;
        electron.dir = energy * currentTrack.dir - newEnergyGamma * newDirGamma;
        electron.dir.Normalize();
      } else {
        atomicAdd(&scoring->energyDeposit, energyEl);
      }

      // Check the new gamma energy and deposit if below threshold.
      if (newEnergyGamma > LowEnergyThreshold) {
        currentTrack.energy = newEnergyGamma;
        currentTrack.dir = newDirGamma;

        // The current track continues to live.
        activeQueue->push_back(slot);
      } else {
        atomicAdd(&scoring->energyDeposit, newEnergyGamma);
        // The current track is killed by not enqueuing into the next activeQueue.
      }
      break;
    }
    case 2: {
      // Invoke photoelectric process: right now only absorb the gamma.
      atomicAdd(&scoring->energyDeposit, energy);
      // The current track is killed by not enqueuing into the next activeQueue.
      break;
    }
    }
  }
}
