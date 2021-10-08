// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include "example10.cuh"

#include <AdePT/LoopNavigator.h>
#include <CopCore/PhysicalConstants.h>

#include <G4HepEmGammaManager.hh>
#include <G4HepEmGammaTrack.hh>
#include <G4HepEmTrack.hh>
#include <G4HepEmGammaInteractionCompton.hh>
#include <G4HepEmGammaInteractionConversion.hh>
#include <G4HepEmGammaInteractionPhotoelectric.hh>
// Pull in implementation.
#include <G4HepEmGammaManager.icc>
#include <G4HepEmGammaInteractionCompton.icc>
#include <G4HepEmGammaInteractionConversion.icc>
#include <G4HepEmGammaInteractionPhotoelectric.icc>

__global__ void TransportGammas(Track *gammas, const adept::MParray *active, Secondaries secondaries,
                                adept::MParray *activeQueue, adept::MParray *relocateQueue, GlobalScoring *scoring,
                                int maxSteps)
{
  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    Track &currentTrack = gammas[slot];

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmGammaTrack gammaTrack;
    G4HepEmTrack *theTrack = gammaTrack.GetTrack();
    theTrack->SetEKin(currentTrack.energy);
    // For now, just assume a single material.
    int theMCIndex = 1;
    theTrack->SetMCIndex(theMCIndex);

    bool alive = true;
    for (int s = 0; alive && s < maxSteps; s++) {

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
      G4HepEmGammaManager::HowFar(&g4HepEmData, &g4HepEmPars, &gammaTrack);

      // Get result into variables.
      double geometricalStepLengthFromPhysics = theTrack->GetGStepLength();
      int winnerProcessIndex                  = theTrack->GetWinnerProcessIndex();
      // Leave the range and MFP inside the G4HepEmTrack. If we split kernels, we
      // also need to carry them over!

      // Check if there's a volume boundary in between.
      double geometryStepLength =
          LoopNavigator::ComputeStepAndNextVolume(currentTrack.pos, currentTrack.dir, geometricalStepLengthFromPhysics,
                                                  currentTrack.currentState, currentTrack.nextState);
      currentTrack.pos += geometryStepLength * currentTrack.dir;

      if (currentTrack.nextState.IsOnBoundary()) {
        theTrack->SetGStepLength(geometryStepLength);
        theTrack->SetOnBoundary(true);
      }

      G4HepEmGammaManager::UpdateNumIALeft(theTrack);

      // Save the `number-of-interaction-left` in our track.
      for (int ip = 0; ip < 3; ++ip) {
        double numIALeft           = theTrack->GetNumIALeft(ip);
        currentTrack.numIALeft[ip] = numIALeft;
      }

      if (currentTrack.nextState.IsOnBoundary()) {
        // For now, just count that we hit something.
        atomicAdd(&scoring->hits, 1);

        // Kill the particle if it left the world.
        if (currentTrack.nextState.Top() != nullptr) {
          alive = true;
          relocateQueue->push_back(slot);

          // Move to the next boundary.
          currentTrack.SwapStates();
        } else {
          alive = false;
        }

        // Cannot continue for now: either the particles left the world, or we
        // need to relocate it to the next volume.
        break;
      } else if (winnerProcessIndex < 0) {
        // No discrete process, move on.
        continue;
      }

      // Reset number of interaction left for the winner discrete process.
      // (Will be resampled in the next iteration.)
      currentTrack.numIALeft[winnerProcessIndex] = -1.0;

      // Perform the discrete interaction.
      RanluxppDoubleEngine rnge(&currentTrack.rngState);
      // We might need one branched RNG state, prepare while threads are synchronized.
      RanluxppDouble newRNG(currentTrack.rngState.Branch());

      const double energy = currentTrack.energy;

      switch (winnerProcessIndex) {
      case 0: {
        // Invoke gamma conversion to e-/e+ pairs, if the energy is above the threshold.
        if (energy < 2 * copcore::units::kElectronMassC2) {
          alive = true;
          continue;
        }

        double logEnergy = std::log(energy);
        double elKinEnergy, posKinEnergy;
        G4HepEmGammaInteractionConversion::SampleKinEnergies(&g4HepEmData, energy, logEnergy, theMCIndex, elKinEnergy,
                                                             posKinEnergy, &rnge);

        double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
        double dirSecondaryEl[3], dirSecondaryPos[3];
        G4HepEmGammaInteractionConversion::SampleDirections(dirPrimary, dirSecondaryEl, dirSecondaryPos, elKinEnergy,
                                                            posKinEnergy, &rnge);

        Track &electron = secondaries.electrons.NextTrack();
        Track &positron = secondaries.positrons.NextTrack();
        atomicAdd(&scoring->secondaries, 2);

        electron.InitAsSecondary(/*parent=*/currentTrack);
        electron.rngState = newRNG;
        electron.energy   = elKinEnergy;
        electron.dir.Set(dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]);

        positron.InitAsSecondary(/*parent=*/currentTrack);
        // Reuse the RNG state of the dying track.
        positron.rngState = currentTrack.rngState;
        positron.energy   = posKinEnergy;
        positron.dir.Set(dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]);

        alive = false;
        break;
      }
      case 1: {
        // Invoke Compton scattering of gamma.
        constexpr double LowEnergyThreshold = 100 * copcore::units::eV;
        if (energy < LowEnergyThreshold) {
          alive = true;
          continue;
        }
        const double origDirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
        double dirPrimary[3];
        const double newEnergyGamma =
            G4HepEmGammaInteractionCompton::SamplePhotonEnergyAndDirection(energy, dirPrimary, origDirPrimary, &rnge);
        vecgeom::Vector3D<double> newDirGamma(dirPrimary[0], dirPrimary[1], dirPrimary[2]);

        const double energyEl = energy - newEnergyGamma;
        if (energyEl > LowEnergyThreshold) {
          // Create a secondary electron and sample/compute directions.
          Track &electron = secondaries.electrons.NextTrack();
          atomicAdd(&scoring->secondaries, 1);

          electron.InitAsSecondary(/*parent=*/currentTrack);
          electron.rngState = newRNG;
          electron.energy   = energyEl;
          electron.dir      = energy * currentTrack.dir - newEnergyGamma * newDirGamma;
          electron.dir.Normalize();
        } else {
          atomicAdd(&scoring->energyDeposit, energyEl);
        }

        // Check the new gamma energy and deposit if below threshold.
        if (newEnergyGamma > LowEnergyThreshold) {
          currentTrack.energy = newEnergyGamma;
          theTrack->SetEKin(currentTrack.energy);
          currentTrack.dir = newDirGamma;

          // The current track continues to live.
          alive = true;
        } else {
          alive = false;
          atomicAdd(&scoring->energyDeposit, newEnergyGamma);
        }
        break;
      }
      case 2: {
        // Invoke photoelectric process.
        const double theLowEnergyThreshold = 1 * copcore::units::eV;

        const double bindingEnergy = G4HepEmGammaInteractionPhotoelectric::SelectElementBindingEnergy(
            &g4HepEmData, theMCIndex, gammaTrack.GetPEmxSec(), energy, &rnge);

        double edep             = bindingEnergy;
        const double photoElecE = energy - edep;
        if (photoElecE > theLowEnergyThreshold) {
          // Create a secondary electron and sample directions.
          Track &electron = secondaries.electrons.NextTrack();
          atomicAdd(&scoring->secondaries, 1);

          double dirGamma[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
          double dirPhotoElec[3];
          G4HepEmGammaInteractionPhotoelectric::SamplePhotoElectronDirection(photoElecE, dirGamma, dirPhotoElec, &rnge);

          electron.InitAsSecondary(/*parent=*/currentTrack);
          electron.rngState = newRNG;
          electron.energy   = photoElecE;
          electron.dir.Set(dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]);
        } else {
          edep = energy;
        }
        atomicAdd(&scoring->energyDeposit, edep);
        alive = false;
        break;
      }
      }
    }

    if (alive) {
      activeQueue->push_back(slot);
    }
  }
}
