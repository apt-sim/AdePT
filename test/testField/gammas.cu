// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include "testField.cuh"

#include <AdePT/navigation/AdePTNavigator.h>

#include <AdePT/copcore/PhysicalConstants.h>

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
                                adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                ScoringPerVolume *scoringPerVolume)
{
#ifdef VECGEOM_FLOAT_PRECISION
  const Precision kPush = 10 * vecgeom::kTolerance;
#else
  const Precision kPush = 0.;
#endif
  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    Track &currentTrack = gammas[slot];
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
    G4HepEmGammaTrack gammaTrack;
    G4HepEmTrack *theTrack = gammaTrack.GetTrack();
    theTrack->SetEKin(energy);
    theTrack->SetMCIndex(theMCIndex);

    // Sample the `number-of-interaction-left` and put it into the track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft = currentTrack.numIALeft[ip];
      if (numIALeft <= 0) {
        numIALeft = -std::log(currentTrack.Uniform());
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
    vecgeom::NavigationState nextState;
#ifdef ADEPT_USE_SURF
    long hitsurf_index = -1;
    double geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics,
                                                                         navState, nextState, hitsurf_index, kPush);
#else
    double geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics,
                                                                         navState, nextState, kPush);
#endif
    pos += geometryStepLength * dir;
    atomicAdd(&globalScoring->neutralSteps, 1);

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    navState.SetBoundaryState(nextState.IsOnBoundary());

    // Propagate information from geometrical step to G4HepEm.
    theTrack->SetGStepLength(geometryStepLength);
    theTrack->SetOnBoundary(nextState.IsOnBoundary());

    G4HepEmGammaManager::UpdateNumIALeft(theTrack);

    // Save the `number-of-interaction-left` in our track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft           = theTrack->GetNumIALeft(ip);
      currentTrack.numIALeft[ip] = numIALeft;
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
    } else if (winnerProcessIndex < 0) {
      // No discrete process, move on.
      survive();
      continue;
    }

    // Reset number of interaction left for the winner discrete process.
    // (Will be resampled in the next iteration.)
    currentTrack.numIALeft[winnerProcessIndex] = -1.0;

    // Perform the discrete interaction.
    G4HepEmRandomEngine rnge(&currentTrack.rngState);
    // We might need one branched RNG state, prepare while threads are synchronized.
    RanluxppDouble newRNG(currentTrack.rngState.Branch());

    switch (winnerProcessIndex) {
    case 0: {
      // Invoke gamma conversion to e-/e+ pairs, if the energy is above the threshold.
      if (energy < 2 * copcore::units::kElectronMassC2) {
        survive();
        continue;
      }

      double logEnergy = std::log(energy);
      double elKinEnergy, posKinEnergy;
      G4HepEmGammaInteractionConversion::SampleKinEnergies(&g4HepEmData, energy, logEnergy, theMCIndex, elKinEnergy,
                                                           posKinEnergy, &rnge);

      double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
      double dirSecondaryEl[3], dirSecondaryPos[3];
      G4HepEmGammaInteractionConversion::SampleDirections(dirPrimary, dirSecondaryEl, dirSecondaryPos, elKinEnergy,
                                                          posKinEnergy, &rnge);

      Track &electron = secondaries.electrons.NextTrack();
      Track &positron = secondaries.positrons.NextTrack();
      atomicAdd(&globalScoring->numElectrons, 1);
      atomicAdd(&globalScoring->numPositrons, 1);

      electron.InitAsSecondary(pos, navState);
      electron.rngState = newRNG;
      electron.energy   = elKinEnergy;
      electron.dir.Set(dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]);

      positron.InitAsSecondary(pos, navState);
      // Reuse the RNG state of the dying track.
      positron.rngState = currentTrack.rngState;
      positron.energy   = posKinEnergy;
      positron.dir.Set(dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]);

      // The current track is killed by not enqueuing into the next activeQueue.
      break;
    }
    case 1: {
      // Invoke Compton scattering of gamma.
      constexpr double LowEnergyThreshold = 100 * copcore::units::eV;
      if (energy < LowEnergyThreshold) {
        survive();
        continue;
      }
      const double origDirPrimary[] = {dir.x(), dir.y(), dir.z()};
      double dirPrimary[3];
      const double newEnergyGamma =
          G4HepEmGammaInteractionCompton::SamplePhotonEnergyAndDirection(energy, dirPrimary, origDirPrimary, &rnge);
      vecgeom::Vector3D<double> newDirGamma(dirPrimary[0], dirPrimary[1], dirPrimary[2]);

      const double energyEl = energy - newEnergyGamma;
      if (energyEl > LowEnergyThreshold) {
        // Create a secondary electron and sample/compute directions.
        Track &electron = secondaries.electrons.NextTrack();
        atomicAdd(&globalScoring->numElectrons, 1);

        electron.InitAsSecondary(pos, navState);
        electron.rngState = newRNG;
        electron.energy   = energyEl;
        electron.dir      = energy * dir - newEnergyGamma * newDirGamma;
        electron.dir.Normalize();
      } else {
        atomicAdd(&globalScoring->energyDeposit, energyEl);
        atomicAdd(&scoringPerVolume->energyDeposit[volumeID], energyEl);
      }

      // Check the new gamma energy and deposit if below threshold.
      if (newEnergyGamma > LowEnergyThreshold) {
        energy = newEnergyGamma;
        dir    = newDirGamma;
        survive();
      } else {
        atomicAdd(&globalScoring->energyDeposit, newEnergyGamma);
        atomicAdd(&scoringPerVolume->energyDeposit[volumeID], newEnergyGamma);
        // The current track is killed by not enqueuing into the next activeQueue.
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
        atomicAdd(&globalScoring->numElectrons, 1);

        double dirGamma[] = {dir.x(), dir.y(), dir.z()};
        double dirPhotoElec[3];
        G4HepEmGammaInteractionPhotoelectric::SamplePhotoElectronDirection(photoElecE, dirGamma, dirPhotoElec, &rnge);

        electron.InitAsSecondary(pos, navState);
        electron.rngState = newRNG;
        electron.energy   = photoElecE;
        electron.dir.Set(dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]);
      } else {
        edep = energy;
      }
      atomicAdd(&globalScoring->energyDeposit, edep);
      // The current track is killed by not enqueuing into the next activeQueue.
      break;
    }
    }
  }
}
