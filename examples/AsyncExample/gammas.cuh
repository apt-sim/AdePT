// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include "AdeptIntegration.cuh"

#include <AdePT/BVHNavigator.h>

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

template <typename Scoring>
__global__ void __launch_bounds__(256, 1)
    TransportGammas(Track *gammas, const adept::MParray *active, Secondaries secondaries, adept::MParray *activeQueue,
                    adept::MParray *leakedQueue, Scoring *userScoring, GammaInteractions gammaInteractions)
{
  using VolAuxData = AdeptIntegration::VolAuxData;
#ifdef VECGEOM_FLOAT_PRECISION
  constexpr Precision kPush = 10 * vecgeom::kTolerance;
#else
  constexpr Precision kPush = 0.;
#endif
  constexpr Precision kPushOutRegion = 10 * vecgeom::kTolerance;
  const int activeSize               = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    Track &currentTrack = gammas[slot];
    const auto energy         = currentTrack.energy;
    auto pos            = currentTrack.pos;
    const auto dir            = currentTrack.dir;
    auto navState       = currentTrack.navState;
    const auto volume   = navState.Top();
    const int lvolID          = volume->GetLogicalVolume()->id(); // the MCC vector is indexed by the logical volume id
    VolAuxData const &auxData = userScoring[currentTrack.threadId].GetAuxData_dev(lvolID);
    assert(auxData.fGPUregion > 0); // make sure we don't get inconsistent region here
    auto &slotManager = *secondaries.gammas.fSlotManager;

    // Write local variables back into track and enqueue
    auto survive = [&](adept::MParray *const nextQueue) {
      currentTrack.pos      = pos;
      currentTrack.dir      = dir;
      currentTrack.navState = navState;
      if (nextQueue) nextQueue->push_back(slot);
    };

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmGammaTrack gammaTrack;
    G4HepEmTrack *theTrack = gammaTrack.GetTrack();
    theTrack->SetEKin(energy);
    theTrack->SetMCIndex(auxData.fMCIndex);

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
    vecgeom::NavStateIndex nextState;
    double geometryStepLength =
        BVHNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics, navState, nextState, kPush);
    pos += geometryStepLength * dir;
    userScoring[currentTrack.threadId].AccountChargedStep(0);

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
      userScoring[currentTrack.threadId].AccountHit();

      // Kill the particle if it left the world.
      if (nextState.Top() != nullptr) {
        BVHNavigator::RelocateToNextVolume(pos, dir, nextState);

        // Move to the next boundary.
        navState = nextState;
        // Check if the next volume belongs to the GPU region and push it to the appropriate queue
        const auto nextvolume         = navState.Top();
        const int nextlvolID          = nextvolume->GetLogicalVolume()->id();
        VolAuxData const &nextauxData = userScoring[currentTrack.threadId].GetAuxData_dev(nextlvolID);
        if (nextauxData.fGPUregion > 0)
          survive(activeQueue);
        else {
          // To be safe, just push a bit the track exiting the GPU region to make sure
          // Geant4 does not relocate it again inside the same region
          pos += kPushOutRegion * dir;
          survive(leakedQueue);
        }
      } else {
        slotManager.MarkSlotForFreeing(slot);
      }
      continue;
    } else if (winnerProcessIndex < 0) {
      // No discrete process, move on.
      survive(activeQueue);
      continue;
    }

    // Reset number of interaction left for the winner discrete process.
    // (Will be resampled in the next iteration.)
    currentTrack.numIALeft[winnerProcessIndex] = -1.0;
    userScoring[currentTrack.threadId].AccountGammaInteraction(winnerProcessIndex);

    assert(winnerProcessIndex < gammaInteractions.NInt);

    // Enqueue track in special interaction queue
    survive(nullptr);
    GammaInteractions::Data si{
        geometryStepLength,
        gammaTrack.GetPEmxSec(),
        static_cast<unsigned int>(slot),
    };
    gammaInteractions.queues[winnerProcessIndex]->push_back(std::move(si));
  }
}

template <typename Scoring>
__global__ void __launch_bounds__(256, 1)
    ApplyGammaInteractions(Track *gammas, Secondaries secondaries, adept::MParray *activeQueue, Scoring *userScoring,
                           GammaInteractions gammaInteractions)
{
  using VolAuxData = AdeptIntegration::VolAuxData;

  for (unsigned int interactionType = blockIdx.y; interactionType < GammaInteractions::NInt;
       interactionType += gridDim.y) {

    const auto &queue             = *gammaInteractions.queues[interactionType];
    const unsigned int activeSize = queue.size();
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
      const unsigned int slot         = queue[i].slot;
      const double geometryStepLength = queue[i].geometryStepLength;
      Track &currentTrack             = gammas[slot];
      auto &slotManager               = *secondaries.gammas.fSlotManager;
      const auto energy               = currentTrack.energy;
      const auto &dir                 = currentTrack.dir;

      VolAuxData const &auxData =
          userScoring[currentTrack.threadId].GetAuxData_dev(currentTrack.navState.Top()->GetLogicalVolume()->id());

      auto survive = [&]() { activeQueue->push_back(slot); };

      // Perform the discrete interaction.
      G4HepEmRandomEngine rnge(&currentTrack.rngState);
      // We might need one branched RNG state, prepare while threads are synchronized.
      RanluxppDouble newRNG(currentTrack.rngState.Branch());

      if (interactionType == GammaInteractions::PairCreation) {
        // Invoke gamma conversion to e-/e+ pairs, if the energy is above the threshold.
        if (energy < 2 * copcore::units::kElectronMassC2) {
          survive();
          continue;
        }

        const double logEnergy = std::log(energy);
        double elKinEnergy, posKinEnergy;
        G4HepEmGammaInteractionConversion::SampleKinEnergies(&g4HepEmData, energy, logEnergy, auxData.fMCIndex,
                                                             elKinEnergy, posKinEnergy, &rnge);

        double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
        double dirSecondaryEl[3], dirSecondaryPos[3];
        G4HepEmGammaInteractionConversion::SampleDirections(dirPrimary, dirSecondaryEl, dirSecondaryPos, elKinEnergy,
                                                            posKinEnergy, &rnge);

        Track &electron = secondaries.electrons.NextTrack();
        Track &positron = secondaries.positrons.NextTrack();

        userScoring[currentTrack.threadId].AccountProduced(/*numElectrons*/ 1, /*numPositrons*/ 1, /*numGammas*/ 0);

        electron.InitAsSecondary(currentTrack.pos, currentTrack.navState, currentTrack);
        electron.rngState = newRNG;
        electron.energy   = elKinEnergy;
        electron.dir.Set(dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]);

        positron.InitAsSecondary(currentTrack.pos, currentTrack.navState, currentTrack);
        // Reuse the RNG state of the dying track.
        positron.rngState = currentTrack.rngState;
        positron.energy   = posKinEnergy;
        positron.dir.Set(dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]);

        // Kill the original track.
        slotManager.MarkSlotForFreeing(slot);
      }

      if (interactionType == GammaInteractions::ComptonScattering) {
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
          userScoring[currentTrack.threadId].AccountProduced(/*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

          electron.InitAsSecondary(currentTrack.pos, currentTrack.navState, currentTrack);
          electron.rngState = newRNG;
          electron.energy   = energyEl;
          electron.dir      = energy * dir - newEnergyGamma * newDirGamma;
          electron.dir.Normalize();
        } else {
          if (auxData.fSensIndex >= 0)
            userScoring[currentTrack.threadId].Score(currentTrack.navState, 0, geometryStepLength, energyEl);
        }

        // Check the new gamma energy and deposit if below threshold.
        if (newEnergyGamma > LowEnergyThreshold) {
          currentTrack.energy = newEnergyGamma;
          currentTrack.dir    = newDirGamma;
          survive();
        } else {
          if (auxData.fSensIndex >= 0)
            userScoring[currentTrack.threadId].Score(currentTrack.navState, 0, geometryStepLength, newEnergyGamma);
          // The current track is killed by not enqueuing into the next activeQueue.
          slotManager.MarkSlotForFreeing(slot);
        }
      }

      if (interactionType == GammaInteractions::PhotoelectricProcess) {
        // Invoke photoelectric process.
        constexpr double theLowEnergyThreshold = 1 * copcore::units::eV;

        const double bindingEnergy = G4HepEmGammaInteractionPhotoelectric::SelectElementBindingEnergy(
            &g4HepEmData, auxData.fMCIndex, queue[i].PEmxSec, energy, &rnge);

        double edep             = bindingEnergy;
        const double photoElecE = energy - edep;
        if (photoElecE > theLowEnergyThreshold) {
          // Create a secondary electron and sample directions.
          Track &electron = secondaries.electrons.NextTrack();
          userScoring[currentTrack.threadId].AccountProduced(/*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

          double dirGamma[] = {dir.x(), dir.y(), dir.z()};
          double dirPhotoElec[3];
          G4HepEmGammaInteractionPhotoelectric::SamplePhotoElectronDirection(photoElecE, dirGamma, dirPhotoElec, &rnge);

          electron.InitAsSecondary(currentTrack.pos, currentTrack.navState, currentTrack);
          electron.rngState = newRNG;
          electron.energy   = photoElecE;
          electron.dir.Set(dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]);
        } else {
          edep = energy;
        }
        if (auxData.fSensIndex >= 0)
          userScoring[currentTrack.threadId].Score(currentTrack.navState, 0, geometryStepLength, edep);
        // The current track is killed by not enqueuing into the next activeQueue.
        slotManager.MarkSlotForFreeing(slot);
      }
    }
  }
}
