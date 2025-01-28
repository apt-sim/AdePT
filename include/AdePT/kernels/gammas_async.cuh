// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/core/AsyncAdePTTransportStruct.cuh>

#include <AdePT/navigation/BVHNavigator.h>

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

namespace AsyncAdePT {

template <typename Scoring>
__global__ void __launch_bounds__(256, 1)
    TransportGammas(Track *gammas, const adept::MParray *active, Secondaries secondaries, adept::MParray *activeQueue,
                    adept::MParray *leakedQueue, Scoring *userScoring, GammaInteractions gammaInteractions)
{
  using VolAuxData = adeptint::VolAuxData;
#ifdef VECGEOM_FLOAT_PRECISION
  constexpr Precision kPush = 10 * vecgeom::kTolerance;
#else
  constexpr Precision kPush = 0.;
#endif
  constexpr Precision kPushOutRegion = 10 * vecgeom::kTolerance;
  const int activeSize               = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot           = (*active)[i];
    Track &currentTrack      = gammas[slot];
    const auto eKin          = currentTrack.eKin;
    const auto preStepEnergy = eKin;
    auto pos                 = currentTrack.pos;
    const auto preStepPos{pos};
    const auto dir = currentTrack.dir;
    const auto preStepDir{dir};
    auto navState              = currentTrack.navState;
    const auto preStepNavState = navState;
    VolAuxData const &auxData  = AsyncAdePT::gVolAuxData[currentTrack.navState.Top()->GetLogicalVolume()->id()];
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
    theTrack->SetEKin(eKin);
    theTrack->SetMCIndex(auxData.fMCIndex);

    // Re-sample the `number-of-interaction-left` (if needed, otherwise use stored numIALeft) and put it into the
    // G4HepEmTrack. Use index 0 since numIALeft for gammas is based only on the total macroscopic cross section. The
    // currentTrack.numIALeft[0] are updated later
    if (currentTrack.numIALeft[0] <= 0.0) {
      theTrack->SetNumIALeft(-std::log(currentTrack.Uniform()), 0);
    } else {
      theTrack->SetNumIALeft(currentTrack.numIALeft[0], 0);
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
    double geometryStepLength =
        BVHNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics, navState, nextState, kPush);
    pos += geometryStepLength * dir;

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    navState.SetBoundaryState(nextState.IsOnBoundary());

    // Propagate information from geometrical step to G4HepEm.
    theTrack->SetGStepLength(geometryStepLength);
    theTrack->SetOnBoundary(nextState.IsOnBoundary());

    // Update the flight times of the particle
    const double deltaTime = theTrack->GetGStepLength() / copcore::units::kCLight;
    currentTrack.globalTime += deltaTime;
    currentTrack.localTime += deltaTime;

    if (nextState.IsOnBoundary()) {
      // Kill the particle if it left the world.
      if (nextState.Top() != nullptr) {

        G4HepEmGammaManager::UpdateNumIALeft(theTrack);

        // Save the `number-of-interaction-left` in our track.
        double numIALeft          = theTrack->GetNumIALeft(0);
        currentTrack.numIALeft[0] = numIALeft; // double to float conversion

        BVHNavigator::RelocateToNextVolume(pos, dir, nextState);

        // Move to the next boundary.
        navState = nextState;
        // Check if the next volume belongs to the GPU region and push it to the appropriate queue
        const auto nextvolume         = navState.Top();
        const int nextlvolID          = nextvolume->GetLogicalVolume()->id();
        VolAuxData const &nextauxData = AsyncAdePT::gVolAuxData[nextlvolID];
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
    } else {

      G4HepEmGammaManager::SampleInteraction(&g4HepEmData, &gammaTrack, currentTrack.Uniform());
      winnerProcessIndex = theTrack->GetWinnerProcessIndex();
      // NOTE: no simple re-drawing is possible for gamma-nuclear, since HowFar returns now smaller steps due to the
      // gamma-nuclear reactions in comparison to without gamma-nuclear reactions. Thus, an empty step without a
      // reaction is needed to compensate for the smaller step size returned by HowFar.

      // Reset number of interaction left for the winner discrete process also in the currentTrack (SampleInteraction()
      // resets it for theTrack), will be resampled in the next iteration.
      currentTrack.numIALeft[0] = -1.0;

      if (winnerProcessIndex == 3) {
        // Gamma-nuclear must be handled by G4, move on.
        survive(activeQueue);
        continue;
      }
    }

    assert(winnerProcessIndex < gammaInteractions.NInt);

    // Enqueue track in special interaction queue
    survive(nullptr);
    GammaInteractions::Data si{geometryStepLength,
                               gammaTrack.GetPEmxSec(),
                               static_cast<unsigned int>(slot),
                               preStepNavState,
                               preStepPos,
                               preStepDir,
                               preStepEnergy};
    gammaInteractions.queues[winnerProcessIndex]->push_back(std::move(si));
  }
}

template <typename Scoring>
__global__ void __launch_bounds__(256, 1)
    ApplyGammaInteractions(Track *gammas, Secondaries secondaries, adept::MParray *activeQueue, Scoring *userScoring,
                           GammaInteractions gammaInteractions)
{
  using VolAuxData = adeptint::VolAuxData;

  for (unsigned int interactionType = blockIdx.y; interactionType < GammaInteractions::NInt;
       interactionType += gridDim.y) {

    const auto &queue             = *gammaInteractions.queues[interactionType];
    const unsigned int activeSize = queue.size();
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
      const unsigned int slot         = queue[i].slot;
      const double geometryStepLength = queue[i].geometryStepLength;
      Track &currentTrack             = gammas[slot];
      auto &slotManager               = *secondaries.gammas.fSlotManager;
      const auto eKin                 = currentTrack.eKin;
      const auto &dir                 = currentTrack.dir;

      VolAuxData const &auxData = AsyncAdePT::gVolAuxData[currentTrack.navState.Top()->GetLogicalVolume()->id()];

      auto survive = [&]() { activeQueue->push_back(slot); };

      // Perform the discrete interaction.
      G4HepEmRandomEngine rnge(&currentTrack.rngState);
      // We might need one branched RNG state, prepare while threads are synchronized.
      RanluxppDouble newRNG(currentTrack.rngState.Branch());

      if (interactionType == GammaInteractions::PairCreation) {
        // Invoke gamma conversion to e-/e+ pairs, if the energy is above the threshold.
        if (eKin < 2 * copcore::units::kElectronMassC2) {
          survive();
          continue;
        }

        const double logEnergy = std::log(eKin);
        double elKinEnergy, posKinEnergy;
        G4HepEmGammaInteractionConversion::SampleKinEnergies(&g4HepEmData, eKin, logEnergy, auxData.fMCIndex,
                                                             elKinEnergy, posKinEnergy, &rnge);

        double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
        double dirSecondaryEl[3], dirSecondaryPos[3];
        G4HepEmGammaInteractionConversion::SampleDirections(dirPrimary, dirSecondaryEl, dirSecondaryPos, elKinEnergy,
                                                            posKinEnergy, &rnge);

        adept_scoring::AccountProduced(userScoring + currentTrack.threadId, /*numElectrons*/ 1, /*numPositrons*/ 1,
                                       /*numGammas*/ 0);

        secondaries.electrons.NextTrack(
            newRNG, elKinEnergy, currentTrack.pos,
            vecgeom::Vector3D<Precision>{dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]},
            currentTrack.navState, currentTrack);
        // Reuse the RNG state of the dying track.
        secondaries.positrons.NextTrack(
            currentTrack.rngState, posKinEnergy, currentTrack.pos,
            vecgeom::Vector3D<Precision>{dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]},
            currentTrack.navState, currentTrack);

        // Kill the original track.
        slotManager.MarkSlotForFreeing(slot);
      }

      if (interactionType == GammaInteractions::ComptonScattering) {
        // Invoke Compton scattering of gamma.
        constexpr double LowEnergyThreshold = 100 * copcore::units::eV;
        if (eKin < LowEnergyThreshold) {
          survive();
          continue;
        }
        const double origDirPrimary[] = {dir.x(), dir.y(), dir.z()};
        double dirPrimary[3];
        const double newEnergyGamma =
            G4HepEmGammaInteractionCompton::SamplePhotonEnergyAndDirection(eKin, dirPrimary, origDirPrimary, &rnge);
        vecgeom::Vector3D<double> newDirGamma(dirPrimary[0], dirPrimary[1], dirPrimary[2]);

        const double energyEl = eKin - newEnergyGamma;
        if (energyEl > LowEnergyThreshold) {
          // Create a secondary electron and sample/compute directions.
          adept_scoring::AccountProduced(userScoring + currentTrack.threadId, /*numElectrons*/ 1, /*numPositrons*/ 0,
                                         /*numGammas*/ 0);

          Track &electron = secondaries.electrons.NextTrack(newRNG, energyEl, currentTrack.pos,
                                                            eKin * dir - newEnergyGamma * newDirGamma,
                                                            currentTrack.navState, currentTrack);
          electron.dir.Normalize();
        } else {
          if (auxData.fSensIndex >= 0)
            adept_scoring::RecordHit(&userScoring[currentTrack.threadId], currentTrack.parentId,
                                     ParticleType::Gamma,       // Particle type
                                     geometryStepLength,        // Step length
                                     0,                         // Total Edep
                                     &queue[i].preStepNavState, // Pre-step point navstate
                                     &queue[i].preStepPos,      // Pre-step point position
                                     &queue[i].preStepDir,      // Pre-step point momentum direction
                                     nullptr,                   // Pre-step point polarization
                                     queue[i].preStepEnergy,    // Pre-step point kinetic energy
                                     0,                         // Pre-step point charge
                                     &currentTrack.navState,    // Post-step point navstate
                                     &currentTrack.pos,         // Post-step point position
                                     &currentTrack.dir,         // Post-step point momentum direction
                                     nullptr,                   // Post-step point polarization
                                     newEnergyGamma,            // Post-step point kinetic energy
                                     0,                         // Post-step point charge
                                     currentTrack.eventId, currentTrack.threadId);
        }

        // Check the new gamma energy and deposit if below threshold.
        if (newEnergyGamma > LowEnergyThreshold) {
          currentTrack.eKin = newEnergyGamma;
          currentTrack.dir  = newDirGamma;
          survive();
        } else {
          if (auxData.fSensIndex >= 0)
            adept_scoring::RecordHit(userScoring + currentTrack.threadId, currentTrack.parentId,
                                     ParticleType::Gamma,       // Particle type
                                     geometryStepLength,        // Step length
                                     0,                         // Total Edep
                                     &queue[i].preStepNavState, // Pre-step point navstate
                                     &queue[i].preStepPos,      // Pre-step point position
                                     &queue[i].preStepDir,      // Pre-step point momentum direction
                                     nullptr,                   // Pre-step point polarization
                                     queue[i].preStepEnergy,    // Pre-step point kinetic energy
                                     0,                         // Pre-step point charge
                                     &currentTrack.navState,    // Post-step point navstate
                                     &currentTrack.pos,         // Post-step point position
                                     &currentTrack.dir,         // Post-step point momentum direction
                                     nullptr,                   // Post-step point polarization
                                     newEnergyGamma,            // Post-step point kinetic energy
                                     0,                         // Post-step point charge
                                     currentTrack.eventId, currentTrack.threadId);

          // The current track is killed by not enqueuing into the next activeQueue.
          slotManager.MarkSlotForFreeing(slot);
        }
      }

      if (interactionType == GammaInteractions::PhotoelectricProcess) {
        // Invoke photoelectric process.
        constexpr double theLowEnergyThreshold = 1 * copcore::units::eV;

        const double bindingEnergy = G4HepEmGammaInteractionPhotoelectric::SelectElementBindingEnergy(
            &g4HepEmData, auxData.fMCIndex, queue[i].PEmxSec, eKin, &rnge);

        double edep             = bindingEnergy;
        const double photoElecE = eKin - edep;
        if (photoElecE > theLowEnergyThreshold) {
          // Create a secondary electron and sample directions.
          adept_scoring::AccountProduced(userScoring + currentTrack.threadId, /*numElectrons*/ 1, /*numPositrons*/ 0,
                                         /*numGammas*/ 0);

          double dirGamma[] = {dir.x(), dir.y(), dir.z()};
          double dirPhotoElec[3];
          G4HepEmGammaInteractionPhotoelectric::SamplePhotoElectronDirection(photoElecE, dirGamma, dirPhotoElec, &rnge);

          secondaries.electrons.NextTrack(
              newRNG, photoElecE, currentTrack.pos,
              vecgeom::Vector3D<Precision>{dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]}, currentTrack.navState,
              currentTrack);
        } else {
          edep = eKin;
        }
        if (auxData.fSensIndex >= 0)
          adept_scoring::RecordHit(userScoring + currentTrack.threadId, currentTrack.parentId,
                                   ParticleType::Gamma,       // Particle type
                                   geometryStepLength,        // Step length
                                   edep,                      // Total Edep
                                   &queue[i].preStepNavState, // Pre-step point navstate
                                   &queue[i].preStepPos,      // Pre-step point position
                                   &queue[i].preStepDir,      // Pre-step point momentum direction
                                   nullptr,                   // Pre-step point polarization
                                   queue[i].preStepEnergy,    // Pre-step point kinetic energy
                                   0,                         // Pre-step point charge
                                   &currentTrack.navState,    // Post-step point navstate
                                   &currentTrack.pos,         // Post-step point position
                                   &currentTrack.dir,         // Post-step point momentum direction
                                   nullptr,                   // Post-step point polarization
                                   0,                         // Post-step point kinetic energy
                                   0,                         // Post-step point charge
                                   currentTrack.eventId, currentTrack.threadId);
        // The current track is killed by not enqueuing into the next activeQueue.
        slotManager.MarkSlotForFreeing(slot);
      }
    }
  }
}

} // namespace AsyncAdePT
