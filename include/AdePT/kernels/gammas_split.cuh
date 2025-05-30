// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

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

using VolAuxData = adeptint::VolAuxData;

namespace AsyncAdePT {

__global__ void GammaHowFar(Track *gammas, G4HepEmGammaTrack *hepEMTracks, const adept::MParray *active,
                            adept::MParray *leakedQueue, Stats *InFlightStats,
                            AllowFinishOffEventArray allowFinishOffEvent)
{
  constexpr unsigned short maxSteps = 10'000;
  int activeSize                    = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    Track &currentTrack = gammas[slot];

#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    int lvolID = currentTrack.navState.GetLogicalId();
#endif
    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    currentTrack.leaked      = false;
    currentTrack.preStepEKin = currentTrack.eKin;
    currentTrack.preStepPos  = currentTrack.pos;
    currentTrack.preStepDir  = currentTrack.dir;
    // the MCC vector is indexed by the logical volume id

    currentTrack.stepCounter++;
    if (currentTrack.stepCounter >= maxSteps) {
      printf("Killing gamma event %d E=%f lvol=%d after %d steps. This indicates a stuck particle!\n",
             currentTrack.eventId, currentTrack.eKin, lvolID, currentTrack.stepCounter);
      continue;
    }

    // Write local variables back into track and enqueue
    auto survive = [&](bool leak = false) {
      if (leak) {
        auto success = leakedQueue->push_back(slot);
        if (!success) {
          printf("ERROR: No space left in gammas leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
      }
      // else
      //   nextActiveQueue->push_back(slot);
    };

    if (InFlightStats->perEventInFlightPrevious[currentTrack.threadId] < allowFinishOffEvent[currentTrack.threadId] &&
        InFlightStats->perEventInFlightPrevious[currentTrack.threadId] != 0) {
      printf("Thread %d Finishing gamma of the %d last particles of event %d on CPU E=%f lvol=%d after %d steps.\n",
             currentTrack.threadId, InFlightStats->perEventInFlightPrevious[currentTrack.threadId],
             currentTrack.eventId, currentTrack.eKin, lvolID, currentTrack.stepCounter);
      survive(/*leak*/ true);
      continue;
    }

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    gammaTrack.ReSet();
    G4HepEmTrack *theTrack = gammaTrack.GetTrack();
    theTrack->SetEKin(currentTrack.eKin);
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
  }
}

__global__ void GammaPropagation(Track *gammas, G4HepEmGammaTrack *hepEMTracks, const adept::MParray *active)
{
  constexpr Precision kPushDistance = 1000 * vecgeom::kTolerance;
  int activeSize                    = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    Track &currentTrack = gammas[slot];

#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    int lvolID = currentTrack.navState.GetLogicalId();
#endif

    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = gammaTrack.GetTrack();

    // Check if there's a volume boundary in between.
#ifdef ADEPT_USE_SURF
    long hitsurf_index = -1;
    currentTrack.geometryStepLength =
        AdePTNavigator::ComputeStepAndNextVolume(currentTrack.pos, currentTrack.dir, theTrack->GetGStepLength(),
                                                 currentTrack.navState, currentTrack.nextState, hitsurf_index);
#else
    currentTrack.geometryStepLength =
        AdePTNavigator::ComputeStepAndNextVolume(currentTrack.pos, currentTrack.dir, theTrack->GetGStepLength(),
                                                 currentTrack.navState, currentTrack.nextState, kPushDistance);
#endif
    //  printf("pvol=%d  step=%g  onboundary=%d  pos={%g, %g, %g}  dir={%g, %g, %g}\n", navState.TopId(),
    //  geometryStepLength,
    //         nextState.IsOnBoundary(), pos[0], pos[1], pos[2], dir[0], dir[1], dir[2]);
    currentTrack.pos += currentTrack.geometryStepLength * currentTrack.dir;

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    currentTrack.navState.SetBoundaryState(currentTrack.nextState.IsOnBoundary());

    // Propagate information from geometrical step to G4HepEm.
    theTrack->SetGStepLength(currentTrack.geometryStepLength);
    theTrack->SetOnBoundary(currentTrack.nextState.IsOnBoundary());

    // Update the flight times of the particle
    double deltaTime = theTrack->GetGStepLength() / copcore::units::kCLight;
    currentTrack.globalTime += deltaTime;
    currentTrack.localTime += deltaTime;
  }
}

template <typename Scoring>
__global__ void GammaRelocation(Track *gammas, G4HepEmGammaTrack *hepEMTracks, const adept::MParray *active,
                                Secondaries secondaries, adept::MParray *nextActiveQueue,
                                adept::MParray *reachedInteractionQueue, AllInteractionQueues interactionQueues,
                                adept::MParray *leakedQueue, Scoring *userScoring, bool returnAllSteps,
                                bool returnLastStep)
{
  constexpr Precision kPushDistance = 1000 * vecgeom::kTolerance;
  int activeSize                    = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    auto &slotManager   = *secondaries.gammas.fSlotManager;
    Track &currentTrack = gammas[slot];

#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    int lvolID = currentTrack.navState.GetLogicalId();
#endif

    // Write local variables back into track and enqueue
    auto survive = [&](bool leak = false) {
      returnLastStep = false; // particle survived, do not force return of step
      if (leak) {
        currentTrack.leaked = true;
        auto success        = leakedQueue->push_back(slot);
        if (!success) {
          printf("ERROR: No space left in gammas leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
      } else
        nextActiveQueue->push_back(slot);
    };

    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = gammaTrack.GetTrack();

    if (currentTrack.nextState.IsOnBoundary()) {
      currentTrack.restrictedPhysicalStepLength = true;
      // For now, just count that we hit something.

      // Kill the particle if it left the world.
      if (!currentTrack.nextState.IsOutside()) {

        G4HepEmGammaManager::UpdateNumIALeft(theTrack);

        // Save the `number-of-interaction-left` in our track.
        // Use index 0 since numIALeft stores for gammas only the total macroscopic cross section
        double numIALeft          = theTrack->GetNumIALeft(0);
        currentTrack.numIALeft[0] = numIALeft;

#ifdef ADEPT_USE_SURF
        AdePTNavigator::RelocateToNextVolume(currentTrack.pos, currentTrack.dir, hitsurf_index, currentTrack.nextState);
#else
        AdePTNavigator::RelocateToNextVolume(currentTrack.pos, currentTrack.dir, currentTrack.nextState);
#endif

        // if all steps are returned, we need to record the hit here,
        // as now the nextState is defined, but the navState is not yet replaced
        if (returnAllSteps)
          adept_scoring::RecordHit(userScoring,
                                   currentTrack.parentId,                       // Track ID
                                   2,                                           // Particle type
                                   currentTrack.geometryStepLength,             // Step length
                                   0,                                           // Total Edep
                                   currentTrack.weight,                         // Track weight
                                   currentTrack.navState,                       // Pre-step point navstate
                                   currentTrack.preStepPos,                     // Pre-step point position
                                   currentTrack.preStepDir,                     // Pre-step point momentum direction
                                   currentTrack.preStepEKin,                    // Pre-step point kinetic energy
                                   0,                                           // Pre-step point charge
                                   currentTrack.nextState,                      // Post-step point navstate
                                   currentTrack.pos,                            // Post-step point position
                                   currentTrack.dir,                            // Post-step point momentum direction
                                   currentTrack.eKin,                           // Post-step point kinetic energy
                                   0,                                           // Post-step point charge
                                   currentTrack.eventId, currentTrack.threadId, // event and thread ID
                                   returnLastStep,
                                   currentTrack.stepCounter == 1 ? true
                                                                 : false); // whether this is the last step of the track

        // Move to the next boundary.
        currentTrack.navState = currentTrack.nextState;
        // printf("  -> pvol=%d pos={%g, %g, %g} \n", navState.TopId(), pos[0], pos[1], pos[2]);
        //  Check if the next volume belongs to the GPU region and push it to the appropriate queue
#ifndef ADEPT_USE_SURF
        const int nextlvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
        const int nextlvolID = currentTrack.navState.GetLogicalId();
#endif
        VolAuxData const &nextauxData = AsyncAdePT::gVolAuxData[nextlvolID];
        if (nextauxData.fGPUregion > 0)
          survive();
        else {
          // To be safe, just push a bit the track exiting the GPU region to make sure
          // Geant4 does not relocate it again inside the same region
          currentTrack.pos += kPushDistance * currentTrack.dir;
          survive(/*leak*/ true);
        }
      } else {
        // release slot for particle that has left the world
        slotManager.MarkSlotForFreeing(slot);

        // particle has left the world, record hit if last or all steps are returned
        if (returnAllSteps || returnLastStep)
          adept_scoring::RecordHit(userScoring,
                                   currentTrack.parentId,                       // Track ID
                                   2,                                           // Particle type
                                   currentTrack.geometryStepLength,             // Step length
                                   0,                                           // Total Edep
                                   currentTrack.weight,                         // Track weight
                                   currentTrack.navState,                       // Pre-step point navstate
                                   currentTrack.preStepPos,                     // Pre-step point position
                                   currentTrack.preStepDir,                     // Pre-step point momentum direction
                                   currentTrack.preStepEKin,                    // Pre-step point kinetic energy
                                   0,                                           // Pre-step point charge
                                   currentTrack.nextState,                      // Post-step point navstate
                                   currentTrack.pos,                            // Post-step point position
                                   currentTrack.dir,                            // Post-step point momentum direction
                                   currentTrack.eKin,                           // Post-step point kinetic energy
                                   0,                                           // Post-step point charge
                                   currentTrack.eventId, currentTrack.threadId, // event and thread ID
                                   returnLastStep, // whether this is the last step of the track
                                   currentTrack.stepCounter == 1 ? true : false); // whether this is the first step
      }
      continue;
    } else {
      currentTrack.restrictedPhysicalStepLength = false;

      // This track will go to the interactions kernel
      reachedInteractionQueue->push_back(slot);

      // NOTE: This may be moved to the next kernel
      G4HepEmGammaManager::SampleInteraction(&g4HepEmData, &gammaTrack, currentTrack.Uniform());
      // NOTE: no simple re-drawing is possible for gamma-nuclear, since HowFar returns now smaller steps due to the
      // gamma-nuclear reactions in comparison to without gamma-nuclear reactions. Thus, an empty step without a
      // reaction is needed to compensate for the smaller step size returned by HowFar.

      // Reset number of interaction left for the winner discrete process also in the currentTrack
      // (SampleInteraction() resets it for theTrack), will be resampled in the next iteration.
      currentTrack.numIALeft[0] = -1.0;

      if (theTrack->GetWinnerProcessIndex() < 3) {
        interactionQueues.queues[theTrack->GetWinnerProcessIndex()]->push_back(slot);
      } else {
        // IMPORTANT: This is necessary just for getting numerically identical results,
        // but should be removed once confirmed that results are good
        G4HepEmRandomEngine rnge(&currentTrack.rngState);
        RanluxppDouble newRNG(currentTrack.rngState.Branch());
        // Gamma-nuclear not implemented, track survives
        survive();
        ;
      }
    }
  }
}

// Asynchronous TransportGammas Interface
template <typename Scoring>
// __global__ void GammaInteractions(Track *gammas, G4HepEmGammaTrack *hepEMTracks, const adept::MParray *active,
__global__ void GammaInteractions(Track *gammas, G4HepEmGammaTrack *hepEMTracks, Secondaries secondaries,
                                  adept::MParray *nextActiveQueue, adept::MParray *reachedInteractionQueue,
                                  adept::MParray *leakedQueue, Scoring *userScoring, bool returnAllSteps,
                                  bool returnLastStep)
{
  // int activeSize = active->size();
  int activeSize = reachedInteractionQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot      = (*active)[i];
    const int slot      = (*reachedInteractionQueue)[i];
    auto &slotManager   = *secondaries.gammas.fSlotManager;
    Track &currentTrack = gammas[slot];

#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    int lvolID = currentTrack.navState.GetLogicalId();
#endif
    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    // Write local variables back into track and enqueue
    auto survive = [&](bool leak = false) {
      returnLastStep = false; // particle survived, do not force return of step
      if (leak) {
        auto success = leakedQueue->push_back(slot);
        if (!success) {
          printf("ERROR: No space left in gammas leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
      } else
        nextActiveQueue->push_back(slot);
    };

    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = gammaTrack.GetTrack();

    // Perform the discrete interaction.
    G4HepEmRandomEngine rnge(&currentTrack.rngState);
    // We might need one branched RNG state, prepare while threads are synchronized.
    RanluxppDouble newRNG(currentTrack.rngState.Branch());

    const double theElCut    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecElProdCutE;
    const double thePosCut   = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecPosProdCutE;
    const double theGammaCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecGamProdCutE;

    const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
    const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

    double edep           = 0.;
    double newEnergyGamma = 0.; // gamma energy after compton scattering

    switch (theTrack->GetWinnerProcessIndex()) {
    case 0: {
      // Invoke gamma conversion to e-/e+ pairs, if the energy is above the threshold.
      if (currentTrack.eKin < 2 * copcore::units::kElectronMassC2) {
        survive();
        break;
      }

      double logEnergy = std::log(currentTrack.eKin);
      double elKinEnergy, posKinEnergy;
      G4HepEmGammaInteractionConversion::SampleKinEnergies(&g4HepEmData, currentTrack.eKin, logEnergy, auxData.fMCIndex,
                                                           elKinEnergy, posKinEnergy, &rnge);

      double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirSecondaryEl[3], dirSecondaryPos[3];
      G4HepEmGammaInteractionConversion::SampleDirections(dirPrimary, dirSecondaryEl, dirSecondaryPos, elKinEnergy,
                                                          posKinEnergy, &rnge);

      adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 1, /*numGammas*/ 0);

      // Check the cuts and deposit energy in this volume if needed
      if (ApplyCuts && elKinEnergy < theElCut) {
        // Deposit the energy here and kill the secondary
        edep = elKinEnergy;
      } else {
        secondaries.electrons.NextTrack(
            newRNG, elKinEnergy, currentTrack.pos,
            vecgeom::Vector3D<Precision>{dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]},
            currentTrack.navState, currentTrack);
      }

      if (ApplyCuts && (copcore::units::kElectronMassC2 < theGammaCut && posKinEnergy < thePosCut)) {
        // Deposit: posKinEnergy + 2 * copcore::units::kElectronMassC2 and kill the secondary
        edep += posKinEnergy + 2 * copcore::units::kElectronMassC2;
      } else {
        secondaries.positrons.NextTrack(
            currentTrack.rngState, posKinEnergy, currentTrack.pos,
            vecgeom::Vector3D<Precision>{dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]},
            currentTrack.navState, currentTrack);
      }

      // The current track is killed by not enqueuing into the next activeQueue and the slot is released
      slotManager.MarkSlotForFreeing(slot);
      break;
    }
    case 1: {
      // Invoke Compton scattering of gamma.
      constexpr double LowEnergyThreshold = 100 * copcore::units::eV;
      if (currentTrack.eKin < LowEnergyThreshold) {
        survive();
        break;
      }
      const double origDirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirPrimary[3];
      newEnergyGamma = G4HepEmGammaInteractionCompton::SamplePhotonEnergyAndDirection(currentTrack.eKin, dirPrimary,
                                                                                      origDirPrimary, &rnge);
      vecgeom::Vector3D<double> newDirGamma(dirPrimary[0], dirPrimary[1], dirPrimary[2]);

      const double energyEl = currentTrack.eKin - newEnergyGamma;

      adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

      // Check the cuts and deposit energy in this volume if needed
      if (ApplyCuts ? energyEl > theElCut : energyEl > LowEnergyThreshold) {
        // Create a secondary electron and sample/compute directions.
        Track &electron = secondaries.electrons.NextTrack(
            newRNG, energyEl, currentTrack.pos, currentTrack.eKin * currentTrack.dir - newEnergyGamma * newDirGamma,
            currentTrack.navState, currentTrack);
        electron.dir.Normalize();
      } else {
        edep = energyEl;
      }

      // Check the new gamma energy and deposit if below threshold.
      // Using same hardcoded very LowEnergyThreshold as G4HepEm
      if (newEnergyGamma > LowEnergyThreshold) {
        currentTrack.eKin = newEnergyGamma;
        currentTrack.dir  = newDirGamma;
        survive();
      } else {
        edep += newEnergyGamma;
        newEnergyGamma = 0.;
        // The current track is killed by not enqueuing into the next activeQueue and the slot is released
        slotManager.MarkSlotForFreeing(slot);
      }
      break;
    }
    case 2: {
      // Invoke photoelectric process.
      const double theLowEnergyThreshold = 1 * copcore::units::eV;

      const double bindingEnergy = G4HepEmGammaInteractionPhotoelectric::SelectElementBindingEnergy(
          &g4HepEmData, auxData.fMCIndex, gammaTrack.GetPEmxSec(), currentTrack.eKin, &rnge);

      edep                    = bindingEnergy;
      const double photoElecE = currentTrack.eKin - edep;
      if (ApplyCuts ? photoElecE > theElCut : photoElecE > theLowEnergyThreshold) {

        adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

        double dirGamma[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
        double dirPhotoElec[3];
        G4HepEmGammaInteractionPhotoelectric::SamplePhotoElectronDirection(photoElecE, dirGamma, dirPhotoElec, &rnge);

        // Create a secondary electron and sample directions.
        secondaries.electrons.NextTrack(newRNG, photoElecE, currentTrack.pos,
                                        vecgeom::Vector3D<Precision>{dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]},
                                        currentTrack.navState, currentTrack);

      } else {
        // If the secondary electron is cut, deposit all the energy of the gamma in this volume
        edep = currentTrack.eKin;
      }
      // The current track is killed by not enqueuing into the next activeQueue and the slot is released
      slotManager.MarkSlotForFreeing(slot);
      break;
    }
    case 3: {
      // Invoke gamma nuclear needs to be handled by Geant4 directly, to be implemented
      // Just keep particle alive
      survive();
    }
    }

    // If there is some edep from cutting particles, record the step
    if ((edep > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep) {
      adept_scoring::RecordHit(userScoring,
                               currentTrack.parentId,                       // Track ID
                               2,                                           // Particle type
                               currentTrack.geometryStepLength,             // Step length
                               edep,                                        // Total Edep
                               currentTrack.weight,                         // Track weight
                               currentTrack.navState,                       // Pre-step point navstate
                               currentTrack.preStepPos,                     // Pre-step point position
                               currentTrack.preStepDir,                     // Pre-step point momentum direction
                               currentTrack.preStepEKin,                    // Pre-step point kinetic energy
                               0,                                           // Pre-step point charge
                               currentTrack.nextState,                      // Post-step point navstate
                               currentTrack.pos,                            // Post-step point position
                               currentTrack.dir,                            // Post-step point momentum direction
                               newEnergyGamma,                              // Post-step point kinetic energy
                               0,                                           // Post-step point charge
                               currentTrack.eventId, currentTrack.threadId, // event and thread ID
                               returnLastStep, // whether this is the last step of the track
                               currentTrack.stepCounter == 1 ? true : false); // whether this is the first step
    }
  }
}

template <typename Scoring>
__global__ void GammaConversion(Track *gammas, G4HepEmGammaTrack *hepEMTracks, Secondaries secondaries,
                                adept::MParray *nextActiveQueue, adept::MParray *interactingQueue,
                                adept::MParray *leakedQueue, Scoring *userScoring, bool returnAllSteps,
                                bool returnLastStep)
{
  // int activeSize = active->size();
  int activeSize = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot      = (*active)[i];
    const int slot      = (*interactingQueue)[i];
    auto &slotManager   = *secondaries.gammas.fSlotManager;
    Track &currentTrack = gammas[slot];

#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    int lvolID = currentTrack.navState.GetLogicalId();
#endif
    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    // Write local variables back into track and enqueue
    auto survive = [&](bool leak = false) {
      returnLastStep = false; // particle survived, do not force return of step
      if (leak) {
        auto success = leakedQueue->push_back(slot);
        if (!success) {
          printf("ERROR: No space left in gammas leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
      } else
        nextActiveQueue->push_back(slot);
    };

    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = gammaTrack.GetTrack();

    // Perform the discrete interaction.
    G4HepEmRandomEngine rnge(&currentTrack.rngState);
    // We might need one branched RNG state, prepare while threads are synchronized.
    RanluxppDouble newRNG(currentTrack.rngState.Branch());

    const double theElCut    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecElProdCutE;
    const double thePosCut   = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecPosProdCutE;
    const double theGammaCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecGamProdCutE;

    const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
    const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

    double edep           = 0.;
    double newEnergyGamma = 0.; // gamma energy after compton scattering

    // Interaction

    // Invoke gamma conversion to e-/e+ pairs, if the energy is above the threshold.
    if (currentTrack.eKin < 2 * copcore::units::kElectronMassC2) {
      survive();
      break;
    }

    double logEnergy = std::log(currentTrack.eKin);
    double elKinEnergy, posKinEnergy;
    G4HepEmGammaInteractionConversion::SampleKinEnergies(&g4HepEmData, currentTrack.eKin, logEnergy, auxData.fMCIndex,
                                                         elKinEnergy, posKinEnergy, &rnge);

    double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
    double dirSecondaryEl[3], dirSecondaryPos[3];
    G4HepEmGammaInteractionConversion::SampleDirections(dirPrimary, dirSecondaryEl, dirSecondaryPos, elKinEnergy,
                                                        posKinEnergy, &rnge);

    adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 1, /*numGammas*/ 0);

    // Check the cuts and deposit energy in this volume if needed
    if (ApplyCuts && elKinEnergy < theElCut) {
      // Deposit the energy here and kill the secondary
      edep = elKinEnergy;
    } else {
      secondaries.electrons.NextTrack(
          newRNG, elKinEnergy, currentTrack.pos,
          vecgeom::Vector3D<Precision>{dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]}, currentTrack.navState,
          currentTrack);
    }

    if (ApplyCuts && (copcore::units::kElectronMassC2 < theGammaCut && posKinEnergy < thePosCut)) {
      // Deposit: posKinEnergy + 2 * copcore::units::kElectronMassC2 and kill the secondary
      edep += posKinEnergy + 2 * copcore::units::kElectronMassC2;
    } else {
      secondaries.positrons.NextTrack(
          currentTrack.rngState, posKinEnergy, currentTrack.pos,
          vecgeom::Vector3D<Precision>{dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]},
          currentTrack.navState, currentTrack);
    }

    // The current track is killed by not enqueuing into the next activeQueue and the slot is released
    slotManager.MarkSlotForFreeing(slot);

    //////////////

    // If there is some edep from cutting particles, record the step
    if ((edep > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep) {
      adept_scoring::RecordHit(userScoring,
                               currentTrack.parentId,                       // Track ID
                               2,                                           // Particle type
                               currentTrack.geometryStepLength,             // Step length
                               edep,                                        // Total Edep
                               currentTrack.weight,                         // Track weight
                               currentTrack.navState,                       // Pre-step point navstate
                               currentTrack.preStepPos,                     // Pre-step point position
                               currentTrack.preStepDir,                     // Pre-step point momentum direction
                               currentTrack.preStepEKin,                    // Pre-step point kinetic energy
                               0,                                           // Pre-step point charge
                               currentTrack.nextState,                      // Post-step point navstate
                               currentTrack.pos,                            // Post-step point position
                               currentTrack.dir,                            // Post-step point momentum direction
                               newEnergyGamma,                              // Post-step point kinetic energy
                               0,                                           // Post-step point charge
                               currentTrack.eventId, currentTrack.threadId, // event and thread ID
                               returnLastStep, // whether this is the last step of the track
                               currentTrack.stepCounter == 1 ? true : false); // whether this is the first step
    }
  }
}

template <typename Scoring>
__global__ void GammaCompton(Track *gammas, G4HepEmGammaTrack *hepEMTracks, Secondaries secondaries,
                             adept::MParray *nextActiveQueue, adept::MParray *interactingQueue,
                             adept::MParray *leakedQueue, Scoring *userScoring, bool returnAllSteps,
                             bool returnLastStep)
{
  // int activeSize = active->size();
  int activeSize = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot      = (*active)[i];
    const int slot      = (*interactingQueue)[i];
    auto &slotManager   = *secondaries.gammas.fSlotManager;
    Track &currentTrack = gammas[slot];

#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    int lvolID = currentTrack.navState.GetLogicalId();
#endif
    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    // Write local variables back into track and enqueue
    auto survive = [&](bool leak = false) {
      returnLastStep = false; // particle survived, do not force return of step
      if (leak) {
        auto success = leakedQueue->push_back(slot);
        if (!success) {
          printf("ERROR: No space left in gammas leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
      } else
        nextActiveQueue->push_back(slot);
    };

    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = gammaTrack.GetTrack();

    // Perform the discrete interaction.
    G4HepEmRandomEngine rnge(&currentTrack.rngState);
    // We might need one branched RNG state, prepare while threads are synchronized.
    RanluxppDouble newRNG(currentTrack.rngState.Branch());

    const double theElCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecElProdCutE;

    const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
    const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

    double edep           = 0.;
    double newEnergyGamma = 0.; // gamma energy after compton scattering

    // Interaction

    // Invoke Compton scattering of gamma.
    constexpr double LowEnergyThreshold = 100 * copcore::units::eV;
    if (currentTrack.eKin < LowEnergyThreshold) {
      survive();
      break;
    }
    const double origDirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
    double dirPrimary[3];
    newEnergyGamma = G4HepEmGammaInteractionCompton::SamplePhotonEnergyAndDirection(currentTrack.eKin, dirPrimary,
                                                                                    origDirPrimary, &rnge);
    vecgeom::Vector3D<double> newDirGamma(dirPrimary[0], dirPrimary[1], dirPrimary[2]);

    const double energyEl = currentTrack.eKin - newEnergyGamma;

    adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

    // Check the cuts and deposit energy in this volume if needed
    if (ApplyCuts ? energyEl > theElCut : energyEl > LowEnergyThreshold) {
      // Create a secondary electron and sample/compute directions.
      Track &electron = secondaries.electrons.NextTrack(
          newRNG, energyEl, currentTrack.pos, currentTrack.eKin * currentTrack.dir - newEnergyGamma * newDirGamma,
          currentTrack.navState, currentTrack);
      electron.dir.Normalize();
    } else {
      edep = energyEl;
    }

    // Check the new gamma energy and deposit if below threshold.
    // Using same hardcoded very LowEnergyThreshold as G4HepEm
    if (newEnergyGamma > LowEnergyThreshold) {
      currentTrack.eKin = newEnergyGamma;
      currentTrack.dir  = newDirGamma;
      survive();
    } else {
      edep += newEnergyGamma;
      newEnergyGamma = 0.;
      // The current track is killed by not enqueuing into the next activeQueue and the slot is released
      slotManager.MarkSlotForFreeing(slot);
    }

    //////////////

    // If there is some edep from cutting particles, record the step
    if ((edep > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep) {
      adept_scoring::RecordHit(userScoring,
                               currentTrack.parentId,                       // Track ID
                               2,                                           // Particle type
                               currentTrack.geometryStepLength,             // Step length
                               edep,                                        // Total Edep
                               currentTrack.weight,                         // Track weight
                               currentTrack.navState,                       // Pre-step point navstate
                               currentTrack.preStepPos,                     // Pre-step point position
                               currentTrack.preStepDir,                     // Pre-step point momentum direction
                               currentTrack.preStepEKin,                    // Pre-step point kinetic energy
                               0,                                           // Pre-step point charge
                               currentTrack.nextState,                      // Post-step point navstate
                               currentTrack.pos,                            // Post-step point position
                               currentTrack.dir,                            // Post-step point momentum direction
                               newEnergyGamma,                              // Post-step point kinetic energy
                               0,                                           // Post-step point charge
                               currentTrack.eventId, currentTrack.threadId, // event and thread ID
                               returnLastStep, // whether this is the last step of the track
                               currentTrack.stepCounter == 1 ? true : false); // whether this is the first step
    }
  }
}

template <typename Scoring>
__global__ void GammaPhotoelectric(Track *gammas, G4HepEmGammaTrack *hepEMTracks, Secondaries secondaries,
                                   adept::MParray *nextActiveQueue, adept::MParray *interactingQueue,
                                   adept::MParray *leakedQueue, Scoring *userScoring, bool returnAllSteps,
                                   bool returnLastStep)
{
  // int activeSize = active->size();
  int activeSize = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot      = (*active)[i];
    const int slot      = (*interactingQueue)[i];
    auto &slotManager   = *secondaries.gammas.fSlotManager;
    Track &currentTrack = gammas[slot];

#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    int lvolID = currentTrack.navState.GetLogicalId();
#endif
    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    // Write local variables back into track and enqueue
    auto survive = [&](bool leak = false) {
      returnLastStep = false; // particle survived, do not force return of step
      if (leak) {
        auto success = leakedQueue->push_back(slot);
        if (!success) {
          printf("ERROR: No space left in gammas leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
      } else
        nextActiveQueue->push_back(slot);
    };

    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = gammaTrack.GetTrack();

    // Perform the discrete interaction.
    G4HepEmRandomEngine rnge(&currentTrack.rngState);
    // We might need one branched RNG state, prepare while threads are synchronized.
    RanluxppDouble newRNG(currentTrack.rngState.Branch());

    const double theElCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecElProdCutE;

    const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
    const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

    double edep           = 0.;
    double newEnergyGamma = 0.; // gamma energy after compton scattering

    // Interaction

    // Invoke photoelectric process.
    const double theLowEnergyThreshold = 1 * copcore::units::eV;

    const double bindingEnergy = G4HepEmGammaInteractionPhotoelectric::SelectElementBindingEnergy(
        &g4HepEmData, auxData.fMCIndex, gammaTrack.GetPEmxSec(), currentTrack.eKin, &rnge);

    edep                    = bindingEnergy;
    const double photoElecE = currentTrack.eKin - edep;
    if (ApplyCuts ? photoElecE > theElCut : photoElecE > theLowEnergyThreshold) {

      adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

      double dirGamma[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirPhotoElec[3];
      G4HepEmGammaInteractionPhotoelectric::SamplePhotoElectronDirection(photoElecE, dirGamma, dirPhotoElec, &rnge);

      // Create a secondary electron and sample directions.
      secondaries.electrons.NextTrack(newRNG, photoElecE, currentTrack.pos,
                                      vecgeom::Vector3D<Precision>{dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]},
                                      currentTrack.navState, currentTrack);

    } else {
      // If the secondary electron is cut, deposit all the energy of the gamma in this volume
      edep = currentTrack.eKin;
    }
    // The current track is killed by not enqueuing into the next activeQueue and the slot is released
    slotManager.MarkSlotForFreeing(slot);

    //////////////

    // If there is some edep from cutting particles, record the step
    if ((edep > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep) {
      adept_scoring::RecordHit(userScoring,
                               currentTrack.parentId,                       // Track ID
                               2,                                           // Particle type
                               currentTrack.geometryStepLength,             // Step length
                               edep,                                        // Total Edep
                               currentTrack.weight,                         // Track weight
                               currentTrack.navState,                       // Pre-step point navstate
                               currentTrack.preStepPos,                     // Pre-step point position
                               currentTrack.preStepDir,                     // Pre-step point momentum direction
                               currentTrack.preStepEKin,                    // Pre-step point kinetic energy
                               0,                                           // Pre-step point charge
                               currentTrack.nextState,                      // Post-step point navstate
                               currentTrack.pos,                            // Post-step point position
                               currentTrack.dir,                            // Post-step point momentum direction
                               newEnergyGamma,                              // Post-step point kinetic energy
                               0,                                           // Post-step point charge
                               currentTrack.eventId, currentTrack.threadId, // event and thread ID
                               returnLastStep, // whether this is the last step of the track
                               currentTrack.stepCounter == 1 ? true : false); // whether this is the first step
    }
  }
}

} // namespace AsyncAdePT
