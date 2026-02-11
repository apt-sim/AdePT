// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/navigation/AdePTNavigator.h>

#include <AdePT/copcore/PhysicalConstants.h>
#include <AdePT/kernels/AdePTSteppingActionSelector.cuh>
#include <AdePT/kernels/gammasWDT_split.cuh>

#include <G4HepEmGammaManager.hh>
#include <G4HepEmGammaTrack.hh>
#include <G4HepEmTrack.hh>
#include <G4HepEmGammaInteractionCompton.hh>
#include <G4HepEmGammaInteractionConversion.hh>
#include <G4HepEmGammaInteractionPhotoelectric.hh>

using StepActionParam = adept::SteppingAction::Params;
using VolAuxData      = adeptint::VolAuxData;

namespace AsyncAdePT {

template <typename Scoring, class SteppingActionT>
__global__ void GammaHowFar(G4HepEmGammaTrack *hepEMTracks, ParticleManager particleManager,
                            adept::MParray *propagationQueue, Stats *InFlightStats, const StepActionParam params,
                            Scoring *userScoring, AllowFinishOffEventArray allowFinishOffEvent,
                            const bool returnAllSteps, const bool returnLastStep)
{
  constexpr unsigned short maxSteps        = 10'000;
  constexpr unsigned short kStepsStuckKill = 25;
  auto &slotManager                        = *particleManager.gammas.fSlotManager;

  const int activeSize = particleManager.gammas.ActiveSize();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const auto slot     = particleManager.gammas.ActiveAt(i);
    Track &currentTrack = particleManager.gammas.TrackAt(slot);

    int lvolID                = currentTrack.navState.GetLogicalId();
    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID];

    currentTrack.preStepEKin       = currentTrack.eKin;
    currentTrack.preStepGlobalTime = currentTrack.globalTime;
    currentTrack.preStepPos        = currentTrack.pos;
    currentTrack.preStepDir        = currentTrack.dir;
    // the MCC vector is indexed by the logical volume id

    currentTrack.stepCounter++;
    bool printErrors = false;

    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = gammaTrack.GetTrack();
    if (!currentTrack.hepEmTrackExists) {
      // Init a track with the needed data to call into G4HepEm.
      gammaTrack.ReSet();
      theTrack->SetEKin(currentTrack.eKin);
      theTrack->SetMCIndex(auxData.fMCIndex);
      currentTrack.hepEmTrackExists = true;
    }
    // Re-sample the `number-of-interaction-left` (if needed, otherwise use stored numIALeft) and put it into the
    // G4HepEmTrack. Use index 0 since numIALeft for gammas is based only on the total macroscopic cross section. The
    if (theTrack->GetNumIALeft(0) <= 0.0) {
      theTrack->SetNumIALeft(-std::log(currentTrack.Uniform()), 0);
    }

    {
      // ---- Begin of SteppingAction:
      // Kill various tracks based on looper criteria, or via an experiment-specific SteppingAction

      // Unlike the monolithic kernels, the SteppingAction in the split kernels is done at the beginning of the step, as
      // this is one central place to do it This is similar but not the same as killing them at the end of the
      // monolithic kernels, as the NavState and the preStepPoints are already updated. Doing the stepping action before
      // updating the variables has the disadvantage that the NavigationState would need to be updated by the
      // NextNavState at the beginning of each step, which means that the NextNavState would have to be initialized as
      // well. Given the fact, that the killed tracks should not play a relevant role in the user code, this was not a
      // priority
      bool trackSurvives   = true;
      double energyDeposit = 0.;

      // check for max steps and stuck tracks
      if (currentTrack.stepCounter >= maxSteps || currentTrack.zeroStepCounter > kStepsStuckKill) {
        if (printErrors)
          printf("Killing gamma event %d track %lu E=%f lvol=%d after %d steps with zeroStepCounter %u\n",
                 currentTrack.eventId, currentTrack.trackId, currentTrack.eKin, lvolID, currentTrack.stepCounter,
                 currentTrack.zeroStepCounter);
        trackSurvives = false;
        energyDeposit += currentTrack.eKin;
        currentTrack.eKin = 0.;
        // check for experiment-specific SteppingAction
      } else {
        SteppingActionT::GammaAction(trackSurvives, currentTrack.eKin, energyDeposit, currentTrack.pos,
                                     currentTrack.globalTime, auxData.fMCIndex, &g4HepEmData, params);
      }

      // this one always needs to be last as it needs to be done only if the track survives
      if (trackSurvives) {
        if (InFlightStats->perEventInFlightPrevious[currentTrack.threadId] <
                allowFinishOffEvent[currentTrack.threadId] &&
            InFlightStats->perEventInFlightPrevious[currentTrack.threadId] != 0) {
          if (printErrors) {
            printf(
                "Thread %d Finishing gamma of the %d last particles of event %d on CPU E=%f lvol=%d after %d steps.\n",
                currentTrack.threadId, InFlightStats->perEventInFlightPrevious[currentTrack.threadId],
                currentTrack.eventId, currentTrack.eKin, lvolID, currentTrack.stepCounter);
          }

          // Set LeakStatus and copy to leaked queue
          currentTrack.leakStatus = LeakStatus::FinishEventOnCPU;
          particleManager.gammas.CopyTrackToLeaked(slot);
          continue;
        }
      } else {
        // Free the slot of the killed track
        slotManager.MarkSlotForFreeing(slot);

        // In case the last steps are recorded, record it now, as this track is killed
        if (returnLastStep) {
          adept_scoring::RecordHit(userScoring,
                                   currentTrack.trackId,                        // Track ID
                                   currentTrack.parentId,                       // parent Track ID
                                   static_cast<short>(10),                      // step limiting process ID
                                   2,                                           // Particle type
                                   theTrack->GetGStepLength(),                  // Step length
                                   energyDeposit,                               // Total Edep
                                   currentTrack.weight,                         // Track weight
                                   currentTrack.navState,                       // Pre-step point navstate
                                   currentTrack.preStepPos,                     // Pre-step point position
                                   currentTrack.preStepDir,                     // Pre-step point momentum direction
                                   currentTrack.preStepEKin,                    // Pre-step point kinetic energy
                                   currentTrack.navState,                       // Post-step point navstate
                                   currentTrack.pos,                            // Post-step point position
                                   currentTrack.dir,                            // Post-step point momentum direction
                                   currentTrack.eKin,                           // Post-step point kinetic energy
                                   currentTrack.globalTime,                     // global time
                                   currentTrack.localTime,                      // local time
                                   currentTrack.preStepGlobalTime,              // preStep global time
                                   currentTrack.eventId, currentTrack.threadId, // eventID and threadID
                                   true,                                        // whether this was the last step
                                   currentTrack.stepCounter,                    // stepcounter
                                   nullptr,                                     // pointer to secondary init data
                                   0);                                          // number of secondaries
        }
        continue; // track is killed, can stop here
      }

      // ---- End of SteppingAction
    }

    // Call G4HepEm to compute the physics step limit.
    G4HepEmGammaManager::HowFar(&g4HepEmData, &g4HepEmPars, &gammaTrack);

    // Particles that were not cut or leaked are added to the queue used by the next kernels
    propagationQueue->push_back(slot);
  }
}

__global__ void GammaPropagation(Track *gammas, G4HepEmGammaTrack *hepEMTracks, const adept::MParray *active)
{
  constexpr double kPushDistance           = 1000 * vecgeom::kTolerance;
  constexpr double kPushStuck              = 100 * vecgeom::kTolerance;
  constexpr unsigned short kStepsStuckPush = 5;

  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    Track &currentTrack = gammas[slot];

    int lvolID = currentTrack.navState.GetLogicalId();

    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = gammaTrack.GetTrack();

    // Check if there's a volume boundary in between.
#ifdef ADEPT_USE_SURF
    currentTrack.hitsurfID = -1;
    auto geometryStepLength =
        AdePTNavigator::ComputeStepAndNextVolume(currentTrack.pos, currentTrack.dir, theTrack->GetGStepLength(),
                                                 currentTrack.navState, currentTrack.nextState, currentTrack.hitsurfID);
#else
    auto geometryStepLength =
        AdePTNavigator::ComputeStepAndNextVolume(currentTrack.pos, currentTrack.dir, theTrack->GetGStepLength(),
                                                 currentTrack.navState, currentTrack.nextState, kPushDistance);
#endif
    //  printf("pvol=%d  step=%g  onboundary=%d  pos={%g, %g, %g}  dir={%g, %g, %g}\n", navState.TopId(),
    //  geometryStepLength,
    //         nextState.IsOnBoundary(), pos[0], pos[1], pos[2], dir[0], dir[1], dir[2]);

    if (geometryStepLength < kPushStuck && geometryStepLength < theTrack->GetGStepLength()) {
      currentTrack.zeroStepCounter++;
      if (currentTrack.zeroStepCounter > kStepsStuckPush) currentTrack.pos += kPushStuck * currentTrack.dir;
    } else
      currentTrack.zeroStepCounter = 0;

    currentTrack.pos += geometryStepLength * currentTrack.dir;

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    currentTrack.navState.SetBoundaryState(currentTrack.nextState.IsOnBoundary());

    // Propagate information from geometrical step to G4HepEm.
    theTrack->SetGStepLength(geometryStepLength);
    theTrack->SetOnBoundary(currentTrack.nextState.IsOnBoundary());

    // Update the flight times of the particle
    double deltaTime = geometryStepLength / copcore::units::kCLight;
    currentTrack.globalTime += deltaTime;
    currentTrack.localTime += deltaTime;
  }
}

template <typename Scoring>
__global__ void GammaSetupInteractions(G4HepEmGammaTrack *hepEMTracks, const adept::MParray *propagationQueue,
                                       ParticleManager particleManager, AllInteractionQueues interactionQueues,
                                       Scoring *userScoring, const bool returnAllSteps, const bool returnLastStep)
{
  int activeSize = propagationQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*propagationQueue)[i];
    Track &currentTrack = particleManager.gammas.TrackAt(slot);

    int lvolID = currentTrack.navState.GetLogicalId();

    // Write local variables back into track and enqueue
    auto survive = [&](LeakStatus leakReason = LeakStatus::NoLeak) {
      currentTrack.leakStatus = leakReason;
      if (leakReason != LeakStatus::NoLeak) {
        // Copy track at slot to the leaked tracks
        particleManager.gammas.CopyTrackToLeaked(slot);
      } else {
        particleManager.gammas.EnqueueNext(slot);
      }
    };

    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = gammaTrack.GetTrack();

    if (currentTrack.nextState.IsOnBoundary()) {
      interactionQueues.queues[4]->push_back(slot);
      continue;
    } else {
      currentTrack.restrictedPhysicalStepLength = false;

      // NOTE: This may be moved to the next kernel
      G4HepEmGammaManager::SampleInteraction(&g4HepEmData, &gammaTrack, currentTrack.Uniform());

      // Reset number of interaction left for the winner discrete process also in the currentTrack
      // (SampleInteraction() resets it for theTrack), will be resampled in the next iteration.
      theTrack->SetNumIALeft(-1.0, 0);

      if (theTrack->GetWinnerProcessIndex() < 3) {
        interactionQueues.queues[theTrack->GetWinnerProcessIndex()]->push_back(slot);
      } else {
        // Gamma nuclear needs to be handled by Geant4 directly, passing track back to CPU
        survive(LeakStatus::GammaNuclear);

        // Record last step to enable UserPostTrackingAction to be called
        if (returnAllSteps || returnLastStep) {
          adept_scoring::RecordHit(
              userScoring,
              currentTrack.trackId,                        // Track ID
              currentTrack.parentId,                       // parent Track ID
              static_cast<short>(3),                       // step defining process ID
              2,                                           // Particle type
              theTrack->GetGStepLength(),                  // Step length
              0,                                           // Total Edep
              currentTrack.weight,                         // Track weight
              currentTrack.navState,                       // Pre-step point navstate
              currentTrack.preStepPos,                     // Pre-step point position
              currentTrack.preStepDir,                     // Pre-step point momentum direction
              currentTrack.preStepEKin,                    // Pre-step point kinetic energy
              currentTrack.nextState,                      // Post-step point navstate
              currentTrack.pos,                            // Post-step point position
              currentTrack.dir,                            // Post-step point momentum direction
              0,                                           // Post-step point kinetic energy
              currentTrack.globalTime,                     // global time
              currentTrack.localTime,                      // local time
              currentTrack.preStepGlobalTime,              // preStep global time
              currentTrack.eventId, currentTrack.threadId, // event and thread ID
              true, // whether this is the last step of the track: true as gamma nuclear kills the gamma
              currentTrack.stepCounter, // stepcounter
              nullptr,                  // pointer to secondary init data
              0);                       // number of secondaries
        }
      }
    }
  }
}

template <typename Scoring>
__global__ void GammaRelocation(G4HepEmGammaTrack *hepEMTracks, ParticleManager particleManager,
                                adept::MParray *relocatingQueue, Scoring *userScoring, const bool returnAllSteps,
                                const bool returnLastStep)
{
  constexpr double kPushDistance = 1000 * vecgeom::kTolerance;
  auto &slotManager              = *particleManager.gammas.fSlotManager;
  int activeSize                 = relocatingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*relocatingQueue)[i];
    Track &currentTrack = particleManager.gammas.TrackAt(slot);

    int lvolID          = currentTrack.navState.GetLogicalId();
    bool enterWDTRegion = false;

    // Write local variables back into track and enqueue
    auto survive = [&](LeakStatus leakReason = LeakStatus::NoLeak) {
      currentTrack.leakStatus = leakReason;
      if (leakReason != LeakStatus::NoLeak) {
        // Copy track at slot to the leaked tracks
        particleManager.gammas.CopyTrackToLeaked(slot);
      } else {
        if (!enterWDTRegion) {
          particleManager.gammas.EnqueueNext(slot);
        } else {
          particleManager.gammasWDT.EnqueueNext(slot);
        }
      }
    };

    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = gammaTrack.GetTrack();

    currentTrack.restrictedPhysicalStepLength = true;
    // For now, just count that we hit something.

    // Kill the particle if it left the world.
    if (!currentTrack.nextState.IsOutside()) {

      G4HepEmGammaManager::UpdateNumIALeft(theTrack);

#ifdef ADEPT_USE_SURF
      AdePTNavigator::RelocateToNextVolume(currentTrack.pos, currentTrack.dir, currentTrack.hitsurfID,
                                           currentTrack.nextState);
#else
      AdePTNavigator::RelocateToNextVolume(currentTrack.pos, currentTrack.dir, currentTrack.nextState);
#endif

      // if all steps are returned, we need to record the hit here,
      // as now the nextState is defined, but the navState is not yet replaced
      if (returnAllSteps)
        adept_scoring::RecordHit(userScoring,
                                 currentTrack.trackId,                        // Track ID
                                 currentTrack.parentId,                       // parent Track ID
                                 static_cast<short>(10),                      // step defining process ID
                                 2,                                           // Particle type
                                 theTrack->GetGStepLength(),                  // Step length
                                 0,                                           // Total Edep
                                 currentTrack.weight,                         // Track weight
                                 currentTrack.navState,                       // Pre-step point navstate
                                 currentTrack.preStepPos,                     // Pre-step point position
                                 currentTrack.preStepDir,                     // Pre-step point momentum direction
                                 currentTrack.preStepEKin,                    // Pre-step point kinetic energy
                                 currentTrack.nextState,                      // Post-step point navstate
                                 currentTrack.pos,                            // Post-step point position
                                 currentTrack.dir,                            // Post-step point momentum direction
                                 currentTrack.eKin,                           // Post-step point kinetic energy
                                 currentTrack.globalTime,                     // global time
                                 currentTrack.localTime,                      // local time
                                 currentTrack.preStepGlobalTime,              // preStep global time
                                 currentTrack.eventId, currentTrack.threadId, // event and thread ID
                                 false,                    // whether this is the last step of the track
                                 currentTrack.stepCounter, // stepcounter
                                 nullptr,                  // pointer to secondary init data
                                 0);                       // number of secondaries

      // Move to the next boundary.
      currentTrack.navState = currentTrack.nextState;
      // printf("  -> pvol=%d pos={%g, %g, %g} \n", navState.TopId(), pos[0], pos[1], pos[2]);
      //  Check if the next volume belongs to the GPU region and push it to the appropriate queue
      const int nextlvolID          = currentTrack.nextState.GetLogicalId();
      VolAuxData const &nextauxData = AsyncAdePT::gVolAuxData[nextlvolID];
      if (nextauxData.fGPUregionId >= 0) {

        // Check whether next region is a Woodcock tracking region
        const adeptint::WDTDeviceView &view = gWDTData;
        const int wdtIdx = view.regionToWDT[nextauxData.fGPUregionId]; // index into view.regions (or -1)
        if (wdtIdx >= 0) {
          const adeptint::WDTRegion reg = view.regions[wdtIdx];
          // minimal energy for Woodcock tracking succeeded, do Woodcock tracking
          if (currentTrack.eKin > reg.ekinMin) {
            enterWDTRegion = true;
          }
        }

        theTrack->SetMCIndex(nextauxData.fMCIndex);
        survive();
      } else {
        // To be safe, just push a bit the track exiting the GPU region to make sure
        // Geant4 does not relocate it again inside the same region
        currentTrack.pos += kPushDistance * currentTrack.dir;
        survive(LeakStatus::OutOfGPURegion);
      }
    } else {
      // release slot for particle that has left the world
      slotManager.MarkSlotForFreeing(slot);

      // particle has left the world, record hit if last or all steps are returned
      if (returnAllSteps || returnLastStep)
        adept_scoring::RecordHit(
            userScoring,
            currentTrack.trackId,                        // Track ID
            currentTrack.parentId,                       // parent Track ID
            static_cast<short>(10),                      // step defining process ID
            2,                                           // Particle type
            theTrack->GetGStepLength(),                  // Step length
            0,                                           // Total Edep
            currentTrack.weight,                         // Track weight
            currentTrack.navState,                       // Pre-step point navstate
            currentTrack.preStepPos,                     // Pre-step point position
            currentTrack.preStepDir,                     // Pre-step point momentum direction
            currentTrack.preStepEKin,                    // Pre-step point kinetic energy
            currentTrack.nextState,                      // Post-step point navstate
            currentTrack.pos,                            // Post-step point position
            currentTrack.dir,                            // Post-step point momentum direction
            currentTrack.eKin,                           // Post-step point kinetic energy
            currentTrack.globalTime,                     // global time
            currentTrack.localTime,                      // local time
            currentTrack.preStepGlobalTime,              // preStep global time
            currentTrack.eventId, currentTrack.threadId, // event and thread ID
            true, // whether this is the last step of the track: true, as particle has left the world
            currentTrack.stepCounter, // stepcounter
            nullptr,                  // pointer to secondary init data
            0);                       // number of secondaries
    }
    continue;
  }
}

template <typename Scoring>
__global__ void GammaConversion(G4HepEmGammaTrack *hepEMTracks, ParticleManager particleManager,
                                adept::MParray *interactingQueue, Scoring *userScoring, const bool returnAllSteps,
                                const bool returnLastStep)
{
  auto &slotManager = *particleManager.gammas.fSlotManager;
  int activeSize    = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*interactingQueue)[i];
    Track &currentTrack = particleManager.gammas.TrackAt(slot);

    int lvolID                = currentTrack.navState.GetLogicalId();
    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID];

    // Write local variables back into track and enqueue
    auto survive = [&]() {
      const bool useWDT = ShouldUseWDT(currentTrack.navState, currentTrack.eKin);
      if (!useWDT) {
        particleManager.gammas.EnqueueNext(slot);
      } else {
        particleManager.gammasWDT.EnqueueNext(slot);
      }
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

    double edep = 0.;

    // Interaction

    // Invoke gamma conversion to e-/e+ pairs, if the energy is above the threshold.
    if (currentTrack.eKin < 2 * copcore::units::kElectronMassC2) {
      survive();
      continue;
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

    // data structure for possible secondaries that are generated
    SecondaryInitData secondaryData[2];
    unsigned int nSecondaries = 0;

    // Check the cuts and deposit energy in this volume if needed
    if (ApplyCuts && elKinEnergy < theElCut) {
      // Deposit the energy here and kill the secondary
      edep = elKinEnergy;
    } else {
      Track &electron = particleManager.electrons.NextTrack(
          newRNG, elKinEnergy, currentTrack.pos,
          vecgeom::Vector3D<double>{dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]}, currentTrack.navState,
          currentTrack, currentTrack.globalTime);

      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        secondaryData[nSecondaries++] = {electron.trackId, electron.dir, electron.eKin, /*particle type*/ char(0)};
      }
    }

    if (ApplyCuts && (copcore::units::kElectronMassC2 < theGammaCut && posKinEnergy < thePosCut)) {
      // Deposit: posKinEnergy + 2 * copcore::units::kElectronMassC2 and kill the secondary
      edep += posKinEnergy + 2 * copcore::units::kElectronMassC2;
    } else {
      Track &positron = particleManager.positrons.NextTrack(
          currentTrack.rngState, posKinEnergy, currentTrack.pos,
          vecgeom::Vector3D<double>{dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]}, currentTrack.navState,
          currentTrack, currentTrack.globalTime);

      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        secondaryData[nSecondaries++] = {positron.trackId, positron.dir, positron.eKin, /*particle type*/ char(1)};
      }
    }

    // The current track is killed by not enqueuing into the next activeQueue and the slot is released
    slotManager.MarkSlotForFreeing(slot);

    assert(nSecondaries <= 2);

    // If there is some edep from cutting particles, record the step
    if ((edep > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep) {
      adept_scoring::RecordHit(
          userScoring,
          currentTrack.trackId,                        // Track ID
          currentTrack.parentId,                       // parent Track ID
          static_cast<short>(0),                       // step defining process ID
          2,                                           // Particle type
          theTrack->GetGStepLength(),                  // Step length
          edep,                                        // Total Edep
          currentTrack.weight,                         // Track weight
          currentTrack.navState,                       // Pre-step point navstate
          currentTrack.preStepPos,                     // Pre-step point position
          currentTrack.preStepDir,                     // Pre-step point momentum direction
          currentTrack.preStepEKin,                    // Pre-step point kinetic energy
          currentTrack.nextState,                      // Post-step point navstate
          currentTrack.pos,                            // Post-step point position
          currentTrack.dir,                            // Post-step point momentum direction
          0.,                                          // Post-step point kinetic energy (0 after conversion)
          currentTrack.globalTime,                     // global time
          currentTrack.localTime,                      // local time
          currentTrack.preStepGlobalTime,              // preStep global time
          currentTrack.eventId, currentTrack.threadId, // event and thread ID
          true, // whether this is the last step of the track: always true as gammas undergoing conversion are killed
          currentTrack.stepCounter, // stepcounter
          secondaryData,            // pointer to secondary init data
          nSecondaries);            // number of secondaries
    }
  }
}

template <typename Scoring>
__global__ void GammaCompton(G4HepEmGammaTrack *hepEMTracks, ParticleManager particleManager,
                             adept::MParray *interactingQueue, Scoring *userScoring, const bool returnAllSteps,
                             const bool returnLastStep)
{
  auto &slotManager = *particleManager.gammas.fSlotManager;
  int activeSize    = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*interactingQueue)[i];
    Track &currentTrack = particleManager.gammas.TrackAt(slot);

    int lvolID                = currentTrack.navState.GetLogicalId();
    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID];
    bool trackSurvives        = false;

    // Write local variables back into track and enqueue
    auto survive = [&]() {
      trackSurvives     = true; // particle survived
      const bool useWDT = ShouldUseWDT(currentTrack.navState, currentTrack.eKin);
      if (!useWDT) {
        particleManager.gammas.EnqueueNext(slot);
      } else {
        particleManager.gammasWDT.EnqueueNext(slot);
      }
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
      continue;
    }
    const double origDirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
    double dirPrimary[3];
    newEnergyGamma = G4HepEmGammaInteractionCompton::SamplePhotonEnergyAndDirection(currentTrack.eKin, dirPrimary,
                                                                                    origDirPrimary, &rnge);
    vecgeom::Vector3D<double> newDirGamma(dirPrimary[0], dirPrimary[1], dirPrimary[2]);

    const double energyEl = currentTrack.eKin - newEnergyGamma;

    adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

    // data structure for possible secondaries that are generated
    SecondaryInitData secondaryData[1];
    unsigned int nSecondaries = 0;

    // Check the cuts and deposit energy in this volume if needed
    if (ApplyCuts ? energyEl > theElCut : energyEl > LowEnergyThreshold) {
      // Create a secondary electron and sample/compute directions.
      Track &electron = particleManager.electrons.NextTrack(
          newRNG, energyEl, currentTrack.pos, currentTrack.eKin * currentTrack.dir - newEnergyGamma * newDirGamma,
          currentTrack.navState, currentTrack, currentTrack.globalTime);
      electron.dir.Normalize();

      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        secondaryData[nSecondaries++] = {electron.trackId, electron.dir, electron.eKin, /*particle type*/ char(0)};
      }

    } else {
      edep = energyEl;
    }

    // Check the new gamma energy and deposit if below threshold.
    // Using same hardcoded very LowEnergyThreshold as G4HepEm
    if (newEnergyGamma > LowEnergyThreshold) {
      currentTrack.eKin = newEnergyGamma;
      theTrack->SetEKin(currentTrack.eKin);
      currentTrack.dir = newDirGamma;
      survive();
    } else {
      edep += newEnergyGamma;
      newEnergyGamma = 0.;
      // The current track is killed by not enqueuing into the next activeQueue and the slot is released
      slotManager.MarkSlotForFreeing(slot);
    }

    //////////////

    assert(nSecondaries <= 1);

    // If there is some edep from cutting particles, record the step
    // Note: step must be returned even if track dies or secondaries are generated
    if ((edep > 0 && auxData.fSensIndex >= 0) || returnAllSteps ||
        (returnLastStep && (nSecondaries > 0 || !trackSurvives))) {
      adept_scoring::RecordHit(userScoring,
                               currentTrack.trackId,                        // Track ID
                               currentTrack.parentId,                       // parent Track ID
                               static_cast<short>(1),                       // step defining process ID
                               2,                                           // Particle type
                               theTrack->GetGStepLength(),                  // Step length
                               edep,                                        // Total Edep
                               currentTrack.weight,                         // Track weight
                               currentTrack.navState,                       // Pre-step point navstate
                               currentTrack.preStepPos,                     // Pre-step point position
                               currentTrack.preStepDir,                     // Pre-step point momentum direction
                               currentTrack.preStepEKin,                    // Pre-step point kinetic energy
                               currentTrack.nextState,                      // Post-step point navstate
                               currentTrack.pos,                            // Post-step point position
                               currentTrack.dir,                            // Post-step point momentum direction
                               newEnergyGamma,                              // Post-step point kinetic energy
                               currentTrack.globalTime,                     // global time
                               currentTrack.localTime,                      // local time
                               currentTrack.preStepGlobalTime,              // preStep global time
                               currentTrack.eventId, currentTrack.threadId, // event and thread ID
                               !trackSurvives,           // whether this is the last step of the track
                               currentTrack.stepCounter, // stepcounter
                               secondaryData,            // pointer to secondary init data
                               nSecondaries);            // number of secondaries
    }
  }
}

template <typename Scoring>
__global__ void GammaPhotoelectric(G4HepEmGammaTrack *hepEMTracks, ParticleManager particleManager,
                                   adept::MParray *interactingQueue, Scoring *userScoring, const bool returnAllSteps,
                                   const bool returnLastStep)
{
  auto &slotManager = *particleManager.gammas.fSlotManager;
  int activeSize    = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*interactingQueue)[i];
    Track &currentTrack = particleManager.gammas.TrackAt(slot);

    int lvolID                = currentTrack.navState.GetLogicalId();
    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID];

    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = gammaTrack.GetTrack();

    // Perform the discrete interaction.
    G4HepEmRandomEngine rnge(&currentTrack.rngState);
    // We might need one branched RNG state, prepare while threads are synchronized.
    RanluxppDouble newRNG(currentTrack.rngState.Branch());

    const double theElCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecElProdCutE;

    const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
    const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

    double edep = 0.;

    // Interaction

    // Invoke photoelectric process.
    const double theLowEnergyThreshold = 1 * copcore::units::eV;

    const double bindingEnergy = G4HepEmGammaInteractionPhotoelectric::SelectElementBindingEnergy(
        &g4HepEmData, auxData.fMCIndex, gammaTrack.GetPEmxSec(), currentTrack.eKin, &rnge);

    edep                    = bindingEnergy;
    const double photoElecE = currentTrack.eKin - edep;

    // data structure for possible secondaries that are generated
    SecondaryInitData secondaryData[1];
    unsigned int nSecondaries = 0;

    if (ApplyCuts ? photoElecE > theElCut : photoElecE > theLowEnergyThreshold) {

      adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

      double dirGamma[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirPhotoElec[3];
      G4HepEmGammaInteractionPhotoelectric::SamplePhotoElectronDirection(photoElecE, dirGamma, dirPhotoElec, &rnge);

      // Create a secondary electron and sample directions.
      Track &electron = particleManager.electrons.NextTrack(
          newRNG, photoElecE, currentTrack.pos,
          vecgeom::Vector3D<double>{dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]}, currentTrack.navState,
          currentTrack, currentTrack.globalTime);

      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        secondaryData[nSecondaries++] = {electron.trackId, electron.dir, electron.eKin, /*particle type*/ char(0)};
      }

    } else {
      // If the secondary electron is cut, deposit all the energy of the gamma in this volume
      edep = currentTrack.eKin;
    }
    // The current track is killed by not enqueuing into the next activeQueue and the slot is released
    slotManager.MarkSlotForFreeing(slot);

    //////////////

    assert(nSecondaries <= 1);

    // If there is some edep from cutting particles, record the step
    if ((edep > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep) {
      adept_scoring::RecordHit(userScoring,
                               currentTrack.trackId,           // Track ID
                               currentTrack.parentId,          // parent Track ID
                               static_cast<short>(2),          // step defining process ID
                               2,                              // Particle type
                               theTrack->GetGStepLength(),     // Step length
                               edep,                           // Total Edep
                               currentTrack.weight,            // Track weight
                               currentTrack.navState,          // Pre-step point navstate
                               currentTrack.preStepPos,        // Pre-step point position
                               currentTrack.preStepDir,        // Pre-step point momentum direction
                               currentTrack.preStepEKin,       // Pre-step point kinetic energy
                               currentTrack.nextState,         // Post-step point navstate
                               currentTrack.pos,               // Post-step point position
                               currentTrack.dir,               // Post-step point momentum direction
                               0.,                             // Post-step point kinetic energy (0 after photoelectric)
                               currentTrack.globalTime,        // global time
                               currentTrack.localTime,         // local time
                               currentTrack.preStepGlobalTime, // preStep global time
                               currentTrack.eventId, currentTrack.threadId, // event and thread ID
                               true, // whether this is the last step of the track: always true as gammas undergoing the
                                     // PhotoElectric effect are killed
                               currentTrack.stepCounter, // stepcounter
                               secondaryData,            // pointer to secondary init data
                               nSecondaries);            // number of secondaries
    }
  }
}

} // namespace AsyncAdePT
