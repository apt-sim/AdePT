// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/navigation/AdePTNavigator.h>
#include <AdePT/kernels/gammasWDT.cuh>

#include <AdePT/copcore/PhysicalConstants.h>
#include <AdePT/core/TrackDebug.cuh>

#include <G4HepEmGammaManager.hh>
#include <G4HepEmGammaTrack.hh>
#include <G4HepEmTrack.hh>
#include <G4HepEmGammaInteractionCompton.hh>
#include <G4HepEmGammaInteractionConversion.hh>
#include <G4HepEmGammaInteractionPhotoelectric.hh>

using VolAuxData      = adeptint::VolAuxData;
using StepActionParam = adept::SteppingAction::Params;

namespace AsyncAdePT {
// Asynchronous TransportGammas Interface
template <typename Scoring, class SteppingActionT>
__global__ void __launch_bounds__(256, 1)
    TransportGammas(ParticleManager particleManager, Scoring *userScoring, Stats *InFlightStats,
                    const StepActionParam params, AllowFinishOffEventArray allowFinishOffEvent,
                    const bool returnAllSteps, const bool returnLastStep)
{
  constexpr double kPushDistance    = 1000 * vecgeom::kTolerance;
  constexpr unsigned short maxSteps = 10'000;
  auto &slotManager                 = *particleManager.gammas.fSlotManager;
  const int activeSize              = particleManager.gammas.ActiveSize();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const auto slot     = particleManager.gammas.ActiveAt(i);
    Track &currentTrack = particleManager.gammas.TrackAt(slot);

    auto navState             = currentTrack.navState;
    int lvolID                = navState.GetLogicalId();
    VolAuxData const &auxData = gVolAuxData[lvolID];

    bool trackSurvives                       = false;
    bool enterWDTRegion                      = false;
    LeakStatus leakReason                    = LeakStatus::NoLeak;
    short stepDefinedProcessId               = 10; // default for transportation
    double edep                              = 0.;
    constexpr double kPushStuck              = 100 * vecgeom::kTolerance;
    constexpr unsigned short kStepsStuckPush = 5;
    constexpr unsigned short kStepsStuckKill = 25;
    auto eKin                                = currentTrack.eKin;
    auto preStepEnergy                       = eKin;
    auto pos                                 = currentTrack.pos;
    vecgeom::Vector3D<double> preStepPos(pos);
    auto dir = currentTrack.dir;
    vecgeom::Vector3D<double> preStepDir(dir);
    double globalTime        = currentTrack.globalTime;
    double preStepGlobalTime = currentTrack.globalTime;
    double localTime         = currentTrack.localTime;
    double properTime        = currentTrack.properTime;
    vecgeom::NavigationState nextState;

    currentTrack.stepCounter++;
    bool printErrors = true;
#if ADEPT_DEBUG_TRACK > 0
    bool verbose = false;
    if (gTrackDebug.active) {
      verbose =
          currentTrack.Matches(gTrackDebug.event_id, gTrackDebug.track_id, gTrackDebug.min_step, gTrackDebug.max_step);
      if (verbose) currentTrack.Print("gamma");
      printErrors = !gTrackDebug.active || verbose;
    }
#endif

    // Write local variables back into track and enqueue
    auto survive = [&]() {
      currentTrack.eKin       = eKin;
      currentTrack.pos        = pos;
      currentTrack.dir        = dir;
      currentTrack.globalTime = globalTime;
      currentTrack.localTime  = localTime;
      currentTrack.properTime = properTime;
      currentTrack.navState   = nextState;
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

#if ADEPT_DEBUG_TRACK > 0
    if (verbose) printf("| geometricalStepLengthFromPhysics %g ", geometricalStepLengthFromPhysics);
#endif

    // Leave the range and MFP inside the G4HepEmTrack. If we split kernels, we
    // also need to carry them over!

    // Check if there's a volume boundary in between.
    double geometryStepLength;
#ifdef ADEPT_USE_SURF
    long hitsurf_index = -1;
    geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics, navState,
                                                                  nextState, hitsurf_index);
#else
    geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics, navState,
                                                                  nextState, kPushDistance);
#endif
    if (geometryStepLength < kPushStuck && geometryStepLength < geometricalStepLengthFromPhysics) {
      currentTrack.zeroStepCounter++;
      if (currentTrack.zeroStepCounter > kStepsStuckPush) geometryStepLength = kPushStuck;
    } else
      currentTrack.zeroStepCounter = 0;

    pos += geometryStepLength * dir;

#if ADEPT_DEBUG_TRACK > 0
    if (verbose) {
      if (currentTrack.zeroStepCounter > kStepsStuckPush) printf("| STUCK TRACK PUSHED ");
      printf("| geometryStepLength %g | propagated_pos {%.19f, %.19f, %.19f} ", geometryStepLength, pos[0], pos[1],
             pos[2]);
      if (geometryStepLength < 100 * vecgeom::kTolerance) printf("| SMALL STEP ");
      nextState.Print();
    }
#endif

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    navState.SetBoundaryState(nextState.IsOnBoundary());

    // Propagate information from geometrical step to G4HepEm.
    theTrack->SetGStepLength(geometryStepLength);
    theTrack->SetOnBoundary(nextState.IsOnBoundary());

    // Update the flight times of the particle
    double deltaTime = theTrack->GetGStepLength() / copcore::units::kCLight;
    globalTime += deltaTime;
    localTime += deltaTime;

    int winnerProcessIndex;
    if (nextState.IsOnBoundary()) {
      // For now, just count that we hit something.

      // Kill the particle if it left the world.
      if (!nextState.IsOutside()) {

        G4HepEmGammaManager::UpdateNumIALeft(theTrack);

        // Save the `number-of-interaction-left` in our track.
        // Use index 0 since numIALeft stores for gammas only the total macroscopic cross section
        double numIALeft          = theTrack->GetNumIALeft(0);
        currentTrack.numIALeft[0] = numIALeft;

#ifdef ADEPT_USE_SURF
        AdePTNavigator::RelocateToNextVolume(pos, dir, hitsurf_index, nextState);
#else
        AdePTNavigator::RelocateToNextVolume(pos, dir, nextState);
#endif

#if ADEPT_DEBUG_TRACK > 0
        if (verbose) {
          printf("| CROSSED into ");
          nextState.Print();
        }
#endif

        //  Check if the next volume belongs to the GPU region and push it to the appropriate queue
        const int nextlvolID          = nextState.GetLogicalId();
        VolAuxData const &nextauxData = gVolAuxData[nextlvolID];
        const auto regionId           = nextauxData.fGPUregionId;

        // next region is a GPU region
        if (regionId >= 0) {

          const adeptint::WDTDeviceView &view = gWDTData;
          const int wdtIdx                    = view.regionToWDT[regionId]; // index into view.regions (or -1)

          // next region is a Woodcock tracking region
          if (wdtIdx >= 0) {
            const adeptint::WDTRegion reg = view.regions[wdtIdx];
            // minimal energy for Woodcock tracking succeeded, do Woodcock tracking
            if (eKin > reg.ekinMin) {
              enterWDTRegion = true;
            }
          }
          trackSurvives = true;
        } else {
          // To be safe, just push a bit the track exiting the GPU region to make sure
          // Geant4 does not relocate it again inside the same region
          pos += kPushDistance * dir;

#if ADEPT_DEBUG_TRACK > 0
          if (verbose) printf("\n| track leaked to Geant4\n");
#endif

          trackSurvives = true;
          leakReason    = LeakStatus::OutOfGPURegion;
        }
      } // else particle has left the world

    } else {
      // a gamma that is not on boundary does an interaction

      G4HepEmGammaManager::SampleInteraction(&g4HepEmData, &gammaTrack, currentTrack.Uniform());
      winnerProcessIndex = theTrack->GetWinnerProcessIndex();

#if ADEPT_DEBUG_TRACK > 0
      if (verbose) printf("| winnerProc %d\n", winnerProcessIndex);
#endif

      // Reset number of interaction left for the winner discrete process also in the currentTrack
      // (SampleInteraction() resets it for theTrack), will be resampled in the next iteration.
      currentTrack.numIALeft[0] = -1.0;

      // Perform the discrete interaction.
      G4HepEmRandomEngine rnge(&currentTrack.rngState);
      // We might need one branched RNG state, prepare while threads are synchronized.
      RanluxppDouble newRNG(currentTrack.rngState.Branch());

      const double theElCut    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecElProdCutE;
      const double thePosCut   = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecPosProdCutE;
      const double theGammaCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecGamProdCutE;

      const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
      const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

      // discrete interaction reached, setting the stepDefinedProcess to the winnerProcess
      stepDefinedProcessId = winnerProcessIndex;

      switch (winnerProcessIndex) {
      case 0: {
        // Invoke gamma conversion to e-/e+ pairs, if the energy is above the threshold.
        if (eKin < 2 * copcore::units::kElectronMassC2) {
          trackSurvives = true;
          break;
        }

#if ADEPT_DEBUG_TRACK > 0
        if (verbose) printf("| GAMMA CONVERSION ");
#endif

        double logEnergy = std::log(eKin);
        double elKinEnergy, posKinEnergy;
        G4HepEmGammaInteractionConversion::SampleKinEnergies(&g4HepEmData, eKin, logEnergy, auxData.fMCIndex,
                                                             elKinEnergy, posKinEnergy, &rnge);

        double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
        double dirSecondaryEl[3], dirSecondaryPos[3];
        G4HepEmGammaInteractionConversion::SampleDirections(dirPrimary, dirSecondaryEl, dirSecondaryPos, elKinEnergy,
                                                            posKinEnergy, &rnge);

        adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 1, /*numGammas*/ 0);

        // Check the cuts and deposit energy in this volume if needed
        if (ApplyCuts && elKinEnergy < theElCut) {
          // Deposit the energy here and kill the secondary
          edep = elKinEnergy;
        } else {
          Track &electron = particleManager.electrons.NextTrack(
              newRNG, elKinEnergy, pos,
              vecgeom::Vector3D<double>{dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]}, navState,
              currentTrack, globalTime);

          // if tracking or stepping action is called, return initial step
          if (returnLastStep) {
            adept_scoring::RecordHit(userScoring, electron.trackId, electron.parentId,
                                     /*CreatorProcessId*/ short(winnerProcessIndex),
                                     /* electron*/ 0, // Particle type
                                     0,               // Step length
                                     0,               // Total Edep
                                     electron.weight, // Track weight
                                     navState,        // Pre-step point navstate
                                     electron.pos,    // Pre-step point position
                                     electron.dir,    // Pre-step point momentum direction
                                     electron.eKin,   // Pre-step point kinetic energy
                                     navState,        // Post-step point navstate
                                     electron.pos,    // Post-step point position
                                     electron.dir,    // Post-step point momentum direction
                                     electron.eKin,   // Post-step point kinetic energy
                                     globalTime,      // global time
                                     0.,              // local time
                                     globalTime, // global time at preStepPoint, for initializingStep its the globalTime
                                     electron.eventId, electron.threadId, // eventID and threadID
                                     false,                               // whether this was the last step
                                     electron.stepCounter);               // whether this was the first step
          }
        }

        if (ApplyCuts && (copcore::units::kElectronMassC2 < theGammaCut && posKinEnergy < thePosCut)) {
          // Deposit: posKinEnergy + 2 * copcore::units::kElectronMassC2 and kill the secondary
          edep += posKinEnergy + 2 * copcore::units::kElectronMassC2;
        } else {
          Track &positron = particleManager.positrons.NextTrack(
              currentTrack.rngState, posKinEnergy, pos,
              vecgeom::Vector3D<double>{dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]}, navState,
              currentTrack, globalTime);

          // if tracking or stepping action is called, return initial step
          if (returnLastStep) {
            adept_scoring::RecordHit(userScoring, positron.trackId, positron.parentId,
                                     /*CreatorProcessId*/ short(winnerProcessIndex),
                                     /* positron*/ 1, // Particle type
                                     0,               // Step length
                                     0,               // Total Edep
                                     positron.weight, // Track weight
                                     navState,        // Pre-step point navstate
                                     positron.pos,    // Pre-step point position
                                     positron.dir,    // Pre-step point momentum direction
                                     positron.eKin,   // Pre-step point kinetic energy
                                     navState,        // Post-step point navstate
                                     positron.pos,    // Post-step point position
                                     positron.dir,    // Post-step point momentum direction
                                     positron.eKin,   // Post-step point kinetic energy
                                     globalTime,      // global time
                                     0.,              // local time
                                     globalTime, // global time at preStepPoint, for initializingStep its the globalTime
                                     positron.eventId, positron.threadId, // eventID and threadID
                                     false,                               // whether this was the last step
                                     positron.stepCounter);               // whether this was the first step
          }
        }
        eKin = 0.;
        break;
      }
      case 1: {
        // Invoke Compton scattering of gamma.

#if ADEPT_DEBUG_TRACK > 0
        if (verbose) printf("| COMPTON ");
#endif

        constexpr double LowEnergyThreshold = 100 * copcore::units::eV;
        if (eKin < LowEnergyThreshold) {
          trackSurvives = true;
          break;
        }
        const double origDirPrimary[] = {dir.x(), dir.y(), dir.z()};
        double dirPrimary[3];
        double newEnergyGamma =
            G4HepEmGammaInteractionCompton::SamplePhotonEnergyAndDirection(eKin, dirPrimary, origDirPrimary, &rnge);
        vecgeom::Vector3D<double> newDirGamma(dirPrimary[0], dirPrimary[1], dirPrimary[2]);

        const double energyEl = eKin - newEnergyGamma;

        adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

        // Check the cuts and deposit energy in this volume if needed
        if (ApplyCuts ? energyEl > theElCut : energyEl > LowEnergyThreshold) {
          // Create a secondary electron and sample/compute directions.
          Track &electron = particleManager.electrons.NextTrack(
              newRNG, energyEl, pos, eKin * dir - newEnergyGamma * newDirGamma, navState, currentTrack, globalTime);

          electron.dir.Normalize();

          // if tracking or stepping action is called, return initial step
          if (returnLastStep) {
            adept_scoring::RecordHit(userScoring, electron.trackId, electron.parentId,
                                     /*CreatorProcessId*/ short(winnerProcessIndex),
                                     /* electron*/ 0, // Particle type
                                     0,               // Step length
                                     0,               // Total Edep
                                     electron.weight, // Track weight
                                     navState,        // Pre-step point navstate
                                     electron.pos,    // Pre-step point position
                                     electron.dir,    // Pre-step point momentum direction
                                     electron.eKin,   // Pre-step point kinetic energy
                                     navState,        // Post-step point navstate
                                     electron.pos,    // Post-step point position
                                     electron.dir,    // Post-step point momentum direction
                                     electron.eKin,   // Post-step point kinetic energy
                                     globalTime,      // global time
                                     0.,              // local time
                                     globalTime, // global time at preStepPoint, for initializingStep its the globalTime
                                     electron.eventId, electron.threadId, // eventID and threadID
                                     false,                               // whether this was the last step
                                     electron.stepCounter);               // whether this was the first step
          }
        } else {
          edep = energyEl;
        }

        // Check the new gamma energy and deposit if below threshold.
        // Using same hardcoded very LowEnergyThreshold as G4HepEm
        if (newEnergyGamma > LowEnergyThreshold) {
          eKin          = newEnergyGamma;
          dir           = newDirGamma;
          trackSurvives = true;
        } else {
          edep += newEnergyGamma;
          eKin = 0.;
        }
        break;
      }
      case 2: {
        // Invoke photoelectric process.

#if ADEPT_DEBUG_TRACK > 0
        if (verbose) printf("| PHOTOELECTRIC ");
#endif

        const double theLowEnergyThreshold = 1 * copcore::units::eV;

        const double bindingEnergy = G4HepEmGammaInteractionPhotoelectric::SelectElementBindingEnergy(
            &g4HepEmData, auxData.fMCIndex, gammaTrack.GetPEmxSec(), eKin, &rnge);

        edep                    = bindingEnergy;
        const double photoElecE = eKin - edep;
        if (ApplyCuts ? photoElecE > theElCut : photoElecE > theLowEnergyThreshold) {

          adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

          double dirGamma[] = {dir.x(), dir.y(), dir.z()};
          double dirPhotoElec[3];
          G4HepEmGammaInteractionPhotoelectric::SamplePhotoElectronDirection(photoElecE, dirGamma, dirPhotoElec, &rnge);

          // Create a secondary electron and sample directions.
          Track &electron = particleManager.electrons.NextTrack(
              newRNG, photoElecE, pos, vecgeom::Vector3D<double>{dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]},
              navState, currentTrack, globalTime);

          // if tracking or stepping action is called, return initial step
          if (returnLastStep) {
            adept_scoring::RecordHit(userScoring, electron.trackId, electron.parentId,
                                     /*CreatorProcessId*/ short(winnerProcessIndex),
                                     /* electron*/ 0, // Particle type
                                     0,               // Step length
                                     0,               // Total Edep
                                     electron.weight, // Track weight
                                     navState,        // Pre-step point navstate
                                     electron.pos,    // Pre-step point position
                                     electron.dir,    // Pre-step point momentum direction
                                     electron.eKin,   // Pre-step point kinetic energy
                                     navState,        // Post-step point navstate
                                     electron.pos,    // Post-step point position
                                     electron.dir,    // Post-step point momentum direction
                                     electron.eKin,   // Post-step point kinetic energy
                                     globalTime,      // global time
                                     0.,              // local time
                                     globalTime, // global time at preStepPoint, for initializingStep its the globalTime
                                     electron.eventId, electron.threadId, // eventID and threadID
                                     false,                               // whether this was the last step
                                     electron.stepCounter);               // whether this was the first step
          }
        } else {
          // If the secondary electron is cut, deposit all the energy of the gamma in this volume
          edep = eKin;
          eKin = 0;
        }
        break;
      }
      case 3: {

#if ADEPT_DEBUG_TRACK > 0
        if (verbose) printf("| GAMMA-NUCLEAR ");
#endif
        // Gamma nuclear needs to be handled by Geant4 directly, passing track back to CPU
        leakReason = LeakStatus::GammaNuclear;
      }
      } // end switch (winnerProcessIndex)

    } // end if !onBoundary

    if (trackSurvives) {
      if (currentTrack.stepCounter >= maxSteps || currentTrack.zeroStepCounter > kStepsStuckKill) {
        if (printErrors)
          printf(
              "Killing gamma event %d track %lu E=%f lvol=%d after %d steps with zeroStepCounter %u. This indicates a "
              "stuck particle!\n",
              currentTrack.eventId, currentTrack.trackId, eKin, lvolID, currentTrack.stepCounter,
              currentTrack.zeroStepCounter);
        trackSurvives = false;
        edep += eKin;
        eKin = 0.;
      } else {
        // call experiment-specific SteppingAction:
        SteppingActionT::GammaAction(trackSurvives, eKin, edep, leakReason, pos, globalTime, auxData.fMCIndex,
                                     &g4HepEmData, params);
      }
    }

    // finishing on CPU must be last one only sets the LeakStatus but does not affect survival of the track
    if (trackSurvives && leakReason == LeakStatus::NoLeak) {
      if (InFlightStats->perEventInFlightPrevious[currentTrack.threadId] < allowFinishOffEvent[currentTrack.threadId] &&
          InFlightStats->perEventInFlightPrevious[currentTrack.threadId] != 0) {
        printf("Thread %d Finishing gamma of the %d last particles of event %d on CPU E=%f lvol=%d after %d steps.\n",
               currentTrack.threadId, InFlightStats->perEventInFlightPrevious[currentTrack.threadId],
               currentTrack.eventId, eKin, lvolID, currentTrack.stepCounter);

        leakReason = LeakStatus::FinishEventOnCPU;
      }
    }

    __syncwarp();

    // A track that survives must be enqueued to the leaks or the next queue.
    // Note: gamma nuclear does not survive but must still be leaked to the CPU, which is done
    // inside survive()
    if (trackSurvives || leakReason == LeakStatus::GammaNuclear) {
      survive();
    } else {
      // particles that don't survive are killed by not enqueing them to the next queue and freeing the slot
      slotManager.MarkSlotForFreeing(slot);
    }

    // If there is some edep from cutting particles, record the step
    if ((edep > 0 && auxData.fSensIndex >= 0) || returnAllSteps || (returnLastStep && !trackSurvives)) {
      adept_scoring::RecordHit(userScoring,
                               currentTrack.trackId,                        // Track ID
                               currentTrack.parentId,                       // parent Track ID
                               stepDefinedProcessId,                        // step-defining process id
                               2,                                           // Particle type
                               geometryStepLength,                          // Step length
                               edep,                                        // Total Edep
                               currentTrack.weight,                         // Track weight
                               navState,                                    // Pre-step point navstate
                               preStepPos,                                  // Pre-step point position
                               preStepDir,                                  // Pre-step point momentum direction
                               preStepEnergy,                               // Pre-step point kinetic energy
                               nextState,                                   // Post-step point navstate
                               pos,                                         // Post-step point position
                               dir,                                         // Post-step point momentum direction
                               eKin,                                        // Post-step point kinetic energy
                               globalTime,                                  // global time
                               localTime,                                   // local time
                               preStepGlobalTime,                           // global time at preStepPoint
                               currentTrack.eventId, currentTrack.threadId, // event and thread ID
                               !trackSurvives,            // whether this is the last step of the track
                               currentTrack.stepCounter); // stepcounter
    }
  } // end for loop over tracks
}

} // namespace AsyncAdePT
