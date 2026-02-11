// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/navigation/AdePTNavigator.h>

#include <AdePT/copcore/PhysicalConstants.h>
#include <AdePT/kernels/AdePTSteppingActionSelector.cuh>

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

using StepActionParam = adept::SteppingAction::Params;
using VolAuxData      = adeptint::VolAuxData;

namespace AsyncAdePT {

// Asynchronous TransportGammasWoodcock Interface
template <typename Scoring, class SteppingActionT>
__global__ void __launch_bounds__(256, 1)
    GammaWoodcock(G4HepEmGammaTrack *hepEMTracks, ParticleManager particleManager,
                  adept::MParray *setupInteractionQueue, Scoring *userScoring, Stats *InFlightStats,
                  const StepActionParam params, AllowFinishOffEventArray allowFinishOffEvent, const bool returnAllSteps,
                  const bool returnLastStep)
{
  // Implementation of the gamma transport using Woodcock tracking. The implementation is taken from
  // Mihaly Novak's G4HepEm (https://github.com/mnovak42/g4hepem), see G4HepEmWoodcockHelper.hh/cc and
  // G4HepEmTrackingManager.cc. Here, it is adopted and adjusted to AdePT's GPU track structures and using VecGeom
  // instead of G4 navigation

  constexpr double kPushDistance           = 1000 * vecgeom::kTolerance;
  constexpr unsigned short kStepsStuckKill = 25;
  constexpr unsigned short maxSteps        = 10'000;
  auto &slotManager                        = *particleManager.gammasWDT.fSlotManager;
  const int activeSize                     = particleManager.gammasWDT.ActiveSize();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const auto slot     = particleManager.gammasWDT.ActiveAt(i);
    Track &currentTrack = particleManager.gammasWDT.TrackAt(slot);

    // Setup of the advanced debug printouts
    bool printErrors = true;
#if ADEPT_DEBUG_TRACK > 0
    bool verbose = false;
    if (gTrackDebug.active) {
      verbose =
          currentTrack.Matches(gTrackDebug.event_id, gTrackDebug.track_id, gTrackDebug.min_step, gTrackDebug.max_step);
      if (verbose) currentTrack.Print("gamma during WDT");
      printErrors = !gTrackDebug.active || verbose;
    }
#endif

    // find region index for current volume
    const int lvolID          = currentTrack.navState.GetLogicalId();
    const VolAuxData &auxData = gVolAuxData[lvolID];

    // Update preStep variables
    currentTrack.preStepEKin       = currentTrack.eKin;
    currentTrack.preStepGlobalTime = currentTrack.globalTime;
    currentTrack.preStepPos        = currentTrack.pos;
    currentTrack.preStepDir        = currentTrack.dir;

    bool leftWDTRegion = false;

    // survive: decide whether to continue woodcock tracking or not:
    // Write local variables back into track and enqueue to correct queue
    auto survive = [&]() {
      if (currentTrack.leakStatus != LeakStatus::NoLeak) {
        // Copy track at slot to the leaked tracks
        particleManager.gammas.CopyTrackToLeaked(slot);
      } else {
        if (leftWDTRegion) {
          particleManager.gammas.EnqueueNext(slot);
        } else {
          particleManager.gammasWDT.EnqueueNext(slot);
        }
      }
    };

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *thePrimaryTrack = gammaTrack.GetTrack();
    if (!currentTrack.hepEmTrackExists) {
      // Init a track with the needed data to call into G4HepEm.
      gammaTrack.ReSet();
      thePrimaryTrack->SetEKin(currentTrack.eKin);
      currentTrack.hepEmTrackExists = true;
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
                                   thePrimaryTrack->GetGStepLength(),           // Step length
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

    //
    // SETUP OF WOODCOCK TRACKING
    //

    // Each gamma that is tracked via Woodcock tracking, must be in one of the root logical volumes of the Woodcock
    // region First, the root logical volume must be identified, to find the DistanceToOut

    // get region ID and Woodcock tracking data for this region
    int regionId = auxData.fGPUregionId;
    assert(regionId >= 0);
    const adeptint::WDTDeviceView &view = gWDTData;
    // get index in the Woodcock tracking data array for the region
    int wdtIdx = view.regionToWDT[regionId];

    // in WDT kernel, the region must be a WDT region (wdtIdx = -1 if not)
    assert(wdtIdx >= 0);

    // get woodcock region and access the root volumes of that region
    const adeptint::WDTRegion reg  = view.regions[wdtIdx];
    const adeptint::WDTRoot *roots = &view.roots[reg.offset];
    // Store matching ID. Don't copy states directly to avoid multiple copies of navStates
    int matchIdx = -1;
    for (int i = 0; i < reg.count; ++i) {
      const auto &rootVol = roots[i];
      // if the current volume is the root volume or descending from the root volume of that WDT region
      // then this is correct matching WDT root volume
      if (currentTrack.navState.GetState() == rootVol.root.GetState() ||
          currentTrack.navState.IsDescendent(rootVol.root.GetState())) {
        matchIdx = i;
        break;
      }
    }
    assert(matchIdx >= 0);

    // Now copy the final navigation state and save the G4HepEm material cut couple of root volume
    const vecgeom::NavigationState rootState = roots[matchIdx].root;
    const int rootHepemIMC                   = roots[matchIdx].hepemIMC;

#if ADEPT_DEBUG_TRACK > 0
    if (verbose) {
      printf("Initial WDT: region %d -> matched root #%d (hepemIMC=%d), lvolID=%d\n", regionId, matchIdx,
             roots[matchIdx].hepemIMC, lvolID);
    }
#endif

    // calculate local point/dir in root volume from global point/dir
    vecgeom::Vector3D<double> localpoint;
    vecgeom::Vector3D<double> localdir;
    vecgeom::Transformation3D m;
    rootState.TopMatrix(m);
    localpoint = m.Transform(currentTrack.pos);
    localdir   = m.TransformDirection(currentTrack.dir);

    // calculate distance to leave the root Woodcock tracking volume
    vecgeom::VPlacedVolume const *rootVolume = rootState.Top();
    double distToBoundary                    = vecgeom::Max(rootVolume->DistanceToOut(localpoint, localdir), 0.0);
    // Difference to the original G4HepEm Woodcock tracking loop:
    // In G4, the boundary must be approached (e.g. to 1e-3) and then the step hitting the boundary + the relocation
    // must be invoked by G4. This is also error prone, as tracks close to the boundary might already be located in a
    // different volume. In VecGeom, one can directly calculate the distance to the boundary (and not just approaching
    // it to 1e-3) because VecGeom allows to do a direct local relocation after leaving a volume, which ensures the
    // correct navigation state Then, one can directly do the Woodcock tracking until hitting the boundary and do the
    // local relocation there.

    thePrimaryTrack->SetMCIndex(rootHepemIMC); // always reset MC index to root WDT volume

    // Compute the WDT reference mxsec (i.e. maximum total mxsec along this step).
    const double wdtMXsec   = G4HepEmGammaManager::GetTotalMacXSec(&g4HepEmData, &gammaTrack);
    const double wdtMFP     = wdtMXsec > 0.0 ? 1.0 / wdtMXsec : vecgeom::InfinityLength<double>();
    const double wdtPEmxSec = gammaTrack.GetPEmxSec();

    //
    // ACTUAL WOODCOCK TRACKING LOOP
    //

    // Init some variables before starting Woodcock tracking of the gamma
    double geometryStep = vecgeom::InfinityLength<double>();
    double mxsec        = 0.0;
    int hepEmIMC        = -1;
    bool realStep       = false;

    // While either interacts or hits the boundary of the actual root volume:
    double wdtStepLength      = 0.0;
    bool isWDTReachedBoundary = false;

    // Note: in the original implementation, a while loop started here. The while loop finished
    // only after a real interaction or boundary was hit. The while loop could be restricted if too many fake
    // interactions happened. However, it was found most beneficial for the GPU (tested in Athena in the EMEC), if this
    // is kept divergence free: each kernel launch does only a single WDT step. For a fake interaction, it is just
    // requeued to the WDT queue. Additionally, there is still the option to put the gamma to the normal gamma queue, in
    // case of too many fake interactions This is tracked with the looperCounter (otherwise not used for gammas). For
    // each fake interaction, the looper counter is increased. Above a certain value, the gamma is queued to the normal
    // gamma kernel

    // Compute the step length till the next interaction in the WDT material
    const double pstep = wdtMFP < vecgeom::InfinityLength<double>() ? -std::log(currentTrack.Uniform()) * wdtMFP
                                                                    : vecgeom::InfinityLength<double>();
    // Take the minimum of this and the distance to the WDT root volume boundary
    // while checking if this step hits the volume boundary

    if (distToBoundary < pstep) {
      wdtStepLength += distToBoundary;
      isWDTReachedBoundary = true;
      realStep             = true;

#if ADEPT_DEBUG_TRACK > 0
      if (verbose) {
        printf("| distToBoundary %.10f < pstep %.10f isWDTReachedBoundary %d \n", distToBoundary, pstep,
               isWDTReachedBoundary);
      }
#endif

    } else {
      // Particle will be moved by a step length of `pstep` so we reduce the
      // distance to boundary accordingly.
      wdtStepLength += pstep;
      distToBoundary -= pstep;

#if ADEPT_DEBUG_TRACK > 0
      if (verbose) {
        printf("| Doing WDT SubStep!  pstep %.10f wdtStepLength %.10f distToBoundary left %.10f\n", pstep,
               wdtStepLength, distToBoundary);
      }
#endif

      // Locate the actual post step point in order to get the real material.
      // This can be done in VecGeom based on the NavState: given a local point within the reference frame of the
      // NavState, it yields the deepest NavState that contains the point.
      currentTrack.nextState.Clear();
      currentTrack.nextState = rootState;
      AdePTNavigator::LocatePointInNavState(localpoint + wdtStepLength * localdir, currentTrack.nextState,
                                            /*top=*/false);

      const int actualLvolID          = currentTrack.nextState.GetLogicalId();
      const VolAuxData &actualAuxData = gVolAuxData[actualLvolID];
      hepEmIMC                        = actualAuxData.fMCIndex;

      // Check if the real material of the post-step point is the WDT one?
      if (hepEmIMC != rootHepemIMC) {
        // Post step point is NOT in the WDT material: need to check if interacts.
        // Compute the total macroscopic cross section for that material.
        thePrimaryTrack->SetMCIndex(hepEmIMC);
        mxsec = G4HepEmGammaManager::GetTotalMacXSec(&g4HepEmData, &gammaTrack);

        // Sample if interaction happens at this post step point:
        // P(interact) = preStepLambda/wdckMXsec note: preStepLambda <= wdckMXsec
        realStep = (mxsec * wdtMFP > currentTrack.Uniform());
        if (realStep) {
          // Interaction happens: set the track fields required later.
          // Set the total MFP of the track that will be needed when sampling
          // the type of the interaction. The HepEm MC index is already set
          // above while the g4 one is also set here.
          const double mfp = mxsec > 0.0 ? 1.0 / mxsec : vecgeom::InfinityLength<double>();
          thePrimaryTrack->SetMFP(mfp, 0);
          // NOTE: PE mxsec is correct as the last call to `GetTotalMacXSec`
          // was done above for this material.
        }
      } else {
        // Post step point is in the WDT material: interacts for sure (prob.=1)
        // Set the total MFP and MC index of the track that will be needed when
        // sampling the type of the interaction. The g4 MC index is also set here.

        realStep = true;
        thePrimaryTrack->SetMCIndex(rootHepemIMC);
        thePrimaryTrack->SetMFP(wdtMFP, 0);
        // Reset the PE mxsec: set in `G4HepEmGammaManager::GetTotalMacXSec`
        // that might have been called for an other material above (without
        // resulting in interaction). Needed for the interaction type sampling.
        gammaTrack.SetPEmxSec(wdtPEmxSec);
      }
    }

    // Single WDT iteration implementation:
    // If a genuine step (boundary reached or real interaction) happened, the looperCounter is reset.
    // For a fake interaction, the looper counter is increased.
    currentTrack.looperCounter = realStep ? 0 : currentTrack.looperCounter + 1;

    // Update the track with its final position and relocate if it hit the boundary.
    currentTrack.pos += wdtStepLength * currentTrack.dir;

    // If it hit the boundary, it just left the Root logical volume. Then, one can call the RelocatePoint function
    // in VecGeom, which relocates the point to the correct new state after leaving the current volume. For this,
    // the Root logical volume must be set as the last exited state, then one need to pass the final local point of
    // the root logical volume. Since the boundary was crossed, the boundary status is set to true. From this, the
    // final state is obtained and no AdePTNavigator::RelocateToNextVolume(pos, dir, nextState) needs to be called
    // anymore
    if (isWDTReachedBoundary) {
      currentTrack.nextState = rootState;
      currentTrack.nextState.SetLastExited();
      AdePTNavigator::RelocatePoint(localpoint + wdtStepLength * localdir, currentTrack.nextState);
      currentTrack.nextState.SetBoundaryState(true);
      // Note: setting of the thePrimaryTrack->SetMCIndex must be done before relocating
    }
    // else: Did not hit the boundary, so the last state that was located within the root volume is the next state and
    // is already set

#if ADEPT_DEBUG_TRACK > 0
    if (verbose) {
      currentTrack.Print("\n Track hit interaction / relocation in WDT tracking \n");
      printf(" isWDTReachedBoundary=%u final after WDT:\n", isWDTReachedBoundary);
      nextState.Print();
    }
#endif

    // Set pre/post step point location and touchable to be the same (as we
    // might have moved the track from a far away volume).
    // NOTE: as energy deposit happens only at discrete interactions for gamma,
    // in case of non-zero energy deposit, i.e. when SD codes are invoked,
    // the track is never on boundary. So all SD code should work fine with
    // identical pre- and post-step points.
    currentTrack.navState   = currentTrack.nextState;
    currentTrack.preStepPos = currentTrack.pos;
    currentTrack.preStepDir = currentTrack.dir;
    // Set all track properties needed later: all pre-step point information are
    // actually set to be their post step point values!

    // Note, the material is already set correctly via hepEmIMC except for the case where the gamma leaves the root WDT
    // volume

    // NOTE: the number of interaction length left will be cleared in all
    // cases when WDT tracking happened (see below).

    // If the WDT region boundary has not been reached in this step then delta
    // interaction happend so just keep moving the post-step point toward the
    // WDT (root) volume boundary.

    // Update the time based on the accumulated WDT step length
    double deltaTime = wdtStepLength / copcore::units::kCLight;
    currentTrack.globalTime += deltaTime;
    currentTrack.localTime += deltaTime;

    // After Woodock tracking: interacts or reached the volume boundary
    // In both cases: reset the number of interaction length left to trigger
    // resampling in the next call to `HowFar` and prevent its update in this step.
    thePrimaryTrack->SetNumIALeft(-1, 0);

    // The track has the total, i.e. the WDT step length.
    // Note: in the original implementation, the GStepLength is set to 0, as this prevents the update of the
    // number of interaction length left when invoking `UpdateNumIALeft`. To avoid caching the geometricStepLength in
    // the AdePT track, we fully rely on propagating the geometric step length in the G4HepEmTrack. Therefore, by
    // setting it to the wdt step length, we have to ensure that `UpdateNumIALeft` is not called. This is only called in
    // the GammaRelocation kernel, which is never accessed by Woodcock tracked gammas.
    thePrimaryTrack->SetGStepLength(wdtStepLength);
    thePrimaryTrack->SetOnBoundary(currentTrack.nextState.IsOnBoundary());

    // END OF WOODCOCK TRACKING
    // here, the original while loop finished. Now, as it is only a single step.
    // Woodcock step has finished, now either a boundary is hit, a discrete process must be invoked,
    // or a fake step happened. In case of the fake step, the track is just requeued to the WDT gammas.
    // For the discrete process, the MFP and EKin of thePrimaryTrack must be set

    // helper variables needed for the processes
    bool reached_interaction = true;

    currentTrack.stepCounter++;

    if (currentTrack.looperCounter > 0) {
      // Case 1: fake interaction, give particle back to WDT gammas unless it did more fake interactions than allowed,
      // in that case give back to normal gammas
      reached_interaction = false;
      leftWDTRegion       = (view.maxIter == currentTrack.looperCounter) ? true : false;
    } else if (isWDTReachedBoundary) {
      // Case 2: Gamma has hit the boundary of the root WDT volume.

      // nextAuxData is needed for either finding the correct GPU region for relocation, or the sensitive detector code,
      // as the initial auxData is obsolete after WDT
      const int nextlvolID          = currentTrack.nextState.GetLogicalId();
      VolAuxData const &nextauxData = gVolAuxData[nextlvolID];

      // Setting the MCindex of the next volume
      thePrimaryTrack->SetMCIndex(nextauxData.fMCIndex);

      // Check for next region if not outside
      if (!currentTrack.nextState.IsOutside()) {

#if ADEPT_DEBUG_TRACK > 0
        if (verbose) {
          printf("| CROSSED into ");
          nextState.Print();
        }
#endif

        //  Check if the next volume belongs to the GPU region and push it to the appropriate queue
        regionId = nextauxData.fGPUregionId;
        // regionId >= 0 is still on GPU, check for Woodcock tracking region
        if (regionId >= 0) {
          // index into view.regions. -1 if region is not a Woodcock tracking region
          wdtIdx = view.regionToWDT[regionId];
          if (wdtIdx < 0) {
            // next region is not in WDT regions, so it left the WDT tracking
            leftWDTRegion = true;
          }

#if ADEPT_DEBUG_TRACK > 0
          if (verbose) {
            printf("| After WDT check: leftWDTRegion %d \n", leftWDTRegion);
          }
#endif
        } else {
          // To be safe, just push a bit the track exiting the GPU region to make sure
          // Geant4 does not relocate it again inside the same region
          currentTrack.pos += kPushDistance * currentTrack.dir;

#if ADEPT_DEBUG_TRACK > 0
          if (verbose) printf("\n| track leaked to Geant4\n");
#endif

          currentTrack.leakStatus = LeakStatus::OutOfGPURegion;
        }
      } // else particle has left the world

      reached_interaction = false;
    } // end if !isWDTReachedBoundary

    if (reached_interaction) {
      // particle has reached physics interaction
      setupInteractionQueue->push_back(slot);
      // To avoid that the SetupInteractions kernel moves the track to the Relocation kernel, ensure that
      // isOnBoundary is false (relocation happened already in this kernel)
      assert(!currentTrack.nextState.IsOnBoundary());

    } else {
      // track survives and is given to either to the WDT gammas, the normal gammas, or leaked out of the GPU
      survive();
    }

  } // end for loop over tracks
}

} // namespace AsyncAdePT
