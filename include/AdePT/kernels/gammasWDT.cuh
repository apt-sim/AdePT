// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/navigation/AdePTNavigator.h>

#include <AdePT/copcore/PhysicalConstants.h>
#include <AdePT/core/TrackDebug.cuh>

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

using VolAuxData      = adeptint::VolAuxData;
using StepActionParam = adept::SteppingAction::Params;

namespace AsyncAdePT {
// Asynchronous TransportGammasWoodcock Interface
template <typename Scoring, class SteppingActionT>
__global__ void __launch_bounds__(256, 1)
    TransportGammasWoodcock(ParticleManager particleManager, Scoring *userScoring, Stats *InFlightStats,
                            const StepActionParam params, AllowFinishOffEventArray allowFinishOffEvent,
                            const bool returnAllSteps, const bool returnLastStep)
{
  // Implementation of the gamma transport using Woodcock tracking. The implementation is taken from
  // Mihaly Novak's G4HepEm (https://github.com/mnovak42/g4hepem), see G4HepEmWoodcockHelper.hh/cc and
  // G4HepEmTrackingManager.cc. Here, it is adopted and adjusted to AdePT's GPU track structures and using VecGeom
  // instead of G4 navigation

  constexpr double kPushDistance    = 1000 * vecgeom::kTolerance;
  constexpr unsigned short maxSteps = 10'000;
  auto &slotManager                 = *particleManager.gammasWDT.fSlotManager;
  const int activeSize              = particleManager.gammasWDT.ActiveSize();
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

    // Local variables that are needed:
    // get global point from track
    auto pos = currentTrack.pos;
    auto dir = currentTrack.dir;
    vecgeom::Vector3D<double> preStepPos(pos);
    vecgeom::Vector3D<double> preStepDir(dir);

    auto eKin = currentTrack.eKin;

    LeakStatus leakReason = LeakStatus::NoLeak;
    bool leftWDTRegion    = false;
    // initialize nextState to current state
    vecgeom::NavigationState nextState = currentTrack.navState;
    double globalTime                  = currentTrack.globalTime;
    double preStepGlobalTime           = currentTrack.globalTime;
    double localTime                   = currentTrack.localTime;

    //
    // survive: decide whether to continue woodcock tracking or not:
    // Write local variables back into track and enqueue to correct queue
    auto survive = [&]() {
      currentTrack.eKin       = eKin;
      currentTrack.pos        = pos;
      currentTrack.dir        = dir;
      currentTrack.globalTime = globalTime;
      currentTrack.localTime  = localTime;
      currentTrack.navState   = nextState;
      currentTrack.leakStatus = leakReason;
      if (leakReason != LeakStatus::NoLeak) {
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

    //
    // SETUP OF WOODCOCK TRACKING
    //

    // Each gamma that is tracked via Woodcock tracking, must be in one of the root logical volumes of the Woodcock
    // region First, the root logical volume must be identified, to find the DistanceToOut

    // find region index for current volume
    auto navState             = currentTrack.navState;
    const int lvolID          = navState.GetLogicalId();
    const VolAuxData &auxData = gVolAuxData[lvolID];

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
      if (navState.GetState() == rootVol.root.GetState() || navState.IsDescendent(rootVol.root.GetState())) {
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
    localpoint = m.Transform(pos);
    localdir   = m.TransformDirection(dir);

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

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmGammaTrack gammaTrack;
    G4HepEmTrack *thePrimaryTrack = gammaTrack.GetTrack();
    thePrimaryTrack->SetEKin(eKin);
    thePrimaryTrack->SetMCIndex(rootHepemIMC);

    // Compute the WDT reference mxsec (i.e. maximum total mxsec along this step).
    const double wdtMXsec   = G4HepEmGammaManager::GetTotalMacXSec(&g4HepEmData, &gammaTrack);
    const double wdtMFP     = wdtMXsec > 0.0 ? 1.0 / wdtMXsec : vecgeom::InfinityLength<double>();
    const double wdtPEmxSec = gammaTrack.GetPEmxSec();

    //
    // ACTUAL WOODCOCK TRACKING LOOP
    //

    // Init some variables before starting Woodcock tracking of the gamma
    vecgeom::NavigationState wdtNavState; // intermediate navigation state during Woodcock tracking
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
      wdtNavState.Clear();
      wdtNavState = rootState;
      AdePTNavigator::LocatePointInNavState(localpoint + wdtStepLength * localdir, wdtNavState, /*top=*/false);

      const int actualLvolID          = wdtNavState.GetLogicalId();
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
    pos += wdtStepLength * dir;

    // If it hit the boundary, it just left the Root logical volume. Then, one can call the RelocatePoint function
    // in VecGeom, which relocates the point to the correct new state after leaving the current volume. For this,
    // the Root logical volume must be set as the last exited state, then one need to pass the final local point of
    // the root logical volume. Since the boundary was crossed, the boundary status is set to true. From this, the
    // final state is obtained and no AdePTNavigator::RelocateToNextVolume(pos, dir, nextState) needs to be called
    // anymore
    if (isWDTReachedBoundary) {
      nextState = rootState;
      nextState.SetLastExited();
      AdePTNavigator::RelocatePoint(localpoint + wdtStepLength * localdir, nextState);
      nextState.SetBoundaryState(true);
    } else {
      // Did not hit the boundary, so the last state when locating within the root volume is the next state
      nextState = wdtNavState;
    }

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
    navState   = nextState;
    preStepPos = pos;
    preStepDir = dir;
    // Set all track properties needed later: all pre-step point information are
    // actually set to be their post step point values!

    // Note, the material is already set correctly via hepEmIMC

    // NOTE: the number of interaction length left will be cleared in all
    // cases when WDT tracking happened (see below).

    // If the WDT region boundary has not been reached in this step then delta
    // interaction happend so just keep moving the post-step point toward the
    // WDT (root) volume boundary.

    // Update the time based on the accumulated WDT step length
    double deltaTime = wdtStepLength / copcore::units::kCLight;
    globalTime += deltaTime;
    localTime += deltaTime;

    // After Woodock tracking: interacts or reached the volume boundary
    // In both cases: reset the number of interaction length left to trigger
    // resampling in the next call to `HowFar` and prevent its update in this step.
    thePrimaryTrack->SetNumIALeft(-1, 0);
    currentTrack.numIALeft[0] = -1; // reset also in the GPU track

    // The track has the total, i.e. the WDT step length.
    // However, `thePrimaryTrack` has zero which results in: the number of interaction length left is
    // not updated when invoking `UpdateNumIALeft`.
    thePrimaryTrack->SetGStepLength(0.0);
    thePrimaryTrack->SetOnBoundary(nextState.IsOnBoundary());

    // END OF WOODCOCK TRACKING
    // here, the original while loop finished. Now, as it is only a single step.
    // Woodcock step has finished, now either a boundary is hit, a discrete process must be invoked,
    // or a fake step happened. In case of the fake step, the track is just requeued to the WDT gammas.
    // For the discrete process, the MFP and EKin of thePrimaryTrack must be set

    // helper variables needed for the processes
    bool trackSurvives         = false;
    short stepDefinedProcessId = 10; // default for transportation
    double edep                = 0.;
    auto preStepEnergy         = eKin;

    currentTrack.stepCounter++;

    // nextAuxData is needed for either finding the correct GPU region for relocation, or the sensitive detector code,
    // as the initial auxData is obsolete after WDT
    const int nextlvolID          = nextState.GetLogicalId();
    VolAuxData const &nextauxData = gVolAuxData[nextlvolID];

    int winnerProcessIndex;
    // data structure for possible secondaries that are generated
    SecondaryInitData secondaryData[3];
    unsigned int nSecondaries = 0;
    if (currentTrack.looperCounter > 0) {
      // Case 1: fake interaction, give particle back to WDT gammas unless it did more fake interactions than allowed,
      // in that case give back to normal gammas
      trackSurvives = true;
      leftWDTRegion = (view.maxIter == currentTrack.looperCounter) ? true : false;
    } else if (isWDTReachedBoundary) {
      // Case 2: Gamma has hit the boundary of the root WDT volume.

      // Kill the particle if it left the world.
      if (!nextState.IsOutside()) {

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

          // particle is still alive
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
      // Case 3: Woodcock tracking gamma undergoes a real interaction

      G4HepEmGammaManager::SampleInteraction(&g4HepEmData, &gammaTrack, currentTrack.Uniform());
      winnerProcessIndex = thePrimaryTrack->GetWinnerProcessIndex();
      // Note: SampleInteraction resets numIALeft, but for WDT it is reset anyway, so it is already reset in
      // currentTrack

#if ADEPT_DEBUG_TRACK > 0
      if (verbose) printf("| winnerProc %d\n", winnerProcessIndex);
#endif

      // Perform the discrete interaction.
      G4HepEmRandomEngine rnge(&currentTrack.rngState);
      // We might need one branched RNG state, prepare while threads are synchronized.
      RanluxppDouble newRNG(currentTrack.rngState.Branch());

      assert(hepEmIMC >= 0);

      const double theElCut    = g4HepEmData.fTheMatCutData->fMatCutData[hepEmIMC].fSecElProdCutE;
      const double thePosCut   = g4HepEmData.fTheMatCutData->fMatCutData[hepEmIMC].fSecPosProdCutE;
      const double theGammaCut = g4HepEmData.fTheMatCutData->fMatCutData[hepEmIMC].fSecGamProdCutE;

      const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[hepEmIMC].fG4RegionIndex;
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
        G4HepEmGammaInteractionConversion::SampleKinEnergies(&g4HepEmData, eKin, logEnergy, hepEmIMC, elKinEnergy,
                                                             posKinEnergy, &rnge);

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
            secondaryData[nSecondaries++] = {electron.trackId, electron.dir, electron.eKin, /*particle type*/ char(0)};
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
            secondaryData[nSecondaries++] = {positron.trackId, positron.dir, positron.eKin, /*particle type*/ char(1)};
          }
        }
        eKin = 0.;
        break;
      }
      case 1: {
        // Invoke Compton scattering of gamma.

#if ADEPT_DEBUG_TRACK > 0
        if (verbose) printf("| COMPTON \n");
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
            secondaryData[nSecondaries++] = {electron.trackId, electron.dir, electron.eKin, /*particle type*/ char(0)};
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
            &g4HepEmData, hepEmIMC, gammaTrack.GetPEmxSec(), eKin, &rnge);

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
            secondaryData[nSecondaries++] = {electron.trackId, electron.dir, electron.eKin, /*particle type*/ char(0)};
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

    } // end if !isWDTReachedBoundary

    if (trackSurvives) {
      if (currentTrack.stepCounter >= maxSteps) {
        if (printErrors)
          printf("Killing gamma event %d track %lu E=%f lvol=%d after %d steps. This indicates a "
                 "stuck particle!\n",
                 currentTrack.eventId, currentTrack.trackId, eKin, lvolID, currentTrack.stepCounter);
        trackSurvives = false;
        edep += eKin;
        eKin = 0.;
      } else {
        // call experiment-specific SteppingAction:
        SteppingActionT::GammaAction(trackSurvives, eKin, edep, pos, globalTime, hepEmIMC, &g4HepEmData, params);
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

    // check for minimal energy for WDT tracking. If too low, move to normal tracking
    if (eKin < reg.ekinMin) {
      leftWDTRegion = true;
    }

    // A track that survives must be enqueued to the leaks or the next queue.
    // Note: gamma nuclear does not survive but must still be leaked to the CPU, which is done
    // inside survive()
    if (trackSurvives || leakReason == LeakStatus::GammaNuclear) {
      survive();
    } else {
      // particles that don't survive are killed by not enqueing them to the next queue and freeing the slot
      slotManager.MarkSlotForFreeing(slot);
    }

    assert(nSecondaries <= 3);

    // If there is some edep from cutting particles, record the step
    // Note: record only real steps that either interacted or hit a boundary
    if (realStep && ((edep > 0 && nextauxData.fSensIndex >= 0) || returnAllSteps ||
                     (returnLastStep && (nSecondaries > 0 || !trackSurvives)))) {
      adept_scoring::RecordHit(userScoring,
                               currentTrack.trackId,                        // Track ID
                               currentTrack.parentId,                       // parent Track ID
                               stepDefinedProcessId,                        // step-defining process id
                               2,                                           // Particle type
                               wdtStepLength,                               // Step length
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
                               !trackSurvives,           // whether this is the last step of the track
                               currentTrack.stepCounter, // stepcounter
                               secondaryData,            // pointer to secondary init data
                               nSecondaries);            // number of secondaries
    }
  } // end for loop over tracks
}

} // namespace AsyncAdePT
