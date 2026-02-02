// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/navigation/AdePTNavigator.h>

// Classes for Runge-Kutta integration
#include <AdePT/magneticfield/MagneticFieldEquation.h>
#include <AdePT/magneticfield/DormandPrinceRK45.h>
#include <AdePT/magneticfield/fieldPropagatorRungeKutta.h>

#include <AdePT/copcore/PhysicalConstants.h>
#include <AdePT/core/TrackDebug.cuh>
#include <AdePT/core/AdePTPrecision.hh>
#include <AdePT/kernels/AdePTSteppingActionSelector.cuh>
#include <AdePT/kernels/WoodcockHelper.cuh>

#include <G4HepEmElectronManager.hh>
#include <G4HepEmElectronTrack.hh>
#include <G4HepEmElectronInteractionBrem.hh>
#include <G4HepEmElectronInteractionIoni.hh>
#include <G4HepEmElectronInteractionUMSC.hh>
#include <G4HepEmPositronInteractionAnnihilation.hh>
// Pull in implementation.
#include <G4HepEmRunUtils.icc>
#include <G4HepEmInteractionUtils.icc>
#include <G4HepEmElectronManager.icc>
#include <G4HepEmElectronInteractionBrem.icc>
#include <G4HepEmElectronInteractionIoni.icc>
#include <G4HepEmElectronInteractionUMSC.icc>
#include <G4HepEmPositronInteractionAnnihilation.icc>
#include <G4HepEmElectronEnergyLossFluctuation.icc>

using StepActionParam = adept::SteppingAction::Params;
using VolAuxData      = adeptint::VolAuxData;

// Compute velocity based on the kinetic energy of the particle
__device__ double GetVelocity(double eKin)
{
  // Taken from G4DynamicParticle::ComputeBeta
  double T    = eKin / copcore::units::kElectronMassC2;
  double beta = sqrt(T * (T + 2.)) / (T + 1.0);
  return copcore::units::kCLight * beta;
}

namespace AsyncAdePT {

// Compute the physics and geometry step limit, transport the electrons while
// applying the continuous effects and maybe a discrete process that could
// generate secondaries.
template <bool IsElectron, typename Scoring, class SteppingActionT>
static __device__ __forceinline__ void TransportElectrons(ParticleManager &particleManager, Scoring *userScoring,
                                                          Stats *InFlightStats, const StepActionParam params,
                                                          AllowFinishOffEventArray allowFinishOffEvent,
                                                          const bool returnAllSteps, const bool returnLastStep)
{
  constexpr double kPushDistance    = 1000 * vecgeom::kTolerance;
  constexpr unsigned short maxSteps = 10'000;
  constexpr int Charge              = IsElectron ? -1 : 1;
  constexpr double restMass         = copcore::units::kElectronMassC2;
  constexpr int Nvar                = 6;
  constexpr int max_iterations      = 10;

#ifdef ADEPT_USE_EXT_BFIELD
  using Field_t = GeneralMagneticField;
#else
  using Field_t = UniformMagneticField;
#endif
  using Equation_t = MagneticFieldEquation<Field_t>;
  using Stepper_t  = DormandPrinceRK45<Equation_t, Field_t, Nvar, rk_integration_t>;
  using RkDriver_t = RkIntegrationDriver<Stepper_t, rk_integration_t, int, Equation_t, Field_t>;

  auto &magneticField = *gMagneticField;

  auto &electronsOrPositrons = (IsElectron ? particleManager.electrons : particleManager.positrons);
  SlotManager &slotManager   = *electronsOrPositrons.fSlotManager;

  const int activeSize = electronsOrPositrons.ActiveSize();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const auto slot     = electronsOrPositrons.ActiveAt(i);
    Track &currentTrack = electronsOrPositrons.TrackAt(slot);

    auto navState = currentTrack.navState;
    // the MCC vector is indexed by the logical volume id
    const int lvolID          = navState.GetLogicalId();
    VolAuxData const &auxData = gVolAuxData[lvolID];

    bool trackSurvives                       = false;
    LeakStatus leakReason                    = LeakStatus::NoLeak;
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
    bool verbose     = false;
#if ADEPT_DEBUG_TRACK > 0
    const char *pname[2] = {"e+", "e-"};
    if (gTrackDebug.active) {
      verbose =
          currentTrack.Matches(gTrackDebug.event_id, gTrackDebug.track_id, gTrackDebug.min_step, gTrackDebug.max_step);
      if (verbose) {
        currentTrack.Print(pname[IsElectron]);
      }
    }
    printErrors = !gTrackDebug.active || verbose;
#endif

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
        electronsOrPositrons.CopyTrackToLeaked(slot);
      } else {
        electronsOrPositrons.EnqueueNext(slot);
      }
    };

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmElectronTrack elTrack;
    G4HepEmTrack *theTrack = elTrack.GetTrack();
    theTrack->SetEKin(eKin);
    theTrack->SetMCIndex(auxData.fMCIndex);
    theTrack->SetOnBoundary(navState.IsOnBoundary());
    theTrack->SetCharge(Charge);
    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    // the default is 1.0e21 but there are float vs double conversions, so we check for 1e20
    mscData->fIsFirstStep        = currentTrack.initialRange > 1.0e+20;
    mscData->fInitialRange       = currentTrack.initialRange;
    mscData->fDynamicRangeFactor = currentTrack.dynamicRangeFactor;
    mscData->fTlimitMin          = currentTrack.tlimitMin;

    // Prepare a branched RNG state while threads are synchronized. Even if not
    // used, this provides a fresh round of random numbers and reduces thread
    // divergence because the RNG state doesn't need to be advanced later.
    RanluxppDouble newRNG(currentTrack.rngState.BranchNoAdvance());
    G4HepEmRandomEngine rnge(&currentTrack.rngState);

    // Sample the `number-of-interaction-left` and put it into the track.
    for (int ip = 0; ip < 4; ++ip) {
      double numIALeft = currentTrack.numIALeft[ip];
      if (numIALeft <= 0) {
        numIALeft = -std::log(currentTrack.Uniform());
      }
      theTrack->SetNumIALeft(numIALeft, ip);
    }

    G4HepEmElectronManager::HowFarToDiscreteInteraction(&g4HepEmData, &g4HepEmPars, &elTrack);

    auto physicalStepLength = elTrack.GetPStepLength();

#if ADEPT_DEBUG_TRACK > 0
    if (verbose) printf("| physStep discrete %g ", physicalStepLength);
#endif

    // Compute safety, needed for MSC step limit. The accuracy range is physicalStepLength
    double safety = 0.;
    if (!navState.IsOnBoundary()) {
      // Get the remaining safety only if larger than physicalStepLength
      safety = currentTrack.GetSafety(pos);
      if (safety < physicalStepLength) {
        // Recompute safety and update it in the track.
        // Use maximum accuracy only if safety is samller than physicalStepLength
        safety = AdePTNavigator::ComputeSafety(pos, navState, physicalStepLength);
        currentTrack.SetSafety(pos, safety);
#if ADEPT_DEBUG_TRACK > 0
        if (verbose) printf("| new safety %g ", safety);
#endif
      }
    }
    theTrack->SetSafety(safety);
    bool restrictedPhysicalStepLength = false;

    double safeLength = 0.;

    if (gMagneticField) {

      const double momentumMag = sqrt(eKin * (eKin + 2.0 * restMass));
      // Distance along the track direction to reach the maximum allowed error

      // SEVERIN: to be checked if we can use float
      vecgeom::Vector3D<double> momentumVec = momentumMag * dir;
      vecgeom::Vector3D<rk_integration_t> B0fieldVec =
          magneticField.Evaluate(pos[0], pos[1], pos[2]); // Field value at starting point
      safeLength =
          fieldPropagatorRungeKutta<Field_t, RkDriver_t, rk_integration_t,
                                    AdePTNavigator>::ComputeSafeLength /*<Real_t>*/ (momentumVec, B0fieldVec, Charge);

#if ADEPT_DEBUG_TRACK > 0
      if (verbose) printf("| safeField %g ", safeLength);
#endif

      constexpr int MaxSafeLength = 10;
      double limit                = MaxSafeLength * safeLength;
      limit                       = safety > limit ? safety : limit;

      if (physicalStepLength > limit) {
        physicalStepLength           = limit;
        restrictedPhysicalStepLength = true;
        elTrack.SetPStepLength(physicalStepLength);

#if ADEPT_DEBUG_TRACK > 0
        if (verbose) printf("| restricted_physStep %g ", physicalStepLength);
#endif

        // Note: We are limiting the true step length, which is converted to
        // a shorter geometry step length in HowFarToMSC. In that sense, the
        // limit is an over-approximation, but that is fine for our purpose.
      }
    }

    __syncwarp(); // was found to be beneficial after divergent calls

    G4HepEmElectronManager::HowFarToMSC(&g4HepEmData, &g4HepEmPars, &elTrack, &rnge);

    // Remember MSC values for the next step(s).
    currentTrack.initialRange       = mscData->fInitialRange;
    currentTrack.dynamicRangeFactor = mscData->fDynamicRangeFactor;
    currentTrack.tlimitMin          = mscData->fTlimitMin;

    // Get result into variables.
    double geometricalStepLengthFromPhysics = theTrack->GetGStepLength();

#if ADEPT_DEBUG_TRACK > 0
    if (verbose) printf("| geometricalStepLengthFromPhysics %g ", geometricalStepLengthFromPhysics);
#endif

    // The phyiscal step length is the amount that the particle experiences
    // which might be longer than the geometrical step length due to MSC. As
    // long as we call PerformContinuous in the same kernel we don't need to
    // care, but we need to make this available when splitting the operations.
    // double physicalStepLength = elTrack.GetPStepLength();
    int winnerProcessIndex = theTrack->GetWinnerProcessIndex();

#if ADEPT_DEBUG_TRACK > 0
    if (verbose) printf("| winnerProc %d\n", winnerProcessIndex);
#endif

    // Leave the range and MFP inside the G4HepEmTrack. If we split kernels, we
    // also need to carry them over!

    // Check if there's a volume boundary in between.
    bool propagated    = true;
    long hitsurf_index = -1;
    double geometryStepLength;
    bool zero_first_step = false;

    if (gMagneticField) {
      int iterDone = -1;
      geometryStepLength =
          fieldPropagatorRungeKutta<Field_t, RkDriver_t, rk_integration_t, AdePTNavigator>::ComputeStepAndNextVolume(
              magneticField, eKin, restMass, Charge, geometricalStepLengthFromPhysics, safeLength, pos, dir, navState,
              nextState, hitsurf_index, propagated, /*lengthDone,*/ safety,
              // activeSize < 100 ? max_iterations : max_iters_tail ), // Was
              max_iterations, iterDone, slot, zero_first_step, verbose);
      // In case of zero step detected by the field propagator this could be due to back scattering, or wrong relocation
      // in the previous step.
      // - In case of BS we should just restore the last exited one for the nextState. For now we cannot detect BS.
      // if (zero_first_step) nextState.SetNavIndex(navState.GetLastExitedState());
    } else {
#ifdef ADEPT_USE_SURF
      geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics,
                                                                    navState, nextState, hitsurf_index);
#else
      geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics,
                                                                    navState, nextState, kPushDistance);
#endif
      pos += geometryStepLength * dir;
    }

    if (geometryStepLength < kPushStuck && geometryStepLength < geometricalStepLengthFromPhysics) {
      currentTrack.zeroStepCounter++;
      if (currentTrack.zeroStepCounter > kStepsStuckPush) pos += kPushStuck * dir;
    } else {
      currentTrack.zeroStepCounter = 0;
    }

    __syncwarp(); // was found to be beneficial after divergent calls

#if ADEPT_DEBUG_TRACK > 0
    if (verbose) {
      printf("| geometryStepLength %g | propagated_pos {%.19f, %.19f, %.19f} dir {%.19f, %.19f, %.19f}",
             geometryStepLength, pos[0], pos[1], pos[2], dir[0], dir[1], dir[2]);
      if (geometryStepLength < 100 * vecgeom::kTolerance) printf("| SMALL STEP ");
      nextState.Print();
    }
#endif

    // punish miniscule steps by increasing the looperCounter by 10
    if (geometryStepLength < 100 * vecgeom::kTolerance) currentTrack.looperCounter += 10;

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    navState.SetBoundaryState(nextState.IsOnBoundary());
    if (nextState.IsOnBoundary()) currentTrack.SetSafety(pos, 0.);

    // Propagate information from geometrical step to MSC.
    theTrack->SetDirection(dir.x(), dir.y(), dir.z());
    theTrack->SetGStepLength(geometryStepLength);
    theTrack->SetOnBoundary(nextState.IsOnBoundary());

    // Apply continuous effects.
    bool stopped = G4HepEmElectronManager::PerformContinuous(&g4HepEmData, &g4HepEmPars, &elTrack, &rnge);

    // Collect the direction change and displacement by MSC.
    const double *direction = theTrack->GetDirection();
    dir.Set(direction[0], direction[1], direction[2]);

#if ADEPT_DEBUG_TRACK > 1
    if (verbose) {
      printf("| after MSC: newdir {%.19f, %.19f, %.19f} ", dir[0], dir[1], dir[2]);
      if (stopped) printf("| particle STOPPED ");
    }
#endif

    if (!nextState.IsOnBoundary()) {
      const double *mscDisplacement = mscData->GetDisplacement();
      vecgeom::Vector3D<double> displacement(mscDisplacement[0], mscDisplacement[1], mscDisplacement[2]);
      const double dLength2            = displacement.Length2();
      constexpr double kGeomMinLength  = 5 * copcore::units::nm;          // 0.05 [nm]
      constexpr double kGeomMinLength2 = kGeomMinLength * kGeomMinLength; // (0.05 [nm])^2
      if (dLength2 > kGeomMinLength2) {
        const double dispR = std::sqrt(dLength2);
        // Estimate safety by subtracting the geometrical step length.
        safety                 = currentTrack.GetSafety(pos);
        constexpr double sFact = 0.99;
        double reducedSafety   = sFact * safety;

        // Apply displacement, depending on how close we are to a boundary.
        // 1a. Far away from geometry boundary:
        if (reducedSafety > 0.0 && dispR <= reducedSafety) {
          pos += displacement;
        } else {
          // Recompute safety.
          // Use maximum accuracy only if safety is smaller than physicalStepLength
          safety = AdePTNavigator::ComputeSafety(pos, navState, dispR);
          currentTrack.SetSafety(pos, safety);
          reducedSafety = sFact * safety;

          // 1b. Far away from geometry boundary:
          if (reducedSafety > 0.0 && dispR <= reducedSafety) {
            pos += displacement;
            // 2. Push to boundary:
          } else if (reducedSafety > kGeomMinLength) {
            pos += displacement * (reducedSafety / dispR);
          }
          // 3. Very small safety: do nothing.
        }
      }
#if ADEPT_DEBUG_TRACK > 0
      if (verbose) {
        printf("| displaced_pos {%.19f, %.19f, %.19f} | ekin %g | edep %g | safety %g\n", pos[0], pos[1], pos[2],
               theTrack->GetEKin(), theTrack->GetEnergyDeposit(), safety);
      }
#endif
    }

    // Update the flight times of the particle
    // To conform with Geant4, we use the initial velocity of the particle, before eKin is updated after the energy loss
    double deltaTime = elTrack.GetPStepLength() / GetVelocity(eKin);
    globalTime += deltaTime;
    localTime += deltaTime;
    properTime += deltaTime * (restMass / (restMass + eKin));

    // Collect the charged step length (might be changed by MSC). Collect the changes in energy and deposit.
    eKin                 = theTrack->GetEKin();
    double energyDeposit = theTrack->GetEnergyDeposit();

    if (nextState.IsOnBoundary()) {
      // if the particle hit a boundary, and is neither stopped or outside, relocate to have the correct next state
      // before RecordHit is called
      if (!stopped && !nextState.IsOutside()) {
#if ADEPT_DEBUG_TRACK > 0
        if (verbose) {
          printf("\n| +++ RelocateToNextVolume -position %.17f, %.17f, %.17f -direction %.17f, %.17f, %.17f ", pos[0],
                 pos[1], pos[2], dir[0], dir[1], dir[2]);
          nextState.Print();
        }
#endif
#ifdef ADEPT_USE_SURF
        AdePTNavigator::RelocateToNextVolume(pos, dir, hitsurf_index, nextState);
#else
        AdePTNavigator::RelocateToNextVolume(pos, dir, nextState);
#endif
        // Set the last exited state to be the one before crossing
        nextState.SetLastExited(navState.GetState());
#if ADEPT_DEBUG_TRACK > 0
        if (verbose) {
          printf("\n| CROSSED into ");
          nextState.Print();
        }
#endif
      }
    }

    // Save the `number-of-interaction-left` in our track.
    for (int ip = 0; ip < 4; ++ip) {
      double numIALeft           = theTrack->GetNumIALeft(ip);
      currentTrack.numIALeft[ip] = numIALeft;
    }

    bool reached_interaction = true;

    const double theElCut    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecElProdCutE;
    const double theGammaCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecGamProdCutE;

    const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
    const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

    if (!stopped) {
      if (nextState.IsOnBoundary()) {
        // For now, just count that we hit something.
        reached_interaction = false;
        // Kill the particle if it left the world.

        if (!nextState.IsOutside()) {

          // Check if the next volume belongs to the GPU region and push it to the appropriate queue
          const int nextlvolID          = nextState.GetLogicalId();
          VolAuxData const &nextauxData = gVolAuxData[nextlvolID];
          // track has left GPU region
          if (nextauxData.fGPUregionId < 0) {
            // To be safe, just push a bit the track exiting the GPU region to make sure
            // Geant4 does not relocate it again inside the same region
            pos += kPushDistance * dir;

#if ADEPT_DEBUG_TRACK > 0
            if (verbose) printf("\n| track leaked to Geant4\n");
#endif
            leakReason = LeakStatus::OutOfGPURegion;
          }

          // the track survives, do not force return of step
          trackSurvives = true;
        } // else particle has left the world

        winnerProcessIndex = 10; // mark winner process to be transport
      } else if (!propagated || restrictedPhysicalStepLength) {
        // Did not yet reach the interaction point due to error in the magnetic
        // field propagation. Try again next time.

        // mark winner process to be transport, although this is not strictly true
        winnerProcessIndex = 10;

        trackSurvives       = true;
        reached_interaction = false;
      } else if (winnerProcessIndex < 0) {
        // No discrete process, move on.
        trackSurvives       = true;
        reached_interaction = false;
      }
    }

    // reset Looper counter if limited by discrete interaction or MSC
    if (reached_interaction) currentTrack.looperCounter = 0;

    __syncwarp(); // was found to be beneficial after divergent calls

    if (reached_interaction && !stopped) {
      // Reset number of interaction left for the winner discrete process.
      // (Will be resampled in the next iteration.)
      currentTrack.numIALeft[winnerProcessIndex] = -1.0;

      // Check if a delta interaction happens instead of the real discrete process.
      if (G4HepEmElectronManager::CheckDelta(&g4HepEmData, theTrack, currentTrack.Uniform())) {
        // A delta interaction happened, move on.
#if ADEPT_DEBUG_TRACK > 0
        if (verbose) printf("| delta interaction\n");
#endif
        trackSurvives = true;
      } else {
        // Perform the discrete interaction, make sure the branched RNG state is
        // ready to be used.
        newRNG.Advance();
        // Also advance the current RNG state to provide a fresh round of random
        // numbers after MSC used up a fair share for sampling the displacement.
        currentTrack.rngState.Advance();

        switch (winnerProcessIndex) {
        case 0: {
          // Invoke ionization (for e-/e+):
          double deltaEkin = (IsElectron)
                                 ? G4HepEmElectronInteractionIoni::SampleETransferMoller(theElCut, eKin, &rnge)
                                 : G4HepEmElectronInteractionIoni::SampleETransferBhabha(theElCut, eKin, &rnge);

#if ADEPT_DEBUG_TRACK > 0
          if (verbose) printf("| IONIZATION: deltaEkin %g \n", deltaEkin);
#endif

          double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
          double dirSecondary[3];
          G4HepEmElectronInteractionIoni::SampleDirections(eKin, deltaEkin, dirSecondary, dirPrimary, &rnge);

          adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

          // Apply cuts
          if (ApplyCuts && (deltaEkin < theElCut)) {
            // Deposit the energy here and kill the secondary
            energyDeposit += deltaEkin;

#if ADEPT_DEBUG_TRACK > 0
            if (verbose) printf("| secondary killed by cut \n");
#endif

          } else {
            Track &secondary = particleManager.electrons.NextTrack(
                newRNG, deltaEkin, pos, vecgeom::Vector3D<double>{dirSecondary[0], dirSecondary[1], dirSecondary[2]},
                navState, currentTrack, globalTime);

            // if tracking or stepping action is called, return initial step
            if (returnLastStep) {
              adept_scoring::RecordHit(
                  userScoring, secondary.trackId, secondary.parentId,
                  /*CreatorProcessId*/ short(0),
                  /* electron*/ 0,  // Particle type
                  0,                // Step length
                  0,                // Total Edep
                  secondary.weight, // Track weight
                  navState,         // Pre-step point navstate
                  secondary.pos,    // Pre-step point position
                  secondary.dir,    // Pre-step point momentum direction
                  secondary.eKin,   // Pre-step point kinetic energy
                  navState,         // Post-step point navstate
                  secondary.pos,    // Post-step point position
                  secondary.dir,    // Post-step point momentum direction
                  secondary.eKin,   // Post-step point kinetic energy
                  globalTime,       // global time
                  0.,               // local time
                  globalTime,       // global time at preStepPoint, for initializingStep its the globalTime
                  secondary.eventId, secondary.threadId, // eventID and threadID
                  false,                                 // whether this was the last step
                  secondary.stepCounter);                // stepcounter
            }
          }

          eKin -= deltaEkin;

          // if below tracking cut, deposit energy for electrons (positrons are annihilated later) and stop particles
          if (eKin < g4HepEmPars.fElectronTrackingCut) {
            energyDeposit += eKin;
            stopped = true;
#if ADEPT_DEBUG_TRACK > 0
            if (verbose) printf("| STOPPED by tracking cut\n");
#endif
            break;
          }

          dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
          trackSurvives = true;
          break;
        }
        case 1: {
          // Invoke model for Bremsstrahlung: either SB- or Rel-Brem.
          double logEnergy = std::log(eKin);
          double deltaEkin = eKin < g4HepEmPars.fElectronBremModelLim
                                 ? G4HepEmElectronInteractionBrem::SampleETransferSB(
                                       &g4HepEmData, eKin, logEnergy, auxData.fMCIndex, &rnge, IsElectron)
                                 : G4HepEmElectronInteractionBrem::SampleETransferRB(
                                       &g4HepEmData, eKin, logEnergy, auxData.fMCIndex, &rnge, IsElectron);

          double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
          double dirSecondary[3];
          G4HepEmElectronInteractionBrem::SampleDirections(eKin, deltaEkin, dirSecondary, dirPrimary, &rnge);

          adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 0, /*numPositrons*/ 0, /*numGammas*/ 1);
#if ADEPT_DEBUG_TRACK > 0
          if (verbose) printf("| BREMSSTRAHLUNG: deltaEkin %g \n", deltaEkin);
#endif
          // Apply cuts
          if (ApplyCuts && (deltaEkin < theGammaCut)) {
            // Deposit the energy here and kill the secondary
            energyDeposit += deltaEkin;
#if ADEPT_DEBUG_TRACK > 0
            if (verbose) printf("| secondary killed by cut \n");
#endif
          } else {
            // check for Woodcock tracking
            const bool useWDT      = ShouldUseWDT(navState, deltaEkin);
            auto &gammaPartManager = useWDT ? particleManager.gammasWDT : particleManager.gammas;

            Track &gamma = gammaPartManager.NextTrack(
                newRNG, deltaEkin, pos, vecgeom::Vector3D<double>{dirSecondary[0], dirSecondary[1], dirSecondary[2]},
                navState, currentTrack, globalTime);

            // if tracking or stepping action is called, return initial step
            if (returnLastStep) {
              adept_scoring::RecordHit(
                  userScoring, gamma.trackId, gamma.parentId, /*CreatorProcessId*/ short(1),
                  /* gamma*/ 2,                  // Particle type
                  0,                             // Step length
                  0,                             // Total Edep
                  gamma.weight,                  // Track weight
                  navState,                      // Pre-step point navstate
                  gamma.pos,                     // Pre-step point position
                  gamma.dir,                     // Pre-step point momentum direction
                  gamma.eKin,                    // Pre-step point kinetic energy
                  navState,                      // Post-step point navstate
                  gamma.pos,                     // Post-step point position
                  gamma.dir,                     // Post-step point momentum direction
                  gamma.eKin,                    // Post-step point kinetic energy
                  globalTime,                    // global time
                  0.,                            // local time
                  globalTime,                    // global time at preStepPoint, for initializingStep its the globalTime
                  gamma.eventId, gamma.threadId, // eventID and threadID
                  false,                         // whether this was the last step
                  gamma.stepCounter);            // stepcounter
            }
          }

          eKin -= deltaEkin;

          // if below tracking cut, deposit energy for electrons (positrons are annihilated later) and stop particles
          if (eKin < g4HepEmPars.fElectronTrackingCut) {
            energyDeposit += eKin;
            stopped = true;
#if ADEPT_DEBUG_TRACK > 0
            if (verbose) printf("| STOPPED by tracking cut\n");
#endif
            break;
          }

          dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
#if ADEPT_DEBUG_TRACK > 0
          if (verbose) printf("| new_dir {%.19f, %.19f, %.19f}\n", dir[0], dir[1], dir[2]);
#endif
          trackSurvives = true;
          break;
        }
        case 2: {
          // Invoke annihilation (in-flight) for e+
#if ADEPT_DEBUG_TRACK > 0
          if (verbose) printf("| ANIHHILATION\n");
#endif
          double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
          double theGamma1Ekin, theGamma2Ekin;
          double theGamma1Dir[3], theGamma2Dir[3];
          G4HepEmPositronInteractionAnnihilation::SampleEnergyAndDirectionsInFlight(
              eKin, dirPrimary, &theGamma1Ekin, theGamma1Dir, &theGamma2Ekin, theGamma2Dir, &rnge);

          // TODO: In principle particles are produced, then cut before stacking them. It seems correct to count them
          // here
          adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 0, /*numPositrons*/ 0, /*numGammas*/ 2);

          // Apply cuts
          if (ApplyCuts && (theGamma1Ekin < theGammaCut)) {
            // Deposit the energy here and kill the secondaries
            energyDeposit += theGamma1Ekin;

          } else {
            const bool useWDT      = ShouldUseWDT(navState, theGamma1Ekin);
            auto &gammaPartManager = useWDT ? particleManager.gammasWDT : particleManager.gammas;
            Track &gamma1 =
                gammaPartManager.NextTrack(newRNG, theGamma1Ekin, pos,
                                           vecgeom::Vector3D<double>{theGamma1Dir[0], theGamma1Dir[1], theGamma1Dir[2]},
                                           navState, currentTrack, globalTime);
            // if tracking or stepping action is called, return initial step
            if (returnLastStep) {
              adept_scoring::RecordHit(
                  userScoring, gamma1.trackId, gamma1.parentId, /*CreatorProcessId*/ short(2),
                  /* gamma*/ 2,  // Particle type
                  0,             // Step length
                  0,             // Total Edep
                  gamma1.weight, // Track weight
                  navState,      // Pre-step point navstate
                  gamma1.pos,    // Pre-step point position
                  gamma1.dir,    // Pre-step point momentum direction
                  gamma1.eKin,   // Pre-step point kinetic energy
                  navState,      // Post-step point navstate
                  gamma1.pos,    // Post-step point position
                  gamma1.dir,    // Post-step point momentum direction
                  gamma1.eKin,   // Post-step point kinetic energy
                  globalTime,    // global time
                  0.,            // local time
                  globalTime,    // global time at preStepPoint, for initializingStep its the globalTime
                  gamma1.eventId, gamma1.threadId, // eventID and threadID
                  false,                           // whether this was the last step
                  gamma1.stepCounter);             // stepcounter
            }
          }
          if (ApplyCuts && (theGamma2Ekin < theGammaCut)) {
            // Deposit the energy here and kill the secondaries
            energyDeposit += theGamma2Ekin;

          } else {
            const bool useWDT      = ShouldUseWDT(navState, theGamma2Ekin);
            auto &gammaPartManager = useWDT ? particleManager.gammasWDT : particleManager.gammas;
            Track &gamma2 =
                gammaPartManager.NextTrack(currentTrack.rngState, theGamma2Ekin, pos,
                                           vecgeom::Vector3D<double>{theGamma2Dir[0], theGamma2Dir[1], theGamma2Dir[2]},
                                           navState, currentTrack, globalTime);
            // if tracking or stepping action is called, return initial step
            if (returnLastStep) {
              adept_scoring::RecordHit(
                  userScoring, gamma2.trackId, gamma2.parentId, /*CreatorProcessId*/ short(2),
                  /* gamma*/ 2,  // Particle type
                  0,             // Step length
                  0,             // Total Edep
                  gamma2.weight, // Track weight
                  navState,      // Pre-step point navstate
                  gamma2.pos,    // Pre-step point position
                  gamma2.dir,    // Pre-step point momentum direction
                  gamma2.eKin,   // Pre-step point kinetic energy
                  navState,      // Post-step point navstate
                  gamma2.pos,    // Post-step point position
                  gamma2.dir,    // Post-step point momentum direction
                  gamma2.eKin,   // Post-step point kinetic energy
                  globalTime,    // global time
                  0.,            // local time
                  globalTime,    // global time at preStepPoint, for initializingStep its the globalTime
                  gamma2.eventId, gamma2.threadId, // eventID and threadID
                  false,                           // whether this was the last step
                  gamma2.stepCounter);             // stepcounter
            }
          }
          break;
        }
        case 3: {
          // Lepton nuclear needs to be handled by Geant4 directly, passing track back to CPU
          trackSurvives = true;
          leakReason    = LeakStatus::LeptonNuclear;
          break;
        }
        }
      }
    }

    __syncwarp(); // sync warp before atomics in secondary generation

    if (stopped) {
      eKin = 0;
      if (!IsElectron) {
        // Annihilate the stopped positron into two gammas heading to opposite
        // directions (isotropic).

        // Apply cuts
        if (ApplyCuts && (copcore::units::kElectronMassC2 < theGammaCut)) {
          // Deposit the energy here and don't initialize any secondaries
          energyDeposit += 2 * copcore::units::kElectronMassC2;
        } else {

          adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 0, /*numPositrons*/ 0, /*numGammas*/ 2);

          const double cost = 2 * currentTrack.Uniform() - 1;
          const double sint = sqrt(1 - cost * cost);
          const double phi  = k2Pi * currentTrack.Uniform();
          double sinPhi, cosPhi;
          sincos(phi, &sinPhi, &cosPhi);

          // as the branched newRNG may have already been used by interactions before, we need to create a new one
          RanluxppDouble newRNG2(currentTrack.rngState.Branch());

          const bool useWDT      = ShouldUseWDT(navState, double{copcore::units::kElectronMassC2});
          auto &gammaPartManager = useWDT ? particleManager.gammasWDT : particleManager.gammas;

          Track &gamma1 = gammaPartManager.NextTrack(newRNG2, double{copcore::units::kElectronMassC2}, pos,
                                                     vecgeom::Vector3D<double>{sint * cosPhi, sint * sinPhi, cost},
                                                     navState, currentTrack, globalTime);

          // Reuse the RNG state of the dying track.
          Track &gamma2 = gammaPartManager.NextTrack(currentTrack.rngState, double{copcore::units::kElectronMassC2},
                                                     pos, -gamma1.dir, navState, currentTrack, globalTime);

          // if tracking or stepping action is called, return initial step
          if (returnLastStep) {
            adept_scoring::RecordHit(userScoring, gamma1.trackId, gamma1.parentId, /*CreatorProcessId*/ short(2),
                                     /* gamma*/ 2,  // Particle type
                                     0,             // Step length
                                     0,             // Total Edep
                                     gamma1.weight, // Track weight
                                     navState,      // Pre-step point navstate
                                     gamma1.pos,    // Pre-step point position
                                     gamma1.dir,    // Pre-step point momentum direction
                                     gamma1.eKin,   // Pre-step point kinetic energy
                                     navState,      // Post-step point navstate
                                     gamma1.pos,    // Post-step point position
                                     gamma1.dir,    // Post-step point momentum direction
                                     gamma1.eKin,   // Post-step point kinetic energy
                                     globalTime,    // global time
                                     0.,            // local time
                                     globalTime, // global time at preStepPoint, for initializingStep its the globalTime
                                     gamma1.eventId, gamma1.threadId, // eventID and threadID
                                     false,                           // whether this was the last step
                                     gamma1.stepCounter);             // stepcounter
            adept_scoring::RecordHit(userScoring, gamma2.trackId, gamma2.parentId, /*CreatorProcessId*/ short(2),
                                     /* gamma*/ 2,  // Particle type
                                     0,             // Step length
                                     0,             // Total Edep
                                     gamma2.weight, // Track weight
                                     navState,      // Pre-step point navstate
                                     gamma2.pos,    // Pre-step point position
                                     gamma2.dir,    // Pre-step point momentum direction
                                     gamma2.eKin,   // Pre-step point kinetic energy
                                     navState,      // Post-step point navstate
                                     gamma2.pos,    // Post-step point position
                                     gamma2.dir,    // Post-step point momentum direction
                                     gamma2.eKin,   // Post-step point kinetic energy
                                     globalTime,    // global time
                                     0.,            // local time
                                     globalTime, // global time at preStepPoint, for initializingStep its the globalTime
                                     gamma2.eventId, gamma2.threadId, // eventID and threadID
                                     false,                           // whether this was the last step
                                     gamma2.stepCounter);             // stepcounter
          }
        }
      }
    }

    if (trackSurvives) {
      if (++currentTrack.looperCounter > 500) {
        // Kill loopers that are not advancing in free space or are scraping at a boundary
        if (printErrors)
          printf("Killing looper due to lack of advance or scraping at a boundary: E=%E event=%d track=%lu loop=%d "
                 "energyDeposit=%E "
                 "geoStepLength=%E "
                 "physicsStepLength=%E "
                 "safety=%E\n",
                 eKin, currentTrack.eventId, currentTrack.trackId, currentTrack.looperCounter, energyDeposit,
                 geometryStepLength, geometricalStepLengthFromPhysics, safety);
        trackSurvives = false;
        // For electrons, simply deposit the kinetic energy, for positrons also the 2 m_e that would be added from
        // annihilation at rest
        energyDeposit += IsElectron ? eKin : eKin + 2 * copcore::units::kElectronMassC2;
        eKin = 0.;
      } else if (currentTrack.stepCounter >= maxSteps || currentTrack.zeroStepCounter > kStepsStuckKill) {
        if (printErrors)
          printf("Killing e-/+ event %d track %lu E=%f lvol=%d after %d steps with zeroStepCounter %u\n",
                 currentTrack.eventId, currentTrack.trackId, eKin, lvolID, currentTrack.stepCounter,
                 currentTrack.zeroStepCounter);
        trackSurvives = false;
        // For electrons, simply deposit the kinetic energy, for positrons also the 2 m_e that would be added from
        // annihilation at rest
        energyDeposit += IsElectron ? eKin : eKin + 2 * copcore::units::kElectronMassC2;
        eKin = 0.;
      } else {
        // call experiment-specific SteppingAction:
        SteppingActionT::ElectronAction(trackSurvives, eKin, energyDeposit, leakReason, pos, globalTime,
                                        auxData.fMCIndex, &g4HepEmData, params);
      }
    }

    // this one always needs to be last as it needs to be done only if the track survives
    if (trackSurvives) {
      if (InFlightStats->perEventInFlightPrevious[currentTrack.threadId] < allowFinishOffEvent[currentTrack.threadId] &&
          InFlightStats->perEventInFlightPrevious[currentTrack.threadId] != 0) {
        if (printErrors)
          printf("Thread %d Finishing e-/e+ of the %d last particles of event %d on CPU E=%f lvol=%d after %d steps.\n",
                 currentTrack.threadId, InFlightStats->perEventInFlightPrevious[currentTrack.threadId],
                 currentTrack.eventId, eKin, lvolID, currentTrack.stepCounter);
        leakReason = LeakStatus::FinishEventOnCPU;
      }
    }

    __syncwarp(); // was found to be beneficial after divergent calls

    if (trackSurvives) {
      survive();
    } else {
      // particles that don't survive are killed by not enqueing them to the next queue and freeing the slot
      slotManager.MarkSlotForFreeing(slot);
    }

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || (returnLastStep && !trackSurvives)) {
      adept_scoring::RecordHit(userScoring, currentTrack.trackId, currentTrack.parentId, short(winnerProcessIndex),
                               static_cast<char>(IsElectron ? 0 : 1),       // Particle type
                               elTrack.GetPStepLength(),                    // Step length
                               energyDeposit,                               // Total Edep
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
                               currentTrack.eventId, currentTrack.threadId, // eventID and threadID
                               !trackSurvives,                              // whether this was the last step
                               currentTrack.stepCounter);                   // stepcounter
    }
  }
}

// Instantiate kernels for electrons and positrons.
template <typename Scoring, class SteppingActionT>
__global__ void TransportElectrons(ParticleManager particleManager, Scoring *userScoring, Stats *InFlightStats,
                                   const StepActionParam params, AllowFinishOffEventArray allowFinishOffEvent,
                                   const bool returnAllSteps, const bool returnLastStep)
{
  TransportElectrons</*IsElectron*/ true, Scoring, SteppingActionT>(
      particleManager, userScoring, InFlightStats, params, allowFinishOffEvent, returnAllSteps, returnLastStep);
}
template <typename Scoring, class SteppingActionT>
__global__ void TransportPositrons(ParticleManager particleManager, Scoring *userScoring, Stats *InFlightStats,
                                   const StepActionParam params, AllowFinishOffEventArray allowFinishOffEvent,
                                   const bool returnAllSteps, const bool returnLastStep)
{
  TransportElectrons</*IsElectron*/ false, Scoring, SteppingActionT>(
      particleManager, userScoring, InFlightStats, params, allowFinishOffEvent, returnAllSteps, returnLastStep);
}

} // namespace AsyncAdePT
