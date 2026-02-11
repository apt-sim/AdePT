// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/navigation/AdePTNavigator.h>

// Classes for Runge-Kutta integration
#include <AdePT/magneticfield/MagneticFieldEquation.h>
#include <AdePT/magneticfield/DormandPrinceRK45.h>
#include <AdePT/magneticfield/fieldPropagatorRungeKutta.h>

#include <AdePT/copcore/PhysicalConstants.h>
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

template <bool IsElectron, typename Scoring, class SteppingActionT>
__global__ void ElectronHowFar(ParticleManager particleManager, G4HepEmElectronTrack *hepEMTracks,
                               adept::MParray *propagationQueue, Stats *InFlightStats, const StepActionParam params,
                               Scoring *userScoring, AllowFinishOffEventArray allowFinishOffEvent,
                               const bool returnAllSteps, const bool returnLastStep)
{
  constexpr unsigned short maxSteps        = 10'000;
  constexpr int Charge                     = IsElectron ? -1 : 1;
  constexpr double restMass                = copcore::units::kElectronMassC2;
  constexpr int Nvar                       = 6;
  constexpr unsigned short kStepsStuckKill = 25;

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
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID];

    currentTrack.preStepEKin       = currentTrack.eKin;
    currentTrack.preStepGlobalTime = currentTrack.globalTime;
    currentTrack.preStepPos        = currentTrack.pos;
    currentTrack.preStepDir        = currentTrack.dir;
    currentTrack.stepCounter++;
    bool printErrors = false;

    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();
    G4HepEmMSCTrackData *mscData  = elTrack.GetMSCTrackData();
    if (!currentTrack.hepEmTrackExists) {
      // Init a track with the needed data to call into G4HepEm.
      elTrack.ReSet();
      theTrack->SetEKin(currentTrack.eKin);
      theTrack->SetMCIndex(auxData.fMCIndex);
      theTrack->SetOnBoundary(currentTrack.navState.IsOnBoundary());
      theTrack->SetCharge(Charge);
      mscData->ReSet();
      currentTrack.hepEmTrackExists = true;
    } else {
      theTrack->SetEnergyDeposit(0);
      mscData->fIsFirstStep = false;
    }

    // ---- Begin of SteppingAction:
    {
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

      // check for loopers
      if (currentTrack.looperCounter > 500) {
        // Kill loopers that are not advancing in free space or are scraping at a boundary
        if (printErrors)
          printf("Killing looper due to lack of advance or scraping at a boundary: E=%E event=%d track=%lu loop=%d "
                 "energyDeposit=%E "
                 "geoStepLength=%E "
                 "safety=%E\n",
                 currentTrack.eKin, currentTrack.eventId, currentTrack.trackId, currentTrack.looperCounter,
                 energyDeposit, theTrack->GetGStepLength(), currentTrack.safety);
        trackSurvives = false;
        energyDeposit += IsElectron ? currentTrack.eKin : currentTrack.eKin + 2 * copcore::units::kElectronMassC2;
        currentTrack.eKin = 0.;
        // check for max steps and stuck tracks
      } else if (currentTrack.stepCounter >= maxSteps || currentTrack.zeroStepCounter > kStepsStuckKill) {
        if (printErrors)
          printf("Killing e-/+ event %d track %lu E=%f lvol=%d after %d steps with zeroStepCounter %u\n",
                 currentTrack.eventId, currentTrack.trackId, currentTrack.eKin, lvolID, currentTrack.stepCounter,
                 currentTrack.zeroStepCounter);
        trackSurvives = false;
        energyDeposit += IsElectron ? currentTrack.eKin : currentTrack.eKin + 2 * copcore::units::kElectronMassC2;
        currentTrack.eKin = 0.;
        // check for experiment-specific SteppingAction
      } else {
        SteppingActionT::ElectronAction(trackSurvives, currentTrack.eKin, energyDeposit, currentTrack.pos,
                                        currentTrack.globalTime, auxData.fMCIndex, &g4HepEmData, params);
      }

      // this one always needs to be last as it needs to be done only if the track survives
      if (trackSurvives) {
        if (InFlightStats->perEventInFlightPrevious[currentTrack.threadId] <
                allowFinishOffEvent[currentTrack.threadId] &&
            InFlightStats->perEventInFlightPrevious[currentTrack.threadId] != 0) {
          if (printErrors) {
            printf(
                "Thread %d Finishing e-/e+ of the %d last particles of event %d on CPU E=%f lvol=%d after %d steps.\n",
                currentTrack.threadId, InFlightStats->perEventInFlightPrevious[currentTrack.threadId],
                currentTrack.eventId, currentTrack.eKin, lvolID, currentTrack.stepCounter);
          }

          // Set LeakStatus and copy to leaked queue
          currentTrack.leakStatus = LeakStatus::FinishEventOnCPU;
          electronsOrPositrons.CopyTrackToLeaked(slot);
          continue;
        }
      } else {
        // Free the slot of the killed track
        slotManager.MarkSlotForFreeing(slot);

        // In case the last steps are recorded, record it now, as this track is killed
        if (returnLastStep) {
          adept_scoring::RecordHit(userScoring,
                                   currentTrack.trackId,                  // Track ID
                                   currentTrack.parentId,                 // parent Track ID
                                   static_cast<short>(10),                // step limiting process ID
                                   static_cast<char>(IsElectron ? 0 : 1), // Particle type
                                   0.,                       // Step length is 0, as post and prestep point are the same
                                   energyDeposit,            // Total Edep
                                   currentTrack.weight,      // Track weight
                                   currentTrack.navState,    // Pre-step point navstate
                                   currentTrack.preStepPos,  // Pre-step point position
                                   currentTrack.preStepDir,  // Pre-step point momentum direction
                                   currentTrack.preStepEKin, // Pre-step point kinetic energy
                                   currentTrack.navState,    // Post-step point navstate
                                   currentTrack.pos,         // Post-step point position
                                   currentTrack.dir,         // Post-step point momentum direction
                                   currentTrack.eKin,        // Post-step point kinetic energy
                                   currentTrack.globalTime,  // global time
                                   currentTrack.localTime,   // local time
                                   currentTrack.preStepGlobalTime, // preStep global time
                                   currentTrack.eventId,           // eventID
                                   currentTrack.threadId,          // threadID
                                   true,                           // whether this was the last step
                                   currentTrack.stepCounter,       // stepcounter
                                   nullptr,                        // pointer to secondary init data
                                   0);                             // number of secondaries
        }
        continue; // track is killed, can stop here
      }
    }
    // ---- End of SteppingAction

    G4HepEmRandomEngine rnge(&currentTrack.rngState);

    // Sample the `number-of-interaction-left` and put it into the track.
    for (int ip = 0; ip < 4; ++ip) {
      if (theTrack->GetNumIALeft(ip) <= 0) {
        theTrack->SetNumIALeft(-std::log(currentTrack.Uniform()), ip);
      }
    }

    G4HepEmElectronManager::HowFarToDiscreteInteraction(&g4HepEmData, &g4HepEmPars, &elTrack);

    auto physicalStepLength = elTrack.GetPStepLength();
    // Compute safety, needed for MSC step limit. The accuracy range is physicalStepLength
    double safety = 0.;
    if (!currentTrack.navState.IsOnBoundary()) {
      // Get the remaining safety only if larger than physicalStepLength
      safety = currentTrack.GetSafety(currentTrack.pos);
      if (safety < physicalStepLength) {
        // Recompute safety and update it in the track.
        // Use maximum accuracy only if safety is samller than physicalStepLength
        safety = AdePTNavigator::ComputeSafety(currentTrack.pos, currentTrack.navState, physicalStepLength);
        currentTrack.SetSafety(currentTrack.pos, safety);
      }
    }
    theTrack->SetSafety(safety);
    currentTrack.restrictedPhysicalStepLength = false;

    currentTrack.safeLength = 0.;

    if (gMagneticField) {

      const double momentumMag = sqrt(currentTrack.eKin * (currentTrack.eKin + 2.0 * restMass));
      // Distance along the track direction to reach the maximum allowed error

      // SEVERIN: to be checked if we can use float
      vecgeom::Vector3D<double> momentumVec          = momentumMag * currentTrack.dir;
      vecgeom::Vector3D<rk_integration_t> B0fieldVec = magneticField.Evaluate(
          currentTrack.pos[0], currentTrack.pos[1], currentTrack.pos[2]); // Field value at starting point
      currentTrack.safeLength =
          fieldPropagatorRungeKutta<Field_t, RkDriver_t, rk_integration_t,
                                    AdePTNavigator>::ComputeSafeLength /*<Real_t>*/ (momentumVec, B0fieldVec, Charge);

      constexpr int MaxSafeLength = 10;
      double limit                = MaxSafeLength * currentTrack.safeLength;
      limit                       = safety > limit ? safety : limit;

      if (physicalStepLength > limit) {
        physicalStepLength                        = limit;
        currentTrack.restrictedPhysicalStepLength = true;
        elTrack.SetPStepLength(physicalStepLength);

        // Note: We are limiting the true step length, which is converted to
        // a shorter geometry step length in HowFarToMSC. In that sense, the
        // limit is an over-approximation, but that is fine for our purpose.
      }
    }

    G4HepEmElectronManager::HowFarToMSC(&g4HepEmData, &g4HepEmPars, &elTrack, &rnge);

    // Particles that were not cut or leaked are added to the queue used by the next kernels
    propagationQueue->push_back(slot);
  }
}

template <bool IsElectron>
__global__ void ElectronPropagation(ParticleManager particleManager, G4HepEmElectronTrack *hepEMTracks)
{
  constexpr double kPushDistance           = 1000 * vecgeom::kTolerance;
  constexpr int Charge                     = IsElectron ? -1 : 1;
  constexpr double restMass                = copcore::units::kElectronMassC2;
  constexpr int Nvar                       = 6;
  constexpr int max_iterations             = 10;
  constexpr double kPushStuck              = 100 * vecgeom::kTolerance;
  constexpr unsigned short kStepsStuckPush = 5;

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

  const int activeSize = electronsOrPositrons.ActiveSize();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const auto slot     = electronsOrPositrons.ActiveAt(i);
    Track &currentTrack = electronsOrPositrons.TrackAt(slot);
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    G4HepEmRandomEngine rnge(&currentTrack.rngState);

    // Check if there's a volume boundary in between.
    currentTrack.propagated = true;
    currentTrack.hitsurfID  = -1;
    double geometryStepLength;
    bool zero_first_step = false;

    if (gMagneticField) {
      int iterDone = -1;
      geometryStepLength =
          fieldPropagatorRungeKutta<Field_t, RkDriver_t, rk_integration_t, AdePTNavigator>::ComputeStepAndNextVolume(
              magneticField, currentTrack.eKin, restMass, Charge, theTrack->GetGStepLength(), currentTrack.safeLength,
              currentTrack.pos, currentTrack.dir, currentTrack.navState, currentTrack.nextState, currentTrack.hitsurfID,
              currentTrack.propagated,
              /*lengthDone,*/ currentTrack.safety,
              // activeSize < 100 ? max_iterations : max_iters_tail ), // Was
              max_iterations, iterDone, slot, zero_first_step);
      // In case of zero step detected by the field propagator this could be due to back scattering, or wrong relocation
      // in the previous step.
      // - In case of BS we should just restore the last exited one for the nextState. For now we cannot detect BS.
      // if (zero_first_step) nextState.SetNavIndex(navState.GetLastExitedState());

    } else {
#ifdef ADEPT_USE_SURF
      geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(currentTrack.pos, currentTrack.dir,
                                                                    theTrack->GetGStepLength(), currentTrack.navState,
                                                                    currentTrack.nextState, currentTrack.hitsurfID);
#else
      geometryStepLength =
          AdePTNavigator::ComputeStepAndNextVolume(currentTrack.pos, currentTrack.dir, theTrack->GetGStepLength(),
                                                   currentTrack.navState, currentTrack.nextState, kPushDistance);
#endif
      currentTrack.pos += geometryStepLength * currentTrack.dir;
    }

    if (geometryStepLength < kPushStuck && geometryStepLength < theTrack->GetGStepLength()) {
      currentTrack.zeroStepCounter++;
      if (currentTrack.zeroStepCounter > kStepsStuckPush) currentTrack.pos += kPushStuck * currentTrack.dir;
    } else
      currentTrack.zeroStepCounter = 0;

    // punish miniscule steps by increasing the looperCounter by 10
    if (geometryStepLength < 100 * vecgeom::kTolerance) currentTrack.looperCounter += 10;

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    currentTrack.navState.SetBoundaryState(currentTrack.nextState.IsOnBoundary());
    if (currentTrack.nextState.IsOnBoundary()) currentTrack.SetSafety(currentTrack.pos, 0.);

    // Propagate information from geometrical step to MSC.
    theTrack->SetDirection(currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z());
    theTrack->SetGStepLength(geometryStepLength);
    theTrack->SetOnBoundary(currentTrack.nextState.IsOnBoundary());
  }
}

template <bool IsElectron>
__global__ void ElectronMSC(Track *electrons, G4HepEmElectronTrack *hepEMTracks, const adept::MParray *active)
{
  constexpr double restMass = copcore::units::kElectronMassC2;

  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot = (*active)[i];

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    G4HepEmRandomEngine rnge(&currentTrack.rngState);

    // Apply continuous effects.
    currentTrack.stopped = G4HepEmElectronManager::PerformContinuous(&g4HepEmData, &g4HepEmPars, &elTrack, &rnge);

    // Collect the direction change and displacement by MSC.
    const double *direction = theTrack->GetDirection();
    currentTrack.dir.Set(direction[0], direction[1], direction[2]);
    if (!currentTrack.nextState.IsOnBoundary()) {
      const double *mscDisplacement = mscData->GetDisplacement();
      vecgeom::Vector3D<double> displacement(mscDisplacement[0], mscDisplacement[1], mscDisplacement[2]);
      const double dLength2            = displacement.Length2();
      constexpr double kGeomMinLength  = 5 * copcore::units::nm;          // 0.05 [nm]
      constexpr double kGeomMinLength2 = kGeomMinLength * kGeomMinLength; // (0.05 [nm])^2
      if (dLength2 > kGeomMinLength2) {
        const double dispR = std::sqrt(dLength2);
        // Estimate safety by subtracting the geometrical step length.
        double safety          = currentTrack.GetSafety(currentTrack.pos);
        constexpr double sFact = 0.99;
        double reducedSafety   = sFact * safety;

        // Apply displacement, depending on how close we are to a boundary.
        // 1a. Far away from geometry boundary:
        if (reducedSafety > 0.0 && dispR <= reducedSafety) {
          currentTrack.pos += displacement;
        } else {
          // Recompute safety.
          // Use maximum accuracy only if safety is samller than physicalStepLength
          safety = AdePTNavigator::ComputeSafety(currentTrack.pos, currentTrack.navState, dispR);
          currentTrack.SetSafety(currentTrack.pos, safety);
          reducedSafety = sFact * safety;

          // 1b. Far away from geometry boundary:
          if (reducedSafety > 0.0 && dispR <= reducedSafety) {
            currentTrack.pos += displacement;
            // 2. Push to boundary:
          } else if (reducedSafety > kGeomMinLength) {
            currentTrack.pos += displacement * (reducedSafety / dispR);
          }
          // 3. Very small safety: do nothing.
        }
      }
    }

    // Collect the charged step length (might be changed by MSC). Collect the changes in energy and deposit.
    currentTrack.eKin = theTrack->GetEKin();
    theTrack->SetEKin(currentTrack.eKin);

    // Update the flight times of the particle
    // By calculating the velocity here, we assume that all the energy deposit is done at the PreStepPoint, and
    // the velocity depends on the remaining energy
    double deltaTime = elTrack.GetPStepLength() / GetVelocity(currentTrack.eKin);
    currentTrack.globalTime += deltaTime;
    currentTrack.localTime += deltaTime;
    currentTrack.properTime += deltaTime * (restMass / (restMass + currentTrack.eKin));
  }
}

/***
 * @brief Adds tracks to interaction and relocation queues depending on their state
 */
template <bool IsElectron, typename Scoring>
__global__ void ElectronSetupInteractions(G4HepEmElectronTrack *hepEMTracks, const adept::MParray *propagationQueue,
                                          ParticleManager particleManager, AllInteractionQueues interactionQueues,
                                          Scoring *userScoring, const bool returnAllSteps, const bool returnLastStep)
{
  auto &electronsOrPositrons = (IsElectron ? particleManager.electrons : particleManager.positrons);
  SlotManager &slotManager   = *electronsOrPositrons.fSlotManager;

  int activeSize = propagationQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*propagationQueue)[i];
    Track &currentTrack = electronsOrPositrons.TrackAt(slot);
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID];

    bool trackSurvives = true;

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();

    double energyDeposit = theTrack->GetEnergyDeposit();

    bool reached_interaction = true;

    // Set Non-stopped, on-boundary tracks for relocation
    if (currentTrack.nextState.IsOnBoundary() && !currentTrack.stopped) {
      // Add particle to relocation queue
      interactionQueues.queues[4]->push_back(slot);
      continue;
    }

    auto winnerProcessIndex = theTrack->GetWinnerProcessIndex();

    // Now check whether the non-relocating tracks reached an interaction
    if (!currentTrack.stopped) {
      if (!currentTrack.propagated || currentTrack.restrictedPhysicalStepLength) {
        // Did not yet reach the interaction point due to error in the magnetic
        // field propagation. Try again next time.

        ++currentTrack.looperCounter;

        // mark winner process to be transport, although this is not strictly true
        winnerProcessIndex = 10;

        reached_interaction = false;
      } else if (winnerProcessIndex < 0) {
        // No discrete process, move on.
        reached_interaction = false;
      } else if (G4HepEmElectronManager::CheckDelta(&g4HepEmData, theTrack, currentTrack.Uniform())) {
        // If there was a delta interaction, the track survives but does not move onto the next kernel
        reached_interaction = false;
        // Reset number of interaction left for the winner discrete process.
        // (Will be resampled in the next iteration.)
        theTrack->SetNumIALeft(-1.0, winnerProcessIndex);
      }
    } else {
      // Stopped positrons annihilate, stopped electrons score and die
      if (IsElectron) {
        reached_interaction = false;
        trackSurvives       = false;
        // Ekin = 0 for correct scoring
        currentTrack.eKin = 0;
        // Particle is killed by not enqueuing it for the next iteration. Free the slot it occupies
        slotManager.MarkSlotForFreeing(slot);
      }
    }

    // Now push the particles that reached their interaction into the per-interaction queues,
    // except for lepton nuclear (winnerProcessIndex == 3), which is sent back to the CPU
    if (reached_interaction && winnerProcessIndex != 3) {
      // reset Looper counter if limited by discrete interaction or MSC
      currentTrack.looperCounter = 0;

      if (!currentTrack.stopped) {
        // Reset number of interaction left for the winner discrete process.
        // (Will be resampled in the next iteration.)
        theTrack->SetNumIALeft(-1.0, winnerProcessIndex);
        // Enqueue the particles
        interactionQueues.queues[winnerProcessIndex]->push_back(slot);
      } else {
        // Stopped positron
        interactionQueues.queues[3]->push_back(slot);
      }

    } else {

      // if not already dead, check for SteppingAction and survive
      if (trackSurvives) {

        // possible hook to SteppingAction here

        // Lepton nuclear needs to be handled by Geant4 directly, passing track back to CPU
        auto leakReason = winnerProcessIndex == 3 ? LeakStatus::LeptonNuclear : LeakStatus::NoLeak;

        // --- Survive --- //
        currentTrack.leakStatus = leakReason;
        if (leakReason == LeakStatus::LeptonNuclear) {
          // Copy track at slot to the leaked tracks
          electronsOrPositrons.CopyTrackToLeaked(slot);
        } else {
          electronsOrPositrons.EnqueueNext(slot);
        }
      }

      // Only non-interacting, non-relocating tracks score here
      // Score the edep for particles that didn't reach the interaction
      if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || (returnLastStep && !trackSurvives)) {
        adept_scoring::RecordHit(userScoring,
                                 currentTrack.trackId,                        // Track ID
                                 currentTrack.parentId,                       // parent Track ID
                                 static_cast<short>(winnerProcessIndex),      // step defining process
                                 static_cast<char>(IsElectron ? 0 : 1),       // Particle type
                                 elTrack.GetPStepLength(),                    // Step length
                                 energyDeposit,                               // Total Edep
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
                                 currentTrack.eventId, currentTrack.threadId, // eventID and threadID
                                 !trackSurvives,                              // whether this was the last step
                                 currentTrack.stepCounter,                    // stepcounter
                                 nullptr,                                     // pointer to secondary init data
                                 0);                                          // number of secondaries
      }
    }
  }
}

template <bool IsElectron, typename Scoring>
__global__ void ElectronRelocation(G4HepEmElectronTrack *hepEMTracks, ParticleManager particleManager,
                                   adept::MParray *relocatingQueue, Scoring *userScoring, const bool returnAllSteps,
                                   const bool returnLastStep)
{
  constexpr double kPushDistance = 1000 * vecgeom::kTolerance;
  auto &electronsOrPositrons     = (IsElectron ? particleManager.electrons : particleManager.positrons);

  SlotManager &slotManager = *electronsOrPositrons.fSlotManager;
  int activeSize           = relocatingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*relocatingQueue)[i];
    Track &currentTrack = electronsOrPositrons.TrackAt(slot);
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID];

    auto survive = [&](LeakStatus leakReason = LeakStatus::NoLeak) {
      // NOTE: When adapting the split kernels for async mode this won't
      // work if we want to re-use slots on the fly. Directly copying to
      // a trackdata struct would be better
      currentTrack.leakStatus = leakReason;
      if (leakReason != LeakStatus::NoLeak) {
        // Copy track at slot to the leaked tracks
        electronsOrPositrons.CopyTrackToLeaked(slot);
      } else {
        electronsOrPositrons.EnqueueNext(slot);
      }
    };

    bool trackSurvives = true;

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    double energyDeposit = theTrack->GetEnergyDeposit();

    bool cross_boundary = false;

    // Relocate to have the correct next state before RecordHit is called

    // - Kill loopers stuck at a boundary
    // - Set cross boundary flag in order to set the correct navstate after scoring
    // - Kill particles that left the world

    ++currentTrack.looperCounter;

    if (!currentTrack.nextState.IsOutside()) {
      // Mark the particle. We need to change its navigation state to the next volume before enqueuing it
      // This will happen after recording the step
      // Relocate
      cross_boundary = true;
#ifdef ADEPT_USE_SURF
      AdePTNavigator::RelocateToNextVolume(currentTrack.pos, currentTrack.dir, currentTrack.hitsurfID,
                                           currentTrack.nextState);
#else
      AdePTNavigator::RelocateToNextVolume(currentTrack.pos, currentTrack.dir, currentTrack.nextState);
#endif
      // Set the last exited state to be the one before crossing
      currentTrack.nextState.SetLastExited(currentTrack.navState.GetState());
    } else {
      // Particle left the world, don't enqueue it and release the slot
      slotManager.MarkSlotForFreeing(slot);
      trackSurvives = false;
    }

    // Score
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || (!trackSurvives && returnLastStep))
      adept_scoring::RecordHit(userScoring,
                               currentTrack.trackId,                        // Track ID
                               currentTrack.parentId,                       // parent Track ID
                               static_cast<short>(/*transport*/ 10),        // step limiting process ID
                               static_cast<char>(IsElectron ? 0 : 1),       // Particle type
                               elTrack.GetPStepLength(),                    // Step length
                               energyDeposit,                               // Total Edep
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
                               currentTrack.eventId, currentTrack.threadId, // eventID and threadID
                               !trackSurvives,                              // whether this was the last step
                               currentTrack.stepCounter,                    // stepcounter
                               nullptr,                                     // pointer to secondary init data
                               0);                                          // number of secondaries

    if (cross_boundary) {
      // Move to the next boundary now that the Step is recorded
      currentTrack.navState = currentTrack.nextState;
      // Check if the next volume belongs to the GPU region and push it to the appropriate queue
      const int nextlvolID          = currentTrack.nextState.GetLogicalId();
      VolAuxData const &nextauxData = AsyncAdePT::gVolAuxData[nextlvolID];
      if (nextauxData.fGPUregionId >= 0) {
        theTrack->SetMCIndex(nextauxData.fMCIndex);
        survive();
      } else {
        // To be safe, just push a bit the track exiting the GPU region to make sure
        // Geant4 does not relocate it again inside the same region
        currentTrack.pos += kPushDistance * currentTrack.dir;
        survive(LeakStatus::OutOfGPURegion);
      }
    }
  }
}

template <typename Scoring>
__device__ __forceinline__ void PerformStoppedAnnihilation(const int slot, Track &currentTrack,
                                                           ParticleManager &particleManager, double &energyDeposit,
                                                           const bool ApplyCuts, const double theGammaCut,
                                                           Scoring *userScoring, SecondaryInitData *secondaryData,
                                                           unsigned int &nSecondaries,
                                                           const bool returnLastStep = false)
{
  currentTrack.eKin = 0;
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

    // as the other branched newRNG may have already been used by interactions before, we need to advance and create a
    // new one
    currentTrack.rngState.Advance();
    RanluxppDouble newRNG(currentTrack.rngState.Branch());

    const bool useWDT      = ShouldUseWDT(currentTrack.navState, double{copcore::units::kElectronMassC2});
    auto &gammaPartManager = useWDT ? particleManager.gammasWDT : particleManager.gammas;
    Track &gamma1 = gammaPartManager.NextTrack(newRNG, double{copcore::units::kElectronMassC2}, currentTrack.pos,
                                               vecgeom::Vector3D<double>{sint * cosPhi, sint * sinPhi, cost},
                                               currentTrack.navState, currentTrack, currentTrack.globalTime);

    // Reuse the RNG state of the dying track.
    Track &gamma2 =
        gammaPartManager.NextTrack(currentTrack.rngState, double{copcore::units::kElectronMassC2}, currentTrack.pos,
                                   -gamma1.dir, currentTrack.navState, currentTrack, currentTrack.globalTime);

    // if tracking or stepping action is called, return initial step
    if (returnLastStep) {
      secondaryData[nSecondaries++] = {gamma1.trackId, gamma1.dir, gamma1.eKin, /*particle type*/ char(2)};
      secondaryData[nSecondaries++] = {gamma2.trackId, gamma2.dir, gamma2.eKin, /*particle type*/ char(2)};
    }
  }
}

template <bool IsElectron, typename Scoring>
__global__ void ElectronIonization(G4HepEmElectronTrack *hepEMTracks, ParticleManager particleManager,
                                   adept::MParray *interactingQueue, Scoring *userScoring, const bool returnAllSteps,
                                   const bool returnLastStep)
{
  auto &electronsOrPositrons = (IsElectron ? particleManager.electrons : particleManager.positrons);
  SlotManager &slotManager   = *electronsOrPositrons.fSlotManager;
  int activeSize             = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot           = (*active)[i];
    const int slot      = (*interactingQueue)[i];
    Track &currentTrack = electronsOrPositrons.TrackAt(slot);
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID];
    bool trackSurvives        = false;

    auto survive = [&]() {
      trackSurvives = true;
      electronsOrPositrons.EnqueueNext(slot);
    };

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    G4HepEmRandomEngine rnge(&currentTrack.rngState);

    double energyDeposit = theTrack->GetEnergyDeposit();

    const double theElCut    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecElProdCutE;
    const double theGammaCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecGamProdCutE;

    const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
    const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

    // Perform the discrete interaction, branch a new RNG state with advance so it is
    // ready to be used.
    auto newRNG = RanluxppDouble(currentTrack.rngState.Branch());
    // Also advance the current RNG state to provide a fresh round of random
    // numbers after MSC used up a fair share for sampling the displacement.
    currentTrack.rngState.Advance();

    // Invoke ionization (for e-/e+):
    double deltaEkin = (IsElectron)
                           ? G4HepEmElectronInteractionIoni::SampleETransferMoller(theElCut, currentTrack.eKin, &rnge)
                           : G4HepEmElectronInteractionIoni::SampleETransferBhabha(theElCut, currentTrack.eKin, &rnge);

    double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
    double dirSecondary[3];
    G4HepEmElectronInteractionIoni::SampleDirections(currentTrack.eKin, deltaEkin, dirSecondary, dirPrimary, &rnge);

    adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

    // data structure for possible secondaries that are generated
    SecondaryInitData secondaryData[3];
    unsigned int nSecondaries = 0;

    // Apply cuts
    if (ApplyCuts && (deltaEkin < theElCut)) {
      // Deposit the energy here and kill the secondary
      energyDeposit += deltaEkin;

    } else {
      Track &secondary = particleManager.electrons.NextTrack(
          newRNG, deltaEkin, currentTrack.pos,
          vecgeom::Vector3D<double>{dirSecondary[0], dirSecondary[1], dirSecondary[2]}, currentTrack.navState,
          currentTrack, currentTrack.globalTime);

      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        secondaryData[nSecondaries++] = {secondary.trackId, secondary.dir, secondary.eKin, /*particle type*/ char(0)};
      }
    }

    currentTrack.eKin -= deltaEkin;
    theTrack->SetEKin(currentTrack.eKin);

    // if below tracking cut, deposit energy for electrons (positrons are annihilated later) and stop particles
    if (currentTrack.eKin < g4HepEmPars.fElectronTrackingCut) {
      if (IsElectron) {
        energyDeposit += currentTrack.eKin;
      }
      if (!IsElectron) {
        PerformStoppedAnnihilation<Scoring>(slot, currentTrack, particleManager, energyDeposit, ApplyCuts, theGammaCut,
                                            userScoring, secondaryData, nSecondaries, returnLastStep);
      }
      slotManager.MarkSlotForFreeing(slot);
    } else {
      currentTrack.dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
      survive();
    }
    assert(nSecondaries <= 3);

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    // Note: step must be returned if track dies or secondaries have been generated
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps ||
        (returnLastStep && (nSecondaries > 0 || !trackSurvives))) {
      adept_scoring::RecordHit(userScoring,
                               currentTrack.trackId,                        // Track ID
                               currentTrack.parentId,                       // parent Track ID
                               static_cast<short>(0),                       // step limiting process ID
                               static_cast<char>(IsElectron ? 0 : 1),       // Particle type
                               elTrack.GetPStepLength(),                    // Step length
                               energyDeposit,                               // Total Edep
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
                               currentTrack.eventId, currentTrack.threadId, // eventID and threadID
                               !trackSurvives,                              // whether this was the last step
                               currentTrack.stepCounter,                    // stepcounter
                               secondaryData,                               // pointer to secondary init data
                               nSecondaries);                               // number of secondaries
    }
  }
}

template <bool IsElectron, typename Scoring>
__global__ void ElectronBremsstrahlung(G4HepEmElectronTrack *hepEMTracks, ParticleManager particleManager,
                                       adept::MParray *interactingQueue, Scoring *userScoring,
                                       const bool returnAllSteps, const bool returnLastStep)
{
  auto &electronsOrPositrons = (IsElectron ? particleManager.electrons : particleManager.positrons);
  SlotManager &slotManager   = *electronsOrPositrons.fSlotManager;
  int activeSize             = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot           = (*active)[i];
    const int slot      = (*interactingQueue)[i];
    Track &currentTrack = electronsOrPositrons.TrackAt(slot);
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID];
    bool trackSurvives        = false;

    auto survive = [&]() {
      trackSurvives = true;
      electronsOrPositrons.EnqueueNext(slot);
    };

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    G4HepEmRandomEngine rnge(&currentTrack.rngState);

    double energyDeposit = theTrack->GetEnergyDeposit();

    const double theGammaCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecGamProdCutE;

    const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
    const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

    // Perform the discrete interaction, branch a new RNG state with advance so it is
    // ready to be used.
    auto newRNG = RanluxppDouble(currentTrack.rngState.Branch());
    // Also advance the current RNG state to provide a fresh round of random
    // numbers after MSC used up a fair share for sampling the displacement.
    currentTrack.rngState.Advance();

    // Invoke model for Bremsstrahlung: either SB- or Rel-Brem.
    double logEnergy = std::log(currentTrack.eKin);
    double deltaEkin = currentTrack.eKin < g4HepEmPars.fElectronBremModelLim
                           ? G4HepEmElectronInteractionBrem::SampleETransferSB(
                                 &g4HepEmData, currentTrack.eKin, logEnergy, auxData.fMCIndex, &rnge, IsElectron)
                           : G4HepEmElectronInteractionBrem::SampleETransferRB(
                                 &g4HepEmData, currentTrack.eKin, logEnergy, auxData.fMCIndex, &rnge, IsElectron);

    double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
    double dirSecondary[3];
    G4HepEmElectronInteractionBrem::SampleDirections(currentTrack.eKin, deltaEkin, dirSecondary, dirPrimary, &rnge);

    adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 0, /*numPositrons*/ 0, /*numGammas*/ 1);

    // data structure for possible secondaries that are generated
    SecondaryInitData secondaryData[3];
    unsigned int nSecondaries = 0;

    // Apply cuts
    if (ApplyCuts && (deltaEkin < theGammaCut)) {
      // Deposit the energy here and kill the secondary
      energyDeposit += deltaEkin;

    } else {
      const bool useWDT      = ShouldUseWDT(currentTrack.navState, deltaEkin);
      auto &gammaPartManager = useWDT ? particleManager.gammasWDT : particleManager.gammas;
      Track &gamma =
          gammaPartManager.NextTrack(newRNG, deltaEkin, currentTrack.pos,
                                     vecgeom::Vector3D<double>{dirSecondary[0], dirSecondary[1], dirSecondary[2]},
                                     currentTrack.navState, currentTrack, currentTrack.globalTime);
      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        secondaryData[nSecondaries++] = {gamma.trackId, gamma.dir, gamma.eKin, /*particle type*/ char(2)};
      }
    }

    currentTrack.eKin -= deltaEkin;
    theTrack->SetEKin(currentTrack.eKin);

    // if below tracking cut, deposit energy for electrons (positrons are annihilated later) and stop particles
    if (currentTrack.eKin < g4HepEmPars.fElectronTrackingCut) {
      if (IsElectron) {
        energyDeposit += currentTrack.eKin;
      }
      if (!IsElectron) {
        PerformStoppedAnnihilation<Scoring>(slot, currentTrack, particleManager, energyDeposit, ApplyCuts, theGammaCut,
                                            userScoring, secondaryData, nSecondaries, returnLastStep);
      }
      slotManager.MarkSlotForFreeing(slot);
    } else {
      currentTrack.dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
      survive();
    }

    assert(nSecondaries <= 3);

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    // Note: step must be returned if track dies or secondaries were generated
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps ||
        (returnLastStep && (nSecondaries > 0 || !trackSurvives))) {
      adept_scoring::RecordHit(userScoring,
                               currentTrack.trackId,                        // Track ID
                               currentTrack.parentId,                       // parent Track ID
                               static_cast<short>(1),                       // step limiting process ID
                               static_cast<char>(IsElectron ? 0 : 1),       // Particle type
                               elTrack.GetPStepLength(),                    // Step length
                               energyDeposit,                               // Total Edep
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
                               currentTrack.eventId, currentTrack.threadId, // eventID and threadID
                               !trackSurvives,                              // whether this was the last step
                               currentTrack.stepCounter,                    // stepcounter
                               secondaryData,                               // pointer to secondary init data
                               nSecondaries);                               // number of secondaries
    }
  }
}

template <typename Scoring>
__global__ void PositronAnnihilation(G4HepEmElectronTrack *hepEMTracks, ParticleManager particleManager,
                                     adept::MParray *interactingQueue, Scoring *userScoring, const bool returnAllSteps,
                                     const bool returnLastStep)
{
  SlotManager &slotManager = *particleManager.positrons.fSlotManager;
  int activeSize           = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot = (*interactingQueue)[i];

    Track &currentTrack = particleManager.positrons.TrackAt(slot);
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID];

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    G4HepEmRandomEngine rnge(&currentTrack.rngState);

    double energyDeposit = theTrack->GetEnergyDeposit();

    const double theGammaCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecGamProdCutE;

    const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
    const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

    // Perform the discrete interaction, branch a new RNG state with advance so it is
    // ready to be used.
    auto newRNG = RanluxppDouble(currentTrack.rngState.Branch());
    // Also advance the current RNG state to provide a fresh round of random
    // numbers after MSC used up a fair share for sampling the displacement.
    currentTrack.rngState.Advance();

    // Invoke annihilation (in-flight) for e+
    double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
    double theGamma1Ekin, theGamma2Ekin;
    double theGamma1Dir[3], theGamma2Dir[3];
    G4HepEmPositronInteractionAnnihilation::SampleEnergyAndDirectionsInFlight(
        currentTrack.eKin, dirPrimary, &theGamma1Ekin, theGamma1Dir, &theGamma2Ekin, theGamma2Dir, &rnge);

    // TODO: In principle particles are produced, then cut before stacking them. It seems correct to count them
    // here
    adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 0, /*numPositrons*/ 0, /*numGammas*/ 2);

    // data structure for possible secondaries that are generated
    SecondaryInitData secondaryData[2];
    unsigned int nSecondaries = 0;

    // Apply cuts
    if (ApplyCuts && (theGamma1Ekin < theGammaCut)) {
      // Deposit the energy here and kill the secondaries
      energyDeposit += theGamma1Ekin;

    } else {
      const bool useWDT      = ShouldUseWDT(currentTrack.navState, theGamma1Ekin);
      auto &gammaPartManager = useWDT ? particleManager.gammasWDT : particleManager.gammas;
      Track &gamma1 =
          gammaPartManager.NextTrack(newRNG, theGamma1Ekin, currentTrack.pos,
                                     vecgeom::Vector3D<double>{theGamma1Dir[0], theGamma1Dir[1], theGamma1Dir[2]},
                                     currentTrack.navState, currentTrack, currentTrack.globalTime);
      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        secondaryData[nSecondaries++] = {gamma1.trackId, gamma1.dir, gamma1.eKin, /*particle type*/ char(2)};
      }
    }
    if (ApplyCuts && (theGamma2Ekin < theGammaCut)) {
      // Deposit the energy here and kill the secondaries
      energyDeposit += theGamma2Ekin;

    } else {
      const bool useWDT      = ShouldUseWDT(currentTrack.navState, theGamma2Ekin);
      auto &gammaPartManager = useWDT ? particleManager.gammasWDT : particleManager.gammas;
      Track &gamma2 =
          gammaPartManager.NextTrack(currentTrack.rngState, theGamma2Ekin, currentTrack.pos,
                                     vecgeom::Vector3D<double>{theGamma2Dir[0], theGamma2Dir[1], theGamma2Dir[2]},
                                     currentTrack.navState, currentTrack, currentTrack.globalTime);
      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        secondaryData[nSecondaries++] = {gamma2.trackId, gamma2.dir, gamma2.eKin, /*particle type*/ char(2)};
      }
    }

    // The current track is killed by not enqueuing into the next activeQueue.
    slotManager.MarkSlotForFreeing(slot);

    assert(nSecondaries <= 2);

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep) {
      adept_scoring::RecordHit(userScoring,
                               currentTrack.trackId,                        // Track ID
                               currentTrack.parentId,                       // parent Track ID
                               static_cast<short>(2),                       // step limiting process ID
                               static_cast<char>(1),                        // Particle type
                               elTrack.GetPStepLength(),                    // Step length
                               energyDeposit,                               // Total Edep
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
                               currentTrack.eventId, currentTrack.threadId, // eventID and threadID
                               true, // whether this was the last step: always true for annihilating positrons
                               currentTrack.stepCounter, // stepcounter
                               secondaryData,            // pointer to secondary init data
                               nSecondaries);            // number of secondaries
    }
  }
}

template <typename Scoring>
__global__ void PositronStoppedAnnihilation(G4HepEmElectronTrack *hepEMTracks, ParticleManager particleManager,
                                            adept::MParray *interactingQueue, Scoring *userScoring,
                                            const bool returnAllSteps, const bool returnLastStep)
{
  SlotManager &slotManager = *particleManager.positrons.fSlotManager;
  int activeSize           = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot = (*interactingQueue)[i];

    Track &currentTrack = particleManager.positrons.TrackAt(slot);
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID];

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    G4HepEmRandomEngine rnge(&currentTrack.rngState);

    double energyDeposit = theTrack->GetEnergyDeposit();

    const double theGammaCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecGamProdCutE;

    const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
    const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

    // Annihilate the stopped positron into two gammas heading to opposite
    // directions (isotropic).

    // data structure for possible secondaries that are generated
    SecondaryInitData secondaryData[2];
    unsigned int nSecondaries = 0;

    PerformStoppedAnnihilation<Scoring>(slot, currentTrack, particleManager, energyDeposit, ApplyCuts, theGammaCut,
                                        userScoring, secondaryData, nSecondaries, returnLastStep);
    slotManager.MarkSlotForFreeing(slot);

    assert(nSecondaries <= 2);

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep) {
      adept_scoring::RecordHit(userScoring,
                               currentTrack.trackId,                        // Track ID
                               currentTrack.parentId,                       // parent Track ID
                               static_cast<short>(2),                       // step limiting process ID
                               static_cast<char>(1),                        // Particle type
                               elTrack.GetPStepLength(),                    // Step length
                               energyDeposit,                               // Total Edep
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
                               currentTrack.eventId, currentTrack.threadId, // eventID and threadID
                               true, // whether this was the last step: always true for annihilating positrons
                               currentTrack.stepCounter, // stepcounter
                               secondaryData,            // pointer to secondary init data
                               nSecondaries);            // number of secondaries
    }
  }
}

} // namespace AsyncAdePT
