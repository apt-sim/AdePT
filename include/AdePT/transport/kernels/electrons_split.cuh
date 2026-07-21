// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <AdePT/transport/navigation/AdePTNavigator.h>
#include <AdePT/transport/config/TransportKernelOptions.hh>

// Classes for Runge-Kutta integration
#include <AdePT/transport/magneticfield/MagneticFieldEquation.h>
#include <AdePT/transport/magneticfield/DormandPrinceRK45.h>
#include <AdePT/transport/magneticfield/fieldPropagatorRungeKutta.h>

#include <AdePT/transport/support/PhysicalConstants.h>
#include <AdePT/transport/support/AdePTPrecision.hh>
#include <AdePT/transport/kernels/AdePTSteppingActionSelector.cuh>
#include <AdePT/transport/kernels/WoodcockHelper.cuh>
#include <AdePT/transport/queues/ParticleManager.cuh>
#include <AdePT/transport/state/DeviceGlobals.cuh>
#include <AdePT/transport/state/TransportStats.hh>
#include <AdePT/transport/steps/GPUStepRecording.cuh>

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

using StepActionParam        = adept::SteppingAction::Params;
using SelectedSteppingAction = adept::SteppingAction::Action;
using adept::SteppingAction::SetSecondaryHostData;
using VolAuxData = adeptint::VolAuxData;

// Compute velocity based on the kinetic energy of the particle
__device__ double GetVelocity(double eKin)
{
  // Taken from G4DynamicParticle::ComputeBeta
  double T    = eKin / copcore::units::kElectronMassC2;
  double beta = sqrt(T * (T + 2.)) / (T + 1.0);
  return copcore::units::kCLight * beta;
}

namespace adept::transport {

template <bool IsElectron, class SteppingActionT>
__global__ void ElectronHowFar(ParticleManager particleManager, G4HepEmElectronTrack *hepEMTracks,
                               adept::MParray *propagationQueue, Stats *InFlightStats, const StepActionParam params,
                               AllowFinishOffEventArray allowFinishOffEvent, const TransportKernelOptions options)
{
  constexpr unsigned short maxSteps        = 10'000;
  constexpr int Charge                     = IsElectron ? -1 : 1;
  constexpr double restMass                = copcore::units::kElectronMassC2;
  constexpr int Nvar                       = 6;
  constexpr unsigned short kStepsStuckKill = 25;
  const bool returnLastStep                = options.returnLastStep;

#ifdef ADEPT_USE_EXT_BFIELD
  using Field_t = GeneralMagneticField;
#else
  using Field_t = UniformMagneticField;
#endif
  using Equation_t = MagneticFieldEquation<Field_t>;
  using Stepper_t  = DormandPrinceRK45<Equation_t, Field_t, Nvar, rk_integration_t>;
  using RkDriver_t = RkIntegrationDriver<Stepper_t, rk_integration_t, int, Equation_t, Field_t>;

  auto &electronsOrPositrons = (IsElectron ? particleManager.electrons : particleManager.positrons);
  SlotManager &slotManager   = *electronsOrPositrons.fSlotManager;

  const int activeSize = electronsOrPositrons.ActiveSize();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const auto slot            = electronsOrPositrons.ActiveAt(i);
    ChargedTrack &currentTrack = electronsOrPositrons.TrackAt(slot);
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = adept::transport::gVolAuxData[lvolID];

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
      if (currentTrack.looperCounter > options.maxChargedLooperCount) {
        // Kill loopers that are not advancing in free space or are scraping at a boundary
        if (printErrors)
          printf("Killing looper due to lack of advance or scraping at a boundary: E=%E event=%d track=%lu loop=%d "
                 "energyDeposit=%E "
                 "geoStepLength=%E "
                 "safety=%E\n",
                 currentTrack.eKin, currentTrack.eventId, currentTrack.trackId, currentTrack.looperCounter,
                 energyDeposit, theTrack->GetGStepLength(), currentTrack.GetSafety(currentTrack.pos));
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
                                        currentTrack.globalTime, auxData, &g4HepEmData, params);
      }

      // this one always needs to be last as it needs to be done only if the track survives
      if (trackSurvives) {
        if (InFlightStats->perEventInFlightPrevious[currentTrack.threadId] <
                allowFinishOffEvent[currentTrack.threadId] &&
            InFlightStats->perEventInFlightPrevious[currentTrack.threadId] != 0) {
          slotManager.MarkSlotForFreeing(slot);

          adept_step_recording::RecordGPUStep(currentTrack.trackId,     // Track ID
                                              currentTrack.parentId,    // parent Track ID
                                              kAdePTFinishOnCPUProcess, // step limiting process
                                              IsElectron ? ParticleType::Electron
                                                         : ParticleType::Positron, // Particle type
                                              0.,                                  // Step length
                                              0.,                                  // Total Edep
                                              currentTrack.weight,                 // Track weight
                                              currentTrack.navState,               // Pre-step point navstate
                                              currentTrack.preStepPos,             // Pre-step point position
                                              currentTrack.preStepDir,             // Pre-step point momentum direction
                                              currentTrack.preStepEKin,            // Pre-step point kinetic energy
                                              currentTrack.navState,               // Post-step point navstate
                                              currentTrack.pos,                    // Post-step point position
                                              currentTrack.dir,                    // Post-step point momentum direction
                                              currentTrack.eKin,                   // Post-step point kinetic energy
                                              currentTrack.globalTime,             // global time
                                              currentTrack.localTime,              // local time
                                              currentTrack.properTime,             // proper time
                                              currentTrack.preStepGlobalTime,      // preStep global time
                                              currentTrack.eventId, currentTrack.threadId, // eventID and threadID
                                              false,                                       // parent continues on CPU
                                              currentTrack.hasHostData,
                                              currentTrack.stepCounter, // stepcounter
                                              nullptr,                  // pointer to secondary init data
                                              0);                       // number of secondaries
          continue;
        }
      } else {
        // Free the slot of the killed track
        slotManager.MarkSlotForFreeing(slot);

        // In case the last steps are recorded, record it now, as this track is killed
        if (returnLastStep || currentTrack.hasHostData) {
          adept_step_recording::RecordGPUStep(currentTrack.trackId,   // Track ID
                                              currentTrack.parentId,  // parent Track ID
                                              static_cast<short>(10), // step limiting process ID
                                              IsElectron ? ParticleType::Electron : ParticleType::Positron,
                                              0.,            // Step length is 0, as post and prestep point are the same
                                              energyDeposit, // Total Edep
                                              currentTrack.weight,            // Track weight
                                              currentTrack.navState,          // Pre-step point navstate
                                              currentTrack.preStepPos,        // Pre-step point position
                                              currentTrack.preStepDir,        // Pre-step point momentum direction
                                              currentTrack.preStepEKin,       // Pre-step point kinetic energy
                                              currentTrack.navState,          // Post-step point navstate
                                              currentTrack.pos,               // Post-step point position
                                              currentTrack.dir,               // Post-step point momentum direction
                                              currentTrack.eKin,              // Post-step point kinetic energy
                                              currentTrack.globalTime,        // global time
                                              currentTrack.localTime,         // local time
                                              currentTrack.properTime,        // proper time
                                              currentTrack.preStepGlobalTime, // preStep global time
                                              currentTrack.eventId,           // eventID
                                              currentTrack.threadId,          // threadID
                                              true,                           // whether this was the last step
                                              currentTrack.hasHostData,
                                              currentTrack.stepCounter, // stepcounter
                                              nullptr,                  // pointer to secondary init data
                                              0);                       // number of secondaries
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
        // Use maximum accuracy only if safety is smaller than physicalStepLength
        safety = AdePTNavigator::ComputeSafety(currentTrack.pos, currentTrack.navState, physicalStepLength);
      }
    }
    currentTrack.SetSafety(currentTrack.pos, safety);
    theTrack->SetSafety(safety);
    currentTrack.restrictedPhysicalStepLength = false;

    currentTrack.safeLength = 0.;

    if (gMagneticField) {

      const double momentumMag = sqrt(currentTrack.eKin * (currentTrack.eKin + 2.0 * restMass));
      // Distance along the track direction to reach the maximum allowed error

      // SEVERIN: to be checked if we can use float
      vecgeom::Vector3D<double> momentumVec          = momentumMag * currentTrack.dir;
      vecgeom::Vector3D<rk_integration_t> B0fieldVec = gMagneticField->Evaluate(
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

    // Particles that were not cut are added to the queue used by the next kernels
    propagationQueue->push_back(slot);
  }
}

template <bool IsElectron>
__global__ void ElectronPropagation(ChargedTrack *electronsOrPositrons, G4HepEmElectronTrack *hepEMTracks,
                                    const adept::MParray *propagationQueue)
{
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

  int activeSize = propagationQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot             = (*propagationQueue)[i];
    ChargedTrack &currentTrack = electronsOrPositrons[slot];
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
    SafetyCache geometrySafetyCache(currentTrack.safetyCache);

    if (gMagneticField) {
      int iterDone = -1;
      geometryStepLength =
          fieldPropagatorRungeKutta<Field_t, RkDriver_t, rk_integration_t, AdePTNavigator>::ComputeStepAndNextVolume(
              *gMagneticField, currentTrack.eKin, restMass, Charge, theTrack->GetGStepLength(), currentTrack.safeLength,
              currentTrack.pos, currentTrack.dir, currentTrack.navState, currentTrack.nextState, currentTrack.hitsurfID,
              currentTrack.propagated, geometrySafetyCache,
              // activeSize < 100 ? max_iterations : max_iters_tail ), // Was
              max_iterations, iterDone, slot);

    } else {
#ifdef ADEPT_USE_SURF
      geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(currentTrack.pos, currentTrack.dir,
                                                                    theTrack->GetGStepLength(), currentTrack.navState,
                                                                    currentTrack.nextState, currentTrack.hitsurfID);
#else
      geometryStepLength =
          AdePTNavigator::ComputeStepAndNextVolume(currentTrack.pos, currentTrack.dir, theTrack->GetGStepLength(),
                                                   currentTrack.navState, currentTrack.nextState);
#endif
      currentTrack.pos += geometryStepLength * currentTrack.dir;
    }

    if (geometryStepLength < kPushStuck && geometryStepLength < theTrack->GetGStepLength()) {
      currentTrack.zeroStepCounter++;
      if (currentTrack.zeroStepCounter > kStepsStuckPush) currentTrack.pos += kPushStuck * currentTrack.dir;
    } else {
      currentTrack.zeroStepCounter = 0;
    }

    // punish minuscule steps by increasing the looperCounter by 10
    if (geometryStepLength < 100 * vecgeom::kTolerance) currentTrack.looperCounter += 10;

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    currentTrack.navState.SetBoundaryState(currentTrack.nextState.IsOnBoundary());
    if (currentTrack.nextState.IsOnBoundary()) {
      currentTrack.SetSafety(currentTrack.pos, 0.);
    } else {
      currentTrack.SetSafety(currentTrack.pos, geometrySafetyCache.SafetyAt(currentTrack.pos));
    }

    // Propagate information from geometrical step to MSC.
    theTrack->SetDirection(currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z());
    theTrack->SetGStepLength(geometryStepLength);
    theTrack->SetOnBoundary(currentTrack.nextState.IsOnBoundary());
  }
}

template <bool IsElectron>
__global__ void ElectronMSC(ChargedTrack *electrons, G4HepEmElectronTrack *hepEMTracks, const adept::MParray *active)
{
  constexpr double restMass = copcore::units::kElectronMassC2;

  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot = (*active)[i];

    ChargedTrack &currentTrack = electrons[slot];
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
      constexpr double kGeomMinLength  = 0.05 * copcore::units::nm;       // 0.05 [nm]
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
          // Use maximum accuracy only if safety is smaller than physicalStepLength
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
template <bool IsElectron>
__global__ void ElectronSetupInteractions(G4HepEmElectronTrack *hepEMTracks, const adept::MParray *propagationQueue,
                                          ParticleManager particleManager, SplitQueues splitQueues,
                                          const TransportKernelOptions options)
{
  auto &electronsOrPositrons = (IsElectron ? particleManager.electrons : particleManager.positrons);
  SlotManager &slotManager   = *electronsOrPositrons.fSlotManager;

  const bool returnAllSteps = options.returnAllSteps;
  const bool returnLastStep = options.returnLastStep;

  int activeSize = propagationQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot             = (*propagationQueue)[i];
    ChargedTrack &currentTrack = electronsOrPositrons.TrackAt(slot);
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = adept::transport::gVolAuxData[lvolID];

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
      splitQueues.queues[ParticleQueues::relocation]->push_back(slot);
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
        winnerProcessIndex = kAdePTTransportationProcess;

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
        winnerProcessIndex = kAdePTTransportationProcess;
      }
    } else {
      // Stopped positrons annihilate, stopped electrons score and die
      if (IsElectron) {
        reached_interaction = false;
        trackSurvives       = false;
        // Ekin = 0 for the returned stopping step
        currentTrack.eKin = 0;
        // Particle is killed by not enqueuing it for the next iteration. Free the slot it occupies
        slotManager.MarkSlotForFreeing(slot);
      }
    }

    // Now push the particles that reached their interaction into the
    // per-interaction queues. Lepton nuclear (winnerProcessIndex == 3) is
    // handled on the host from the returned step only.
    if (reached_interaction && winnerProcessIndex != 3) {
      // reset Looper counter if limited by discrete interaction or MSC
      currentTrack.looperCounter = 0;

      if (!currentTrack.stopped) {
        // Reset number of interaction left for the winner discrete process.
        // (Will be resampled in the next iteration.)
        theTrack->SetNumIALeft(-1.0, winnerProcessIndex);
        // Enqueue the particles
        splitQueues.queues[winnerProcessIndex]->push_back(slot);
      } else {
        // Stopped positron
        splitQueues.queues[ParticleQueues::positronStoppedAnnihilation]->push_back(slot);
      }

    } else {

      const bool continuesOnCPU = reached_interaction && winnerProcessIndex == 3;

      if (continuesOnCPU) {
        // The returned hit is sufficient to reconstruct the step and later
        // continue the parent on the CPU in sorted order.
        trackSurvives = false;
        slotManager.MarkSlotForFreeing(slot);
      }

      // if not already dead, check for SteppingAction and survive
      if (trackSurvives) {

        // possible hook to SteppingAction here

        // --- Survive --- //
        electronsOrPositrons.EnqueueNext(slot);
      }

      // Only non-interacting, non-relocating tracks score here
      // Score the edep for particles that didn't reach the interaction
      if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || continuesOnCPU ||
          ((returnLastStep || currentTrack.hasHostData) && (!trackSurvives || continuesOnCPU))) {
        adept_step_recording::RecordGPUStep(currentTrack.trackId,                   // Track ID
                                            currentTrack.parentId,                  // parent Track ID
                                            static_cast<short>(winnerProcessIndex), // step defining process
                                            IsElectron ? ParticleType::Electron
                                                       : ParticleType::Positron, // Particle type
                                            elTrack.GetPStepLength(),            // Step length
                                            energyDeposit,                       // Total Edep
                                            currentTrack.weight,                 // Track weight
                                            currentTrack.navState,               // Pre-step point navstate
                                            currentTrack.preStepPos,             // Pre-step point position
                                            currentTrack.preStepDir,             // Pre-step point momentum direction
                                            currentTrack.preStepEKin,            // Pre-step point kinetic energy
                                            currentTrack.nextState,              // Post-step point navstate
                                            currentTrack.pos,                    // Post-step point position
                                            currentTrack.dir,                    // Post-step point momentum direction
                                            currentTrack.eKin,                   // Post-step point kinetic energy
                                            currentTrack.globalTime,             // global time
                                            currentTrack.localTime,              // local time
                                            currentTrack.properTime,             // proper time
                                            currentTrack.preStepGlobalTime,      // preStep global time
                                            currentTrack.eventId, currentTrack.threadId, // eventID and threadID
                                            !trackSurvives && !continuesOnCPU, // whether this was the last step
                                            currentTrack.hasHostData,
                                            currentTrack.stepCounter, // stepcounter
                                            nullptr,                  // pointer to secondary init data
                                            0);                       // number of secondaries
      }
    }
  }
}

template <bool IsElectron>
__global__ void ElectronRelocation(G4HepEmElectronTrack *hepEMTracks, ParticleManager particleManager,
                                   adept::MParray *relocatingQueue, const TransportKernelOptions options)
{
  auto &electronsOrPositrons = (IsElectron ? particleManager.electrons : particleManager.positrons);

  SlotManager &slotManager  = *electronsOrPositrons.fSlotManager;
  const bool returnAllSteps = options.returnAllSteps;
  const bool returnLastStep = options.returnLastStep;
  int activeSize            = relocatingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot             = (*relocatingQueue)[i];
    ChargedTrack &currentTrack = electronsOrPositrons.TrackAt(slot);
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = adept::transport::gVolAuxData[lvolID];

    auto survive = [&]() { electronsOrPositrons.EnqueueNext(slot); };

    bool trackSurvives = true;

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    double energyDeposit = theTrack->GetEnergyDeposit();

    bool cross_boundary = false;
    bool returnsToCPU   = false;

    // Relocate to have the correct next state before RecordGPUStep is called

    // - Kill loopers stuck at a boundary
    // - Set cross boundary flag in order to set the correct navstate for the returned step
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
    } else {
      // Particle left the world, don't enqueue it and release the slot
      slotManager.MarkSlotForFreeing(slot);
      trackSurvives = false;
    }

    short stepProcessId = kAdePTTransportationProcess;
    bool isLastStep     = !trackSurvives;
    if (cross_boundary) {
      const int nextlvolID          = currentTrack.nextState.GetLogicalId();
      VolAuxData const &nextauxData = adept::transport::gVolAuxData[nextlvolID];
      returnsToCPU                  = nextauxData.fGPUregionId < 0;
      if (returnsToCPU) {
        stepProcessId = kAdePTOutOfGPURegionProcess;
        isLastStep    = false;
      }
    }

    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnsToCPU ||
        (cross_boundary && currentTrack.hasHostData) ||
        (!trackSurvives && (returnLastStep || currentTrack.hasHostData)))
      adept_step_recording::RecordGPUStep(currentTrack.trackId,  // Track ID
                                          currentTrack.parentId, // parent Track ID
                                          stepProcessId,         // step limiting process ID
                                          IsElectron ? ParticleType::Electron : ParticleType::Positron, // Particle type
                                          elTrack.GetPStepLength(),                                     // Step length
                                          energyDeposit,                                                // Total Edep
                                          currentTrack.weight,                                          // Track weight
                                          currentTrack.navState,          // Pre-step point navstate
                                          currentTrack.preStepPos,        // Pre-step point position
                                          currentTrack.preStepDir,        // Pre-step point momentum direction
                                          currentTrack.preStepEKin,       // Pre-step point kinetic energy
                                          currentTrack.nextState,         // Post-step point navstate
                                          currentTrack.pos,               // Post-step point position
                                          currentTrack.dir,               // Post-step point momentum direction
                                          currentTrack.eKin,              // Post-step point kinetic energy
                                          currentTrack.globalTime,        // global time
                                          currentTrack.localTime,         // local time
                                          currentTrack.properTime,        // proper time
                                          currentTrack.preStepGlobalTime, // preStep global time
                                          currentTrack.eventId, currentTrack.threadId, // eventID and threadID
                                          isLastStep,                                  // whether this was the last step
                                          currentTrack.hasHostData,
                                          currentTrack.stepCounter, // stepcounter
                                          nullptr,                  // pointer to secondary init data
                                          0);                       // number of secondaries

    if (cross_boundary) {
      // Check if the next volume belongs to the GPU region and push it to the appropriate queue
      const int nextlvolID          = currentTrack.nextState.GetLogicalId();
      VolAuxData const &nextauxData = adept::transport::gVolAuxData[nextlvolID];
      if (!returnsToCPU) {
        // Move to the next boundary now that the step is recorded.
        currentTrack.navState = currentTrack.nextState;
        theTrack->SetMCIndex(nextauxData.fMCIndex);
        survive();
      } else {
        slotManager.MarkSlotForFreeing(slot);
      }
    }
  }
}

__device__ __forceinline__ void PerformStoppedAnnihilation(const int slot, ChargedTrack &currentTrack,
                                                           ParticleManager &particleManager, double &energyDeposit,
                                                           const bool ApplyCuts, const double theGammaCut,
                                                           SecondaryInitData *secondaryData, unsigned int &nSecondaries,
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
    auto &gamma1 = gammaPartManager.NextTrack(newRNG, double{copcore::units::kElectronMassC2}, currentTrack.pos,
                                              vecgeom::Vector3D<double>{sint * cosPhi, sint * sinPhi, cost},
                                              currentTrack.navState, currentTrack, currentTrack.globalTime);

    // Reuse the RNG state of the dying track.
    auto &gamma2 =
        gammaPartManager.NextTrack(currentTrack.rngState, double{copcore::units::kElectronMassC2}, currentTrack.pos,
                                   -gamma1.dir, currentTrack.navState, currentTrack, currentTrack.globalTime);

    SetSecondaryHostData<SelectedSteppingAction>(gamma1, currentTrack, 0., returnLastStep);
    SetSecondaryHostData<SelectedSteppingAction>(gamma2, currentTrack, 0., returnLastStep);

    // Return the initializing step when HostTrackData is needed for these secondaries.
    if (gamma1.hasHostData) {
      secondaryData[nSecondaries++] = {gamma1.trackId,      gamma1.dir,
                                       gamma1.eKin,         /*creator process*/ short(2),
                                       ParticleType::Gamma, gamma1.hasHostData};
      secondaryData[nSecondaries++] = {gamma2.trackId,      gamma2.dir,
                                       gamma2.eKin,         /*creator process*/ short(2),
                                       ParticleType::Gamma, gamma2.hasHostData};
    }
  }
}

template <bool IsElectron>
__global__ void ElectronIonization(G4HepEmElectronTrack *hepEMTracks, ParticleManager particleManager,
                                   adept::MParray *interactingQueue, const TransportKernelOptions options)
{
  auto &electronsOrPositrons = (IsElectron ? particleManager.electrons : particleManager.positrons);
  SlotManager &slotManager   = *electronsOrPositrons.fSlotManager;
  const bool returnAllSteps  = options.returnAllSteps;
  const bool returnLastStep  = options.returnLastStep;
  int activeSize             = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot           = (*active)[i];
    const int slot             = (*interactingQueue)[i];
    ChargedTrack &currentTrack = electronsOrPositrons.TrackAt(slot);
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = adept::transport::gVolAuxData[lvolID];
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

    // data structure for possible secondaries that are generated
    SecondaryInitData secondaryData[3];
    unsigned int nSecondaries = 0;

    // Apply cuts
    if (ApplyCuts && (deltaEkin < theElCut)) {
      // Deposit the energy here and kill the secondary
      energyDeposit += deltaEkin;

    } else {
      auto &secondary = particleManager.electrons.NextTrack(
          newRNG, deltaEkin, currentTrack.pos,
          vecgeom::Vector3D<double>{dirSecondary[0], dirSecondary[1], dirSecondary[2]}, currentTrack.navState,
          currentTrack, currentTrack.globalTime);

      SetSecondaryHostData<SelectedSteppingAction>(secondary, currentTrack, copcore::units::kElectronMassC2,
                                                   returnLastStep);

      // Return the initializing step when HostTrackData is needed for this secondary.
      if (secondary.hasHostData) {
        secondaryData[nSecondaries++] = {secondary.trackId,      secondary.dir,
                                         secondary.eKin,         /*creator process*/ short(0),
                                         ParticleType::Electron, secondary.hasHostData};
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
        PerformStoppedAnnihilation(slot, currentTrack, particleManager, energyDeposit, ApplyCuts, theGammaCut,
                                   secondaryData, nSecondaries, returnLastStep);
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
        ((returnLastStep || currentTrack.hasHostData) && (nSecondaries > 0 || !trackSurvives))) {
      adept_step_recording::RecordGPUStep(currentTrack.trackId,  // Track ID
                                          currentTrack.parentId, // parent Track ID
                                          static_cast<short>(0), // step limiting process ID
                                          IsElectron ? ParticleType::Electron : ParticleType::Positron, // Particle type
                                          elTrack.GetPStepLength(),                                     // Step length
                                          energyDeposit,                                                // Total Edep
                                          currentTrack.weight,                                          // Track weight
                                          currentTrack.navState,          // Pre-step point navstate
                                          currentTrack.preStepPos,        // Pre-step point position
                                          currentTrack.preStepDir,        // Pre-step point momentum direction
                                          currentTrack.preStepEKin,       // Pre-step point kinetic energy
                                          currentTrack.nextState,         // Post-step point navstate
                                          currentTrack.pos,               // Post-step point position
                                          currentTrack.dir,               // Post-step point momentum direction
                                          currentTrack.eKin,              // Post-step point kinetic energy
                                          currentTrack.globalTime,        // global time
                                          currentTrack.localTime,         // local time
                                          currentTrack.properTime,        // proper time
                                          currentTrack.preStepGlobalTime, // preStep global time
                                          currentTrack.eventId, currentTrack.threadId, // eventID and threadID
                                          !trackSurvives,                              // whether this was the last step
                                          currentTrack.hasHostData,
                                          currentTrack.stepCounter, // stepcounter
                                          secondaryData,            // pointer to secondary init data
                                          nSecondaries);            // number of secondaries
    }
  }
}

template <bool IsElectron>
__global__ void ElectronBremsstrahlung(G4HepEmElectronTrack *hepEMTracks, ParticleManager particleManager,
                                       adept::MParray *interactingQueue, const TransportKernelOptions options)
{
  auto &electronsOrPositrons = (IsElectron ? particleManager.electrons : particleManager.positrons);
  SlotManager &slotManager   = *electronsOrPositrons.fSlotManager;
  const bool returnAllSteps  = options.returnAllSteps;
  const bool returnLastStep  = options.returnLastStep;
  int activeSize             = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot           = (*active)[i];
    const int slot             = (*interactingQueue)[i];
    ChargedTrack &currentTrack = electronsOrPositrons.TrackAt(slot);
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = adept::transport::gVolAuxData[lvolID];
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

    // data structure for possible secondaries that are generated
    SecondaryInitData secondaryData[3];
    unsigned int nSecondaries = 0;

    // Apply cuts
    if (ApplyCuts && (deltaEkin < theGammaCut)) {
      // Deposit the energy here and kill the secondary
      energyDeposit += deltaEkin;

    } else {
      bool createGamma  = true;
      float gammaWeight = currentTrack.weight;
      if constexpr (adept::SteppingAction::Action::kGammaRussianRoulette) {
        const auto gammaRouletteResult = adept::SteppingAction::Action::ApplyGammaRussianRoulette(
            currentTrack.weight, deltaEkin, auxData, currentTrack.rngState);
        createGamma = gammaRouletteResult.create;
        gammaWeight = gammaRouletteResult.weight;
      }
      if (createGamma) {
        const bool useWDT      = ShouldUseWDT(currentTrack.navState, deltaEkin);
        auto &gammaPartManager = useWDT ? particleManager.gammasWDT : particleManager.gammas;
        auto &gamma =
            gammaPartManager.NextTrack(newRNG, deltaEkin, currentTrack.pos,
                                       vecgeom::Vector3D<double>{dirSecondary[0], dirSecondary[1], dirSecondary[2]},
                                       currentTrack.navState, currentTrack, currentTrack.globalTime, gammaWeight);
        SetSecondaryHostData<SelectedSteppingAction>(gamma, currentTrack, 0., returnLastStep);
        // Return the initializing step when HostTrackData is needed for this secondary.
        if (gamma.hasHostData) {
          secondaryData[nSecondaries++] = {gamma.trackId,       gamma.dir,
                                           gamma.eKin,          /*creator process*/ short(1),
                                           ParticleType::Gamma, gamma.hasHostData};
        }
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
        PerformStoppedAnnihilation(slot, currentTrack, particleManager, energyDeposit, ApplyCuts, theGammaCut,
                                   secondaryData, nSecondaries, returnLastStep);
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
        ((returnLastStep || currentTrack.hasHostData) && (nSecondaries > 0 || !trackSurvives))) {
      adept_step_recording::RecordGPUStep(currentTrack.trackId,  // Track ID
                                          currentTrack.parentId, // parent Track ID
                                          static_cast<short>(1), // step limiting process ID
                                          IsElectron ? ParticleType::Electron : ParticleType::Positron, // Particle type
                                          elTrack.GetPStepLength(),                                     // Step length
                                          energyDeposit,                                                // Total Edep
                                          currentTrack.weight,                                          // Track weight
                                          currentTrack.navState,          // Pre-step point navstate
                                          currentTrack.preStepPos,        // Pre-step point position
                                          currentTrack.preStepDir,        // Pre-step point momentum direction
                                          currentTrack.preStepEKin,       // Pre-step point kinetic energy
                                          currentTrack.nextState,         // Post-step point navstate
                                          currentTrack.pos,               // Post-step point position
                                          currentTrack.dir,               // Post-step point momentum direction
                                          currentTrack.eKin,              // Post-step point kinetic energy
                                          currentTrack.globalTime,        // global time
                                          currentTrack.localTime,         // local time
                                          currentTrack.properTime,        // proper time
                                          currentTrack.preStepGlobalTime, // preStep global time
                                          currentTrack.eventId, currentTrack.threadId, // eventID and threadID
                                          !trackSurvives,                              // whether this was the last step
                                          currentTrack.hasHostData,
                                          currentTrack.stepCounter, // stepcounter
                                          secondaryData,            // pointer to secondary init data
                                          nSecondaries);            // number of secondaries
    }
  }
}

__global__ void PositronAnnihilation(G4HepEmElectronTrack *hepEMTracks, ParticleManager particleManager,
                                     adept::MParray *interactingQueue, const TransportKernelOptions options)
{
  SlotManager &slotManager  = *particleManager.positrons.fSlotManager;
  const bool returnAllSteps = options.returnAllSteps;
  const bool returnLastStep = options.returnLastStep;
  int activeSize            = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot = (*interactingQueue)[i];

    ChargedTrack &currentTrack = particleManager.positrons.TrackAt(slot);
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = adept::transport::gVolAuxData[lvolID];

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

    // data structure for possible secondaries that are generated
    SecondaryInitData secondaryData[2];
    unsigned int nSecondaries = 0;

    // Apply cuts
    if (ApplyCuts && (theGamma1Ekin < theGammaCut)) {
      // Deposit the energy here and kill the secondaries
      energyDeposit += theGamma1Ekin;

    } else {
      bool createGamma1  = true;
      float gamma1Weight = currentTrack.weight;
      if constexpr (adept::SteppingAction::Action::kGammaRussianRoulette) {
        const auto gamma1RouletteResult = adept::SteppingAction::Action::ApplyGammaRussianRoulette(
            currentTrack.weight, theGamma1Ekin, auxData, currentTrack.rngState);
        createGamma1 = gamma1RouletteResult.create;
        gamma1Weight = gamma1RouletteResult.weight;
      }
      if (createGamma1) {
        const bool useWDT      = ShouldUseWDT(currentTrack.navState, theGamma1Ekin);
        auto &gammaPartManager = useWDT ? particleManager.gammasWDT : particleManager.gammas;
        auto &gamma1 =
            gammaPartManager.NextTrack(newRNG, theGamma1Ekin, currentTrack.pos,
                                       vecgeom::Vector3D<double>{theGamma1Dir[0], theGamma1Dir[1], theGamma1Dir[2]},
                                       currentTrack.navState, currentTrack, currentTrack.globalTime, gamma1Weight);
        SetSecondaryHostData<SelectedSteppingAction>(gamma1, currentTrack, 0., returnLastStep);
        // Return the initializing step when HostTrackData is needed for this secondary.
        if (gamma1.hasHostData) {
          secondaryData[nSecondaries++] = {gamma1.trackId,      gamma1.dir,
                                           gamma1.eKin,         /*creator process*/ short(2),
                                           ParticleType::Gamma, gamma1.hasHostData};
        }
      }
    }
    if (ApplyCuts && (theGamma2Ekin < theGammaCut)) {
      // Deposit the energy here and kill the secondaries
      energyDeposit += theGamma2Ekin;

    } else {
      bool createGamma2  = true;
      float gamma2Weight = currentTrack.weight;
      if constexpr (adept::SteppingAction::Action::kGammaRussianRoulette) {
        const auto gamma2RouletteResult = adept::SteppingAction::Action::ApplyGammaRussianRoulette(
            currentTrack.weight, theGamma2Ekin, auxData, currentTrack.rngState);
        createGamma2 = gamma2RouletteResult.create;
        gamma2Weight = gamma2RouletteResult.weight;
      }
      if (createGamma2) {
        const bool useWDT      = ShouldUseWDT(currentTrack.navState, theGamma2Ekin);
        auto &gammaPartManager = useWDT ? particleManager.gammasWDT : particleManager.gammas;
        auto &gamma2 =
            gammaPartManager.NextTrack(currentTrack.rngState, theGamma2Ekin, currentTrack.pos,
                                       vecgeom::Vector3D<double>{theGamma2Dir[0], theGamma2Dir[1], theGamma2Dir[2]},
                                       currentTrack.navState, currentTrack, currentTrack.globalTime, gamma2Weight);
        SetSecondaryHostData<SelectedSteppingAction>(gamma2, currentTrack, 0., returnLastStep);
        // Return the initializing step when HostTrackData is needed for this secondary.
        if (gamma2.hasHostData) {
          secondaryData[nSecondaries++] = {gamma2.trackId,      gamma2.dir,
                                           gamma2.eKin,         /*creator process*/ short(2),
                                           ParticleType::Gamma, gamma2.hasHostData};
        }
      }
    }

    // The current track is killed by not enqueuing into the next activeQueue.
    slotManager.MarkSlotForFreeing(slot);

    assert(nSecondaries <= 2);

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep ||
        currentTrack.hasHostData) {
      adept_step_recording::RecordGPUStep(
          currentTrack.trackId,                        // Track ID
          currentTrack.parentId,                       // parent Track ID
          static_cast<short>(2),                       // step limiting process ID
          ParticleType::Positron,                      // Particle type
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
          currentTrack.properTime,                     // proper time
          currentTrack.preStepGlobalTime,              // preStep global time
          currentTrack.eventId, currentTrack.threadId, // eventID and threadID
          true, // whether this was the last step: always true for annihilating positrons
          currentTrack.hasHostData,
          currentTrack.stepCounter, // stepcounter
          secondaryData,            // pointer to secondary init data
          nSecondaries);            // number of secondaries
    }
  }
}

__global__ void PositronStoppedAnnihilation(G4HepEmElectronTrack *hepEMTracks, ParticleManager particleManager,
                                            adept::MParray *interactingQueue, const TransportKernelOptions options)
{
  SlotManager &slotManager  = *particleManager.positrons.fSlotManager;
  const bool returnAllSteps = options.returnAllSteps;
  const bool returnLastStep = options.returnLastStep;
  int activeSize            = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot = (*interactingQueue)[i];

    ChargedTrack &currentTrack = particleManager.positrons.TrackAt(slot);
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = adept::transport::gVolAuxData[lvolID];

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

    PerformStoppedAnnihilation(slot, currentTrack, particleManager, energyDeposit, ApplyCuts, theGammaCut,
                               secondaryData, nSecondaries, returnLastStep);
    slotManager.MarkSlotForFreeing(slot);

    assert(nSecondaries <= 2);

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep ||
        currentTrack.hasHostData) {
      adept_step_recording::RecordGPUStep(
          currentTrack.trackId,                        // Track ID
          currentTrack.parentId,                       // parent Track ID
          static_cast<short>(2),                       // step limiting process ID
          ParticleType::Positron,                      // Particle type
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
          currentTrack.properTime,                     // proper time
          currentTrack.preStepGlobalTime,              // preStep global time
          currentTrack.eventId, currentTrack.threadId, // eventID and threadID
          true, // whether this was the last step: always true for annihilating positrons
          currentTrack.hasHostData,
          currentTrack.stepCounter, // stepcounter
          secondaryData,            // pointer to secondary init data
          nSecondaries);            // number of secondaries
    }
  }
}

} // namespace adept::transport
