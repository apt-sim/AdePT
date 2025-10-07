// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/navigation/AdePTNavigator.h>

// Classes for Runge-Kutta integration
#include <AdePT/magneticfield/MagneticFieldEquation.h>
#include <AdePT/magneticfield/DormandPrinceRK45.h>
#include <AdePT/magneticfield/fieldPropagatorRungeKutta.h>

#include <AdePT/copcore/PhysicalConstants.h>
#include <AdePT/core/AdePTPrecision.hh>

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

using VolAuxData = adeptint::VolAuxData;

// Compute velocity based on the kinetic energy of the particle
__device__ double GetVelocity(double eKin)
{
  // Taken from G4DynamicParticle::ComputeBeta
  double T    = eKin / copcore::units::kElectronMassC2;
  double beta = sqrt(T * (T + 2.)) / (T + 1.0);
  return copcore::units::kCLight * beta;
}

namespace AsyncAdePT {

template <bool IsElectron>
__global__ void ElectronHowFar(Track *electrons, G4HepEmElectronTrack *hepEMTracks, const adept::MParray *active,
                               adept::MParray *nextActiveQueue, adept::MParray *propagationQueue,
                               adept::MParray *leakedQueue, Stats *InFlightStats,
                               AllowFinishOffEventArray allowFinishOffEvent)
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

  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    currentTrack.preStepEKin       = currentTrack.eKin;
    currentTrack.preStepGlobalTime = currentTrack.globalTime;
    currentTrack.preStepPos        = currentTrack.pos;
    currentTrack.preStepDir        = currentTrack.dir;
    currentTrack.stepCounter++;
    bool printErrors = false;
    if (currentTrack.stepCounter >= maxSteps || currentTrack.zeroStepCounter > kStepsStuckKill) {
      if (printErrors)
        printf("Killing e-/+ event %d track %ld E=%f lvol=%d after %d steps with zeroStepCounter %u\n",
               currentTrack.eventId, currentTrack.trackId, currentTrack.eKin, lvolID, currentTrack.stepCounter,
               currentTrack.zeroStepCounter);
      continue;
    }

    auto survive = [&](LeakStatus leakReason = LeakStatus::NoLeak) {
      // NOTE: When adapting the split kernels for async mode this won't
      // work if we want to re-use slots on the fly. Directly copying to
      // a trackdata struct would be better
      currentTrack.leakStatus = leakReason;
      if (leakReason != LeakStatus::NoLeak) {
        auto success = leakedQueue->push_back(slot);
        if (!success) {
          printf("ERROR: No space left in e-/+ leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
      } else {
        nextActiveQueue->push_back(slot);
      }
    };

    if (InFlightStats->perEventInFlightPrevious[currentTrack.threadId] < allowFinishOffEvent[currentTrack.threadId] &&
        InFlightStats->perEventInFlightPrevious[currentTrack.threadId] != 0) {
      printf("Thread %d Finishing e-/e+ of the %d last particles of event %d on CPU E=%f lvol=%d after %d steps.\n",
             currentTrack.threadId, InFlightStats->perEventInFlightPrevious[currentTrack.threadId],
             currentTrack.eventId, currentTrack.eKin, lvolID, currentTrack.stepCounter);
      survive(LeakStatus::FinishEventOnCPU);
      continue;
    }

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
      // the default is 1.0e21 but there are float vs double conversions, so we check for 1e20
      mscData->fIsFirstStep         = currentTrack.initialRange > 1.0e+20;
      mscData->fInitialRange        = currentTrack.initialRange;
      mscData->fDynamicRangeFactor  = currentTrack.dynamicRangeFactor;
      mscData->fTlimitMin           = currentTrack.tlimitMin;
      currentTrack.hepEmTrackExists = true;
    } else {
      theTrack->SetEnergyDeposit(0);
      mscData->fIsFirstStep = false;
    }

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
__global__ void ElectronPropagation(Track *electrons, G4HepEmElectronTrack *hepEMTracks, const adept::MParray *active,
                                    adept::MParray *leakedQueue)
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

    // Check if there's a volume boundary in between.
    currentTrack.propagated = true;
    currentTrack.hitsurfID  = -1;
    bool zero_first_step    = false;

    if (gMagneticField) {
      int iterDone = -1;
      currentTrack.geometryStepLength =
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
      currentTrack.geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(
          currentTrack.pos, currentTrack.dir, theTrack->GetGStepLength(), currentTrack.navState, currentTrack.nextState,
          currentTrack.hitsurfID);
#else
      currentTrack.geometryStepLength =
          AdePTNavigator::ComputeStepAndNextVolume(currentTrack.pos, currentTrack.dir, theTrack->GetGStepLength(),
                                                   currentTrack.navState, currentTrack.nextState, kPushDistance);
#endif
      currentTrack.pos += currentTrack.geometryStepLength * currentTrack.dir;
    }

    if (currentTrack.geometryStepLength < kPushStuck && currentTrack.geometryStepLength < theTrack->GetGStepLength()) {
      currentTrack.zeroStepCounter++;
      if (currentTrack.zeroStepCounter > kStepsStuckPush) currentTrack.pos += kPushStuck * currentTrack.dir;
    } else
      currentTrack.zeroStepCounter = 0;

    // punish miniscule steps by increasing the looperCounter by 10
    if (currentTrack.geometryStepLength < 100 * vecgeom::kTolerance) currentTrack.looperCounter += 10;

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    currentTrack.navState.SetBoundaryState(currentTrack.nextState.IsOnBoundary());
    if (currentTrack.nextState.IsOnBoundary()) currentTrack.SetSafety(currentTrack.pos, 0.);

    // Propagate information from geometrical step to MSC.
    theTrack->SetDirection(currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z());
    theTrack->SetGStepLength(currentTrack.geometryStepLength);
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
__global__ void ElectronSetupInteractions(Track *electrons, Track *leaks, G4HepEmElectronTrack *hepEMTracks,
                                          const adept::MParray *active, Secondaries secondaries,
                                          adept::MParray *nextActiveQueue, AllInteractionQueues interactionQueues,
                                          adept::MParray *leakedQueue, Scoring *userScoring, const bool returnAllSteps,
                                          const bool returnLastStep)
{
  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot           = (*active)[i];
    SlotManager &slotManager = IsElectron ? *secondaries.electrons.fSlotManager : *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    auto survive = [&](LeakStatus leakReason = LeakStatus::NoLeak) {
      // NOTE: When adapting the split kernels for async mode this won't
      // work if we want to re-use slots on the fly. Directly copying to
      // a trackdata struct would be better
      currentTrack.leakStatus = leakReason;
      if (leakReason != LeakStatus::NoLeak) {
        // Get a slot in the leaks array
        int leakSlot;
        if (IsElectron)
          leakSlot = secondaries.electrons.NextLeakSlot();
        else
          leakSlot = secondaries.positrons.NextLeakSlot();
        // Copy the track to the leaks array and store the index in the leak queue
        leaks[leakSlot] = electrons[slot];
        auto success    = leakedQueue->push_back(leakSlot);
        if (!success) {
          printf("ERROR: No space left in e-/+ leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
        // Free the slot in the tracks slot manager
        slotManager.MarkSlotForFreeing(slot);
      } else {
        nextActiveQueue->push_back(slot);
      }
    };

    bool isLastStep = false;

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();

    double energyDeposit = theTrack->GetEnergyDeposit();

    bool reached_interaction = true;
    bool printErrors         = false;

    // Save the `number-of-interaction-left` in our track.
    for (int ip = 0; ip < 4; ++ip) {
      double numIALeft = theTrack->GetNumIALeft(ip);
    }

    // Set Non-stopped, on-boundary tracks for relocation
    if (currentTrack.nextState.IsOnBoundary() && !currentTrack.stopped) {
      // Add particle to relocation queue
      interactionQueues.queues[4]->push_back(slot);
      continue;
    }

    // Now check whether the non-relocating tracks reached an interaction
    if (!currentTrack.stopped) {
      if (!currentTrack.propagated || currentTrack.restrictedPhysicalStepLength) {
        // Did not yet reach the interaction point due to error in the magnetic
        // field propagation. Try again next time.

        if (++currentTrack.looperCounter > 500) {
          // Kill loopers that are not advancing in free space
          if (printErrors)
            printf("Killing looper due to lack of advance: E=%E event=%d loop=%d energyDeposit=%E geoStepLength=%E "
                   "physicsStepLength=%E "
                   "safety=%E\n",
                   currentTrack.eKin, currentTrack.eventId, currentTrack.looperCounter, energyDeposit,
                   currentTrack.geometryStepLength, theTrack->GetGStepLength(), currentTrack.safety);
          continue;
        }

        // mark winner process to be transport, although this is not strictly true
        theTrack->SetWinnerProcessIndex(10);

        survive();
        reached_interaction = false;
      } else if (theTrack->GetWinnerProcessIndex() < 0) {
        // No discrete process, move on.
        survive();
        reached_interaction = false;
      } else if (G4HepEmElectronManager::CheckDelta(&g4HepEmData, theTrack, currentTrack.Uniform())) {
        // If there was a delta interaction, the track survives but does not move onto the next kernel
        survive();
        reached_interaction = false;
        // Reset number of interaction left for the winner discrete process.
        // (Will be resampled in the next iteration.)
        theTrack->SetNumIALeft(-1.0, theTrack->GetWinnerProcessIndex());
      }
    } else {
      // Stopped positrons annihilate, stopped electrons score and die
      if (IsElectron) {
        reached_interaction = false;
        isLastStep          = true;
        // Ekin = 0 for correct scoring
        currentTrack.eKin = 0;
        // Particle is killed by not enqueuing it for the next iteration. Free the slot it occupies
        slotManager.MarkSlotForFreeing(slot);
      }
    }

    // Now push the particles that reached their interaction into the per-interaction queues
    if (reached_interaction) {
      // reset Looper counter if limited by discrete interaction or MSC
      currentTrack.looperCounter = 0;

      if (!currentTrack.stopped) {
        // Reset number of interaction left for the winner discrete process.
        // (Will be resampled in the next iteration.)
        theTrack->SetNumIALeft(-1.0, theTrack->GetWinnerProcessIndex());
        // Enqueue the particles
        if (theTrack->GetWinnerProcessIndex() < 3) {
          interactionQueues.queues[theTrack->GetWinnerProcessIndex()]->push_back(slot);
        } else {
          // Lepton nuclear needs to be handled by Geant4 directly, passing track back to CPU
          survive(LeakStatus::LeptonNuclear);
        }
      } else {
        // Stopped positron
        interactionQueues.queues[3]->push_back(slot);
      }

    } else {
      // Only non-interacting, non-relocating tracks score here
      // Note: In this kernel returnLastStep is only true for particles that left the world
      // Score the edep for particles that didn't reach the interaction
      if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep)
        adept_scoring::RecordHit(userScoring,
                                 currentTrack.trackId,                                  // Track ID
                                 currentTrack.parentId,                                 // parent Track ID
                                 static_cast<short>(theTrack->GetWinnerProcessIndex()), // step defining process
                                 static_cast<char>(IsElectron ? 0 : 1),                 // Particle type
                                 elTrack.GetPStepLength(),                              // Step length
                                 energyDeposit,                                         // Total Edep
                                 currentTrack.weight,                                   // Track weight
                                 currentTrack.navState,                                 // Pre-step point navstate
                                 currentTrack.preStepPos,                               // Pre-step point position
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
                                 isLastStep,                                  // whether this was the last step
                                 currentTrack.stepCounter);                   // stepcounter
    }
  }
}

template <bool IsElectron, typename Scoring>
__global__ void ElectronRelocation(Track *electrons, Track *leaks, G4HepEmElectronTrack *hepEMTracks,
                                   Secondaries secondaries, adept::MParray *nextActiveQueue,
                                   adept::MParray *relocatingQueue, adept::MParray *leakedQueue, Scoring *userScoring,
                                   const bool returnAllSteps, const bool returnLastStep)
{
  constexpr double kPushDistance = 1000 * vecgeom::kTolerance;
  int activeSize                 = relocatingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot           = (*relocatingQueue)[i];
    SlotManager &slotManager = IsElectron ? *secondaries.electrons.fSlotManager : *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    auto survive = [&](LeakStatus leakReason = LeakStatus::NoLeak) {
      // NOTE: When adapting the split kernels for async mode this won't
      // work if we want to re-use slots on the fly. Directly copying to
      // a trackdata struct would be better
      currentTrack.leakStatus = leakReason;
      if (leakReason != LeakStatus::NoLeak) {
        // Get a slot in the leaks array
        int leakSlot;
        if (IsElectron)
          leakSlot = secondaries.electrons.NextLeakSlot();
        else
          leakSlot = secondaries.positrons.NextLeakSlot();
        // Copy the track to the leaks array and store the index in the leak queue
        leaks[leakSlot] = electrons[slot];
        auto success    = leakedQueue->push_back(leakSlot);
        if (!success) {
          printf("ERROR: No space left in e-/+ leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
        // Free the slot in the tracks slot manager
        slotManager.MarkSlotForFreeing(slot);
      } else {
        nextActiveQueue->push_back(slot);
      }
    };

    bool isLastStep = false;

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    double energyDeposit = theTrack->GetEnergyDeposit();

    bool cross_boundary = false;
    bool printErrors    = false;

    // Relocate to have the correct next state before RecordHit is called

    // - Kill loopers stuck at a boundary
    // - Set cross boundary flag in order to set the correct navstate after scoring
    // - Kill particles that left the world

    if (++currentTrack.looperCounter > 500) {
      // Kill loopers that are scraping a boundary
      if (printErrors)
        printf("Killing looper scraping at a boundary: E=%E event=%d loop=%d energyDeposit=%E geoStepLength=%E "
               "physicsStepLength=%E "
               "safety=%E\n",
               currentTrack.eKin, currentTrack.eventId, currentTrack.looperCounter, energyDeposit,
               currentTrack.geometryStepLength, theTrack->GetGStepLength(), currentTrack.safety);
      continue;
    }

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
      isLastStep = true;
    }

    // Score
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep)
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
                               isLastStep,                                  // whether this was the last step
                               currentTrack.stepCounter);                   // stepcounter

    if (cross_boundary) {
      // Move to the next boundary now that the Step is recorded
      currentTrack.navState = currentTrack.nextState;
      // Check if the next volume belongs to the GPU region and push it to the appropriate queue
      const int nextlvolID          = currentTrack.navState.GetLogicalId();
      VolAuxData const &nextauxData = AsyncAdePT::gVolAuxData[nextlvolID];
      if (nextauxData.fGPUregion > 0) {
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

template <bool IsElectron, typename Scoring>
__device__ __forceinline__ void PerformStoppedAnnihilation(const int slot, Track &currentTrack,
                                                           Secondaries &secondaries, double &energyDeposit,
                                                           SlotManager &slotManager, const bool ApplyCuts,
                                                           const double theGammaCut, Scoring *userScoring,
                                                           const bool returnLastStep = false)
{
  currentTrack.eKin = 0;
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

      // as the other branched newRNG may have already been used by interactions before, we need to advance and create a
      // new one
      currentTrack.rngState.Advance();
      RanluxppDouble newRNG(currentTrack.rngState.Branch());

      Track &gamma1 = secondaries.gammas.NextTrack(newRNG, double{copcore::units::kElectronMassC2}, currentTrack.pos,
                                                   vecgeom::Vector3D<double>{sint * cosPhi, sint * sinPhi, cost},
                                                   currentTrack.navState, currentTrack, currentTrack.globalTime);

      // Reuse the RNG state of the dying track.
      Track &gamma2 =
          secondaries.gammas.NextTrack(currentTrack.rngState, double{copcore::units::kElectronMassC2}, currentTrack.pos,
                                       -gamma1.dir, currentTrack.navState, currentTrack, currentTrack.globalTime);

      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        adept_scoring::RecordHit(userScoring, gamma1.trackId, gamma1.parentId, /*CreatorProcessId*/ short(2),
                                 /* gamma*/ 2,                    // Particle type
                                 0,                               // Step length
                                 0,                               // Total Edep
                                 gamma1.weight,                   // Track weight
                                 gamma1.navState,                 // Pre-step point navstate
                                 gamma1.pos,                      // Pre-step point position
                                 gamma1.dir,                      // Pre-step point momentum direction
                                 gamma1.eKin,                     // Pre-step point kinetic energy
                                 gamma1.navState,                 // Post-step point navstate
                                 gamma1.pos,                      // Post-step point position
                                 gamma1.dir,                      // Post-step point momentum direction
                                 gamma1.eKin,                     // Post-step point kinetic energy
                                 gamma1.globalTime,               // global time
                                 0.,                              // local time
                                 gamma1.globalTime,               // preStep global time for initializing step
                                 gamma1.eventId, gamma1.threadId, // eventID and threadID
                                 false,                           // whether this was the last step
                                 gamma1.stepCounter);             // whether this was the first step
        adept_scoring::RecordHit(userScoring, gamma2.trackId, gamma2.parentId, /*CreatorProcessId*/ short(2),
                                 /* gamma*/ 2,                    // Particle type
                                 0,                               // Step length
                                 0,                               // Total Edep
                                 gamma2.weight,                   // Track weight
                                 gamma2.navState,                 // Pre-step point navstate
                                 gamma2.pos,                      // Pre-step point position
                                 gamma2.dir,                      // Pre-step point momentum direction
                                 gamma2.eKin,                     // Pre-step point kinetic energy
                                 gamma2.navState,                 // Post-step point navstate
                                 gamma2.pos,                      // Post-step point position
                                 gamma2.dir,                      // Post-step point momentum direction
                                 gamma2.eKin,                     // Post-step point kinetic energy
                                 gamma2.globalTime,               // global time
                                 0.,                              // local time
                                 gamma2.globalTime,               // preStep global time for initializing step
                                 gamma2.eventId, gamma2.threadId, // eventID and threadID
                                 false,                           // whether this was the last step
                                 gamma2.stepCounter);             // whether this was the first step
      }
    }
  }
  // Particles are killed by not enqueuing them into the new activeQueue (and free the slot in async mode)
  slotManager.MarkSlotForFreeing(slot);
}

template <bool IsElectron, typename Scoring>
__global__ void ElectronIonization(Track *electrons, G4HepEmElectronTrack *hepEMTracks, Secondaries secondaries,
                                   adept::MParray *nextActiveQueue, adept::MParray *interactingQueue,
                                   Scoring *userScoring, const bool returnAllSteps, const bool returnLastStep)
{
  int activeSize = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot           = (*active)[i];
    const int slot           = (*interactingQueue)[i];
    SlotManager &slotManager = IsElectron ? *secondaries.electrons.fSlotManager : *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData
    bool isLastStep           = true;

    auto survive = [&]() {
      isLastStep = false; // track survived, do not force return of step
      nextActiveQueue->push_back(slot);
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

    // Apply cuts
    if (ApplyCuts && (deltaEkin < theElCut)) {
      // Deposit the energy here and kill the secondary
      energyDeposit += deltaEkin;

    } else {
      Track &secondary =
          secondaries.electrons.NextTrack(newRNG, deltaEkin, currentTrack.pos,
                                          vecgeom::Vector3D<double>{dirSecondary[0], dirSecondary[1], dirSecondary[2]},
                                          currentTrack.navState, currentTrack, currentTrack.globalTime);

      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        adept_scoring::RecordHit(userScoring, secondary.trackId, secondary.parentId,
                                 /*CreatorProcessId*/ short(0),
                                 /* electron*/ 0,                       // Particle type
                                 0,                                     // Step length
                                 0,                                     // Total Edep
                                 secondary.weight,                      // Track weight
                                 secondary.navState,                    // Pre-step point navstate
                                 secondary.pos,                         // Pre-step point position
                                 secondary.dir,                         // Pre-step point momentum direction
                                 secondary.eKin,                        // Pre-step point kinetic energy
                                 secondary.navState,                    // Post-step point navstate
                                 secondary.pos,                         // Post-step point position
                                 secondary.dir,                         // Post-step point momentum direction
                                 secondary.eKin,                        // Post-step point kinetic energy
                                 secondary.globalTime,                  // global time
                                 0.,                                    // local time
                                 secondary.globalTime,                  // preStep global time for initializing step
                                 secondary.eventId, secondary.threadId, // eventID and threadID
                                 false,                                 // whether this was the last step
                                 secondary.stepCounter);                // whether this was the first step
      }
    }

    currentTrack.eKin -= deltaEkin;
    theTrack->SetEKin(currentTrack.eKin);

    // if below tracking cut, deposit energy for electrons (positrons are annihilated later) and stop particles
    if (currentTrack.eKin < g4HepEmPars.fElectronTrackingCut) {
      if (IsElectron) {
        energyDeposit += currentTrack.eKin;
      }
      currentTrack.stopped = true;
      PerformStoppedAnnihilation<IsElectron, Scoring>(slot, currentTrack, secondaries, energyDeposit, slotManager,
                                                      ApplyCuts, theGammaCut, userScoring, returnLastStep);
    } else {
      currentTrack.dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
      survive();
    }

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep)
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
                               isLastStep,                                  // whether this was the last step
                               currentTrack.stepCounter);                   // stepcounter
  }
}

template <bool IsElectron, typename Scoring>
__global__ void ElectronBremsstrahlung(Track *electrons, G4HepEmElectronTrack *hepEMTracks, Secondaries secondaries,
                                       adept::MParray *nextActiveQueue, adept::MParray *interactingQueue,
                                       Scoring *userScoring, const bool returnAllSteps, const bool returnLastStep)
{
  int activeSize = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot           = (*active)[i];
    const int slot           = (*interactingQueue)[i];
    SlotManager &slotManager = IsElectron ? *secondaries.electrons.fSlotManager : *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData
    bool isLastStep           = true;

    auto survive = [&]() {
      isLastStep = false; // track survived, do not force return of step
      nextActiveQueue->push_back(slot);
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

    // Apply cuts
    if (ApplyCuts && (deltaEkin < theGammaCut)) {
      // Deposit the energy here and kill the secondary
      energyDeposit += deltaEkin;

    } else {
      Track &gamma =
          secondaries.gammas.NextTrack(newRNG, deltaEkin, currentTrack.pos,
                                       vecgeom::Vector3D<double>{dirSecondary[0], dirSecondary[1], dirSecondary[2]},
                                       currentTrack.navState, currentTrack, currentTrack.globalTime);
      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        adept_scoring::RecordHit(userScoring, gamma.trackId, gamma.parentId, /*CreatorProcessId*/ short(1),
                                 /* gamma*/ 2,                  // Particle type
                                 0,                             // Step length
                                 0,                             // Total Edep
                                 gamma.weight,                  // Track weight
                                 gamma.navState,                // Pre-step point navstate
                                 gamma.pos,                     // Pre-step point position
                                 gamma.dir,                     // Pre-step point momentum direction
                                 gamma.eKin,                    // Pre-step point kinetic energy
                                 gamma.navState,                // Post-step point navstate
                                 gamma.pos,                     // Post-step point position
                                 gamma.dir,                     // Post-step point momentum direction
                                 gamma.eKin,                    // Post-step point kinetic energy
                                 gamma.globalTime,              // global time
                                 0.,                            // local time
                                 gamma.globalTime,              // preStep global time at initializing step
                                 gamma.eventId, gamma.threadId, // eventID and threadID
                                 false,                         // whether this was the last step
                                 gamma.stepCounter);            // whether this was the first step
      }
    }

    currentTrack.eKin -= deltaEkin;
    theTrack->SetEKin(currentTrack.eKin);

    // if below tracking cut, deposit energy for electrons (positrons are annihilated later) and stop particles
    if (currentTrack.eKin < g4HepEmPars.fElectronTrackingCut) {
      if (IsElectron) {
        energyDeposit += currentTrack.eKin;
      }
      currentTrack.stopped = true;
      PerformStoppedAnnihilation<IsElectron, Scoring>(slot, currentTrack, secondaries, energyDeposit, slotManager,
                                                      ApplyCuts, theGammaCut, userScoring, returnLastStep);
    } else {
      currentTrack.dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
      survive();
    }

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep)
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
                               isLastStep,                                  // whether this was the last step
                               currentTrack.stepCounter);                   // stepcounter
  }
}

template <typename Scoring>
__global__ void PositronAnnihilation(Track *electrons, G4HepEmElectronTrack *hepEMTracks, Secondaries secondaries,
                                     adept::MParray *nextActiveQueue, adept::MParray *interactingQueue,
                                     Scoring *userScoring, const bool returnAllSteps, const bool returnLastStep)
{
  int activeSize = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot           = (*active)[i];
    const int slot           = (*interactingQueue)[i];
    SlotManager &slotManager = *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData
    bool isLastStep           = true;

    auto survive = [&]() {
      isLastStep = false; // track survived, do not force return of step
      nextActiveQueue->push_back(slot);
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

    // Invoke annihilation (in-flight) for e+
    double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
    double theGamma1Ekin, theGamma2Ekin;
    double theGamma1Dir[3], theGamma2Dir[3];
    G4HepEmPositronInteractionAnnihilation::SampleEnergyAndDirectionsInFlight(
        currentTrack.eKin, dirPrimary, &theGamma1Ekin, theGamma1Dir, &theGamma2Ekin, theGamma2Dir, &rnge);

    // TODO: In principle particles are produced, then cut before stacking them. It seems correct to count them
    // here
    adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 0, /*numPositrons*/ 0, /*numGammas*/ 2);

    // Apply cuts
    if (ApplyCuts && (theGamma1Ekin < theGammaCut)) {
      // Deposit the energy here and kill the secondaries
      energyDeposit += theGamma1Ekin;

    } else {
      Track &gamma1 =
          secondaries.gammas.NextTrack(newRNG, theGamma1Ekin, currentTrack.pos,
                                       vecgeom::Vector3D<double>{theGamma1Dir[0], theGamma1Dir[1], theGamma1Dir[2]},
                                       currentTrack.navState, currentTrack, currentTrack.globalTime);
      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        adept_scoring::RecordHit(userScoring, gamma1.trackId, gamma1.parentId, /*CreatorProcessId*/ short(2),
                                 /* gamma*/ 2,                    // Particle type
                                 0,                               // Step length
                                 0,                               // Total Edep
                                 gamma1.weight,                   // Track weight
                                 gamma1.navState,                 // Pre-step point navstate
                                 gamma1.pos,                      // Pre-step point position
                                 gamma1.dir,                      // Pre-step point momentum direction
                                 gamma1.eKin,                     // Pre-step point kinetic energy
                                 gamma1.navState,                 // Post-step point navstate
                                 gamma1.pos,                      // Post-step point position
                                 gamma1.dir,                      // Post-step point momentum direction
                                 gamma1.eKin,                     // Post-step point kinetic energy
                                 gamma1.globalTime,               // global time
                                 0.,                              // local time
                                 gamma1.globalTime,               // preStep global time at initializing step
                                 gamma1.eventId, gamma1.threadId, // eventID and threadID
                                 false,                           // whether this was the last step
                                 gamma1.stepCounter);
      }
    }
    if (ApplyCuts && (theGamma2Ekin < theGammaCut)) {
      // Deposit the energy here and kill the secondaries
      energyDeposit += theGamma2Ekin;

    } else {
      Track &gamma2 =
          secondaries.gammas.NextTrack(currentTrack.rngState, theGamma2Ekin, currentTrack.pos,
                                       vecgeom::Vector3D<double>{theGamma2Dir[0], theGamma2Dir[1], theGamma2Dir[2]},
                                       currentTrack.navState, currentTrack, currentTrack.globalTime);
      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        adept_scoring::RecordHit(userScoring, gamma2.trackId, gamma2.parentId, /*CreatorProcessId*/ short(2),
                                 /* gamma*/ 2,                    // Particle type
                                 0,                               // Step length
                                 0,                               // Total Edep
                                 gamma2.weight,                   // Track weight
                                 gamma2.navState,                 // Pre-step point navstate
                                 gamma2.pos,                      // Pre-step point position
                                 gamma2.dir,                      // Pre-step point momentum direction
                                 gamma2.eKin,                     // Pre-step point kinetic energy
                                 gamma2.navState,                 // Post-step point navstate
                                 gamma2.pos,                      // Post-step point position
                                 gamma2.dir,                      // Post-step point momentum direction
                                 gamma2.eKin,                     // Post-step point kinetic energy
                                 gamma2.globalTime,               // global time
                                 0.,                              // local time
                                 gamma2.globalTime,               // preStep global time at initializing step
                                 gamma2.eventId, gamma2.threadId, // eventID and threadID
                                 false,                           // whether this was the last step
                                 gamma2.stepCounter);
      }
    }

    // The current track is killed by not enqueuing into the next activeQueue.
    slotManager.MarkSlotForFreeing(slot);

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep)
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
                               isLastStep,                                  // whether this was the last step
                               currentTrack.stepCounter);                   // stepcounter
  }
}

template <typename Scoring>
__global__ void PositronStoppedAnnihilation(Track *electrons, G4HepEmElectronTrack *hepEMTracks,
                                            Secondaries secondaries, adept::MParray *nextActiveQueue,
                                            adept::MParray *interactingQueue, Scoring *userScoring,
                                            const bool returnAllSteps, const bool returnLastStep)
{
  int activeSize = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot           = (*active)[i];
    const int slot           = (*interactingQueue)[i];
    SlotManager &slotManager = *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
    const int lvolID = currentTrack.navState.GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData
    bool isLastStep           = true;

    auto survive = [&]() {
      isLastStep = false; // track survived, do not force return of step
      nextActiveQueue->push_back(slot);
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

    // Annihilate the stopped positron into two gammas heading to opposite
    // directions (isotropic).

    PerformStoppedAnnihilation<false, Scoring>(slot, currentTrack, secondaries, energyDeposit, slotManager, ApplyCuts,
                                               theGammaCut, userScoring, returnLastStep);

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep)
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
                               isLastStep,                                  // whether this was the last step
                               currentTrack.stepCounter);                   // stepcounter
  }
}

} // namespace AsyncAdePT
