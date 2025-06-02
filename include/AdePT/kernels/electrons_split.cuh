// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/navigation/AdePTNavigator.h>

// Classes for Runge-Kutta integration
#include <AdePT/magneticfield/MagneticFieldEquation.h>
#include <AdePT/magneticfield/DormandPrinceRK45.h>
#include <AdePT/magneticfield/fieldPropagatorRungeKutta.h>

#include <AdePT/copcore/PhysicalConstants.h>

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
                               adept::MParray *nextActiveQueue, adept::MParray *leakedQueue, Stats *InFlightStats,
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
  using Stepper_t  = DormandPrinceRK45<Equation_t, Field_t, Nvar, vecgeom::Precision>;
  using RkDriver_t = RkIntegrationDriver<Stepper_t, vecgeom::Precision, int, Equation_t, Field_t>;

  auto &magneticField = *gMagneticField;

  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    const int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    const int lvolID = currentTrack.navState.GetLogicalId();
#endif

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    currentTrack.preStepEKin = currentTrack.eKin;
    currentTrack.preStepPos  = currentTrack.pos;
    currentTrack.preStepDir  = currentTrack.dir;
    currentTrack.stepCounter++;
    bool printErrors = true;
    if (printErrors && (currentTrack.stepCounter >= maxSteps || currentTrack.zeroStepCounter > kStepsStuckKill)) {
      printf("Killing e-/+ event %d track %ld E=%f lvol=%d after %d steps with zeroStepCounter %u\n",
             currentTrack.eventId, currentTrack.id, currentTrack.eKin, lvolID, currentTrack.stepCounter,
             currentTrack.zeroStepCounter);
      continue;
    }

    auto survive = [&](bool leak = false) {
      // NOTE: When adapting the split kernels for async mode this won't
      // work if we want to re-use slots on the fly. Directly copying to
      // a trackdata struct would be better
      if (leak) {
        auto success = leakedQueue->push_back(slot);
        if (!success) {
          printf("ERROR: No space left in e-/+ leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
      } else
        nextActiveQueue->push_back(slot);
    };

    if (InFlightStats->perEventInFlightPrevious[currentTrack.threadId] < allowFinishOffEvent[currentTrack.threadId] &&
        InFlightStats->perEventInFlightPrevious[currentTrack.threadId] != 0) {
      printf("Thread %d Finishing e-/e+ of the %d last particles of event %d on CPU E=%f lvol=%d after %d steps.\n",
             currentTrack.threadId, InFlightStats->perEventInFlightPrevious[currentTrack.threadId],
             currentTrack.eventId, currentTrack.eKin, lvolID, currentTrack.stepCounter);
      survive(/*leak*/ true);
      continue;
    }

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    elTrack.ReSet();
    G4HepEmTrack *theTrack = elTrack.GetTrack();
    theTrack->SetEKin(currentTrack.eKin);
    theTrack->SetMCIndex(auxData.fMCIndex);
    theTrack->SetOnBoundary(currentTrack.navState.IsOnBoundary());
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
    currentTrack.newRNG = RanluxppDouble(currentTrack.rngState.BranchNoAdvance());
    G4HepEmRandomEngine rnge(&currentTrack.rngState);

    // Sample the `number-of-interaction-left` and put it into the track.
    for (int ip = 0; ip < 4; ++ip) {
      double numIALeft = currentTrack.numIALeft[ip];
      if (numIALeft <= 0) {
        numIALeft = -std::log(currentTrack.Uniform());
      }
      if (ip == 3) numIALeft = vecgeom::kInfLength; // suppress lepton nuclear by infinite length
      theTrack->SetNumIALeft(numIALeft, ip);
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
#ifdef ADEPT_USE_SURF
        // Use maximum accuracy only if safety is samller than physicalStepLength
        safety = AdePTNavigator::ComputeSafety(currentTrack.pos, currentTrack.navState, physicalStepLength);
#else
        safety = AdePTNavigator::ComputeSafety(currentTrack.pos, currentTrack.navState);
#endif
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
      vecgeom::Vector3D<double> momentumVec = momentumMag * currentTrack.dir;
      vecgeom::Vector3D<double> B0fieldVec  = magneticField.Evaluate(
          currentTrack.pos[0], currentTrack.pos[1], currentTrack.pos[2]); // Field value at starting point
      currentTrack.safeLength =
          fieldPropagatorRungeKutta<Field_t, RkDriver_t, Precision, AdePTNavigator>::ComputeSafeLength /*<Real_t>*/ (
              momentumVec, B0fieldVec, Charge);

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

    // Remember MSC values for the next step(s).
    currentTrack.initialRange       = mscData->fInitialRange;
    currentTrack.dynamicRangeFactor = mscData->fDynamicRangeFactor;
    currentTrack.tlimitMin          = mscData->fTlimitMin;
  }
}

template <bool IsElectron>
__global__ void ElectronPropagation(Track *electrons, G4HepEmElectronTrack *hepEMTracks, const adept::MParray *active,
                                    adept::MParray *leakedQueue)
{
  constexpr Precision kPushDistance        = 1000 * vecgeom::kTolerance;
  constexpr int Charge                     = IsElectron ? -1 : 1;
  constexpr double restMass                = copcore::units::kElectronMassC2;
  constexpr int Nvar                       = 6;
  constexpr int max_iterations             = 10;
  constexpr Precision kPushStuck           = 100 * vecgeom::kTolerance;
  constexpr unsigned short kStepsStuckPush = 5;

#ifdef ADEPT_USE_EXT_BFIELD
  using Field_t = GeneralMagneticField;
#else
  using Field_t = UniformMagneticField;
#endif
  using Equation_t = MagneticFieldEquation<Field_t>;
  using Stepper_t  = DormandPrinceRK45<Equation_t, Field_t, Nvar, vecgeom::Precision>;
  using RkDriver_t = RkIntegrationDriver<Stepper_t, vecgeom::Precision, int, Equation_t, Field_t>;

  auto &magneticField = *gMagneticField;

  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot = (*active)[i];

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    const int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    const int lvolID = currentTrack.navState.GetLogicalId();
#endif

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    G4HepEmRandomEngine rnge(&currentTrack.rngState);

    // Check if there's a volume boundary in between.
    currentTrack.propagated = true;
    currentTrack.hitsurfID  = -1;

    if (gMagneticField) {
      int iterDone = -1;
      currentTrack.geometryStepLength =
          fieldPropagatorRungeKutta<Field_t, RkDriver_t, Precision, AdePTNavigator>::ComputeStepAndNextVolume(
              magneticField, currentTrack.eKin, restMass, Charge, theTrack->GetGStepLength(), currentTrack.safeLength,
              currentTrack.pos, currentTrack.dir, currentTrack.navState, currentTrack.nextState, currentTrack.hitsurfID,
              currentTrack.propagated,
              /*lengthDone,*/ currentTrack.safety,
              // activeSize < 100 ? max_iterations : max_iters_tail ), // Was
              max_iterations, iterDone, slot);
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
#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    const int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    const int lvolID = currentTrack.navState.GetLogicalId();
#endif

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
      vecgeom::Vector3D<Precision> displacement(mscDisplacement[0], mscDisplacement[1], mscDisplacement[2]);
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
#ifdef ADEPT_USE_SURF
          // Use maximum accuracy only if safety is samller than physicalStepLength
          safety = AdePTNavigator::ComputeSafety(currentTrack.pos, currentTrack.navState, dispR);
#else
          safety = AdePTNavigator::ComputeSafety(currentTrack.pos, currentTrack.navState);
#endif
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

    // Update the flight times of the particle
    // By calculating the velocity here, we assume that all the energy deposit is done at the PreStepPoint, and
    // the velocity depends on the remaining energy
    double deltaTime = elTrack.GetPStepLength() / GetVelocity(currentTrack.eKin);
    currentTrack.globalTime += deltaTime;
    currentTrack.localTime += deltaTime;
    currentTrack.properTime += deltaTime * (restMass / currentTrack.eKin);
  }
}

template <bool IsElectron, typename Scoring>
__global__ void ElectronRelocation(Track *electrons, G4HepEmElectronTrack *hepEMTracks, const adept::MParray *active,
                                   Secondaries secondaries, adept::MParray *nextActiveQueue,
                                   adept::MParray *reachedInteractionQueue, AllInteractionQueues interactionQueues,
                                   adept::MParray *leakedQueue, Scoring *userScoring, bool returnAllSteps,
                                   bool returnLastStep)
{
  constexpr Precision kPushDistance = 1000 * vecgeom::kTolerance;
  int activeSize                    = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot           = (*active)[i];
    SlotManager &slotManager = IsElectron ? *secondaries.electrons.fSlotManager : *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    const int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    const int lvolID = currentTrack.navState.GetLogicalId();
#endif

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    auto survive = [&](bool leak = false) {
      returnLastStep = false; // track survived, do not force return of step
      // NOTE: When adapting the split kernels for async mode this won't
      // work if we want to re-use slots on the fly. Directly copying to
      // a trackdata struct would be better
      if (leak) {
        auto success = leakedQueue->push_back(slot);
        if (!success) {
          printf("ERROR: No space left in e-/+ leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
      } else
        nextActiveQueue->push_back(slot);
    };

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    G4HepEmRandomEngine rnge(&currentTrack.rngState);

    double energyDeposit = theTrack->GetEnergyDeposit();

    if (currentTrack.nextState.IsOnBoundary()) {
      // if the particle hit a boundary, and is neither stopped or outside, relocate to have the correct next state
      // before RecordHit is called
      if (!currentTrack.stopped && !currentTrack.nextState.IsOutside()) {
#ifdef ADEPT_USE_SURF
        AdePTNavigator::RelocateToNextVolume(currentTrack.pos, currentTrack.dir, currentTrack.hitsurfID,
                                             currentTrack.nextState);
#else
        AdePTNavigator::RelocateToNextVolume(currentTrack.pos, currentTrack.dir, currentTrack.nextState);
#endif
      }
    }

    // Save the `number-of-interaction-left` in our track.
    for (int ip = 0; ip < 4; ++ip) {
      double numIALeft           = theTrack->GetNumIALeft(ip);
      currentTrack.numIALeft[ip] = numIALeft;
    }

    bool reached_interaction = true;
    bool cross_boundary      = false;

    bool printErrors = true;

    if (!currentTrack.stopped) {
      if (currentTrack.nextState.IsOnBoundary()) {
        // For now, just count that we hit something.
        reached_interaction = false;
        // Kill the particle if it left the world.

        if (++currentTrack.looperCounter > 500) {
          // Kill loopers that are scraping a boundary
          if (printErrors)
            printf("Killing looper scraping at a boundary: E=%E event=%d loop=%d energyDeposit=%E geoStepLength=%E "
                   "physicsStepLength=%E "
                   "safety=%E\n",
                   currentTrack.eKin, currentTrack.eventId, currentTrack.looperCounter, energyDeposit,
                   currentTrack.geometryStepLength, theTrack->GetGStepLength(), currentTrack.safety);
          continue;
        } else if (!currentTrack.nextState.IsOutside()) {
          // Mark the particle. We need to change its navigation state to the next volume before enqueuing it
          // This will happen after recording the step
          cross_boundary = true;
          returnLastStep = false; // the track survives, do not force return of step
        } else {
          // Particle left the world, don't enqueue it and release the slot
          slotManager.MarkSlotForFreeing(slot);
        }

      } else if (!currentTrack.propagated || currentTrack.restrictedPhysicalStepLength) {
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

        survive();
        reached_interaction = false;
        // Note, check for process 3 (leptonuclear) should not be hit, here just for safety
      } else if (theTrack->GetWinnerProcessIndex() < 0 || theTrack->GetWinnerProcessIndex() == 3) {
        // No discrete process, move on.
        survive();
        reached_interaction = false;
      } else if (G4HepEmElectronManager::CheckDelta(&g4HepEmData, theTrack, currentTrack.Uniform())) {
        // If there was a delta interaction, the track survives but does not move onto the next kernel
        survive();
        reached_interaction = false;
        // Reset number of interaction left for the winner discrete process.
        // (Will be resampled in the next iteration.)
        currentTrack.numIALeft[theTrack->GetWinnerProcessIndex()] = -1.0;
      }
    } else {
      // Stopped positrons annihilate, stopped electrons score and die
      if (IsElectron) {
        reached_interaction = false;
        // Ekin = 0 for correct scoring
        currentTrack.eKin = 0;
        // Particle is killed by not enqueuing it for the next iteration. Free the slot it occupies
        slotManager.MarkSlotForFreeing(slot);
      }
    }

    // Now push the particles that reached their interaction into the per-interaction queues
    if (reached_interaction) {
      // Add to the queue of particles that will be processed by the next kernel
      reachedInteractionQueue->push_back(slot);
      // reset Looper counter if limited by discrete interaction or MSC
      currentTrack.looperCounter = 0;

      if (!currentTrack.stopped) {
        // Reset number of interaction left for the winner discrete process.
        // (Will be resampled in the next iteration.)
        currentTrack.numIALeft[theTrack->GetWinnerProcessIndex()] = -1.0;
        // Enqueue the particles
        if (theTrack->GetWinnerProcessIndex() < 3) {
          interactionQueues.queues[theTrack->GetWinnerProcessIndex()]->push_back(slot);
        } else {
          // Lepton nuclear, don't do anything for now
          ;
        }
      } else {
        // Stopped positron
        interactionQueues.queues[3]->push_back(slot);
      }

    } else {
      // Note: In this kernel returnLastStep is only true for particles that left the world
      // Score the edep for particles that didn't reach the interaction
      if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep)
        adept_scoring::RecordHit(userScoring, currentTrack.parentId,
                                 static_cast<char>(IsElectron ? 0 : 1),         // Particle type
                                 elTrack.GetPStepLength(),                      // Step length
                                 energyDeposit,                                 // Total Edep
                                 currentTrack.weight,                           // Track weight
                                 currentTrack.navState,                         // Pre-step point navstate
                                 currentTrack.preStepPos,                       // Pre-step point position
                                 currentTrack.preStepDir,                       // Pre-step point momentum direction
                                 currentTrack.preStepEKin,                      // Pre-step point kinetic energy
                                 IsElectron ? -1 : 1,                           // Pre-step point charge
                                 currentTrack.nextState,                        // Post-step point navstate
                                 currentTrack.pos,                              // Post-step point position
                                 currentTrack.dir,                              // Post-step point momentum direction
                                 currentTrack.eKin,                             // Post-step point kinetic energy
                                 IsElectron ? -1 : 1,                           // Post-step point charge
                                 currentTrack.globalTime,                       // global time
                                 currentTrack.eventId, currentTrack.threadId,   // eventID and threadID
                                 returnLastStep,                                // whether this was the last step
                                 currentTrack.stepCounter == 1 ? true : false); // whether this was the first step

      if (cross_boundary) {
        // Move to the next boundary now that the Step is recorded
        currentTrack.navState = currentTrack.nextState;
        // Check if the next volume belongs to the GPU region and push it to the appropriate queue
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
      }
    }
  }
}

template <bool IsElectron, typename Scoring>
__device__ __forceinline__ void PerformStoppedAnnihilation(const int slot, Track &currentTrack,
                                                           Secondaries &secondaries, double &energyDeposit,
                                                           SlotManager &slotManager, const bool ApplyCuts,
                                                           const double theGammaCut, Scoring *userScoring)
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

      currentTrack.newRNG.Advance();
      Track &gamma1 =
          secondaries.gammas.NextTrack(currentTrack.newRNG, double{copcore::units::kElectronMassC2}, currentTrack.pos,
                                       vecgeom::Vector3D<Precision>{sint * cosPhi, sint * sinPhi, cost},
                                       currentTrack.navState, currentTrack, currentTrack.globalTime);

      // Reuse the RNG state of the dying track.
      Track &gamma2 =
          secondaries.gammas.NextTrack(currentTrack.rngState, double{copcore::units::kElectronMassC2}, currentTrack.pos,
                                       -gamma1.dir, currentTrack.navState, currentTrack, currentTrack.globalTime);
    }
  }
  // Particles are killed by not enqueuing them into the new activeQueue (and free the slot in async mode)
  slotManager.MarkSlotForFreeing(slot);
}

// Compute the physics and geometry step limit, transport the electrons while
// applying the continuous effects and maybe a discrete process that could
// generate secondaries.
template <bool IsElectron, typename Scoring>
__global__ void ElectronInteractions(Track *electrons, G4HepEmElectronTrack *hepEMTracks, Secondaries secondaries,
                                     adept::MParray *nextActiveQueue, adept::MParray *reachedInteractionQueue,
                                     adept::MParray *leakedQueue, Scoring *userScoring, bool returnAllSteps,
                                     bool returnLastStep)
{
  // int activeSize = active->size();
  int activeSize = reachedInteractionQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot           = (*active)[i];
    const int slot           = (*reachedInteractionQueue)[i];
    SlotManager &slotManager = IsElectron ? *secondaries.electrons.fSlotManager : *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    const int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    const int lvolID = currentTrack.navState.GetLogicalId();
#endif

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    auto survive = [&](bool leak = false) {
      returnLastStep = false; // track survived, do not force return of step
      // NOTE: When adapting the split kernels for async mode this won't
      // work if we want to re-use slots on the fly. Directly copying to
      // a trackdata struct would be better
      if (leak) {
        auto success = leakedQueue->push_back(slot);
        if (!success) {
          printf("ERROR: No space left in e-/+ leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
      } else
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

    // keep debug printout for now, needed for identifying more slow particles
    // if (activeSize == 1) {
    //   printf("Stuck particle!: E=%E event=%d loop=%d step=%d energyDeposit=%E geoStepLength=%E "
    //     "physicsStepLength=%E safety=%E  reached_interaction %d winnerProcessIndex %d onBoundary %d propagated
    //     %d\n", eKin, currentTrack.eventId, currentTrack.looperCounter, currentTrack.stepCounter, energyDeposit,
    //     geometryStepLength, geometricalStepLengthFromPhysics, safety, reached_interaction, winnerProcessIndex,
    //     nextState.IsOnBoundary(), propagated);
    // }

    // if (reached_interaction && !currentTrack.stopped) { // All tracks in this kernel reached their interaction
    if (!currentTrack.stopped) {
      // Reset number of interaction left for the winner discrete process.
      // (Will be resampled in the next iteration.)
      currentTrack.numIALeft[theTrack->GetWinnerProcessIndex()] = -1.0;

      // Check if a delta interaction happens instead of the real discrete process.
      if (G4HepEmElectronManager::CheckDelta(&g4HepEmData, theTrack, currentTrack.Uniform())) {
        // A delta interaction happened, move on.
        survive();
      } else {
        // Perform the discrete interaction, make sure the branched RNG state is
        // ready to be used.
        currentTrack.newRNG.Advance();
        // Also advance the current RNG state to provide a fresh round of random
        // numbers after MSC used up a fair share for sampling the displacement.
        currentTrack.rngState.Advance();

        switch (theTrack->GetWinnerProcessIndex()) {
        case 0: {
          // Invoke ionization (for e-/e+):
          double deltaEkin =
              (IsElectron) ? G4HepEmElectronInteractionIoni::SampleETransferMoller(theElCut, currentTrack.eKin, &rnge)
                           : G4HepEmElectronInteractionIoni::SampleETransferBhabha(theElCut, currentTrack.eKin, &rnge);

          double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
          double dirSecondary[3];
          G4HepEmElectronInteractionIoni::SampleDirections(currentTrack.eKin, deltaEkin, dirSecondary, dirPrimary,
                                                           &rnge);

          adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

          // Apply cuts
          if (ApplyCuts && (deltaEkin < theElCut)) {
            // Deposit the energy here and kill the secondary
            energyDeposit += deltaEkin;

          } else {
            Track &secondary = secondaries.electrons.NextTrack(
                currentTrack.newRNG, deltaEkin, currentTrack.pos,
                vecgeom::Vector3D<Precision>{dirSecondary[0], dirSecondary[1], dirSecondary[2]}, currentTrack.navState,
                currentTrack, currentTrack.globalTime);
          }

          currentTrack.eKin -= deltaEkin;

          // if below tracking cut, deposit energy for electrons (positrons are annihilated later) and stop particles
          if (currentTrack.eKin < g4HepEmPars.fElectronTrackingCut) {
            if (IsElectron) {
              energyDeposit += currentTrack.eKin;
            }
            currentTrack.stopped = true;
            break;
          }

          currentTrack.dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
          survive();
          break;
        }
        case 1: {
          // Invoke model for Bremsstrahlung: either SB- or Rel-Brem.
          double logEnergy = std::log(currentTrack.eKin);
          double deltaEkin = currentTrack.eKin < g4HepEmPars.fElectronBremModelLim
                                 ? G4HepEmElectronInteractionBrem::SampleETransferSB(
                                       &g4HepEmData, currentTrack.eKin, logEnergy, auxData.fMCIndex, &rnge, IsElectron)
                                 : G4HepEmElectronInteractionBrem::SampleETransferRB(
                                       &g4HepEmData, currentTrack.eKin, logEnergy, auxData.fMCIndex, &rnge, IsElectron);

          double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
          double dirSecondary[3];
          G4HepEmElectronInteractionBrem::SampleDirections(currentTrack.eKin, deltaEkin, dirSecondary, dirPrimary,
                                                           &rnge);

          adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 0, /*numPositrons*/ 0, /*numGammas*/ 1);

          // Apply cuts
          if (ApplyCuts && (deltaEkin < theGammaCut)) {
            // Deposit the energy here and kill the secondary
            energyDeposit += deltaEkin;

          } else {
            secondaries.gammas.NextTrack(
                currentTrack.newRNG, deltaEkin, currentTrack.pos,
                vecgeom::Vector3D<Precision>{dirSecondary[0], dirSecondary[1], dirSecondary[2]}, currentTrack.navState,
                currentTrack, currentTrack.globalTime);
          }

          currentTrack.eKin -= deltaEkin;

          // if below tracking cut, deposit energy for electrons (positrons are annihilated later) and stop particles
          if (currentTrack.eKin < g4HepEmPars.fElectronTrackingCut) {
            if (IsElectron) {
              energyDeposit += currentTrack.eKin;
            }
            currentTrack.stopped = true;
            break;
          }

          currentTrack.dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
          survive();
          break;
        }
        case 2: {
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
            secondaries.gammas.NextTrack(
                currentTrack.newRNG, theGamma1Ekin, currentTrack.pos,
                vecgeom::Vector3D<Precision>{theGamma1Dir[0], theGamma1Dir[1], theGamma1Dir[2]}, currentTrack.navState,
                currentTrack, currentTrack.globalTime);
          }
          if (ApplyCuts && (theGamma2Ekin < theGammaCut)) {
            // Deposit the energy here and kill the secondaries
            energyDeposit += theGamma2Ekin;

          } else {
            secondaries.gammas.NextTrack(
                currentTrack.rngState, theGamma2Ekin, currentTrack.pos,
                vecgeom::Vector3D<Precision>{theGamma2Dir[0], theGamma2Dir[1], theGamma2Dir[2]}, currentTrack.navState,
                currentTrack, currentTrack.globalTime);
          }

          // The current track is killed by not enqueuing into the next activeQueue.
          slotManager.MarkSlotForFreeing(slot);
          break;
        }
        }
      }
    }

    if (currentTrack.stopped) {
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

          currentTrack.newRNG.Advance();
          Track &gamma1 = secondaries.gammas.NextTrack(currentTrack.newRNG, double{copcore::units::kElectronMassC2},
                                                       currentTrack.pos,
                                                       vecgeom::Vector3D<Precision>{sint * cosPhi, sint * sinPhi, cost},
                                                       currentTrack.navState, currentTrack, currentTrack.globalTime);

          // Reuse the RNG state of the dying track.
          Track &gamma2 = secondaries.gammas.NextTrack(currentTrack.rngState, double{copcore::units::kElectronMassC2},
                                                       currentTrack.pos, -gamma1.dir, currentTrack.navState,
                                                       currentTrack, currentTrack.globalTime);
        }
      }
      // Particles are killed by not enqueuing them into the new activeQueue (and free the slot in async mode)
      slotManager.MarkSlotForFreeing(slot);
    }

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep)
      adept_scoring::RecordHit(userScoring, currentTrack.parentId,
                               static_cast<char>(IsElectron ? 0 : 1),         // Particle type
                               elTrack.GetPStepLength(),                      // Step length
                               energyDeposit,                                 // Total Edep
                               currentTrack.weight,                           // Track weight
                               currentTrack.navState,                         // Pre-step point navstate
                               currentTrack.preStepPos,                       // Pre-step point position
                               currentTrack.preStepDir,                       // Pre-step point momentum direction
                               currentTrack.preStepEKin,                      // Pre-step point kinetic energy
                               IsElectron ? -1 : 1,                           // Pre-step point charge
                               currentTrack.nextState,                        // Post-step point navstate
                               currentTrack.pos,                              // Post-step point position
                               currentTrack.dir,                              // Post-step point momentum direction
                               currentTrack.eKin,                             // Post-step point kinetic energy
                               IsElectron ? -1 : 1,                           // Post-step point charge
                               currentTrack.globalTime,                       // global time
                               currentTrack.eventId, currentTrack.threadId,   // eventID and threadID
                               returnLastStep,                                // whether this was the last step
                               currentTrack.stepCounter == 1 ? true : false); // whether this was the first step
  }
}

template <bool IsElectron, typename Scoring>
__global__ void ElectronIonization(Track *electrons, G4HepEmElectronTrack *hepEMTracks, Secondaries secondaries,
                                   adept::MParray *nextActiveQueue, adept::MParray *interactingQueue,
                                   adept::MParray *leakedQueue, Scoring *userScoring, bool returnAllSteps,
                                   bool returnLastStep)
{
  int activeSize = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot           = (*active)[i];
    const int slot           = (*interactingQueue)[i];
    SlotManager &slotManager = IsElectron ? *secondaries.electrons.fSlotManager : *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    const int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    const int lvolID = currentTrack.navState.GetLogicalId();
#endif

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    auto survive = [&](bool leak = false) {
      returnLastStep = false; // track survived, do not force return of step
      // NOTE: When adapting the split kernels for async mode this won't
      // work if we want to re-use slots on the fly. Directly copying to
      // a trackdata struct would be better
      if (leak) {
        auto success = leakedQueue->push_back(slot);
        if (!success) {
          printf("ERROR: No space left in e-/+ leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
      } else
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

    // Perform the discrete interaction, make sure the branched RNG state is
    // ready to be used.
    currentTrack.newRNG.Advance();
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
      Track &secondary = secondaries.electrons.NextTrack(
          currentTrack.newRNG, deltaEkin, currentTrack.pos,
          vecgeom::Vector3D<Precision>{dirSecondary[0], dirSecondary[1], dirSecondary[2]}, currentTrack.navState,
          currentTrack, currentTrack.globalTime);
    }

    currentTrack.eKin -= deltaEkin;

    // if below tracking cut, deposit energy for electrons (positrons are annihilated later) and stop particles
    if (currentTrack.eKin < g4HepEmPars.fElectronTrackingCut) {
      if (IsElectron) {
        energyDeposit += currentTrack.eKin;
      }
      currentTrack.stopped = true;
      PerformStoppedAnnihilation<IsElectron, Scoring>(slot, currentTrack, secondaries, energyDeposit, slotManager,
                                                      ApplyCuts, theGammaCut, userScoring);
    } else {
      currentTrack.dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
      survive();
    }

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep)
      adept_scoring::RecordHit(userScoring, currentTrack.parentId,
                               static_cast<char>(IsElectron ? 0 : 1),         // Particle type
                               elTrack.GetPStepLength(),                      // Step length
                               energyDeposit,                                 // Total Edep
                               currentTrack.weight,                           // Track weight
                               currentTrack.navState,                         // Pre-step point navstate
                               currentTrack.preStepPos,                       // Pre-step point position
                               currentTrack.preStepDir,                       // Pre-step point momentum direction
                               currentTrack.preStepEKin,                      // Pre-step point kinetic energy
                               IsElectron ? -1 : 1,                           // Pre-step point charge
                               currentTrack.nextState,                        // Post-step point navstate
                               currentTrack.pos,                              // Post-step point position
                               currentTrack.dir,                              // Post-step point momentum direction
                               currentTrack.eKin,                             // Post-step point kinetic energy
                               IsElectron ? -1 : 1,                           // Post-step point charge
                               currentTrack.globalTime,                       // global time
                               currentTrack.eventId, currentTrack.threadId,   // eventID and threadID
                               returnLastStep,                                // whether this was the last step
                               currentTrack.stepCounter == 1 ? true : false); // whether this was the first step
  }
}

template <bool IsElectron, typename Scoring>
__global__ void ElectronBremsstrahlung(Track *electrons, G4HepEmElectronTrack *hepEMTracks, Secondaries secondaries,
                                       adept::MParray *nextActiveQueue, adept::MParray *interactingQueue,
                                       adept::MParray *leakedQueue, Scoring *userScoring, bool returnAllSteps,
                                       bool returnLastStep)
{
  int activeSize = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot           = (*active)[i];
    const int slot           = (*interactingQueue)[i];
    SlotManager &slotManager = IsElectron ? *secondaries.electrons.fSlotManager : *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    const int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    const int lvolID = currentTrack.navState.GetLogicalId();
#endif

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    auto survive = [&](bool leak = false) {
      returnLastStep = false; // track survived, do not force return of step
      // NOTE: When adapting the split kernels for async mode this won't
      // work if we want to re-use slots on the fly. Directly copying to
      // a trackdata struct would be better
      if (leak) {
        auto success = leakedQueue->push_back(slot);
        if (!success) {
          printf("ERROR: No space left in e-/+ leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
      } else
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

    // Perform the discrete interaction, make sure the branched RNG state is
    // ready to be used.
    currentTrack.newRNG.Advance();
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
      secondaries.gammas.NextTrack(currentTrack.newRNG, deltaEkin, currentTrack.pos,
                                   vecgeom::Vector3D<Precision>{dirSecondary[0], dirSecondary[1], dirSecondary[2]},
                                   currentTrack.navState, currentTrack, currentTrack.globalTime);
    }

    currentTrack.eKin -= deltaEkin;

    // if below tracking cut, deposit energy for electrons (positrons are annihilated later) and stop particles
    if (currentTrack.eKin < g4HepEmPars.fElectronTrackingCut) {
      if (IsElectron) {
        energyDeposit += currentTrack.eKin;
      }
      currentTrack.stopped = true;
      PerformStoppedAnnihilation<IsElectron, Scoring>(slot, currentTrack, secondaries, energyDeposit, slotManager,
                                                      ApplyCuts, theGammaCut, userScoring);
    } else {
      currentTrack.dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
      survive();
    }

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep)
      adept_scoring::RecordHit(userScoring, currentTrack.parentId,
                               static_cast<char>(IsElectron ? 0 : 1),         // Particle type
                               elTrack.GetPStepLength(),                      // Step length
                               energyDeposit,                                 // Total Edep
                               currentTrack.weight,                           // Track weight
                               currentTrack.navState,                         // Pre-step point navstate
                               currentTrack.preStepPos,                       // Pre-step point position
                               currentTrack.preStepDir,                       // Pre-step point momentum direction
                               currentTrack.preStepEKin,                      // Pre-step point kinetic energy
                               IsElectron ? -1 : 1,                           // Pre-step point charge
                               currentTrack.nextState,                        // Post-step point navstate
                               currentTrack.pos,                              // Post-step point position
                               currentTrack.dir,                              // Post-step point momentum direction
                               currentTrack.eKin,                             // Post-step point kinetic energy
                               IsElectron ? -1 : 1,                           // Post-step point charge
                               currentTrack.globalTime,                       // global time
                               currentTrack.eventId, currentTrack.threadId,   // eventID and threadID
                               returnLastStep,                                // whether this was the last step
                               currentTrack.stepCounter == 1 ? true : false); // whether this was the first step
  }
}

template <typename Scoring>
__global__ void PositronAnnihilation(Track *electrons, G4HepEmElectronTrack *hepEMTracks, Secondaries secondaries,
                                     adept::MParray *nextActiveQueue, adept::MParray *interactingQueue,
                                     adept::MParray *leakedQueue, Scoring *userScoring, bool returnAllSteps,
                                     bool returnLastStep)
{
  int activeSize = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot           = (*active)[i];
    const int slot           = (*interactingQueue)[i];
    SlotManager &slotManager = *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    const int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    const int lvolID = currentTrack.navState.GetLogicalId();
#endif

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    auto survive = [&](bool leak = false) {
      returnLastStep = false; // track survived, do not force return of step
      // NOTE: When adapting the split kernels for async mode this won't
      // work if we want to re-use slots on the fly. Directly copying to
      // a trackdata struct would be better
      if (leak) {
        auto success = leakedQueue->push_back(slot);
        if (!success) {
          printf("ERROR: No space left in e-/+ leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
      } else
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

    // Perform the discrete interaction, make sure the branched RNG state is
    // ready to be used.
    currentTrack.newRNG.Advance();
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
      secondaries.gammas.NextTrack(currentTrack.newRNG, theGamma1Ekin, currentTrack.pos,
                                   vecgeom::Vector3D<Precision>{theGamma1Dir[0], theGamma1Dir[1], theGamma1Dir[2]},
                                   currentTrack.navState, currentTrack, currentTrack.globalTime);
    }
    if (ApplyCuts && (theGamma2Ekin < theGammaCut)) {
      // Deposit the energy here and kill the secondaries
      energyDeposit += theGamma2Ekin;

    } else {
      secondaries.gammas.NextTrack(currentTrack.rngState, theGamma2Ekin, currentTrack.pos,
                                   vecgeom::Vector3D<Precision>{theGamma2Dir[0], theGamma2Dir[1], theGamma2Dir[2]},
                                   currentTrack.navState, currentTrack, currentTrack.globalTime);
    }

    // The current track is killed by not enqueuing into the next activeQueue.
    slotManager.MarkSlotForFreeing(slot);

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep)
      adept_scoring::RecordHit(userScoring, currentTrack.parentId,
                               static_cast<char>(1),                          // Particle type
                               elTrack.GetPStepLength(),                      // Step length
                               energyDeposit,                                 // Total Edep
                               currentTrack.weight,                           // Track weight
                               currentTrack.navState,                         // Pre-step point navstate
                               currentTrack.preStepPos,                       // Pre-step point position
                               currentTrack.preStepDir,                       // Pre-step point momentum direction
                               currentTrack.preStepEKin,                      // Pre-step point kinetic energy
                               1,                                             // Pre-step point charge
                               currentTrack.nextState,                        // Post-step point navstate
                               currentTrack.pos,                              // Post-step point position
                               currentTrack.dir,                              // Post-step point momentum direction
                               currentTrack.eKin,                             // Post-step point kinetic energy
                               1,                                             // Post-step point charge
                               currentTrack.globalTime,                       // global time
                               currentTrack.eventId, currentTrack.threadId,   // eventID and threadID
                               returnLastStep,                                // whether this was the last step
                               currentTrack.stepCounter == 1 ? true : false); // whether this was the first step
  }
}

template <typename Scoring>
__global__ void PositronStoppedAnnihilation(Track *electrons, G4HepEmElectronTrack *hepEMTracks,
                                            Secondaries secondaries, adept::MParray *nextActiveQueue,
                                            adept::MParray *interactingQueue, adept::MParray *leakedQueue,
                                            Scoring *userScoring, bool returnAllSteps, bool returnLastStep)
{
  int activeSize = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot           = (*active)[i];
    const int slot           = (*interactingQueue)[i];
    SlotManager &slotManager = *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    const int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    const int lvolID = currentTrack.navState.GetLogicalId();
#endif

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    auto survive = [&](bool leak = false) {
      returnLastStep = false; // track survived, do not force return of step
      // NOTE: When adapting the split kernels for async mode this won't
      // work if we want to re-use slots on the fly. Directly copying to
      // a trackdata struct would be better
      if (leak) {
        auto success = leakedQueue->push_back(slot);
        if (!success) {
          printf("ERROR: No space left in e-/+ leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
      } else
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
                                               theGammaCut, userScoring);

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep)
      adept_scoring::RecordHit(userScoring, currentTrack.parentId,
                               static_cast<char>(1),                          // Particle type
                               elTrack.GetPStepLength(),                      // Step length
                               energyDeposit,                                 // Total Edep
                               currentTrack.weight,                           // Track weight
                               currentTrack.navState,                         // Pre-step point navstate
                               currentTrack.preStepPos,                       // Pre-step point position
                               currentTrack.preStepDir,                       // Pre-step point momentum direction
                               currentTrack.preStepEKin,                      // Pre-step point kinetic energy
                               1,                                             // Pre-step point charge
                               currentTrack.nextState,                        // Post-step point navstate
                               currentTrack.pos,                              // Post-step point position
                               currentTrack.dir,                              // Post-step point momentum direction
                               currentTrack.eKin,                             // Post-step point kinetic energy
                               1,                                             // Post-step point charge
                               currentTrack.globalTime,                       // global time
                               currentTrack.eventId, currentTrack.threadId,   // eventID and threadID
                               returnLastStep,                                // whether this was the last step
                               currentTrack.stepCounter == 1 ? true : false); // whether this was the first step
  }
}

} // namespace AsyncAdePT
