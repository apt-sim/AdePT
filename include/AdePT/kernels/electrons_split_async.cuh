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

template <bool IsElectron, typename Scoring>
static __device__ __forceinline__ void ElectronHowFar(Track *electrons, G4HepEmElectronTrack *hepEMTracks,
                                                      const adept::MParray *active, Secondaries &secondaries,
                                                      adept::MParray *nextActiveQueue, adept::MParray *leakedQueue,
                                                      Scoring *userScoring, Stats *InFlightStats,
                                                      AllowFinishOffEventArray allowFinishOffEvent, bool returnAllSteps,
                                                      bool returnLastStep)
{
  constexpr Precision kPushDistance = 1000 * vecgeom::kTolerance;
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
  using Stepper_t  = DormandPrinceRK45<Equation_t, Field_t, Nvar, vecgeom::Precision>;
  using RkDriver_t = RkIntegrationDriver<Stepper_t, vecgeom::Precision, int, Equation_t, Field_t>;

  auto &magneticField = *gMagneticField;

  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot           = (*active)[i];
    SlotManager &slotManager = IsElectron ? *secondaries.electrons.fSlotManager : *secondaries.positrons.fSlotManager;

    Track &currentTrack          = electrons[slot];
    currentTrack.preStepEKin     = currentTrack.eKin;
    currentTrack.preStepPos      = currentTrack.pos;
    currentTrack.preStepDir      = currentTrack.dir;
    currentTrack.preStepNavState = currentTrack.navState;
    // the MCC vector is indexed by the logical volume id
#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    const int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    const int lvolID = currentTrack.navState.GetLogicalId();
#endif

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    currentTrack.stepCounter++;
    if (currentTrack.stepCounter >= maxSteps) {
      printf("Killing e-/+ event %d E=%f lvol=%d after %d steps.\n", currentTrack.eventId, currentTrack.eKin, lvolID,
             currentTrack.stepCounter);
      continue;
    }

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
    theTrack->SetSafety(currentTrack.safety);
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
      limit                       = currentTrack.safety > limit ? currentTrack.safety : limit;

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

// Compute the physics and geometry step limit, transport the electrons while
// applying the continuous effects and maybe a discrete process that could
// generate secondaries.
template <bool IsElectron, typename Scoring>
static __device__ __forceinline__ void TransportElectrons(Track *electrons, const adept::MParray *active,
                                                          Secondaries &secondaries, adept::MParray *nextActiveQueue,
                                                          adept::MParray *leakedQueue, Scoring *userScoring,
                                                          Stats *InFlightStats,
                                                          AllowFinishOffEventArray allowFinishOffEvent,
                                                          bool returnAllSteps, bool returnLastStep)
{

  //////////////////////////// TRANSPORT ELECTRONS ////////////////////////////

  constexpr Precision kPushDistance = 1000 * vecgeom::kTolerance;
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
  using Stepper_t  = DormandPrinceRK45<Equation_t, Field_t, Nvar, vecgeom::Precision>;
  using RkDriver_t = RkIntegrationDriver<Stepper_t, vecgeom::Precision, int, Equation_t, Field_t>;

  auto &magneticField = *gMagneticField;

  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot           = (*active)[i];
    SlotManager &slotManager = IsElectron ? *secondaries.electrons.fSlotManager : *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    auto navState       = currentTrack.navState;
    // the MCC vector is indexed by the logical volume id
#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    const int lvolID = navState.Top()->GetLogicalVolume()->id();
#else
    const int lvolID = navState.GetLogicalId();
#endif

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    auto eKin          = currentTrack.eKin;
    auto preStepEnergy = eKin;
    auto pos           = currentTrack.pos;
    vecgeom::Vector3D<Precision> preStepPos(pos);
    auto dir = currentTrack.dir;
    vecgeom::Vector3D<Precision> preStepDir(dir);
    double globalTime = currentTrack.globalTime;
    double localTime  = currentTrack.localTime;
    double properTime = currentTrack.properTime;
    currentTrack.stepCounter++;
    if (currentTrack.stepCounter >= maxSteps) {
      printf("Killing e-/+ event %d E=%f lvol=%d after %d steps.\n", currentTrack.eventId, eKin, lvolID,
             currentTrack.stepCounter);
      continue;
    }

    auto survive = [&](bool leak = false) {
      returnLastStep          = false; // track survived, do not force return of step
      currentTrack.eKin       = eKin;
      currentTrack.pos        = pos;
      currentTrack.dir        = dir;
      currentTrack.globalTime = globalTime;
      currentTrack.localTime  = localTime;
      currentTrack.properTime = properTime;
      currentTrack.navState   = navState;
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
      printf("Thread %d Finishing e-/e+ of the %d last particles of event %d on CPU E=%f lvol=%d after %d steps.\n ",
             currentTrack.threadId, InFlightStats->perEventInFlightPrevious[currentTrack.threadId],
             currentTrack.eventId, eKin, lvolID, currentTrack.stepCounter);
      survive(/*leak*/ true);
      continue;
    }

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

    G4HepEmRandomEngine rnge(&currentTrack.rngState);

    //     // Prepare a branched RNG state while threads are synchronized. Even if not
    //     // used, this provides a fresh round of random numbers and reduces thread
    //     // divergence because the RNG state doesn't need to be advanced later.
    //     RanluxppDouble newRNG(currentTrack.rngState.BranchNoAdvance());
    //     G4HepEmRandomEngine rnge(&currentTrack.rngState);

    //     // Sample the `number-of-interaction-left` and put it into the track.
    //     for (int ip = 0; ip < 4; ++ip) {
    //       double numIALeft = currentTrack.numIALeft[ip];
    //       if (numIALeft <= 0) {
    //         numIALeft = -std::log(currentTrack.Uniform());
    //       }
    //       if (ip == 3) numIALeft = vecgeom::kInfLength; // suppress lepton nuclear by infinite length
    //       theTrack->SetNumIALeft(numIALeft, ip);
    //     }

    //     G4HepEmElectronManager::HowFarToDiscreteInteraction(&g4HepEmData, &g4HepEmPars, &elTrack);

    //     auto physicalStepLength = elTrack.GetPStepLength();
    //     // Compute safety, needed for MSC step limit. The accuracy range is physicalStepLength
    //     double safety = 0.;
    //     if (!navState.IsOnBoundary()) {
    //       // Get the remaining safety only if larger than physicalStepLength
    //       safety = currentTrack.GetSafety(pos);
    //       if (safety < physicalStepLength) {
    //         // Recompute safety and update it in the track.
    // #ifdef ADEPT_USE_SURF
    //         // Use maximum accuracy only if safety is samller than physicalStepLength
    //         safety = AdePTNavigator::ComputeSafety(pos, navState, physicalStepLength);
    // #else
    //         safety = AdePTNavigator::ComputeSafety(pos, navState);
    // #endif
    //         currentTrack.SetSafety(pos, safety);
    //       }
    //     }
    //     theTrack->SetSafety(safety);
    //     bool restrictedPhysicalStepLength = false;

    //     double safeLength = 0.;

    //     if (gMagneticField) {

    //       const double momentumMag = sqrt(eKin * (eKin + 2.0 * restMass));
    //       // Distance along the track direction to reach the maximum allowed error

    //       // SEVERIN: to be checked if we can use float
    //       vecgeom::Vector3D<double> momentumVec = momentumMag * dir;
    //       vecgeom::Vector3D<double> B0fieldVec =
    //           magneticField.Evaluate(pos[0], pos[1], pos[2]); // Field value at starting point
    //       safeLength =
    //           fieldPropagatorRungeKutta<Field_t, RkDriver_t, Precision, AdePTNavigator>::ComputeSafeLength
    //           /*<Real_t>*/ (
    //               momentumVec, B0fieldVec, Charge);

    //       constexpr int MaxSafeLength = 10;
    //       double limit                = MaxSafeLength * safeLength;
    //       limit                       = safety > limit ? safety : limit;

    //       if (physicalStepLength > limit) {
    //         physicalStepLength           = limit;
    //         restrictedPhysicalStepLength = true;
    //         elTrack.SetPStepLength(physicalStepLength);

    //         // Note: We are limiting the true step length, which is converted to
    //         // a shorter geometry step length in HowFarToMSC. In that sense, the
    //         // limit is an over-approximation, but that is fine for our purpose.
    //       }
    //     }

    //     G4HepEmElectronManager::HowFarToMSC(&g4HepEmData, &g4HepEmPars, &elTrack, &rnge);

    //     // Remember MSC values for the next step(s).
    //     currentTrack.initialRange       = mscData->fInitialRange;
    //     currentTrack.dynamicRangeFactor = mscData->fDynamicRangeFactor;
    //     currentTrack.tlimitMin          = mscData->fTlimitMin;

    //////////////////////////// TRANSPORT ELECTRONS END ////////////////////////////

    // Get result into variables.
    double geometricalStepLengthFromPhysics = theTrack->GetGStepLength();
    // The phyiscal step length is the amount that the particle experiences
    // which might be longer than the geometrical step length due to MSC. As
    // long as we call PerformContinuous in the same kernel we don't need to
    // care, but we need to make this available when splitting the operations.
    // double physicalStepLength = elTrack.GetPStepLength();
    int winnerProcessIndex = theTrack->GetWinnerProcessIndex();
    // Leave the range and MFP inside the G4HepEmTrack. If we split kernels, we
    // also need to carry them over!

    // Skip electron/positron-nuclear reaction that would need to be handled by G4 itself
    if (winnerProcessIndex == 3) {
      winnerProcessIndex = -1;
      // Note, this should not be hit at the moment due to the infinite length, this is just for safety
    }

    // Check if there's a volume boundary in between.
    bool propagated    = true;
    long hitsurf_index = -1;
    double geometryStepLength;
    vecgeom::NavigationState nextState;

    if (gMagneticField) {
      int iterDone = -1;
      geometryStepLength =
          fieldPropagatorRungeKutta<Field_t, RkDriver_t, Precision, AdePTNavigator>::ComputeStepAndNextVolume(
              magneticField, eKin, restMass, Charge, geometricalStepLengthFromPhysics, currentTrack.safeLength, pos,
              dir, navState, nextState, hitsurf_index, propagated, /*lengthDone,*/ currentTrack.safety,
              // activeSize < 100 ? max_iterations : max_iters_tail ), // Was
              max_iterations, iterDone, slot);
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
    if (!nextState.IsOnBoundary()) {
      const double *mscDisplacement = mscData->GetDisplacement();
      vecgeom::Vector3D<Precision> displacement(mscDisplacement[0], mscDisplacement[1], mscDisplacement[2]);
      const double dLength2            = displacement.Length2();
      constexpr double kGeomMinLength  = 5 * copcore::units::nm;          // 0.05 [nm]
      constexpr double kGeomMinLength2 = kGeomMinLength * kGeomMinLength; // (0.05 [nm])^2
      if (dLength2 > kGeomMinLength2) {
        const double dispR = std::sqrt(dLength2);
        // Estimate safety by subtracting the geometrical step length.
        auto safety            = currentTrack.GetSafety(pos);
        constexpr double sFact = 0.99;
        double reducedSafety   = sFact * safety;

        // Apply displacement, depending on how close we are to a boundary.
        // 1a. Far away from geometry boundary:
        if (reducedSafety > 0.0 && dispR <= reducedSafety) {
          pos += displacement;
        } else {
          // Recompute safety.
#ifdef ADEPT_USE_SURF
          // Use maximum accuracy only if safety is samller than physicalStepLength
          safety = AdePTNavigator::ComputeSafety(pos, navState, dispR);
#else
          safety = AdePTNavigator::ComputeSafety(pos, navState);
#endif
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
    }

    // Collect the charged step length (might be changed by MSC). Collect the changes in energy and deposit.
    eKin                 = theTrack->GetEKin();
    double energyDeposit = theTrack->GetEnergyDeposit();

    // Update the flight times of the particle
    // By calculating the velocity here, we assume that all the energy deposit is done at the PreStepPoint, and
    // the velocity depends on the remaining energy
    double deltaTime = elTrack.GetPStepLength() / GetVelocity(eKin);
    globalTime += deltaTime;
    localTime += deltaTime;
    properTime += deltaTime * (restMass / eKin);

    if (nextState.IsOnBoundary()) {
      // if the particle hit a boundary, and is neither stopped or outside, relocate to have the correct next state
      // before RecordHit is called
      if (!stopped && !nextState.IsOutside()) {
#ifdef ADEPT_USE_SURF
        AdePTNavigator::RelocateToNextVolume(pos, dir, hitsurf_index, nextState);
#else
        AdePTNavigator::RelocateToNextVolume(pos, dir, nextState);
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

    const double theElCut    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecElProdCutE;
    const double theGammaCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecGamProdCutE;

    const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
    const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

    if (!stopped) {
      if (nextState.IsOnBoundary()) {
        // For now, just count that we hit something.
        reached_interaction = false;
        // Kill the particle if it left the world.

        if (++currentTrack.looperCounter > 500) {
          // Kill loopers that are scraping a boundary
          printf("Killing looper scraping at a boundary: E=%E event=%d loop=%d energyDeposit=%E geoStepLength=%E "
                 "physicsStepLength=%E "
                 "safety=%E\n",
                 eKin, currentTrack.eventId, currentTrack.looperCounter, energyDeposit, geometryStepLength,
                 geometricalStepLengthFromPhysics, currentTrack.safety);
          continue;
        } else if (!nextState.IsOutside()) {
          // Mark the particle. We need to change its navigation state to the next volume before enqueuing it
          // This will happen after recording the step
          cross_boundary = true;
          returnLastStep = false; // the track survives, do not force return of step
        } else {
          // Particle left the world, don't enqueue it and release the slot
          slotManager.MarkSlotForFreeing(slot);
        }

      } else if (!propagated || currentTrack.restrictedPhysicalStepLength) {
        // Did not yet reach the interaction point due to error in the magnetic
        // field propagation. Try again next time.

        if (++currentTrack.looperCounter > 500) {
          // Kill loopers that are not advancing in free space
          printf("Killing looper due to lack of advance: E=%E event=%d loop=%d energyDeposit=%E geoStepLength=%E "
                 "physicsStepLength=%E "
                 "safety=%E\n",
                 eKin, currentTrack.eventId, currentTrack.looperCounter, energyDeposit, geometryStepLength,
                 geometricalStepLengthFromPhysics, currentTrack.safety);
          continue;
        }

        survive();
        reached_interaction = false;
      } else if (winnerProcessIndex < 0) {
        // No discrete process, move on.
        survive();
        reached_interaction = false;
      }
    }

    // reset Looper counter if limited by discrete interaction or MSC
    if (reached_interaction) currentTrack.looperCounter = 0;

    // keep debug printout for now, needed for identifying more slow particles
    // if (activeSize == 1) {
    //   printf("Stuck particle!: E=%E event=%d loop=%d step=%d energyDeposit=%E geoStepLength=%E "
    //     "physicsStepLength=%E safety=%E  reached_interaction %d winnerProcessIndex %d onBoundary %d propagated %d\n",
    //     eKin, currentTrack.eventId, currentTrack.looperCounter, currentTrack.stepCounter, energyDeposit,
    //     geometryStepLength, geometricalStepLengthFromPhysics, safety, reached_interaction, winnerProcessIndex,
    //     nextState.IsOnBoundary(), propagated);
    // }

    if (reached_interaction && !stopped) {
      // Reset number of interaction left for the winner discrete process.
      // (Will be resampled in the next iteration.)
      currentTrack.numIALeft[winnerProcessIndex] = -1.0;

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

        switch (winnerProcessIndex) {
        case 0: {
          // Invoke ionization (for e-/e+):
          double deltaEkin = (IsElectron)
                                 ? G4HepEmElectronInteractionIoni::SampleETransferMoller(theElCut, eKin, &rnge)
                                 : G4HepEmElectronInteractionIoni::SampleETransferBhabha(theElCut, eKin, &rnge);

          double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
          double dirSecondary[3];
          G4HepEmElectronInteractionIoni::SampleDirections(eKin, deltaEkin, dirSecondary, dirPrimary, &rnge);

          adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

          // Apply cuts
          if (ApplyCuts && (deltaEkin < theElCut)) {
            // Deposit the energy here and kill the secondary
            energyDeposit += deltaEkin;

          } else {
            Track &secondary = secondaries.electrons.NextTrack(
                currentTrack.newRNG, deltaEkin, pos,
                vecgeom::Vector3D<Precision>{dirSecondary[0], dirSecondary[1], dirSecondary[2]}, navState,
                currentTrack);
          }

          eKin -= deltaEkin;

          // if below tracking cut, deposit energy for electrons (positrons are annihilated later) and stop particles
          if (eKin < g4HepEmPars.fElectronTrackingCut) {
            if (IsElectron) {
              energyDeposit += eKin;
            }
            stopped = true;
            break;
          }

          dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
          survive();
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

          // Apply cuts
          if (ApplyCuts && (deltaEkin < theGammaCut)) {
            // Deposit the energy here and kill the secondary
            energyDeposit += deltaEkin;

          } else {
            secondaries.gammas.NextTrack(
                currentTrack.newRNG, deltaEkin, pos,
                vecgeom::Vector3D<Precision>{dirSecondary[0], dirSecondary[1], dirSecondary[2]}, navState,
                currentTrack);
          }

          eKin -= deltaEkin;

          // if below tracking cut, deposit energy for electrons (positrons are annihilated later) and stop particles
          if (eKin < g4HepEmPars.fElectronTrackingCut) {
            if (IsElectron) {
              energyDeposit += eKin;
            }
            stopped = true;
            break;
          }

          dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
          survive();
          break;
        }
        case 2: {
          // Invoke annihilation (in-flight) for e+
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
            secondaries.gammas.NextTrack(
                currentTrack.newRNG, theGamma1Ekin, pos,
                vecgeom::Vector3D<Precision>{theGamma1Dir[0], theGamma1Dir[1], theGamma1Dir[2]}, navState,
                currentTrack);
          }
          if (ApplyCuts && (theGamma2Ekin < theGammaCut)) {
            // Deposit the energy here and kill the secondaries
            energyDeposit += theGamma2Ekin;

          } else {
            secondaries.gammas.NextTrack(
                currentTrack.rngState, theGamma2Ekin, pos,
                vecgeom::Vector3D<Precision>{theGamma2Dir[0], theGamma2Dir[1], theGamma2Dir[2]}, navState,
                currentTrack);
          }

          // The current track is killed by not enqueuing into the next activeQueue.
          slotManager.MarkSlotForFreeing(slot);
          break;
        }
        }
      }
    }

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

          currentTrack.newRNG.Advance();
          Track &gamma1 = secondaries.gammas.NextTrack(
              currentTrack.newRNG, double{copcore::units::kElectronMassC2}, pos,
              vecgeom::Vector3D<Precision>{sint * cosPhi, sint * sinPhi, cost}, navState, currentTrack);

          // Reuse the RNG state of the dying track.
          Track &gamma2 = secondaries.gammas.NextTrack(currentTrack.rngState, double{copcore::units::kElectronMassC2},
                                                       pos, -gamma1.dir, navState, currentTrack);
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
                               navState,                                      // Pre-step point navstate
                               preStepPos,                                    // Pre-step point position
                               preStepDir,                                    // Pre-step point momentum direction
                               preStepEnergy,                                 // Pre-step point kinetic energy
                               IsElectron ? -1 : 1,                           // Pre-step point charge
                               nextState,                                     // Post-step point navstate
                               pos,                                           // Post-step point position
                               dir,                                           // Post-step point momentum direction
                               eKin,                                          // Post-step point kinetic energy
                               IsElectron ? -1 : 1,                           // Post-step point charge
                               currentTrack.eventId, currentTrack.threadId,   // eventID and threadID
                               returnLastStep,                                // whether this was the last step
                               currentTrack.stepCounter == 1 ? true : false); // whether this was the first step
    if (cross_boundary) {
      // Move to the next boundary now that the Step is recorded
      navState = nextState;
      // Check if the next volume belongs to the GPU region and push it to the appropriate queue
#ifndef ADEPT_USE_SURF
      const int nextlvolID = navState.Top()->GetLogicalVolume()->id();
#else
      const int nextlvolID = navState.GetLogicalId();
#endif
      VolAuxData const &nextauxData = AsyncAdePT::gVolAuxData[nextlvolID];
      if (nextauxData.fGPUregion > 0)
        survive();
      else {
        // To be safe, just push a bit the track exiting the GPU region to make sure
        // Geant4 does not relocate it again inside the same region
        pos += kPushDistance * dir;
        survive(/*leak*/ true);
      }
    }
  }
}

// Instantiate kernels for electrons and positrons.
template <typename Scoring>
__global__ void TransportElectrons(Track *electrons, const adept::MParray *active, Secondaries secondaries,
                                   adept::MParray *nextActiveQueue, adept::MParray *leakedQueue, Scoring *userScoring,
                                   Stats *InFlightStats, AllowFinishOffEventArray allowFinishOffEvent,
                                   bool returnAllSteps, bool returnLastStep)
{
  TransportElectrons</*IsElectron*/ true, Scoring>(electrons, active, secondaries, nextActiveQueue, leakedQueue,
                                                   userScoring, InFlightStats, allowFinishOffEvent, returnAllSteps,
                                                   returnLastStep);
}
template <typename Scoring>
__global__ void TransportPositrons(Track *positrons, const adept::MParray *active, Secondaries secondaries,
                                   adept::MParray *nextActiveQueue, adept::MParray *leakedQueue, Scoring *userScoring,
                                   Stats *InFlightStats, AllowFinishOffEventArray allowFinishOffEvent,
                                   bool returnAllSteps, bool returnLastStep)
{
  TransportElectrons</*IsElectron*/ false, Scoring>(positrons, active, secondaries, nextActiveQueue, leakedQueue,
                                                    userScoring, InFlightStats, allowFinishOffEvent, returnAllSteps,
                                                    returnLastStep);
}

} // namespace AsyncAdePT
