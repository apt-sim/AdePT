// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/navigation/AdePTNavigator.h>
#include <AdePT/magneticfield/fieldPropagatorConstBz.h>

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

#ifdef ASYNC_MODE
namespace AsyncAdePT {

// Compute the physics and geometry step limit, transport the electrons while
// applying the continuous effects and maybe a discrete process that could
// generate secondaries.
template <bool IsElectron, typename Scoring>
static __device__ __forceinline__ void TransportElectrons(Track *electrons, const adept::MParray *active,
                                                          Secondaries &secondaries, adept::MParray *activeQueue,
                                                          adept::MParray *leakedQueue, Scoring *userScoring,
                                                          bool returnAllSteps)
{
  constexpr Precision kPushDistance = 1000 * vecgeom::kTolerance;
  constexpr int Charge              = IsElectron ? -1 : 1;
  constexpr double restMass         = copcore::units::kElectronMassC2;
  fieldPropagatorConstBz fieldPropagatorBz(BzFieldValue);

  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot           = (*active)[i];
    SlotManager &slotManager = IsElectron ? *secondaries.electrons.fSlotManager : *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    currentTrack.stepCounter++; // to be moved to common part as soon as Tracks are unified
    auto navState = currentTrack.navState;
    // the MCC vector is indexed by the logical volume id
#ifndef ADEPT_USE_SURF // FIXME remove as soon as surface model branch is merged!
    const int lvolID = navState.Top()->GetLogicalVolume()->id();
#else
    const int lvolID = navState.GetLogicalId();
#endif

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

#else
template <bool IsElectron, typename Scoring>
static __device__ __forceinline__ void TransportElectrons(adept::TrackManager<Track> *electrons,
                                                          Secondaries &secondaries, MParrayTracks *leakedQueue,
                                                          Scoring *userScoring, VolAuxData const *auxDataArray)
{
  using namespace adept_impl;
  constexpr bool returnAllSteps     = false;
  constexpr Precision kPushDistance = 1000 * vecgeom::kTolerance;
  constexpr int Charge              = IsElectron ? -1 : 1;
  constexpr double restMass         = copcore::units::kElectronMassC2;
  constexpr int Pdg                 = IsElectron ? 11 : -11;
  fieldPropagatorConstBz fieldPropagatorBz(BzFieldValue);

  int activeSize = electrons->fActiveTracks->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot = (*electrons->fActiveTracks)[i];
    adeptint::TrackData trackdata;
    Track &currentTrack = (*electrons)[slot];
    auto navState       = currentTrack.navState;
    // the MCC vector is indexed by the logical volume id
#ifndef ADEPT_USE_SURF
    const int lvolID = navState.Top()->GetLogicalVolume()->id();
#else
    const int lvolID = navState.GetLogicalId();
#endif

    VolAuxData const &auxData = auxDataArray[lvolID];

#endif
    auto eKin          = currentTrack.eKin;
    auto preStepEnergy = eKin;
    auto pos           = currentTrack.pos;
    vecgeom::Vector3D<Precision> preStepPos(pos);
    auto dir = currentTrack.dir;
    vecgeom::Vector3D<Precision> preStepDir(dir);
    double globalTime = currentTrack.globalTime;
    double localTime  = currentTrack.localTime;
    double properTime = currentTrack.properTime;

    auto survive = [&](bool leak = false) {
      currentTrack.eKin       = eKin;
      currentTrack.pos        = pos;
      currentTrack.dir        = dir;
      currentTrack.globalTime = globalTime;
      currentTrack.localTime  = localTime;
      currentTrack.properTime = properTime;
      currentTrack.navState   = navState;
#ifdef ASYNC_MODE
      // NOTE: When adapting the split kernels for async mode this won't
      // work if we want to re-use slots on the fly. Directly copying to
      // a trackdata struct would be better
      if (leak)
        leakedQueue->push_back(slot);
      else
        activeQueue->push_back(slot);
#else
      currentTrack.CopyTo(trackdata, Pdg);
      if (leak)
        leakedQueue->push_back(trackdata);
      else
        electrons->fNextTracks->push_back(slot);
#endif
    };

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmElectronTrack elTrack;
    G4HepEmTrack *theTrack = elTrack.GetTrack();
    theTrack->SetEKin(eKin);
    theTrack->SetMCIndex(auxData.fMCIndex);
    theTrack->SetOnBoundary(navState.IsOnBoundary());
    theTrack->SetCharge(Charge);
    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    mscData->fIsFirstStep        = currentTrack.initialRange < 0;
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
      if (ip == 3) numIALeft = vecgeom::kInfLength; // suppress lepton nuclear by infinite length
      theTrack->SetNumIALeft(numIALeft, ip);
    }

    G4HepEmElectronManager::HowFarToDiscreteInteraction(&g4HepEmData, &g4HepEmPars, &elTrack);

    auto physicalStepLength = elTrack.GetPStepLength();
    // Compute safety, needed for MSC step limit. The accuracy range is physicalStepLength
    double safety = 0.;
    if (!navState.IsOnBoundary()) {
      // Get the remaining safety only if larger than physicalStepLength
      safety = currentTrack.GetSafety(pos);
      if (safety < physicalStepLength) {
        // Recompute safety and update it in the track.
#ifdef ADEPT_USE_SURF
        // Use maximum accuracy only if safety is samller than physicalStepLength
        safety = AdePTNavigator::ComputeSafety(pos, navState, physicalStepLength);
#else
        safety = AdePTNavigator::ComputeSafety(pos, navState);
#endif
        currentTrack.SetSafety(pos, safety);
      }
    }
    theTrack->SetSafety(safety);
    bool restrictedPhysicalStepLength = false;
    if (BzFieldValue != 0) {
      const double momentumMag = sqrt(eKin * (eKin + 2.0 * restMass));
      // Distance along the track direction to reach the maximum allowed error
      const double safeLength = fieldPropagatorBz.ComputeSafeLength(momentumMag, Charge, dir);

      constexpr int MaxSafeLength = 10;
      double limit                = MaxSafeLength * safeLength;
      limit                       = safety > limit ? safety : limit;

      if (physicalStepLength > limit) {
        physicalStepLength           = limit;
        restrictedPhysicalStepLength = true;
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
    if (BzFieldValue != 0) {
      geometryStepLength = fieldPropagatorBz.ComputeStepAndNextVolume<AdePTNavigator>(
          eKin, restMass, Charge, geometricalStepLengthFromPhysics, pos, dir, navState, nextState, hitsurf_index,
          propagated, safety);
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
        safety                 = currentTrack.GetSafety(pos);
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

    if (!stopped) {
      if (nextState.IsOnBoundary()) {
        // For now, just count that we hit something.
        reached_interaction = false;
        // Kill the particle if it left the world.
        if (!nextState.IsOutside()) {
          // Mark the particle. We need to change its navigation state to the next volume before enqueuing it
          // This will happen after recordinf the step
          cross_boundary = true;
        } else {
          // Particle left the world, don't enqueue it and release the slot
#ifdef ASYNC_MODE
          slotManager.MarkSlotForFreeing(slot);
#endif
        }

      } else if (!propagated || restrictedPhysicalStepLength) {
        // Did not yet reach the interaction point due to error in the magnetic
        // field propagation. Try again next time.
        survive();
        reached_interaction = false;
      } else if (winnerProcessIndex < 0) {
        // No discrete process, move on.
        survive();
        reached_interaction = false;
      }
    }

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

          double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
          double dirSecondary[3];
          G4HepEmElectronInteractionIoni::SampleDirections(eKin, deltaEkin, dirSecondary, dirPrimary, &rnge);

          adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

          // Apply cuts
          if (ApplyCuts && (deltaEkin < theElCut)) {
            // Deposit the energy here and kill the secondary
            energyDeposit += deltaEkin;

          } else {
#ifdef ASYNC_MODE
            Track &secondary = secondaries.electrons.NextTrack(
                newRNG, deltaEkin, pos, vecgeom::Vector3D<Precision>{dirSecondary[0], dirSecondary[1], dirSecondary[2]},
                navState, currentTrack);
#else
            Track &secondary = secondaries.electrons->NextTrack();
            secondary.InitAsSecondary(pos, navState, globalTime);
            secondary.parentId = currentTrack.parentId;
            secondary.rngState = newRNG;
            secondary.eKin = secondary.vertexEkin = deltaEkin;
            secondary.dir.Set(dirSecondary[0], dirSecondary[1], dirSecondary[2]);
            secondary.vertexMomentumDirection.Set(dirSecondary[0], dirSecondary[1], dirSecondary[2]);
#endif
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
#ifdef ASYNC_MODE
            secondaries.gammas.NextTrack(
                newRNG, deltaEkin, pos, vecgeom::Vector3D<Precision>{dirSecondary[0], dirSecondary[1], dirSecondary[2]},
                navState, currentTrack);
#else
            Track &gamma = secondaries.gammas->NextTrack();
            gamma.InitAsSecondary(pos, navState, globalTime);
            gamma.parentId = currentTrack.parentId;
            gamma.rngState = newRNG;
            gamma.eKin = gamma.vertexEkin = deltaEkin;
            gamma.dir.Set(dirSecondary[0], dirSecondary[1], dirSecondary[2]);
            gamma.vertexMomentumDirection.Set(dirSecondary[0], dirSecondary[1], dirSecondary[2]);

#endif
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
#ifdef ASYNC_MODE
            secondaries.gammas.NextTrack(
                newRNG, theGamma1Ekin, pos,
                vecgeom::Vector3D<Precision>{theGamma1Dir[0], theGamma1Dir[1], theGamma1Dir[2]}, navState,
                currentTrack);
#else
            Track &gamma1 = secondaries.gammas->NextTrack();
            gamma1.InitAsSecondary(pos, navState, globalTime);
            gamma1.parentId = currentTrack.parentId;
            gamma1.rngState = newRNG;
            gamma1.eKin = gamma1.vertexEkin = theGamma1Ekin;
            gamma1.dir.Set(theGamma1Dir[0], theGamma1Dir[1], theGamma1Dir[2]);
            gamma1.vertexMomentumDirection.Set(theGamma1Dir[0], theGamma1Dir[1], theGamma1Dir[2]);
#endif
          }
          if (ApplyCuts && (theGamma2Ekin < theGammaCut)) {
            // Deposit the energy here and kill the secondaries
            energyDeposit += theGamma2Ekin;

          } else {
#ifdef ASYNC_MODE
            secondaries.gammas.NextTrack(
                currentTrack.rngState, theGamma2Ekin, pos,
                vecgeom::Vector3D<Precision>{theGamma2Dir[0], theGamma2Dir[1], theGamma2Dir[2]}, navState,
                currentTrack);
#else
            Track &gamma2 = secondaries.gammas->NextTrack();
            gamma2.InitAsSecondary(pos, navState, globalTime);
            // Reuse the RNG state of the dying track. (This is done for efficiency, if the particle is cut
            // the state is not reused, but this shouldn't be an issue)
            gamma2.parentId = currentTrack.parentId;
            gamma2.rngState = currentTrack.rngState;
            gamma2.eKin = gamma2.vertexEkin = theGamma2Ekin;
            gamma2.dir.Set(theGamma2Dir[0], theGamma2Dir[1], theGamma2Dir[2]);
            gamma2.vertexMomentumDirection.Set(theGamma2Dir[0], theGamma2Dir[1], theGamma2Dir[2]);
#endif
          }

          // The current track is killed by not enqueuing into the next activeQueue.
#ifdef ASYNC_MODE
          slotManager.MarkSlotForFreeing(slot);
#endif
          break;
        }
        }
      }
    }

    if (stopped) {
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

          newRNG.Advance();
#ifdef ASYNC_MODE
          Track &gamma1 = secondaries.gammas.NextTrack(newRNG, double{copcore::units::kElectronMassC2}, pos,
                                                       vecgeom::Vector3D<Precision>{sint * cosPhi, sint * sinPhi, cost},
                                                       navState, currentTrack);

          // Reuse the RNG state of the dying track.
          Track &gamma2 = secondaries.gammas.NextTrack(currentTrack.rngState, double{copcore::units::kElectronMassC2},
                                                       pos, -gamma1.dir, navState, currentTrack);
#else
          Track &gamma1 = secondaries.gammas->NextTrack();
          Track &gamma2 = secondaries.gammas->NextTrack();
          gamma1.InitAsSecondary(pos, navState, globalTime);
          newRNG.Advance();
          gamma1.parentId = currentTrack.parentId;
          gamma1.rngState = newRNG;
          gamma1.eKin = gamma1.vertexEkin = copcore::units::kElectronMassC2;
          gamma1.dir.Set(sint * cosPhi, sint * sinPhi, cost);
          gamma1.vertexMomentumDirection.Set(sint * cosPhi, sint * sinPhi, cost);

          gamma2.InitAsSecondary(pos, navState, globalTime);
          // Reuse the RNG state of the dying track.
          gamma2.parentId = currentTrack.parentId;
          gamma2.rngState = currentTrack.rngState;
          gamma2.eKin = gamma2.vertexEkin = copcore::units::kElectronMassC2;
          gamma2.dir                      = -gamma1.dir;
          gamma2.vertexMomentumDirection  = -gamma1.dir;
#endif
        }
      }
      // Particles are killed by not enqueuing them into the new activeQueue (and free the slot in async mode)
#ifdef ASYNC_MODE
      slotManager.MarkSlotForFreeing(slot);
#endif
    }

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps)
      adept_scoring::RecordHit(userScoring, currentTrack.parentId,
                               static_cast<char>(IsElectron ? 0 : 1),        // Particle type
                               elTrack.GetPStepLength(),                     // Step length
                               energyDeposit,                                // Total Edep
                               navState,                                     // Pre-step point navstate
                               preStepPos,                                   // Pre-step point position
                               preStepDir,                                   // Pre-step point momentum direction
                               preStepEnergy,                                // Pre-step point kinetic energy
                               IsElectron ? -1 : 1,                          // Pre-step point charge
                               nextState,                                    // Post-step point navstate
                               pos,                                          // Post-step point position
                               dir,                                          // Post-step point momentum direction
                               eKin,                                         // Post-step point kinetic energy
                               IsElectron ? -1 : 1,                          // Post-step point charge
                               currentTrack.eventId, currentTrack.threadId); // eventID and threadID (not needed here)
    if (cross_boundary) {
      // Move to the next boundary.
      navState = nextState;
      // Check if the next volume belongs to the GPU region and push it to the appropriate queue
#ifndef ADEPT_USE_SURF
      const int nextlvolID = navState.Top()->GetLogicalVolume()->id();
#else
      const int nextlvolID = navState.GetLogicalId();
#endif
#ifdef ASYNC_MODE
      VolAuxData const &nextauxData = AsyncAdePT::gVolAuxData[nextlvolID];
#else
      VolAuxData const &nextauxData = auxDataArray[nextlvolID];
#endif
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
#ifdef ASYNC_MODE
template <typename Scoring>
__global__ void TransportElectrons(Track *electrons, const adept::MParray *active, Secondaries secondaries,
                                   adept::MParray *activeQueue, adept::MParray *leakedQueue, Scoring *userScoring,
                                   bool returnAllSteps)
{
  TransportElectrons</*IsElectron*/ true, Scoring>(electrons, active, secondaries, activeQueue, leakedQueue,
                                                   userScoring, returnAllSteps);
}
template <typename Scoring>
__global__ void TransportPositrons(Track *positrons, const adept::MParray *active, Secondaries secondaries,
                                   adept::MParray *activeQueue, adept::MParray *leakedQueue, Scoring *userScoring,
                                   bool returnAllSteps)
{
  TransportElectrons</*IsElectron*/ false, Scoring>(positrons, active, secondaries, activeQueue, leakedQueue,
                                                    userScoring, returnAllSteps);
}
#else
template <typename Scoring>
__global__ void TransportElectrons(adept::TrackManager<Track> *electrons, Secondaries secondaries,
                                   MParrayTracks *leakedQueue, Scoring *userScoring, VolAuxData const *auxDataArray)
{
  TransportElectrons</*IsElectron*/ true, Scoring>(electrons, secondaries, leakedQueue, userScoring, auxDataArray);
}
template <typename Scoring>
__global__ void TransportPositrons(adept::TrackManager<Track> *positrons, Secondaries secondaries,
                                   MParrayTracks *leakedQueue, Scoring *userScoring, VolAuxData const *auxDataArray)
{
  TransportElectrons</*IsElectron*/ false, Scoring>(positrons, secondaries, leakedQueue, userScoring, auxDataArray);
}
#endif

#ifdef ASYNC_MODE
} // namespace AsyncAdePT
#endif
