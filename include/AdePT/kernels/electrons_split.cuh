// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/core/AdePTTransportStruct.cuh>
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

template <bool IsElectron>
__global__ void ElectronHowFar(adept::TrackManager<Track> *electrons, G4HepEmElectronTrack *hepEMTracks,
                               VolAuxData const *auxDataArray)
{
  constexpr int Charge      = IsElectron ? -1 : 1;
  constexpr double restMass = copcore::units::kElectronMassC2;
  fieldPropagatorConstBz fieldPropagatorBz(BzFieldValue);

  int activeSize = electrons->fActiveTracks->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot               = (*electrons->fActiveTracks)[i];
    Track &currentTrack          = (*electrons)[slot];
    currentTrack.preStepEKin     = currentTrack.eKin;
    currentTrack.preStepPos      = currentTrack.pos;
    currentTrack.preStepDir      = currentTrack.dir;
    currentTrack.preStepNavState = currentTrack.navState;
    // the MCC vector is indexed by the logical volume id
#ifndef ADEPT_USE_SURF
    const int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    const int lvolID = currentTrack.navState.GetLogicalId();
#endif
    VolAuxData const &auxData = auxDataArray[lvolID];

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    elTrack.ReSet();
    G4HepEmTrack *theTrack = elTrack.GetTrack();
    theTrack->SetEKin(currentTrack.eKin);
    theTrack->SetMCIndex(auxData.fMCIndex);
    theTrack->SetOnBoundary(currentTrack.navState.IsOnBoundary());
    theTrack->SetCharge(Charge);
    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    mscData->fIsFirstStep        = currentTrack.initialRange < 0;
    mscData->fInitialRange       = currentTrack.initialRange;
    mscData->fDynamicRangeFactor = currentTrack.dynamicRangeFactor;
    mscData->fTlimitMin          = currentTrack.tlimitMin;

    // Prepare a branched RNG state while threads are synchronized. Even if not
    // used, this provides a fresh round of random numbers and reduces thread
    // divergence because the RNG state doesn't need to be advanced later.
    currentTrack.newRNG = RanluxppDouble(currentTrack.rngState.BranchNoAdvance());

    // Compute safety, needed for MSC step limit.
    currentTrack.safety = 0;
    if (!currentTrack.navState.IsOnBoundary()) {
      currentTrack.safety = AdePTNavigator::ComputeSafety(currentTrack.pos, currentTrack.navState);
    }
    theTrack->SetSafety(currentTrack.safety);

    G4HepEmRandomEngine rnge(&currentTrack.rngState);

    // Sample the `number-of-interaction-left` and put it into the track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft = currentTrack.numIALeft[ip];
      if (numIALeft <= 0) {
        numIALeft = -std::log(currentTrack.Uniform());
      }
      theTrack->SetNumIALeft(numIALeft, ip);
    }

    G4HepEmElectronManager::HowFarToDiscreteInteraction(&g4HepEmData, &g4HepEmPars, &elTrack);

    currentTrack.restrictedPhysicalStepLength = false;
    if (BzFieldValue != 0) {
      const double momentumMag = sqrt(currentTrack.eKin * (currentTrack.eKin + 2.0 * restMass));
      // Distance along the track direction to reach the maximum allowed error
      const double safeLength = fieldPropagatorBz.ComputeSafeLength(momentumMag, Charge, currentTrack.dir);

      constexpr int MaxSafeLength = 10;
      double limit                = MaxSafeLength * safeLength;
      limit                       = currentTrack.safety > limit ? currentTrack.safety : limit;

      double physicalStepLength = elTrack.GetPStepLength();
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
static __global__ void ElectronPropagation(adept::TrackManager<Track> *electrons, G4HepEmElectronTrack *hepEMTracks)
{
#ifdef VECGEOM_FLOAT_PRECISION
  const Precision kPush = 10 * vecgeom::kTolerance;
#else
  const Precision kPush = 0.;
#endif
  constexpr int Charge      = IsElectron ? -1 : 1;
  constexpr double restMass = copcore::units::kElectronMassC2;
  fieldPropagatorConstBz fieldPropagatorBz(BzFieldValue);

  int activeSize = electrons->fActiveTracks->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*electrons->fActiveTracks)[i];
    Track &currentTrack = (*electrons)[slot];

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    // Check if there's a volume boundary in between.
    currentTrack.propagated = true;
    currentTrack.hitsurfID  = -1;
    // double geometryStepLength;
    // vecgeom::NavigationState nextState;
    if (BzFieldValue != 0) {
      currentTrack.geometryStepLength = fieldPropagatorBz.ComputeStepAndNextVolume<AdePTNavigator>(
          currentTrack.eKin, restMass, Charge, theTrack->GetGStepLength(), currentTrack.pos, currentTrack.dir,
          currentTrack.navState, currentTrack.nextState, currentTrack.hitsurfID, currentTrack.propagated,
          currentTrack.safety);
    } else {
#ifdef ADEPT_USE_SURF
      currentTrack.geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(
          currentTrack.pos, currentTrack.dir, theTrack->GetGStepLength(), currentTrack.navState, currentTrack.nextState,
          currentTrack.hitsurfID, kPush);
#else
      currentTrack.geometryStepLength =
          AdePTNavigator::ComputeStepAndNextVolume(currentTrack.pos, currentTrack.dir, theTrack->GetGStepLength(),
                                                   currentTrack.navState, currentTrack.nextState, kPush);
#endif
      currentTrack.pos += currentTrack.geometryStepLength * currentTrack.dir;
    }

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    currentTrack.navState.SetBoundaryState(currentTrack.nextState.IsOnBoundary());

    // Propagate information from geometrical step to MSC.
    theTrack->SetDirection(currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z());
    theTrack->SetGStepLength(currentTrack.geometryStepLength);
    theTrack->SetOnBoundary(currentTrack.nextState.IsOnBoundary());
  }
}

// May work well to separate MSC1 (Continuous effects) and MSC2 (Checks + safety)
template <bool IsElectron>
static __global__ void ElectronMSC(adept::TrackManager<Track> *electrons, G4HepEmElectronTrack *hepEMTracks)
{
  constexpr double restMass = copcore::units::kElectronMassC2;
  int activeSize            = electrons->fActiveTracks->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*electrons->fActiveTracks)[i];
    Track &currentTrack = (*electrons)[slot];

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();
    G4HepEmMSCTrackData *mscData  = elTrack.GetMSCTrackData();

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
        currentTrack.safety -= currentTrack.geometryStepLength;
        constexpr double sFact = 0.99;
        double reducedSafety   = sFact * currentTrack.safety;

        // Apply displacement, depending on how close we are to a boundary.
        // 1a. Far away from geometry boundary:
        if (reducedSafety > 0.0 && dispR <= reducedSafety) {
          currentTrack.pos += displacement;
        } else {
          // Recompute safety.
          currentTrack.safety = AdePTNavigator::ComputeSafety(currentTrack.pos, currentTrack.navState);
          reducedSafety       = sFact * currentTrack.safety;

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

template <bool IsElectron>
static __global__ void ElectronRelocation(adept::TrackManager<Track> *electrons)
{
  int activeSize = electrons->fActiveTracks->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*electrons->fActiveTracks)[i];
    Track &currentTrack = (*electrons)[slot];

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
  }
}

template <bool IsElectron, typename Scoring>
static __global__ void ElectronInteractions(adept::TrackManager<Track> *electrons, G4HepEmElectronTrack *hepEMTracks,
                                            Secondaries secondaries, MParrayTracks *leakedQueue, Scoring *userScoring,
                                            VolAuxData const *auxDataArray)
{
  constexpr Precision kPushOutRegion = 10 * vecgeom::kTolerance;
  constexpr int Pdg                  = IsElectron ? 11 : -11;
  fieldPropagatorConstBz fieldPropagatorBz(BzFieldValue);

  int activeSize = electrons->fActiveTracks->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*electrons->fActiveTracks)[i];
    Track &currentTrack = (*electrons)[slot];
    adeptint::TrackData trackdata;
    // the MCC vector is indexed by the logical volume id
#ifndef ADEPT_USE_SURF
    const int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    const int lvolID = currentTrack.navState.GetLogicalId();
#endif
    VolAuxData const &auxData = auxDataArray[lvolID];

    auto survive = [&](bool leak = false) {
      currentTrack.CopyTo(trackdata, Pdg);
      if (leak)
        leakedQueue->push_back(trackdata);
      else
        electrons->fNextTracks->push_back(slot);
    };

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();
    G4HepEmMSCTrackData *mscData  = elTrack.GetMSCTrackData();

    G4HepEmRandomEngine rnge(&currentTrack.rngState);

    if (auxData.fSensIndex >= 0)
      adept_scoring::RecordHit(userScoring, currentTrack.parentID,
                               IsElectron ? 0 : 1,            // Particle type
                               elTrack.GetPStepLength(),      // Step length
                               theTrack->GetEnergyDeposit(),  // Total Edep
                               &currentTrack.preStepNavState, // Pre-step point navstate
                               &currentTrack.preStepPos,      // Pre-step point position
                               &currentTrack.preStepDir,      // Pre-step point momentum direction
                               nullptr,                       // Pre-step point polarization
                               currentTrack.preStepEKin,      // Pre-step point kinetic energy
                               IsElectron ? -1 : 1,           // Pre-step point charge
                               &currentTrack.navState,        // Post-step point navstate
                               &currentTrack.pos,             // Post-step point position
                               &currentTrack.dir,             // Post-step point momentum direction
                               nullptr,                       // Post-step point polarization
                               currentTrack.eKin,             // Post-step point kinetic energy
                               IsElectron ? -1 : 1);          // Post-step point charge

    // Save the `number-of-interaction-left` in our track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft           = theTrack->GetNumIALeft(ip);
      currentTrack.numIALeft[ip] = numIALeft;
    }

    if (currentTrack.stopped) {
      if (!IsElectron) {
        // Annihilate the stopped positron into two gammas heading to opposite
        // directions (isotropic).
        Track &gamma1 = secondaries.gammas->NextTrack();
        Track &gamma2 = secondaries.gammas->NextTrack();

        adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 0, /*numPositrons*/ 0, /*numGammas*/ 2);

        const double cost = 2 * currentTrack.Uniform() - 1;
        const double sint = sqrt(1 - cost * cost);
        const double phi  = k2Pi * currentTrack.Uniform();
        double sinPhi, cosPhi;
        sincos(phi, &sinPhi, &cosPhi);

        gamma1.InitAsSecondary(currentTrack.pos, currentTrack.navState, currentTrack.globalTime);
        currentTrack.newRNG.Advance();
        gamma1.parentID = currentTrack.parentID;
        gamma1.rngState = currentTrack.newRNG;
        gamma1.eKin     = copcore::units::kElectronMassC2;
        gamma1.dir.Set(sint * cosPhi, sint * sinPhi, cost);

        gamma2.InitAsSecondary(currentTrack.pos, currentTrack.navState, currentTrack.globalTime);
        // Reuse the RNG state of the dying track.
        gamma2.parentID = currentTrack.parentID;
        gamma2.rngState = currentTrack.rngState;
        gamma2.eKin     = copcore::units::kElectronMassC2;
        gamma2.dir      = -gamma1.dir;
      }
      // Particles are killed by not enqueuing them into the new activeQueue.
      continue;
    }

    if (currentTrack.nextState.IsOnBoundary()) {
      // For now, just count that we hit something.

      // Kill the particle if it left the world.
      if (!currentTrack.nextState.IsOutside()) {

        // Move to the next boundary.
        currentTrack.navState = currentTrack.nextState;
        // Check if the next volume belongs to the GPU region and push it to the appropriate queue
#ifndef ADEPT_USE_SURF
        const int nextlvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
        const int nextlvolID = currentTrack.navState.GetLogicalId();
#endif
        VolAuxData const &nextauxData = auxDataArray[nextlvolID];
        if (nextauxData.fGPUregion > 0)
          survive();
        else {
          // To be safe, just push a bit the track exiting the GPU region to make sure
          // Geant4 does not relocate it again inside the same region
          currentTrack.pos += kPushOutRegion * currentTrack.dir;
          survive(/*leak*/ true);
        }
      }
      continue;
    } else if (!currentTrack.propagated || currentTrack.restrictedPhysicalStepLength) {
      // Did not yet reach the interaction point due to error in the magnetic
      // field propagation. Try again next time.
      survive();
      continue;
    } else if (theTrack->GetWinnerProcessIndex() < 0) {
      // No discrete process, move on.
      survive();
      continue;
    }

    // Reset number of interaction left for the winner discrete process.
    // (Will be resampled in the next iteration.)
    currentTrack.numIALeft[theTrack->GetWinnerProcessIndex()] = -1.0;

    // Check if a delta interaction happens instead of the real discrete process.
    if (G4HepEmElectronManager::CheckDelta(&g4HepEmData, theTrack, currentTrack.Uniform())) {
      // A delta interaction happened, move on.
      survive();
      continue;
    }

    // Perform the discrete interaction, make sure the branched RNG state is
    // ready to be used.
    currentTrack.newRNG.Advance();
    // Also advance the current RNG state to provide a fresh round of random
    // numbers after MSC used up a fair share for sampling the displacement.
    currentTrack.rngState.Advance();

    const double theElCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecElProdCutE;

    switch (theTrack->GetWinnerProcessIndex()) {
    case 0: {
      // Invoke ionization (for e-/e+):
      double deltaEkin =
          (IsElectron) ? G4HepEmElectronInteractionIoni::SampleETransferMoller(theElCut, currentTrack.eKin, &rnge)
                       : G4HepEmElectronInteractionIoni::SampleETransferBhabha(theElCut, currentTrack.eKin, &rnge);

      double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirSecondary[3];
      G4HepEmElectronInteractionIoni::SampleDirections(currentTrack.eKin, deltaEkin, dirSecondary, dirPrimary, &rnge);

      Track &secondary = secondaries.electrons->NextTrack();

      adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

      secondary.InitAsSecondary(currentTrack.pos, currentTrack.navState, currentTrack.globalTime);
      secondary.parentID = currentTrack.parentID;
      secondary.rngState = currentTrack.newRNG;
      secondary.eKin     = deltaEkin;
      secondary.dir.Set(dirSecondary[0], dirSecondary[1], dirSecondary[2]);

      currentTrack.eKin -= deltaEkin;
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
      G4HepEmElectronInteractionBrem::SampleDirections(currentTrack.eKin, deltaEkin, dirSecondary, dirPrimary, &rnge);

      Track &gamma = secondaries.gammas->NextTrack();
      adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 0, /*numPositrons*/ 0, /*numGammas*/ 1);

      gamma.InitAsSecondary(currentTrack.pos, currentTrack.navState, currentTrack.globalTime);
      gamma.parentID = currentTrack.parentID;
      gamma.rngState = currentTrack.newRNG;
      gamma.eKin     = deltaEkin;
      gamma.dir.Set(dirSecondary[0], dirSecondary[1], dirSecondary[2]);

      currentTrack.eKin -= deltaEkin;
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

      Track &gamma1 = secondaries.gammas->NextTrack();
      Track &gamma2 = secondaries.gammas->NextTrack();
      adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 0, /*numPositrons*/ 0, /*numGammas*/ 2);

      gamma1.InitAsSecondary(currentTrack.pos, currentTrack.navState, currentTrack.globalTime);
      gamma1.parentID = currentTrack.parentID;
      gamma1.rngState = currentTrack.newRNG;
      gamma1.eKin     = theGamma1Ekin;
      gamma1.dir.Set(theGamma1Dir[0], theGamma1Dir[1], theGamma1Dir[2]);

      gamma2.InitAsSecondary(currentTrack.pos, currentTrack.navState, currentTrack.globalTime);
      // Reuse the RNG state of the dying track.
      gamma2.parentID = currentTrack.parentID;
      gamma2.rngState = currentTrack.rngState;
      gamma2.eKin     = theGamma2Ekin;
      gamma2.dir.Set(theGamma2Dir[0], theGamma2Dir[1], theGamma2Dir[2]);

      // The current track is killed by not enqueuing into the next activeQueue.
      break;
    }
    }
  }
}
