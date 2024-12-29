// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/core/AdePTTransportStruct.cuh>
#include <AdePT/navigation/AdePTNavigator.h>

#include <AdePT/copcore/PhysicalConstants.h>

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

using VolAuxData = adeptint::VolAuxData;

__global__ void GammaHowFar(adept::TrackManager<Track> *gammas, G4HepEmGammaTrack *hepEMTracks,
                            VolAuxData const *auxDataArray)
{
  int activeSize = gammas->fActiveTracks->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*gammas->fActiveTracks)[i];
    Track &currentTrack = (*gammas)[slot];
    // Save current values for scoring
    currentTrack.preStepEKin     = currentTrack.eKin;
    currentTrack.preStepPos      = currentTrack.pos;
    currentTrack.preStepDir      = currentTrack.dir;
    currentTrack.preStepNavState = currentTrack.navState;
    // the MCC vector is indexed by the logical volume id
#ifndef ADEPT_USE_SURF
    int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    int lvolID = currentTrack.navState.GetLogicalId();
#endif
    VolAuxData const &auxData = auxDataArray[lvolID];

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = gammaTrack.GetTrack();
    theTrack->SetEKin(currentTrack.eKin);
    theTrack->SetMCIndex(auxData.fMCIndex);

    // Sample the `number-of-interaction-left` and put it into the track.
    // Use index 0 since numIALeft for gammas is based only on the total macroscopic cross section
    if (theTrack->GetNumIALeft(0) <= 0.0) {
      theTrack->SetNumIALeft(-std::log(currentTrack.Uniform()), 0);
    }

    // Call G4HepEm to compute the physics step limit.
    G4HepEmGammaManager::HowFar(&adept_impl::g4HepEmData, &adept_impl::g4HepEmPars, &gammaTrack);
    G4HepEmGammaManager::SampleInteraction(&adept_impl::g4HepEmData, &gammaTrack, currentTrack.Uniform());

    // Skip electron/positron-nuclear reaction that would need to be handled by G4 itself
    if (theTrack->GetWinnerProcessIndex() == 3) {
      theTrack->SetWinnerProcessIndex(-1);
      assert(0); // currently, the gamma-nuclear processes are not registered in the AdePTPhysicsList, so they should
                 // never be hit.
    }
  }
}

__global__ void GammaPropagation(adept::TrackManager<Track> *gammas, G4HepEmGammaTrack *hepEMTracks,
                                 VolAuxData const *auxDataArray)
{
#ifdef VECGEOM_FLOAT_PRECISION
  const Precision kPush = 10 * vecgeom::kTolerance;
#else
  const Precision kPush = 0.;
#endif
  int activeSize = gammas->fActiveTracks->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*gammas->fActiveTracks)[i];
    Track &currentTrack = (*gammas)[slot];
    // Retrieve the HepEM Track
    G4HepEmTrack *theTrack = hepEMTracks[slot].GetTrack();

    // Check if there's a volume boundary in between.
    vecgeom::NavigationState nextState;
    double geometryStepLength;
#ifdef ADEPT_USE_SURF
    long hitsurf_index = -1;
    geometryStepLength =
        AdePTNavigator::ComputeStepAndNextVolume(currentTrack.pos, currentTrack.dir, theTrack->GetGStepLength(),
                                                 currentTrack.navState, nextState, hitsurf_index, kPush);
    currentTrack.hitsurfID = hitsurf_index;
#else
    geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(
        currentTrack.pos, currentTrack.dir, theTrack->GetGStepLength(), currentTrack.navState, nextState, kPush);
#endif
    currentTrack.pos += geometryStepLength * currentTrack.dir;

    // Store the actual geometrical step length traveled
    currentTrack.geometryStepLength = geometryStepLength;

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    currentTrack.navState.SetBoundaryState(nextState.IsOnBoundary());

    // Propagate information from geometrical step to G4HepEm.
    theTrack->SetGStepLength(geometryStepLength);
    theTrack->SetOnBoundary(nextState.IsOnBoundary());

    // Update the number-of-interaction-left
    G4HepEmGammaManager::UpdateNumIALeft(theTrack);

    // Save the `number-of-interaction-left` in our track.
    // Use index 0 since numIALeft for gammas is based only on the total macroscopic cross section
    double numIALeft          = theTrack->GetNumIALeft(0);
    currentTrack.numIALeft[0] = numIALeft;

    // Update the flight times of the particle
    double deltaTime = theTrack->GetGStepLength() / copcore::units::kCLight;
    currentTrack.globalTime += deltaTime;
    currentTrack.localTime += deltaTime;

    // Save the next state in the track
    currentTrack.nextState = nextState;
  }
}

__global__ void GammaRelocation(adept::TrackManager<Track> *gammas, MParrayTracks *leakedQueue,
                                VolAuxData const *auxDataArray)
{
  constexpr Precision kPushOutRegion = 10 * vecgeom::kTolerance;
  constexpr int Pdg                  = 22;
  int activeSize                     = gammas->fActiveTracks->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*gammas->fActiveTracks)[i];
    Track &currentTrack = (*gammas)[slot];

    auto survive = [&](bool leak = false) {
      adeptint::TrackData trackdata;
      currentTrack.CopyTo(trackdata, Pdg);
      if (leak)
        leakedQueue->push_back(trackdata);
      else
        gammas->fNextTracks->push_back(slot);
    };

    // Last transportation call, in case we are on a boundary we need to relocate to the next volume
    // Every track on a boundary will stop being processed here (no interaction)
    // For now (experimental kernels) I just mark it in the track, and in the next kernel we just
    // return inmediately (but ideally we won't even give those tracks to the kernel)
    if (currentTrack.nextState.IsOnBoundary()) {
      currentTrack.restrictedPhysicalStepLength = true;
      // Kill the particle if it left the world.
      if (!currentTrack.nextState.IsOutside()) {
#ifdef ADEPT_USE_SURF
        AdePTNavigator::RelocateToNextVolume(currentTrack.pos, currentTrack.dir, currentTrack.hitsurfID,
                                             currentTrack.nextState);
        if (currentTrack.nextState.IsOutside()) continue;
#else
        AdePTNavigator::RelocateToNextVolume(currentTrack.pos, currentTrack.dir, currentTrack.nextState);
#endif
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
    } else {
      currentTrack.restrictedPhysicalStepLength = false;
    }
  }
}

template <typename Scoring>
__global__ void GammaInteractions(adept::TrackManager<Track> *gammas, G4HepEmGammaTrack *hepEMTracks,
                                  Secondaries secondaries, Scoring *userScoring, VolAuxData const *auxDataArray)
{
  int activeSize = gammas->fActiveTracks->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*gammas->fActiveTracks)[i];
    Track &currentTrack = (*gammas)[slot];
    // Retrieve the HepEM Track
    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = gammaTrack.GetTrack();

#ifndef ADEPT_USE_SURF
    int lvolID = currentTrack.navState.Top()->GetLogicalVolume()->id();
#else
    int lvolID = currentTrack.navState.GetLogicalId();
#endif
    VolAuxData const &auxData = auxDataArray[lvolID];

    auto survive = [&]() { gammas->fNextTracks->push_back(slot); };

    // Temporary solution
    if (currentTrack.restrictedPhysicalStepLength) {
      continue;
    }

    if (theTrack->GetWinnerProcessIndex() < 0) {
      continue;
    }

    // Reset number of interaction left for the winner discrete process.
    // (Will be resampled in the next iteration.)
    currentTrack.numIALeft[theTrack->GetWinnerProcessIndex()] = -1.0;

    // Perform the discrete interaction.
    G4HepEmRandomEngine rnge(&currentTrack.rngState);
    // We might need one branched RNG state, prepare while threads are synchronized.
    RanluxppDouble newRNG(currentTrack.rngState.Branch());

    switch (theTrack->GetWinnerProcessIndex()) {
    case 0: {
      // Invoke gamma conversion to e-/e+ pairs, if the energy is above the threshold.
      if (currentTrack.eKin < 2 * copcore::units::kElectronMassC2) {
        survive();
        continue;
      }

      double logEnergy = std::log(currentTrack.eKin);
      double elKinEnergy, posKinEnergy;
      G4HepEmGammaInteractionConversion::SampleKinEnergies(&adept_impl::g4HepEmData, currentTrack.eKin, logEnergy,
                                                           auxData.fMCIndex, elKinEnergy, posKinEnergy, &rnge);

      double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirSecondaryEl[3], dirSecondaryPos[3];
      G4HepEmGammaInteractionConversion::SampleDirections(dirPrimary, dirSecondaryEl, dirSecondaryPos, elKinEnergy,
                                                          posKinEnergy, &rnge);

      Track &electron = secondaries.electrons->NextTrack();
      Track &positron = secondaries.positrons->NextTrack();

      adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 1, /*numGammas*/ 0);

      electron.InitAsSecondary(currentTrack.pos, currentTrack.navState, currentTrack.globalTime);
      electron.parentID = currentTrack.parentID;
      electron.rngState = newRNG;
      electron.eKin     = elKinEnergy;
      electron.dir.Set(dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]);

      positron.InitAsSecondary(currentTrack.pos, currentTrack.navState, currentTrack.globalTime);
      // Reuse the RNG state of the dying track.
      positron.parentID = currentTrack.parentID;
      positron.rngState = currentTrack.rngState;
      positron.eKin     = posKinEnergy;
      positron.dir.Set(dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]);

      // The current track is killed by not enqueuing into the next activeQueue.
      break;
    }
    case 1: {
      // Invoke Compton scattering of gamma.
      constexpr double LowEnergyThreshold = 100 * copcore::units::eV;
      if (currentTrack.eKin < LowEnergyThreshold) {
        survive();
        continue;
      }
      const double origDirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirPrimary[3];
      const double newEnergyGamma = G4HepEmGammaInteractionCompton::SamplePhotonEnergyAndDirection(
          currentTrack.eKin, dirPrimary, origDirPrimary, &rnge);
      vecgeom::Vector3D<double> newDirGamma(dirPrimary[0], dirPrimary[1], dirPrimary[2]);

      const double energyEl = currentTrack.eKin - newEnergyGamma;
      if (energyEl > LowEnergyThreshold) {
        // Create a secondary electron and sample/compute directions.
        Track &electron = secondaries.electrons->NextTrack();
        adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

        electron.InitAsSecondary(currentTrack.pos, currentTrack.navState, currentTrack.globalTime);
        electron.parentID = currentTrack.parentID;
        electron.rngState = newRNG;
        electron.eKin     = energyEl;
        electron.dir      = currentTrack.eKin * currentTrack.dir - newEnergyGamma * newDirGamma;
        electron.dir.Normalize();
      } else {
        if (auxData.fSensIndex >= 0)
          adept_scoring::RecordHit(userScoring,
                                   currentTrack.parentID,           // Track ID
                                   2,                               // Particle type
                                   currentTrack.geometryStepLength, // Step length
                                   0,                               // Total Edep
                                   &currentTrack.preStepNavState,   // Pre-step point navstate
                                   &currentTrack.preStepPos,        // Pre-step point position
                                   &currentTrack.preStepDir,        // Pre-step point momentum direction
                                   nullptr,                         // Pre-step point polarization
                                   currentTrack.preStepEKin,        // Pre-step point kinetic energy
                                   0,                               // Pre-step point charge
                                   &currentTrack.navState,          // Post-step point navstate
                                   &currentTrack.pos,               // Post-step point position
                                   &currentTrack.dir,               // Post-step point momentum direction
                                   nullptr,                         // Post-step point polarization
                                   newEnergyGamma,                  // Post-step point kinetic energy
                                   0,                               // Post-step point charge
                                   0, -1);                          // eventID and threadID (not needed here)
      }

      // Check the new gamma energy and deposit if below threshold.
      if (newEnergyGamma > LowEnergyThreshold) {
        currentTrack.eKin = newEnergyGamma;
        currentTrack.dir  = newDirGamma;
        survive();
      } else {
        if (auxData.fSensIndex >= 0)
          adept_scoring::RecordHit(userScoring,
                                   currentTrack.parentID,           // Track ID
                                   2,                               // Particle type
                                   currentTrack.geometryStepLength, // Step length
                                   0,                               // Total Edep
                                   &currentTrack.preStepNavState,   // Pre-step point navstate
                                   &currentTrack.preStepPos,        // Pre-step point position
                                   &currentTrack.preStepDir,        // Pre-step point momentum direction
                                   nullptr,                         // Pre-step point polarization
                                   currentTrack.preStepEKin,        // Pre-step point kinetic energy
                                   0,                               // Pre-step point charge
                                   &currentTrack.navState,          // Post-step point navstate
                                   &currentTrack.pos,               // Post-step point position
                                   &currentTrack.dir,               // Post-step point momentum direction
                                   nullptr,                         // Post-step point polarization
                                   newEnergyGamma,                  // Post-step point kinetic energy
                                   0,                               // Post-step point charge
                                   0, -1);                          // eventID and threadID (not needed here)

        // The current track is killed by not enqueuing into the next activeQueue.
      }
      break;
    }
    case 2: {
      // Invoke photoelectric process.
      const double theLowEnergyThreshold = 1 * copcore::units::eV;

      const double bindingEnergy = G4HepEmGammaInteractionPhotoelectric::SelectElementBindingEnergy(
          &adept_impl::g4HepEmData, auxData.fMCIndex, gammaTrack.GetPEmxSec(), currentTrack.eKin, &rnge);
      double edep = bindingEnergy;

      const double photoElecE = currentTrack.eKin - edep;
      if (photoElecE > theLowEnergyThreshold) {
        // Create a secondary electron and sample directions.
        Track &electron = secondaries.electrons->NextTrack();
        adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

        double dirGamma[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
        double dirPhotoElec[3];
        G4HepEmGammaInteractionPhotoelectric::SamplePhotoElectronDirection(photoElecE, dirGamma, dirPhotoElec, &rnge);

        electron.InitAsSecondary(currentTrack.pos, currentTrack.navState, currentTrack.globalTime);
        electron.parentID = currentTrack.parentID;
        electron.rngState = newRNG;
        electron.eKin     = photoElecE;
        electron.dir.Set(dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]);
      } else {
        edep = currentTrack.eKin;
      }

      if (auxData.fSensIndex >= 0)
        adept_scoring::RecordHit(userScoring,
                                 currentTrack.parentID,           // Track ID
                                 2,                               // Particle type
                                 currentTrack.geometryStepLength, // Step length
                                 edep,                            // Total Edep
                                 &currentTrack.preStepNavState,   // Pre-step point navstate
                                 &currentTrack.preStepPos,        // Pre-step point position
                                 &currentTrack.preStepDir,        // Pre-step point momentum direction
                                 nullptr,                         // Pre-step point polarization
                                 currentTrack.preStepEKin,        // Pre-step point kinetic energy
                                 0,                               // Pre-step point charge
                                 &currentTrack.navState,          // Post-step point navstate
                                 &currentTrack.pos,               // Post-step point position
                                 &currentTrack.dir,               // Post-step point momentum direction
                                 nullptr,                         // Post-step point polarization
                                 0,                               // Post-step point kinetic energy
                                 0,                               // Post-step point charge
                                 0, -1);                          // eventID and threadID (not needed here)

      // The current track is killed by not enqueuing into the next activeQueue.
      break;
    }
    }
  }
}
