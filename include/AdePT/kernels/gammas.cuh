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

template <typename Scoring>
__global__ void TransportGammas(adept::TrackManager<Track> *gammas, Secondaries secondaries, MParrayTracks *leakedQueue,
                                Scoring *userScoring, VolAuxData const *auxDataArray)
{
#ifdef VECGEOM_FLOAT_PRECISION
  const Precision kPush = 10 * vecgeom::kTolerance;
#else
  const Precision kPush = 0.;
#endif
  constexpr Precision kPushOutRegion = 10 * vecgeom::kTolerance;
  constexpr int Pdg                  = 22;
  int activeSize                     = gammas->fActiveTracks->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*gammas->fActiveTracks)[i];
    Track &currentTrack = (*gammas)[slot];
    auto eKin           = currentTrack.eKin;
    auto preStepEnergy  = eKin;
    auto pos            = currentTrack.pos;
    vecgeom::Vector3D<Precision> preStepPos(pos);
    auto dir = currentTrack.dir;
    vecgeom::Vector3D<Precision> preStepDir(dir);
    double globalTime = currentTrack.globalTime;
    double localTime  = currentTrack.localTime;
    double properTime = currentTrack.properTime;
    auto navState     = currentTrack.navState;
    adeptint::TrackData trackdata;
    // the MCC vector is indexed by the logical volume id
#ifndef ADEPT_USE_SURF
    int lvolID = navState.Top()->GetLogicalVolume()->id();
#else
    int lvolID = navState.GetLogicalId();
#endif
    VolAuxData const &auxData = auxDataArray[lvolID];

    auto survive = [&](bool leak = false) {
      currentTrack.eKin       = eKin;
      currentTrack.pos        = pos;
      currentTrack.dir        = dir;
      currentTrack.globalTime = globalTime;
      currentTrack.localTime  = localTime;
      currentTrack.properTime = properTime;
      currentTrack.navState   = navState;
      currentTrack.CopyTo(trackdata, Pdg);
      if (leak)
        leakedQueue->push_back(trackdata);
      else
        gammas->fNextTracks->push_back(slot);
    };

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmGammaTrack gammaTrack;
    G4HepEmTrack *theTrack = gammaTrack.GetTrack();
    theTrack->SetEKin(eKin);
    theTrack->SetMCIndex(auxData.fMCIndex);

    // Sample the `number-of-interaction-left` and put it into the track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft = currentTrack.numIALeft[ip];
      if (numIALeft <= 0) {
        numIALeft = -std::log(currentTrack.Uniform());
      }
      theTrack->SetNumIALeft(numIALeft, ip);
    }

    // Call G4HepEm to compute the physics step limit.
    G4HepEmGammaManager::HowFar(&g4HepEmData, &g4HepEmPars, &gammaTrack);

    // Get result into variables.
    double geometricalStepLengthFromPhysics = theTrack->GetGStepLength();
    int winnerProcessIndex                  = theTrack->GetWinnerProcessIndex();
    // Leave the range and MFP inside the G4HepEmTrack. If we split kernels, we
    // also need to carry them over!

    // Check if there's a volume boundary in between.
    vecgeom::NavigationState nextState;
    double geometryStepLength;
#ifdef ADEPT_USE_SURF
    long hitsurf_index = -1;
    geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics, navState,
                                                                  nextState, hitsurf_index, kPush);
#else
    geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics, navState,
                                                                  nextState, kPush);
#endif
    //  printf("pvol=%d  step=%g  onboundary=%d  pos={%g, %g, %g}  dir={%g, %g, %g}\n", navState.TopId(),
    //  geometryStepLength,
    //         nextState.IsOnBoundary(), pos[0], pos[1], pos[2], dir[0], dir[1], dir[2]);
    pos += geometryStepLength * dir;

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    navState.SetBoundaryState(nextState.IsOnBoundary());

    // Propagate information from geometrical step to G4HepEm.
    theTrack->SetGStepLength(geometryStepLength);
    theTrack->SetOnBoundary(nextState.IsOnBoundary());

    G4HepEmGammaManager::UpdateNumIALeft(theTrack);

    // Save the `number-of-interaction-left` in our track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft           = theTrack->GetNumIALeft(ip);
      currentTrack.numIALeft[ip] = numIALeft;
    }

    if (nextState.IsOnBoundary()) {
      // For now, just count that we hit something.

      // Kill the particle if it left the world.
      if (!nextState.IsOutside()) {
#ifdef ADEPT_USE_SURF
        AdePTNavigator::RelocateToNextVolume(pos, dir, hitsurf_index, nextState);
        if (nextState.IsOutside()) continue;
#else
        AdePTNavigator::RelocateToNextVolume(pos, dir, nextState);
#endif
        // Move to the next boundary.
        navState = nextState;
        // printf("  -> pvol=%d pos={%g, %g, %g} \n", navState.TopId(), pos[0], pos[1], pos[2]);
        //  Check if the next volume belongs to the GPU region and push it to the appropriate queue
#ifndef ADEPT_USE_SURF
        const int nextlvolID = navState.Top()->GetLogicalVolume()->id();
#else
        const int nextlvolID = navState.GetLogicalId();
#endif
        VolAuxData const &nextauxData = auxDataArray[nextlvolID];
        if (nextauxData.fGPUregion > 0)
          survive();
        else {
          // To be safe, just push a bit the track exiting the GPU region to make sure
          // Geant4 does not relocate it again inside the same region
          pos += kPushOutRegion * dir;
          survive(/*leak*/ true);
        }
      }
      continue;
    } else if (winnerProcessIndex < 0) {
      // No discrete process, move on.
      survive();
      continue;
    }

    // Reset number of interaction left for the winner discrete process.
    // (Will be resampled in the next iteration.)
    currentTrack.numIALeft[winnerProcessIndex] = -1.0;

    // Update the flight times of the particle
    double deltaTime = theTrack->GetGStepLength() / copcore::units::kCLight;
    globalTime += deltaTime;
    localTime += deltaTime;

    // Perform the discrete interaction.
    G4HepEmRandomEngine rnge(&currentTrack.rngState);
    // We might need one branched RNG state, prepare while threads are synchronized.
    RanluxppDouble newRNG(currentTrack.rngState.Branch());

    switch (winnerProcessIndex) {
    case 0: {
      // Invoke gamma conversion to e-/e+ pairs, if the energy is above the threshold.
      if (eKin < 2 * copcore::units::kElectronMassC2) {
        survive();
        continue;
      }

      double logEnergy = std::log(eKin);
      double elKinEnergy, posKinEnergy;
      G4HepEmGammaInteractionConversion::SampleKinEnergies(&g4HepEmData, eKin, logEnergy, auxData.fMCIndex, elKinEnergy,
                                                           posKinEnergy, &rnge);

      double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
      double dirSecondaryEl[3], dirSecondaryPos[3];
      G4HepEmGammaInteractionConversion::SampleDirections(dirPrimary, dirSecondaryEl, dirSecondaryPos, elKinEnergy,
                                                          posKinEnergy, &rnge);

      Track &electron = secondaries.electrons->NextTrack();
      Track &positron = secondaries.positrons->NextTrack();

      adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 1, /*numGammas*/ 0);

      electron.InitAsSecondary(pos, navState, globalTime);
      electron.parentID = currentTrack.parentID;
      electron.rngState = newRNG;
      electron.eKin     = elKinEnergy;
      electron.dir.Set(dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]);

      positron.InitAsSecondary(pos, navState, globalTime);
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
      if (eKin < LowEnergyThreshold) {
        survive();
        continue;
      }
      const double origDirPrimary[] = {dir.x(), dir.y(), dir.z()};
      double dirPrimary[3];
      const double newEnergyGamma =
          G4HepEmGammaInteractionCompton::SamplePhotonEnergyAndDirection(eKin, dirPrimary, origDirPrimary, &rnge);
      vecgeom::Vector3D<double> newDirGamma(dirPrimary[0], dirPrimary[1], dirPrimary[2]);

      const double energyEl = eKin - newEnergyGamma;
      if (energyEl > LowEnergyThreshold) {
        // Create a secondary electron and sample/compute directions.
        Track &electron = secondaries.electrons->NextTrack();
        adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

        electron.InitAsSecondary(pos, navState, globalTime);
        electron.parentID = currentTrack.parentID;
        electron.rngState = newRNG;
        electron.eKin     = energyEl;
        electron.dir      = eKin * dir - newEnergyGamma * newDirGamma;
        electron.dir.Normalize();
      } else {
        if (auxData.fSensIndex >= 0)
          adept_scoring::RecordHit(userScoring,
                                   currentTrack.parentID, // Track ID
                                   2,                     // Particle type
                                   geometryStepLength,    // Step length
                                   0,                     // Total Edep
                                   &navState,             // Pre-step point navstate
                                   &preStepPos,           // Pre-step point position
                                   &preStepDir,           // Pre-step point momentum direction
                                   nullptr,               // Pre-step point polarization
                                   preStepEnergy,         // Pre-step point kinetic energy
                                   0,                     // Pre-step point charge
                                   &nextState,            // Post-step point navstate
                                   &pos,                  // Post-step point position
                                   &dir,                  // Post-step point momentum direction
                                   nullptr,               // Post-step point polarization
                                   newEnergyGamma,        // Post-step point kinetic energy
                                   0);                    // Post-step point charge
      }

      // Check the new gamma energy and deposit if below threshold.
      if (newEnergyGamma > LowEnergyThreshold) {
        eKin = newEnergyGamma;
        dir  = newDirGamma;
        survive();
      } else {
        if (auxData.fSensIndex >= 0)
          adept_scoring::RecordHit(userScoring,
                                   currentTrack.parentID, // Track ID
                                   2,                     // Particle type
                                   geometryStepLength,    // Step length
                                   0,                     // Total Edep
                                   &navState,             // Pre-step point navstate
                                   &preStepPos,           // Pre-step point position
                                   &preStepDir,           // Pre-step point momentum direction
                                   nullptr,               // Pre-step point polarization
                                   preStepEnergy,         // Pre-step point kinetic energy
                                   0,                     // Pre-step point charge
                                   &nextState,            // Post-step point navstate
                                   &pos,                  // Post-step point position
                                   &dir,                  // Post-step point momentum direction
                                   nullptr,               // Post-step point polarization
                                   newEnergyGamma,        // Post-step point kinetic energy
                                   0);                    // Post-step point charge
        // The current track is killed by not enqueuing into the next activeQueue.
      }
      break;
    }
    case 2: {
      // Invoke photoelectric process.
      const double theLowEnergyThreshold = 1 * copcore::units::eV;

      const double bindingEnergy = G4HepEmGammaInteractionPhotoelectric::SelectElementBindingEnergy(
          &g4HepEmData, auxData.fMCIndex, gammaTrack.GetPEmxSec(), eKin, &rnge);

      double edep             = bindingEnergy;
      const double photoElecE = eKin - edep;
      if (photoElecE > theLowEnergyThreshold) {
        // Create a secondary electron and sample directions.
        Track &electron = secondaries.electrons->NextTrack();
        adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

        double dirGamma[] = {dir.x(), dir.y(), dir.z()};
        double dirPhotoElec[3];
        G4HepEmGammaInteractionPhotoelectric::SamplePhotoElectronDirection(photoElecE, dirGamma, dirPhotoElec, &rnge);

        electron.InitAsSecondary(pos, navState, globalTime);
        electron.parentID = currentTrack.parentID;
        electron.rngState = newRNG;
        electron.eKin     = photoElecE;
        electron.dir.Set(dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]);
      } else {
        edep = eKin;
      }
      if (auxData.fSensIndex >= 0)
        adept_scoring::RecordHit(userScoring,
                                 currentTrack.parentID, // Track ID
                                 2,                     // Particle type
                                 geometryStepLength,    // Step length
                                 edep,                  // Total Edep
                                 &navState,             // Pre-step point navstate
                                 &preStepPos,           // Pre-step point position
                                 &preStepDir,           // Pre-step point momentum direction
                                 nullptr,               // Pre-step point polarization
                                 preStepEnergy,         // Pre-step point kinetic energy
                                 0,                     // Pre-step point charge
                                 &nextState,            // Post-step point navstate
                                 &pos,                  // Post-step point position
                                 &dir,                  // Post-step point momentum direction
                                 nullptr,               // Post-step point polarization
                                 0,                     // Post-step point kinetic energy
                                 0);                    // Post-step point charge
      // The current track is killed by not enqueuing into the next activeQueue.
      break;
    }
    }
  }
}
