// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef SCORING_COMMONS_HH
#define SCORING_COMMONS_HH

#include "VecGeom/navigation/NavigationState.h"

struct GPUStepPoint {
  vecgeom::Vector3D<Precision> fPosition;
  vecgeom::Vector3D<Precision> fMomentumDirection;
  vecgeom::Vector3D<Precision> fPolarization;
  double fEKin;
  double fCharge;
  // Data needed to reconstruct G4 Touchable history
  vecgeom::NavigationState fNavigationState{0}; // VecGeom navigation state, used to identify the touchable
};

// Stores the necessary data to reconstruct GPU hits on the host , and
// call the user-defined Geant4 sensitive detector code
struct GPUHit {
  int fParentID{0}; // Track ID
  // Data needed to reconstruct G4 Step
  double fStepLength{0};
  double fTotalEnergyDeposit{0};
  double fNonIonizingEnergyDeposit{0};
  // bool fFirstStepInVolume{false};
  // bool fLastStepInVolume{false};
  // Data needed to reconstruct pre-post step points
  GPUStepPoint fPreStepPoint;
  GPUStepPoint fPostStepPoint;
  unsigned int fEventId{0};
  short threadId{-1};
  char fParticleType{0}; // Particle type ID
};

/// @brief Stores information used for comparison with Geant4 (Number of steps, Number of produced particles, etc)
struct GlobalCounters {
  double energyDeposit;
  // Not int to avoid overflows for more than 100,000 events; unsigned long long
  // is the only other data type available for atomicAdd().
  unsigned long long chargedSteps;
  unsigned long long neutralSteps;
  unsigned long long hits;
  unsigned long long numGammas;
  unsigned long long numElectrons;
  unsigned long long numPositrons;
  // Not used on the device, filled in by the host.
  unsigned long long numKilled;

  void Print()
  {
    printf("Global scoring: stpChg=%llu stpNeu=%llu hits=%llu numGam=%llu numEle=%llu numPos=%llu numKilled=%llu\n",
           chargedSteps, neutralSteps, hits, numGammas, numElectrons, numPositrons, numKilled);
  }
};

#endif
