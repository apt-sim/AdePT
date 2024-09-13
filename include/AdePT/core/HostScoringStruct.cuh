// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef HOSTSCORING_H
#define HOSTSCORING_H

#include "VecGeom/navigation/NavigationState.h"
#include <AdePT/base/Atomic.h>
#include <G4ios.hh>

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
  char fParticleType{0}; // Particle type ID
  // Data needed to reconstruct G4 Step
  double fStepLength{0};
  double fTotalEnergyDeposit{0};
  double fNonIonizingEnergyDeposit{0};
  // bool fFirstStepInVolume{false};
  // bool fLastStepInVolume{false};
  // Data needed to reconstruct pre-post step points
  GPUStepPoint fPreStepPoint;
  GPUStepPoint fPostStepPoint;
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
// Contains the necessary information for recording hits on GPU, and reconstructing them
// on the host, in order to call the user-defined sensitive detector code
struct HostScoring {

  /// @brief The data in this struct is copied from device to host after each iteration
  struct Stats {
    unsigned int fUsedSlots;   ///< Number of used hit slots
    unsigned int fNextFreeHit; ///< Index of last used hit slot in the buffer
    unsigned int fBufferStart; ///< Index of first used hit slot in the buffer
  };

  HostScoring(unsigned int aBufferCapacity = 1024 * 1024, float aFlushLimit = 0.8)
      : fBufferCapacity(aBufferCapacity), fFlushLimit(aFlushLimit)
  {
    printf("Initializing scoring with buffer capacity: %d\n", aBufferCapacity);
    // Allocate the hits buffer on Host
    fGPUHitsBuffer_host = (GPUHit *)malloc(sizeof(GPUHit) * fBufferCapacity);
    // Allocate the global counters struct on host
    fGlobalCounters_host = (GlobalCounters *)malloc(sizeof(GlobalCounters));
  };

  ~HostScoring()
  {
    free(fGPUHitsBuffer_host);
    free(fGlobalCounters_host);
  }

  /// @brief Print scoring info
  void Print();

  // Data members
  unsigned int fBufferCapacity{0}; ///< Number of hits to be stored in the buffer
  float fFlushLimit{0};            ///< Proportion of the buffer that needs to be filled to trigger a flush to CPU
  unsigned int fBufferStart{0};    ///< Index of first used slot in the buffer
  GPUHit *fGPUHitsBuffer_dev{nullptr};
  GPUHit *fGPUHitsBuffer_host{nullptr};

  // Atomic variables used on GPU
  adept::Atomic_t<unsigned int> *fUsedSlots_dev;   ///< Number of used hit slots
  adept::Atomic_t<unsigned int> *fNextFreeHit_dev; ///< Index of last used hit slot in the buffer

  // Stats struct, used to transfer information about the state of the buffer
  Stats fStats;
  Stats *fStats_dev;

  // Used to get performance information
  unsigned int fStepsSinceLastFlush{0};

  // Used for comparison with Geant4
  GlobalCounters *fGlobalCounters_dev{nullptr};
  GlobalCounters *fGlobalCounters_host{nullptr};
};

using AdeptScoring = HostScoring;

#endif // HOSTSCORING_H