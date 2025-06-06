// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

// This file contains elements that can be shared between several scoring implementations

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
  // Data needed to reconstruct pre-post step points
  GPUStepPoint fPreStepPoint;
  GPUStepPoint fPostStepPoint;
  // Data needed to reconstruct G4 Step
  vecgeom::Vector3D<vecgeom::Precision> fVertexPosition{0., 0., 0.};
  double fStepLength{0};
  double fTotalEnergyDeposit{0};
  double fNonIonizingEnergyDeposit{0};
  double fGlobalTime{0.};
  double fLocalTime{0.};
  float fTrackWeight{1};
  uint64_t fTrackID{0};  // Track ID
  uint64_t fParentID{0}; // parent Track ID
  short fCreatorProcessID{-1};
  short fStepLimitingProcessID{-1};
  unsigned int fEventId{0};
  short threadId{-1};
  // bool fFirstStepInVolume{false};
  bool fFirstStepOfTrack{false};
  bool fLastStepOfTrack{false};
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

/// @brief Utility function to copy a 3D vector, used for filling the Step Points
__device__ __forceinline__ void Copy3DVector(vecgeom::Vector3D<Precision> const &source,
                                             vecgeom::Vector3D<Precision> &destination)
{
  destination.x() = source.x();
  destination.y() = source.y();
  destination.z() = source.z();
};

/// @brief Fill the provided hit with the given data
__device__ __forceinline__ void FillHit(
    GPUHit &aGPUHit, uint64_t aTrackID, uint64_t aParentID, short aCreatorProcessID, short aStepLimitingProcessID,
    char aParticleType, double aStepLength, double aTotalEnergyDeposit, float aTrackWeight,
    vecgeom::Vector3D<vecgeom::Precision> aVertexPosition, vecgeom::NavigationState const &aPreState,
    vecgeom::Vector3D<Precision> const &aPrePosition, vecgeom::Vector3D<Precision> const &aPreMomentumDirection,
    double aPreEKin, double aPreCharge, vecgeom::NavigationState const &aPostState,
    vecgeom::Vector3D<Precision> const &aPostPosition, vecgeom::Vector3D<Precision> const &aPostMomentumDirection,
    double aPostEKin, double aPostCharge, double aGlobalTime, double aLocalTime, unsigned int eventID, short threadID,
    bool isLastStep, bool isFirstStep)
{
  aGPUHit.fEventId = eventID;
  aGPUHit.threadId = threadID;

  aGPUHit.fFirstStepOfTrack = isFirstStep;
  aGPUHit.fLastStepOfTrack  = isLastStep;
  // Fill the required data
  aGPUHit.fTrackID               = aTrackID;
  aGPUHit.fParentID              = aParentID;
  aGPUHit.fCreatorProcessID      = aCreatorProcessID;
  aGPUHit.fStepLimitingProcessID = aStepLimitingProcessID;
  aGPUHit.fParticleType          = aParticleType;
  aGPUHit.fStepLength            = aStepLength;
  aGPUHit.fTotalEnergyDeposit    = aTotalEnergyDeposit;
  aGPUHit.fTrackWeight           = aTrackWeight;
  aGPUHit.fGlobalTime            = aGlobalTime;
  aGPUHit.fLocalTime             = aLocalTime;
  Copy3DVector(aVertexPosition, aGPUHit.fVertexPosition);
  // Pre step point
  aGPUHit.fPreStepPoint.fNavigationState = aPreState;
  Copy3DVector(aPrePosition, aGPUHit.fPreStepPoint.fPosition);
  Copy3DVector(aPreMomentumDirection, aGPUHit.fPreStepPoint.fMomentumDirection);
  aGPUHit.fPreStepPoint.fEKin   = aPreEKin;
  aGPUHit.fPreStepPoint.fCharge = aPreCharge;
  // Post step point
  aGPUHit.fPostStepPoint.fNavigationState = aPostState;
  Copy3DVector(aPostPosition, aGPUHit.fPostStepPoint.fPosition);
  Copy3DVector(aPostMomentumDirection, aGPUHit.fPostStepPoint.fMomentumDirection);
  aGPUHit.fPostStepPoint.fEKin   = aPostEKin;
  aGPUHit.fPostStepPoint.fCharge = aPostCharge;
};

#endif
