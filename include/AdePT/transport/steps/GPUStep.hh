// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

// This file contains the step data returned from GPU transport to the host.

#ifndef ADEPT_GPU_STEP_HH
#define ADEPT_GPU_STEP_HH

#include <AdePT/transport/tracks/ParticleTypes.hh>
#include "VecGeom/navigation/NavigationState.h"

struct GPUStepPoint {
  vecgeom::Vector3D<double> fPosition;
  vecgeom::Vector3D<double> fMomentumDirection;
  double fEKin;
  // Data needed to reconstruct G4 Touchable history
  vecgeom::NavigationState fNavigationState{0}; // VecGeom navigation state, used to identify the touchable
};

// Stores the data needed to reconstruct a Geant4 step on the host.
struct GPUStep {
  // Data needed to reconstruct pre-post step points
  GPUStepPoint fPreStepPoint;
  GPUStepPoint fPostStepPoint;
  double fStepLength{0};
  double fTotalEnergyDeposit{0};
  // double fNonIonizingEnergyDeposit{0};
  double fGlobalTime{0.};
  float fLocalTime{0.f};
  float fProperTime{0.f};
  double fPreGlobalTime{0.};
  float fTrackWeight{1};
  uint64_t fTrackID{0};  // Track ID
  uint64_t fParentID{0}; // parent Track ID
  short fStepLimProcessId{-1};
  int fEventId{0};
  short threadId{-1};
  unsigned short fStepCounter{0};
  bool fLastStepOfTrack{false};
  ParticleType fParticleType{ParticleType::Electron};
  unsigned char fNumSecondaries{0};

  bool operator<(GPUStep const &other) const
  {
    const auto pdgFromParticleType = [](ParticleType particleType) {
      switch (particleType) {
      case ParticleType::Electron:
        return 11;
      case ParticleType::Positron:
        return -11;
      case ParticleType::Gamma:
        return 22;
      }
      return 0;
    };

    const int thisPDG  = pdgFromParticleType(fParticleType);
    const int otherPDG = pdgFromParticleType(other.fParticleType);

    if (thisPDG != otherPDG) return thisPDG < otherPDG;
    if (fPostStepPoint.fEKin != other.fPostStepPoint.fEKin) return fPostStepPoint.fEKin < other.fPostStepPoint.fEKin;
    if (fPostStepPoint.fPosition.x() != other.fPostStepPoint.fPosition.x()) {
      return fPostStepPoint.fPosition.x() < other.fPostStepPoint.fPosition.x();
    }
    if (fPostStepPoint.fPosition.y() != other.fPostStepPoint.fPosition.y()) {
      return fPostStepPoint.fPosition.y() < other.fPostStepPoint.fPosition.y();
    }
    if (fPostStepPoint.fPosition.z() != other.fPostStepPoint.fPosition.z()) {
      return fPostStepPoint.fPosition.z() < other.fPostStepPoint.fPosition.z();
    }
    if (fPostStepPoint.fMomentumDirection.x() != other.fPostStepPoint.fMomentumDirection.x()) {
      return fPostStepPoint.fMomentumDirection.x() < other.fPostStepPoint.fMomentumDirection.x();
    }
    if (fPostStepPoint.fMomentumDirection.y() != other.fPostStepPoint.fMomentumDirection.y()) {
      return fPostStepPoint.fMomentumDirection.y() < other.fPostStepPoint.fMomentumDirection.y();
    }
    if (fPostStepPoint.fMomentumDirection.z() != other.fPostStepPoint.fMomentumDirection.z()) {
      return fPostStepPoint.fMomentumDirection.z() < other.fPostStepPoint.fMomentumDirection.z();
    }
    return false;
  }
};

/// @brief AdePT-specific step-limiting process ids stored in GPUStep::fStepLimProcessId.
constexpr short kAdePTTransportationProcess = 10;
/// @brief Returned step for a track that leaves the GPU region and continues on the CPU.
constexpr short kAdePTOutOfGPURegionProcess = 11;
/// @brief Returned step for a track that is intentionally finished on the CPU.
constexpr short kAdePTFinishOnCPUProcess = 12;

/// @brief Minimal data struct that is needed along with the parent track to provide the initial track information that
/// is sent back to the CPU
struct SecondaryInitData {
  uint64_t trackId;
  vecgeom::Vector3D<double> dir;
  double eKin;
  short creatorProcessId{-1};
  ParticleType particleType{ParticleType::Electron};
};

/// @brief Utility function to copy a 3D vector, used for filling the Step Points
__device__ __forceinline__ void Copy3DVector(vecgeom::Vector3D<double> const &source,
                                             vecgeom::Vector3D<double> &destination)
{
  destination.x() = source.x();
  destination.y() = source.y();
  destination.z() = source.z();
};

/// @brief Fill the provided GPU step with the given data
__device__ __forceinline__ void FillGPUStep(
    GPUStep &aGPUStep, uint64_t aTrackID, uint64_t aParentID, short aStepLimProcessId, ParticleType aParticleType,
    double aStepLength, double aTotalEnergyDeposit, float aTrackWeight, vecgeom::NavigationState const &aPreState,
    vecgeom::Vector3D<double> const &aPrePosition, vecgeom::Vector3D<double> const &aPreMomentumDirection,
    double aPreEKin, vecgeom::NavigationState const &aPostState, vecgeom::Vector3D<double> const &aPostPosition,
    vecgeom::Vector3D<double> const &aPostMomentumDirection, double aPostEKin, double aGlobalTime, float aLocalTime,
    float aProperTime, double aPreGlobalTime, unsigned int eventID, short threadID, bool isLastStep,
    unsigned short stepCounter, unsigned char aNumSecondaries)
{
  aGPUStep.fEventId = eventID;
  aGPUStep.threadId = threadID;

  aGPUStep.fStepCounter     = stepCounter;
  aGPUStep.fLastStepOfTrack = isLastStep;
  // Fill the required data
  aGPUStep.fTrackID            = aTrackID;
  aGPUStep.fParentID           = aParentID;
  aGPUStep.fStepLimProcessId   = aStepLimProcessId;
  aGPUStep.fParticleType       = aParticleType;
  aGPUStep.fStepLength         = aStepLength;
  aGPUStep.fTotalEnergyDeposit = aTotalEnergyDeposit;
  aGPUStep.fTrackWeight        = aTrackWeight;
  aGPUStep.fGlobalTime         = aGlobalTime;
  aGPUStep.fLocalTime          = aLocalTime;
  aGPUStep.fProperTime         = aProperTime;
  aGPUStep.fPreGlobalTime      = aPreGlobalTime;
  aGPUStep.fNumSecondaries     = aNumSecondaries;
  // Pre step point
  aGPUStep.fPreStepPoint.fNavigationState = aPreState;
  Copy3DVector(aPrePosition, aGPUStep.fPreStepPoint.fPosition);
  Copy3DVector(aPreMomentumDirection, aGPUStep.fPreStepPoint.fMomentumDirection);
  aGPUStep.fPreStepPoint.fEKin = aPreEKin;
  // Post step point
  aGPUStep.fPostStepPoint.fNavigationState = aPostState;
  Copy3DVector(aPostPosition, aGPUStep.fPostStepPoint.fPosition);
  Copy3DVector(aPostMomentumDirection, aGPUStep.fPostStepPoint.fMomentumDirection);
  aGPUStep.fPostStepPoint.fEKin = aPostEKin;
};

#endif
