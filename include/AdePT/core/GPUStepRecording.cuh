// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_GPU_STEP_RECORDING_CUH
#define ADEPT_GPU_STEP_RECORDING_CUH

#include <AdePT/core/DeviceStepBuffer.cuh>
#include <AdePT/core/GPUStep.hh>
#include <AdePT/copcore/Global.h>

#include <cstdint>

namespace adept_step_recording {

/// @brief Record a GPU step
__device__ void RecordGPUStep(uint64_t aTrackID, uint64_t aParentID, short stepLimProcessId, ParticleType aParticleType,
                              double aStepLength, double aTotalEnergyDeposit, float aTrackWeight,
                              vecgeom::NavigationState const &aPreState, vecgeom::Vector3D<double> const &aPrePosition,
                              vecgeom::Vector3D<double> const &aPreMomentumDirection, double aPreEKin,
                              vecgeom::NavigationState const &aPostState,
                              vecgeom::Vector3D<double> const &aPostPosition,
                              vecgeom::Vector3D<double> const &aPostMomentumDirection, double aPostEKin,
                              double aGlobalTime, float aLocalTime, float aProperTime, double aPreGlobalTime,
                              unsigned int eventID, short threadID, bool isLastStep, unsigned short stepCounter,
                              SecondaryInitData const *secondaryData, unsigned int nSecondaries)
{

  // defensive check
  if (nSecondaries > 0 && secondaryData == nullptr) {
    COPCORE_EXCEPTION("secondaryData is null but nSecondaries > 0");
  }

  // allocate step slots: one for the parent and then one for each secondary
  auto slotStartIndex = AsyncAdePT::gDeviceStepBuffer.ReserveStepSlots(threadID, 1u + nSecondaries);

  // The ProcessGPUSteps on the Host expects the step of the parent track first, and then all secondaries
  // that were generated in that step.
  GPUStep &parentStep = AsyncAdePT::gDeviceStepBuffer.GetSlot(threadID, slotStartIndex);
  // Fill the required data for the parent step
  FillGPUStep(parentStep, aTrackID, aParentID, stepLimProcessId, aParticleType, aStepLength, aTotalEnergyDeposit,
              aTrackWeight, aPreState, aPrePosition, aPreMomentumDirection, aPreEKin, aPostState, aPostPosition,
              aPostMomentumDirection, aPostEKin, aGlobalTime, aLocalTime, aProperTime, aPreGlobalTime, eventID,
              threadID, isLastStep, stepCounter, nSecondaries);

  // Fill the steps for the secondaries
  for (unsigned int i = 0; i < nSecondaries; ++i) {
    // The index is the startIndex + 1 (for the parent) + i for the current secondary
    GPUStep &secondaryStep = AsyncAdePT::gDeviceStepBuffer.GetSlot(threadID, slotStartIndex + 1u + i);
    FillGPUStep(secondaryStep, secondaryData[i].trackId, aTrackID, secondaryData[i].creatorProcessId,
                secondaryData[i].particleType,
                /*steplength*/ 0., /*energydeposit*/ 0., aTrackWeight, aPostState, aPostPosition, secondaryData[i].dir,
                secondaryData[i].eKin, aPostState, aPostPosition, secondaryData[i].dir, secondaryData[i].eKin,
                aGlobalTime,
                /*localTime*/ 0.f, /*properTime*/ 0.f, aGlobalTime, eventID, threadID, /*isLastStep*/ false,
                /*stepCounter*/ 0, /*nSecondaries*/ 0);
  }
}

} // namespace adept_step_recording

#endif
