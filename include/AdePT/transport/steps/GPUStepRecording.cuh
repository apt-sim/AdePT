// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <AdePT/transport/steps/DeviceStepBuffer.cuh>
#include <AdePT/transport/steps/GPUStep.hh>
#include <AdePT/transport/support/Global.h>
#include <AdePT/transport/tracks/Track.cuh>

#include <cstdint>

namespace adept_step_recording {

/// @brief Record a GPU step.
///
/// Track identity and bookkeeping are taken from the owning track so callers cannot
/// accidentally combine metadata from different tracks.
__device__ void RecordGPUStep(TrackBase const &track, short stepLimProcessId, ParticleType aParticleType,
                              double aStepLength, double aTotalEnergyDeposit, vecgeom::NavigationState const &aPreState,
                              vecgeom::Vector3D<double> const &aPrePosition,
                              vecgeom::Vector3D<double> const &aPreMomentumDirection, double aPreEKin,
                              vecgeom::NavigationState const &aPostState,
                              vecgeom::Vector3D<double> const &aPostPosition,
                              vecgeom::Vector3D<double> const &aPostMomentumDirection, double aPostEKin,
                              double aGlobalTime, float aLocalTime, float aProperTime, double aPreGlobalTime,
                              bool isLastStep, SecondaryInitData const *secondaryData, unsigned int nSecondaries)
{

  // Snapshot the common metadata once. RecordGPUStep is intentionally out of line,
  // so this avoids repeated loads through the track reference while keeping those
  // values consistent for the complete parent-plus-secondaries record.
  const uint64_t trackId           = track.trackId;
  const uint64_t parentId          = track.parentId;
  const float trackWeight          = track.weight;
  const unsigned int eventId       = track.eventId;
  const short threadId             = track.threadId;
  const bool hasHostData           = track.hasHostData;
  const unsigned short stepCounter = track.stepCounter;

  // defensive check
  if (nSecondaries > 0 && secondaryData == nullptr) {
    COPCORE_EXCEPTION("secondaryData is null but nSecondaries > 0");
  }

  // allocate step slots: one for the parent and then one for each secondary
  auto slotStartIndex = adept::transport::gDeviceStepBuffer.ReserveStepSlots(threadId, 1u + nSecondaries);

  // The ProcessGPUSteps on the Host expects the step of the parent track first, and then all secondaries
  // that were generated in that step.
  GPUStep &parentStep = adept::transport::gDeviceStepBuffer.GetSlot(threadId, slotStartIndex);
  // Fill the required data for the parent step
  FillGPUStep(parentStep, trackId, parentId, stepLimProcessId, aParticleType, aStepLength, aTotalEnergyDeposit,
              trackWeight, aPreState, aPrePosition, aPreMomentumDirection, aPreEKin, aPostState, aPostPosition,
              aPostMomentumDirection, aPostEKin, aGlobalTime, aLocalTime, aProperTime, aPreGlobalTime, eventId,
              threadId, isLastStep, hasHostData, stepCounter, nSecondaries);

  // Fill the steps for the secondaries
  for (unsigned int i = 0; i < nSecondaries; ++i) {
    // The index is the startIndex + 1 (for the parent) + i for the current secondary
    GPUStep &secondaryStep = adept::transport::gDeviceStepBuffer.GetSlot(threadId, slotStartIndex + 1u + i);
    FillGPUStep(secondaryStep, secondaryData[i].trackId, trackId, secondaryData[i].creatorProcessId,
                secondaryData[i].particleType,
                /*steplength*/ 0., /*energydeposit*/ 0., trackWeight, aPostState, aPostPosition, secondaryData[i].dir,
                secondaryData[i].eKin, aPostState, aPostPosition, secondaryData[i].dir, secondaryData[i].eKin,
                aGlobalTime,
                /*localTime*/ 0.f, /*properTime*/ 0.f, aGlobalTime, eventId, threadId, /*isLastStep*/ false,
                secondaryData[i].hasHostData, /*stepCounter*/ 0, /*nSecondaries*/ 0);
  }
}

} // namespace adept_step_recording
