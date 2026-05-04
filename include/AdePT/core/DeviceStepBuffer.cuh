// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_DEVICE_STEP_BUFFER_CUH
#define ADEPT_DEVICE_STEP_BUFFER_CUH

#include <AdePT/core/GPUStep.hh>
#include <AdePT/copcore/Global.h>

#include <VecCore/VecMath.h>

#include <cstdio>

// Comparison for sorting steps into events on device.
struct CompareGPUSteps {
  __device__ bool operator()(const GPUStep &lhs, const GPUStep &rhs) const { return lhs.threadId < rhs.threadId; }
};

namespace AsyncAdePT {

/// Struct holding GPU steps to be used both on host and device.
struct DeviceStepBufferView {
  GPUStep *stepBuffer_dev    = nullptr;
  unsigned int *fSlotCounter = nullptr; // Array of per-thread counters
  unsigned int fNSlot        = 0;
  unsigned int fNThreads     = 0;

  __host__ __device__ unsigned int GetMaxSlotCount()
  {
    unsigned int maxVal = 0;
    for (unsigned int i = 0; i < fNThreads; ++i) {
      maxVal = vecCore::math::Max(maxVal, fSlotCounter[i]);
    }
    return maxVal;
  }

  __device__ unsigned int ReserveStepSlots(unsigned int threadId, unsigned int nSlots)
  {
    const auto slotStartIndex = atomicAdd(&fSlotCounter[threadId], nSlots);
    if (slotStartIndex + nSlots > fNSlot) {
      printf("Trying to record step #%d with only %d slots\n", slotStartIndex, fNSlot);
      COPCORE_EXCEPTION("Out of slots in DeviceStepBufferView::NextSlot");
    }
    return slotStartIndex;
  }

  __device__ GPUStep &GetSlot(unsigned int threadId, unsigned int slot)
  {
    return stepBuffer_dev[threadId * fNSlot + slot];
  }
};

__device__ DeviceStepBufferView gDeviceStepBuffer;

} // namespace AsyncAdePT

#endif
