// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/core/HostScoring.h>
#include "AdePT/copcore/Global.h"

namespace device_utils_hostscoring {

/// @brief Move the buffer start on GPU
__global__ void UpdateBufferStartGPU(HostScoring *aScoring_device, unsigned int aBufferStart)
{
  aScoring_device->fBufferStart = aBufferStart;
}

/// @brief Update the buffer usage on GPU
/// @details This function is to be called after copying the buffer contents back to the host. Updates the
/// usage of the buffer based on usage prior to the copy and current usage.
__global__ void UpdateBufferUsageGPU(HostScoring *aScoring_device, unsigned int aUsedSlotsBeforeCopy)
{
  // Must be performed atomically
  *(aScoring_device->fUsedSlots_d) -= aUsedSlotsBeforeCopy;
}
} // namespace device_utils_hostscoring

HostScoring *HostScoring::InitializeOnGPU()
{
  // Allocate space for the hits buffer
  COPCORE_CUDA_CHECK(cudaMalloc(&fGPUHitsBuffer_device, sizeof(GPUHit) * fBufferCapacity));

  // Allocate space for the global counters
  COPCORE_CUDA_CHECK(cudaMalloc(&fGlobalCounters_dev, sizeof(GlobalCounters)));

  // Allocate space for the atomic variables on device
  COPCORE_CUDA_CHECK(cudaMalloc(&fUsedSlots_d, sizeof(adept::Atomic_t<unsigned int>)));
  COPCORE_CUDA_CHECK(cudaMalloc(&fNextFreeHit_d, sizeof(adept::Atomic_t<unsigned int>)));

  // Allocate space for the instance on GPU and copy the data members from the host
  // Now allocate space for the BasicScoring placeholder on device and copy the device pointers of components
  HostScoring *HostScoring_dev = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&HostScoring_dev, sizeof(HostScoring)));
  COPCORE_CUDA_CHECK(cudaMemcpy(HostScoring_dev, this, sizeof(HostScoring), cudaMemcpyHostToDevice));

  return HostScoring_dev;
}

/// @brief Frees the memory allocated on the GPU
/// @param aHostScoring_dev Pointer to the device instance of this object
void HostScoring::FreeGPU(HostScoring *aHostScoring_dev)
{
  // Free hits buffer
  COPCORE_CUDA_CHECK(cudaFree(fGPUHitsBuffer_device));
  // Free global counters
  COPCORE_CUDA_CHECK(cudaFree(fGlobalCounters_dev));
  // Free atomic variable instances
  COPCORE_CUDA_CHECK(cudaFree(fUsedSlots_d));
  COPCORE_CUDA_CHECK(cudaFree(fNextFreeHit_d));
  // Free the space allocated for the GPU instance of this object
  COPCORE_CUDA_CHECK(cudaFree(aHostScoring_dev));
}

__device__ void HostScoring::RecordHit(
    char aParticleType, double aStepLength, double aTotalEnergyDeposit, vecgeom::NavigationState const *aPreState,
    vecgeom::Vector3D<Precision> *aPrePosition, vecgeom::Vector3D<Precision> *aPreMomentumDirection,
    vecgeom::Vector3D<Precision> *aPrePolarization, double aPreEKin, double aPreCharge,
    vecgeom::NavigationState const *aPostState, vecgeom::Vector3D<Precision> *aPostPosition,
    vecgeom::Vector3D<Precision> *aPostMomentumDirection, vecgeom::Vector3D<Precision> *aPostPolarization,
    double aPostEKin, double aPostCharge)
{
  // Acquire a hit slot
  GPUHit *aGPUHit = GetNextFreeHit();

  // Fill the required data
  aGPUHit->fParticleType       = aParticleType;
  aGPUHit->fStepLength         = aStepLength;
  aGPUHit->fTotalEnergyDeposit = aTotalEnergyDeposit;
  // Pre step point
  aGPUHit->fPreStepPoint.fNavigationStateIndex = aPreState->GetNavIndex();
  Copy3DVector(aPrePosition, &(aGPUHit->fPreStepPoint.fPosition));
  Copy3DVector(aPreMomentumDirection, &(aGPUHit->fPreStepPoint.fMomentumDirection));
  // Copy3DVector(aPrePolarization, aGPUHit.fPreStepPoint.fPolarization);
  aGPUHit->fPreStepPoint.fEKin   = aPreEKin;
  aGPUHit->fPreStepPoint.fCharge = aPreCharge;
  // Post step point
  aGPUHit->fPostStepPoint.fNavigationStateIndex = aPostState->GetNavIndex();
  Copy3DVector(aPostPosition, &(aGPUHit->fPostStepPoint.fPosition));
  Copy3DVector(aPostMomentumDirection, &(aGPUHit->fPostStepPoint.fMomentumDirection));
  // Copy3DVector(aPostPolarization, aGPUHit.fPostStepPoint.fPolarization);
  aGPUHit->fPostStepPoint.fEKin   = aPostEKin;
  aGPUHit->fPostStepPoint.fCharge = aPostCharge;
}

__device__ __forceinline__ unsigned int HostScoring::GetNextFreeHitIndex()
{
  // Atomic addition, each GPU thread accessing concurrently gets a different slot
  unsigned int next = fNextFreeHit_d->fetch_add(1);
  assert(next >= fBufferStart);
  if ((next - fBufferStart) >= fBufferCapacity) {
    COPCORE_EXCEPTION("No slot available in Hit Buffer");
  }
  (*fUsedSlots_d)++;
  // Circular buffer, wrap around to the beginning if needed
  return next % fBufferCapacity;
}

__device__ __forceinline__ GPUHit *HostScoring::GetNextFreeHit()
{
  unsigned int aHitIndex = GetNextFreeHitIndex();
  assert(aHitIndex < fBufferCapacity);
  return &fGPUHitsBuffer_device[aHitIndex];
}

__device__ __forceinline__ void HostScoring::Copy3DVector(vecgeom::Vector3D<Precision> *source,
                                                          vecgeom::Vector3D<Precision> *destination)
{
  destination->x() = source->x();
  destination->y() = source->y();
  destination->z() = source->z();
}

bool HostScoring::CheckAndFlush(Stats &aStats_host, HostScoring *aScoring_device, cudaStream_t stream)
{
  fStepsSinceLastFlush++;
  float aBufferUsage = (float)aStats_host.fUsedSlots / fBufferCapacity;

  if (aBufferUsage > fFlushLimit) {
    fStepsSinceLastFlush = 0;
    CopyHitsToHost(aStats_host, aScoring_device, stream);
    return true;
  }
  return false;
}

void HostScoring::CopyHitsToHost(Stats &aStats_host, HostScoring *aScoring_device, cudaStream_t stream)
{
  // Move the start of the hits buffer on GPU
  device_utils_hostscoring::UpdateBufferStartGPU<<<1, 1, 0, stream>>>(aScoring_device, aStats_host.fNextFreeHit);

  // Copy the hits to the host. We need to copy the entire buffer as the starting index may change
  COPCORE_CUDA_CHECK(cudaMemcpyAsync(fGPUHitsBuffer_host, fGPUHitsBuffer_device, fBufferCapacity * sizeof(GPUHit),
                                     cudaMemcpyDeviceToHost, stream));

  // Update the used slots counter after all data has been copied, taking into account the slots filled in the meantime
  // NOTE: This functionality won't give any benefits as long as we sync & process the returned buffer right
  // after copying it, as the CPU thread will be processing hits instead of launching new kernels.
  device_utils_hostscoring::UpdateBufferUsageGPU<<<1, 1, 0, stream>>>(aScoring_device, aStats_host.fUsedSlots);
}

__device__ void HostScoring::AccountProduced(int num_ele, int num_pos, int num_gam)
{
  // Increment number of secondaries
  atomicAdd(&fGlobalCounters_dev->numElectrons, num_ele);
  atomicAdd(&fGlobalCounters_dev->numPositrons, num_pos);
  atomicAdd(&fGlobalCounters_dev->numGammas, num_gam);
}

void HostScoring::CopyGlobalCountersToHost(cudaStream_t stream)
{
  COPCORE_CUDA_CHECK(cudaMemcpyAsync(fGlobalCounters_host, fGlobalCounters_dev, sizeof(GlobalCounters),
                                     cudaMemcpyDeviceToHost, stream));
}

void HostScoring::Print()
{
  printf("HostScoring: Summary\n");
  printf("HostScoring: Buffer Capacity %d\n", fBufferCapacity);
  printf("HostScoring: Flush Limit %f\n", fFlushLimit);
  printf("HostScoring: Buffer Start %d\n", fBufferStart);
}