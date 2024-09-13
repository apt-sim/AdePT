// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

// This file contains the specialization of the methods defined on AdePTScoringTemplate.cuh
// for doing scoring on the Host, calling the user-defined sensitive detector code

#include <AdePT/core/HostScoringStruct.cuh>

// CUDA Methods specific to HostScoring

/// @brief Move the buffer start on GPU
__global__ void UpdateBufferStartGPU(HostScoring *hostScoring_dev, unsigned int aBufferStart)
{
  hostScoring_dev->fBufferStart = aBufferStart;
}

/// @brief Update the buffer usage on GPU
/// @details This function is to be called after copying the buffer contents back to the host. Updates the
/// usage of the buffer based on usage prior to the copy and current usage.
__global__ void UpdateBufferUsageGPU(HostScoring *hostScoring_dev, unsigned int aUsedSlotsBeforeCopy)
{
  // Must be performed atomically
  *(hostScoring_dev->fUsedSlots_dev) -= aUsedSlotsBeforeCopy;
}

/// @brief Get index of the next free hit slot
__device__ __forceinline__ unsigned int GetNextFreeHitIndex(HostScoring *hostScoring_dev)
{
  // Atomic addition, each GPU thread accessing concurrently gets a different slot
  unsigned int next = hostScoring_dev->fNextFreeHit_dev->fetch_add(1);
  assert(next >= hostScoring_dev->fBufferStart);
  if ((next - hostScoring_dev->fBufferStart) >= hostScoring_dev->fBufferCapacity) {
    COPCORE_EXCEPTION("No slot available in Hit Buffer");
  }
  (*hostScoring_dev->fUsedSlots_dev)++;
  // Circular buffer, wrap around to the beginning if needed
  return next % hostScoring_dev->fBufferCapacity;
}

/// @brief Get reference to the next free hit struct in the buffer
__device__ __forceinline__ GPUHit *GetNextFreeHit(HostScoring *hostScoring_dev)
{
unsigned int aHitIndex = GetNextFreeHitIndex(hostScoring_dev);
  assert(aHitIndex < hostScoring_dev->fBufferCapacity);
  return &hostScoring_dev->fGPUHitsBuffer_dev[aHitIndex];
}

/// @brief Utility function to copy a 3D vector, used for filling the Step Points
__device__ __forceinline__ void Copy3DVector(vecgeom::Vector3D<Precision> *source,
                                            vecgeom::Vector3D<Precision> *destination)
{
  destination->x() = source->x();
  destination->y() = source->y();
  destination->z() = source->z();
}

/// @brief Copy the hits buffer to the host
void CopyHitsToHost(HostScoring &hostScoring, HostScoring::Stats &statsHost, HostScoring *hostScoring_dev, cudaStream_t &stream)
{
  // Move the start of the hits buffer on GPU
  UpdateBufferStartGPU<<<1, 1, 0, stream>>>(hostScoring_dev, statsHost.fNextFreeHit);

  // Copy the hits to the host. We need to copy the entire buffer as the starting index may change
  COPCORE_CUDA_CHECK(cudaMemcpyAsync(hostScoring.fGPUHitsBuffer_host, hostScoring.fGPUHitsBuffer_dev, hostScoring.fBufferCapacity * sizeof(GPUHit),
                                     cudaMemcpyDeviceToHost, stream));

  // Update the used slots counter after all data has been copied, taking into account the slots filled in the meantime
  // NOTE: This functionality won't give any benefits as long as we sync & process the returned buffer right
  // after copying it, as the CPU thread will be processing hits instead of launching new kernels.
  UpdateBufferUsageGPU<<<1, 1, 0, stream>>>(hostScoring_dev, statsHost.fUsedSlots);
}

/// @brief Check if the buffer is filled over a certain capacity and copy the hits to the host if so
/// @return True if the buffer was transferred to the host
bool CheckAndFlush(HostScoring &hostScoring, HostScoring::Stats &statsHost, HostScoring *hostScoring_dev, cudaStream_t &stream)
{
  hostScoring.fStepsSinceLastFlush++;
  float aBufferUsage = (float)statsHost.fUsedSlots / hostScoring.fBufferCapacity;

  if (aBufferUsage > hostScoring.fFlushLimit) {
    hostScoring.fStepsSinceLastFlush = 0;
    CopyHitsToHost(hostScoring, statsHost, hostScoring_dev, stream);
    return true;
  }
  return false;
}

/// @brief Update the stats struct. To be called before copying it back.
/// @details This is an in-device update, meant to copy the state of the member variables we need
/// into a struct that can be copied back to the host
__device__ __forceinline__ void refresh_stats(HostScoring *hostScoring_dev)
{
  hostScoring_dev->fStats_dev->fUsedSlots   = hostScoring_dev->fUsedSlots_dev->load();
  hostScoring_dev->fStats_dev->fNextFreeHit = hostScoring_dev->fNextFreeHit_dev->load();
  hostScoring_dev->fStats_dev->fBufferStart = hostScoring_dev->fBufferStart;
}

/// @brief Copies the global stats struct back to the host, called after finishing a shower
void CopyGlobalCountersToHost(HostScoring &hostScoring, cudaStream_t &stream)
{
  COPCORE_CUDA_CHECK(cudaMemcpyAsync(hostScoring.fGlobalCounters_host, hostScoring.fGlobalCounters_dev, sizeof(GlobalCounters),
                                     cudaMemcpyDeviceToHost, stream));
}

// Specialization of CUDA Methods for HostScoring
#include <AdePT/core/AdePTScoringTemplate.cuh>
namespace adept_scoring
{
  /// @brief Allocate and initialize data structures on device
  template <>
  HostScoring* InitializeOnGPU(HostScoring *hostScoring)
  {
    // Allocate space for the hits buffer
    COPCORE_CUDA_CHECK(cudaMalloc(&hostScoring->fGPUHitsBuffer_dev, sizeof(GPUHit) * hostScoring->fBufferCapacity));

    // Allocate space for the global counters
    COPCORE_CUDA_CHECK(cudaMalloc(&hostScoring->fGlobalCounters_dev, sizeof(GlobalCounters)));

    // Allocate space for the atomic variables on device
    COPCORE_CUDA_CHECK(cudaMalloc(&hostScoring->fUsedSlots_dev, sizeof(adept::Atomic_t<unsigned int>)));
    COPCORE_CUDA_CHECK(cudaMalloc(&hostScoring->fNextFreeHit_dev, sizeof(adept::Atomic_t<unsigned int>)));

    // Allocate space for the stats on device
    // Allocate space for the global counters
    COPCORE_CUDA_CHECK(cudaMalloc(&hostScoring->fStats_dev, sizeof(HostScoring::Stats)));

    // Allocate space for the instance on GPU and copy the data members from the host
    // Now allocate space for the BasicScoring placeholder on device and copy the device pointers of components
    HostScoring *hostScoring_dev = nullptr;
    COPCORE_CUDA_CHECK(cudaMalloc(&hostScoring_dev, sizeof(HostScoring)));
    COPCORE_CUDA_CHECK(cudaMemcpy(hostScoring_dev, hostScoring, sizeof(HostScoring), cudaMemcpyHostToDevice));

    return hostScoring_dev;
  }

  template <>
  void FreeGPU(HostScoring *hostScoring, HostScoring *hostScoring_dev)
  {
    // Free hits buffer
    COPCORE_CUDA_CHECK(cudaFree(hostScoring->fGPUHitsBuffer_dev));
    // Free global counters
    COPCORE_CUDA_CHECK(cudaFree(hostScoring->fGlobalCounters_dev));
    // Free atomic variable instances
    COPCORE_CUDA_CHECK(cudaFree(hostScoring->fUsedSlots_dev));
    COPCORE_CUDA_CHECK(cudaFree(hostScoring->fNextFreeHit_dev));
    // Free Stats
    COPCORE_CUDA_CHECK(cudaFree(hostScoring->fStats_dev));
    // Free the space allocated for the GPU instance of this object
    COPCORE_CUDA_CHECK(cudaFree(hostScoring_dev));
  }

  /// @brief Record a hit
  template <>
  __device__ void RecordHit(HostScoring *hostScoring_dev, int aParentID, char aParticleType, double aStepLength,
                          double aTotalEnergyDeposit, vecgeom::NavigationState const *aPreState,
                          vecgeom::Vector3D<Precision> *aPrePosition,
                          vecgeom::Vector3D<Precision> *aPreMomentumDirection,
                          vecgeom::Vector3D<Precision> *aPrePolarization, double aPreEKin, double aPreCharge,
                          vecgeom::NavigationState const *aPostState, vecgeom::Vector3D<Precision> *aPostPosition,
                          vecgeom::Vector3D<Precision> *aPostMomentumDirection,
                          vecgeom::Vector3D<Precision> *aPostPolarization, double aPostEKin, double aPostCharge)
  {
    // Acquire a hit slot
    GPUHit *aGPUHit = GetNextFreeHit(hostScoring_dev);

    // Fill the required data
    aGPUHit->fParentID           = aParentID;
    aGPUHit->fParticleType       = aParticleType;
    aGPUHit->fStepLength         = aStepLength;
    aGPUHit->fTotalEnergyDeposit = aTotalEnergyDeposit;
    // Pre step point
    aGPUHit->fPreStepPoint.fNavigationState = *aPreState;
    Copy3DVector(aPrePosition, &(aGPUHit->fPreStepPoint.fPosition));
    Copy3DVector(aPreMomentumDirection, &(aGPUHit->fPreStepPoint.fMomentumDirection));
    // Copy3DVector(aPrePolarization, aGPUHit.fPreStepPoint.fPolarization);
    aGPUHit->fPreStepPoint.fEKin   = aPreEKin;
    aGPUHit->fPreStepPoint.fCharge = aPreCharge;
    // Post step point
    aGPUHit->fPostStepPoint.fNavigationState = *aPostState;
    Copy3DVector(aPostPosition, &(aGPUHit->fPostStepPoint.fPosition));
    Copy3DVector(aPostMomentumDirection, &(aGPUHit->fPostStepPoint.fMomentumDirection));
    // Copy3DVector(aPostPolarization, aGPUHit.fPostStepPoint.fPolarization);
    aGPUHit->fPostStepPoint.fEKin   = aPostEKin;
    aGPUHit->fPostStepPoint.fCharge = aPostCharge;
  }

  /// @brief Account for the number of produced secondaries
  /// @details Atomically increase the number of produced secondaries. These numbers are used as another
  /// way to compare the amount of work done with Geant4. This is not part of the scoring per se and is
  /// copied back at the end of a shower
  template <>
  __device__ void AccountProduced(HostScoring *hostScoring_dev, int num_ele, int num_pos, int num_gam)
  {
    // Increment number of secondaries
    atomicAdd(&hostScoring_dev->fGlobalCounters_dev->numElectrons, num_ele);
    atomicAdd(&hostScoring_dev->fGlobalCounters_dev->numPositrons, num_pos);
    atomicAdd(&hostScoring_dev->fGlobalCounters_dev->numGammas, num_gam);
  }

  template <>
  __device__  __forceinline__ void EndOfIterationGPU(HostScoring *hostScoring_dev)
  {
    // Update hit buffer stats
    refresh_stats(hostScoring_dev);
  }

  template <typename IntegrationLayer>
  inline void EndOfIteration(HostScoring &hostScoring, HostScoring *hostScoring_dev, cudaStream_t &stream, IntegrationLayer &integration)
  {
    // Copy host scoring stats from device to host
    // COPCORE_CUDA_CHECK(
    //       cudaMemcpyAsync(&hostScoring.fStats, hostScoring.fStats_dev, sizeof(HostScoring::Stats), cudaMemcpyDeviceToHost, stream));
    // COPCORE_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Check if we need to flush the hits buffer
    if (CheckAndFlush(hostScoring, hostScoring.fStats, hostScoring_dev, stream)) {
      // Synchronize the stream used to copy back the hits
      COPCORE_CUDA_CHECK(cudaStreamSynchronize(stream));
      // Process the hits on CPU
      integration.ProcessGPUHits(hostScoring, hostScoring.fStats);
    }
  }

  template <typename IntegrationLayer>
  inline void EndOfTransport(HostScoring &hostScoring, HostScoring *hostScoring_dev, cudaStream_t &stream, IntegrationLayer &integration)
  {
    // Transfer back scoring.
    CopyHitsToHost(hostScoring, hostScoring.fStats, hostScoring_dev, stream);
    // Transfer back the global counters
    // scoring->fGlobalCounters_dev->numKilled = inFlight;
    CopyGlobalCountersToHost(hostScoring, stream);
    COPCORE_CUDA_CHECK(cudaStreamSynchronize(stream));
    // Process the last hits on CPU
    integration.ProcessGPUHits(hostScoring, hostScoring.fStats);
  }
}



