// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef SCORING_H
#define SCORING_H

// #include "BufferHandle.h" 
#include <AdePT/core/AdePTScoringTemplate.cuh>

#include "ResourceManagement.h"
#include <AdePT/core/HostScoringStruct.cuh>
#include <VecGeom/navigation/NavStateIndex.h>

#include <atomic>
#include <deque>
#include <shared_mutex>
#include <array>

namespace AsyncAdePT {

/// Struct holding GPU hits to be used both on host and device.
struct HitScoringBuffer {
  GPUHit *hitBuffer_dev     = nullptr;
  unsigned int fSlotCounter = 0;
  unsigned int fNSlot       = 0;

  __device__ GPUHit &GetNextSlot();
};

extern __device__ HitScoringBuffer gHitScoringBuffer_dev;

struct BufferHandle {
  HitScoringBuffer hitScoringInfo;
  GPUHit *hostBuffer;
  enum class State { Free, OnDevice, OnDeviceNeedTransferToHost, TransferToHost, NeedHostProcessing };
  std::atomic<State> state;
};

class HitScoring {
  unique_ptr_cuda<GPUHit> fGPUHitBuffer_dev;
  unique_ptr_cuda<GPUHit, CudaHostDeleter<GPUHit>> fGPUHitBuffer_host;

  std::array<BufferHandle, 2> fBuffers;
  // BufferHandle fBuffers[2];  // Replace std::array with a raw array

  void *fHitScoringBuffer_deviceAddress = nullptr;
  unsigned int fHitCapacity;
  unsigned short fActiveBuffer = 0;
  unique_ptr_cuda<std::byte> fGPUSortAuxMemory;
  std::size_t fGPUSortAuxMemorySize;

  std::vector<std::deque<std::shared_ptr<const std::vector<GPUHit>>>> fHitQueues;
  mutable std::shared_mutex fProcessingHitsMutex;

  void ProcessBuffer(BufferHandle &handle);

public:
  HitScoring(unsigned int hitCapacity, unsigned int nThread);
  unsigned int HitCapacity() const { return fHitCapacity; }
  void SwapDeviceBuffers(cudaStream_t cudaStream);
  bool ProcessHits();
  bool ReadyToSwapBuffers() const
  {
    return std::any_of(fBuffers.begin(), fBuffers.end(),
                       [](const auto &handle) { return handle.state == BufferHandle::State::Free; });
  }
  void TransferHitsToHost(cudaStream_t cudaStreamForHitCopy);
  std::shared_ptr<const std::vector<GPUHit>> GetNextHitsVector(unsigned int threadId);
};

struct PerEventScoring {
  GlobalCounters fGlobalCounters;
  PerEventScoring *const fScoring_dev;

  PerEventScoring(PerEventScoring *gpuScoring) : fScoring_dev{gpuScoring} { ClearGPU(); }
  PerEventScoring(PerEventScoring &&other) = default;
  ~PerEventScoring()                       = default;

  /// @brief Copy hits to host for a single event
  void CopyToHost(cudaStream_t cudaStream = 0);

  /// @brief Clear hits on device to reuse for next event
  void ClearGPU(cudaStream_t cudaStream = 0);

  /// @brief Print scoring info
  void Print() { fGlobalCounters.Print(); };
};

} // namespace AsyncAdePT

namespace adept_scoring {

/// @brief Record a hit
template <>
__device__ void RecordHit(AsyncAdePT::PerEventScoring * /*scoring*/, int aParentID, char aParticleType,
                          double aStepLength, double aTotalEnergyDeposit, vecgeom::NavigationState const *aPreState,
                          vecgeom::Vector3D<Precision> const *aPrePosition,
                          vecgeom::Vector3D<Precision> const *aPreMomentumDirection,
                          vecgeom::Vector3D<Precision> const * /*aPrePolarization*/, double aPreEKin, double aPreCharge,
                          vecgeom::NavigationState const *aPostState, vecgeom::Vector3D<Precision> const *aPostPosition,
                          vecgeom::Vector3D<Precision> const *aPostMomentumDirection,
                          vecgeom::Vector3D<Precision> const * /*aPostPolarization*/, double aPostEKin,
                          double aPostCharge, unsigned int eventID, short threadID);

/// @brief Account for the number of produced secondaries
/// @details Atomically increase the number of produced secondaries.
template <>
__device__ void AccountProduced(AsyncAdePT::PerEventScoring *scoring, int num_ele, int num_pos, int num_gam);

} // namespace adept_scoring

#endif
