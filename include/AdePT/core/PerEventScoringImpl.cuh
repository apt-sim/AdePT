// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef PER_EVENT_SCORING_CUH
#define PER_EVENT_SCORING_CUH

#include <AdePT/core/PerEventScoringStruct.cuh>
#include <AdePT/base/ResourceManagement.cuh>
#include <AdePT/core/AdePTScoringTemplate.cuh>
#include <AdePT/core/ScoringCommons.hh>
#include <AdePT/copcore/Global.h>

#include <VecGeom/navigation/NavigationState.h>

#include <atomic>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <array>
#include <chrono>
#include <thread>
#include <condition_variable>

#include <cub/device/device_merge_sort.cuh>

// Comparison for sorting tracks into events on device:
struct CompareGPUHits {
  __device__ bool operator()(const GPUHit &lhs, const GPUHit &rhs) const { return lhs.threadId < rhs.threadId; }
};

namespace AsyncAdePT {

/// Struct holding GPU hits to be used both on host and device.
struct HitScoringBuffer {
  GPUHit *hitBuffer_dev     = nullptr;
  unsigned int *fSlotCounter = nullptr;  // Array of per-thread counters
  unsigned int fNSlot       = 0;
  unsigned int fNThreads    = 0;

  // __host__ __device__ HitScoringBuffer() = default;

  // __host__ void AllocateSlotCounterMemory(unsigned int threads) {
  //   // Allocate per-thread slot counters on device
  //   COPCORE_CUDA_CHECK(cudaMalloc(&fSlotCounter, sizeof(unsigned int) * threads));
  //   COPCORE_CUDA_CHECK(cudaMemset(fSlotCounter, 0, sizeof(unsigned int) * threads));
  // }

  // __host__ ~HitScoringBuffer() {
  //   // Free device memory
  //   if (fSlotCounter) cudaFree(fSlotCounter);
  // }

  __host__ __device__ unsigned int GetMaxSlotCount() {
    unsigned int maxVal = 0;
    for (unsigned int i = 0; i < fNThreads; ++i) {
      maxVal = vecCore::math::Max(maxVal, fSlotCounter[i]);
    }
    return maxVal;
  }

  __device__ GPUHit &GetNextSlot(unsigned int threadId)
  {
    // printf("Thread %u accessing fSlotCounter at address: %p\n", threadId, fSlotCounter);
    if (!fSlotCounter) {
        printf("ERROR: SLOTCOUNTER IS NULLPTR\n");
    }
    const auto slotIndex = atomicAdd(&fSlotCounter[threadId], 1);
    if (slotIndex >= fNSlot) {
      printf("Trying to score hit #%d with only %d slots\n", slotIndex, fNSlot);
      COPCORE_EXCEPTION("Out of slots in HitScoringBuffer::NextSlot");
    }
    return hitBuffer_dev[threadId * fNSlot + slotIndex];
  }
};

__device__ HitScoringBuffer gHitScoringBuffer_dev;

#ifdef __SANITIZE_ADDRESS__
#include <sanitizer/asan_interface.h>
#endif

bool isValidPointer(void* ptr) {
#ifdef __SANITIZE_ADDRESS__
  return !__asan_address_is_poisoned(ptr);
#else
  return ptr != nullptr;
#endif
}

struct BufferHandle {
  HitScoringBuffer hitScoringInfo;
  GPUHit *hostBuffer;
  unsigned int *hostBufferCount; 
  enum class State { Free, OnDevice, OnDeviceNeedTransferToHost, TransferToHost, NeedHostProcessing };
  std::atomic<State> state;
  std::atomic<short> refcount = 0;

  void reset() {
    // std::cout << "Resetting buffer handle: " << this 
    //           << " | Refcount: " << refcount.load() 
    //           << " | State: " << static_cast<int>(state.load())
    //           << " | hitScoringInfo.fSlotCounter: " << (void*)hitScoringInfo.fSlotCounter
    //           << std::endl;

    if (!hitScoringInfo.fSlotCounter) {
        std::cerr << "ERROR: fSlotCounter is NULL at reset!\n";
        return;
    }

    if (refcount.load() != 0) {
      std::cerr << "Error: Attempting to reset a buffer with nonzero refcount!" << std::endl;
      std::abort();
    }

  // if (!isValidPointer(hitScoringInfo.fSlotCounter)) {
  //   std::cerr << "ERROR: Trying to reset invalid memory in BufferHandle!" << std::endl;
  //   return;
  // }

  //   for (int i = 0; i < hitScoringInfo.fNThreads; i++) {
  //     std::cout << "Writing to: " << (void*)&hitScoringInfo.fSlotCounter[i] 
  //                 << " (before=" << hitScoringInfo.fSlotCounter[i] << ")" << std::endl;
  //     hitScoringInfo.fSlotCounter[i] = 0;
  //   }
    state.store(State::Free, std::memory_order_release);  // Mark buffer as free
  }

  void increment() {
    refcount.fetch_add(1, std::memory_order_relaxed);
    // std::cout << " incrementing refcount " << refcount.load() << std::endl; 
  }
  void decrement(unsigned int threadId) {

    int prev = refcount.load();
    // std::cout << "Before decrement: " << prev << " | Thread: " << threadId << std::endl;



    refcount.fetch_sub(1, std::memory_order_acq_rel);
    // std::cout << "After decrement: " << refcount.load() << " | Thread: " << threadId << std::endl;

    // if (refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      // Last worker, reset state for reuse
      // std::cout << std::dec << "worker " << threadId << " releasing and setting to Free " << std::endl;
      // reset();
    // }
    // std::cout << std::dec << "worker " << threadId << " refcount after decreasing:  " << refcount.load() << std::endl;
  }

};

// TODO: Rename this. Maybe ScoringState? Check usage in GPUstate
class HitScoring {
  unique_ptr_cuda<GPUHit> fGPUHitBuffer_dev;
  unique_ptr_cuda<GPUHit, CudaHostDeleter<GPUHit>> fGPUHitBuffer_host;
  unique_ptr_cuda<unsigned int> fGPUHitBufferCount_dev;
  unique_ptr_cuda<unsigned int, CudaHostDeleter<unsigned int>> fGPUHitBufferCount_host;

  std::array<BufferHandle, 2> fBuffers;

  void *fHitScoringBuffer_deviceAddress = nullptr;
  unsigned int fHitCapacity;
  unsigned short fActiveBuffer = 0;
  unique_ptr_cuda<std::byte> fGPUSortAuxMemory;
  std::size_t fGPUSortAuxMemorySize;

  // std::vector<std::deque<std::shared_ptr<const std::vector<GPUHit>>>> fHitQueues;
  std::vector<std::deque<BufferHandle*>> fHitQueues;

  mutable std::shared_mutex fProcessingHitsMutex;

  using GPUHitVectorPtr = std::shared_ptr<const std::vector<GPUHit>>;
  using HitDeque        = std::deque<GPUHitVectorPtr>;
  using HitQueueVector  = std::vector<HitDeque>;

  inline size_t calculateMemoryUsage(const HitQueueVector &fHitQueues)
  {
    size_t totalMemory = 0;

    for (const auto &dq : fHitQueues) {
      for (const auto &ptr : dq) {
        if (ptr) {
          totalMemory += sizeof(*ptr);
          totalMemory += ptr->size() * sizeof(GPUHit); // Actual GPUHit data
        }
      }
    }
    return totalMemory;
  }

  void ProcessBuffer(BufferHandle &handle, std::condition_variable &cvG4Workers, std::unique_lock<std::shared_mutex> &lock)
  {
    // We are assuming that the caller holds a lock on fProcessingHitsMutex.
    if (handle.state == BufferHandle::State::NeedHostProcessing) {

      // std::cout << "Total Memory Used in fHitQueues: " << calculateMemoryUsage(fHitQueues) / 1024.0 / 1024.0 / 1024.0
      // << " GB" << std::endl;

      // FIXME this doesn't work anymore, since the fSlotCounter is now an array!
      // auto begin = handle.hostBuffer;
      // auto end   = handle.hostBuffer + handle.hitScoringInfo.fSlotCounter;

      // OPTIONAL: print size of buffer 
      // size_t memoryUsed = (end - begin) * sizeof(GPUHit);
      // std::cout << "Memory in hit buffer to be scored: " << memoryUsed / 1024. / 1024. /1024. << " GB" << std::endl;

      handle.refcount.store(0, std::memory_order_relaxed);

      // std::cout << " Pushing back handles to queues " << fHitQueues.size() << std::endl;
      for (auto &queues : fHitQueues) {
        queues.push_back(&handle);
        handle.increment();
      }

      lock.unlock();
      // std::cout << "Notifying G4 workers..." << std::endl;
      cvG4Workers.notify_all();

      while (handle.refcount.load(std::memory_order_acquire)  != 0) {
        // std::cout << " Waiting for G4 workers... State: " 
        //           << GetStateName(handle.state.load()) 
        //           << " Refcount: " << handle.refcount.load() << std::endl;
        // std::atomic_thread_fence(std::memory_order_seq_cst);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      lock.lock();
      handle.reset();
      // std::cout << "G4 workers seem to have done their job! Final State: " 
      //           << GetStateName(handle.state.load()) 
      //           << " Refcount: " << handle.refcount.load() << std::endl;
      // PrintBufferStates();

      // while (begin != end) {
      //   short threadId = begin->threadId; // Get threadId of first hit in the range

      //   // linear search, slower, doesn't require a sorted array
      //   // auto threadEnd = std::find_if(begin, end,
      //   //       [threadId](const GPUHit &hit) { return threadId != hit.threadId; });

      //   // binary search, faster but requires a sorted array
      //   auto threadEnd =
      //       std::upper_bound(begin, end, threadId, [](short id, const GPUHit &hit) { return id < hit.threadId; });

      //   // Copy hits into a unique pointer and push it to workers queue
      //   auto HitsPerThread = std::make_unique<std::vector<GPUHit>>(begin, threadEnd);
      //   fHitQueues[threadId].push_back(std::move(HitsPerThread));

      //   begin = threadEnd; // set begin to start of the threadId
      // }

      // handle.hitScoringInfo.fSlotCounter = 0;
      // handle.state                       = BufferHandle::State::Free;

      // std::cout << "After pushing hitVector: Total Memory Used in fHitQueues: " << calculateMemoryUsage(fHitQueues)
      // / 1024.0 / 1024.0 / 1024.0 << " GB" << std::endl;
    }
  }

public:
  HitScoring(unsigned int hitCapacity, unsigned int nThread) : fHitCapacity{hitCapacity}, fHitQueues(nThread)
  {
    // We use a single allocation for both buffers:
    GPUHit *gpuHits = nullptr;
    COPCORE_CUDA_CHECK(cudaMallocHost(&gpuHits, sizeof(GPUHit) * 2 * fHitCapacity));
    fGPUHitBuffer_host.reset(gpuHits);

    auto result = cudaMalloc(&gpuHits, sizeof(GPUHit) * 2 * fHitCapacity);
    if (result != cudaSuccess) throw std::invalid_argument{"No space to allocate hit buffer."};
    fGPUHitBuffer_dev.reset(gpuHits);

    unsigned int *buffer_count = nullptr;
    COPCORE_CUDA_CHECK(cudaMallocHost(&buffer_count, sizeof(unsigned int) * 2 * nThread));
    fGPUHitBufferCount_host.reset(buffer_count);

    result = cudaMalloc(&buffer_count, sizeof(unsigned int) * 2 * nThread);
    if (result != cudaSuccess) throw std::invalid_argument{"No space to allocate hit buffer."};
    fGPUHitBufferCount_dev.reset(buffer_count);


    // Init buffers for on-device sorting of hits:
    // Determine device storage requirements for on-device sorting.
    // result = cub::DeviceMergeSort::SortKeys(nullptr, fGPUSortAuxMemorySize, fGPUHitBuffer_dev.get(), fHitCapacity,
    //                                         CompareGPUHits{});
    // if (result != cudaSuccess) throw std::invalid_argument{"No space for hit sorting on device."};

    // std::byte *gpuSortingMem;
    // result = cudaMalloc(&gpuSortingMem, fGPUSortAuxMemorySize);
    // if (result != cudaSuccess) throw std::invalid_argument{"No space to allocate hit sorting buffer."};
    // fGPUSortAuxMemory.reset(gpuSortingMem);

    // Store buffer data in structs
    // fBuffers[0].hitScoringInfo = HitScoringBuffer{fGPUHitBuffer_dev.get(), 0, fHitCapacity};
    fBuffers[0].hitScoringInfo = HitScoringBuffer{fGPUHitBuffer_dev.get(), fGPUHitBufferCount_dev.get(), fHitCapacity/nThread, nThread};
    fBuffers[0].hostBuffer     = fGPUHitBuffer_host.get();
    fBuffers[0].hostBufferCount = fGPUHitBufferCount_host.get();
    fBuffers[0].state          = BufferHandle::State::OnDevice;

    // fBuffers[1].hitScoringInfo = HitScoringBuffer{fGPUHitBuffer_dev.get() + fHitCapacity, 0, fHitCapacity};
    fBuffers[1].hitScoringInfo = HitScoringBuffer{fGPUHitBuffer_dev.get() + fHitCapacity, fGPUHitBufferCount_dev.get() + nThread, fHitCapacity/nThread, nThread};
    fBuffers[1].hostBuffer     = fGPUHitBuffer_host.get() + fHitCapacity;
    fBuffers[1].hostBufferCount = fGPUHitBufferCount_host.get() + nThread;
    fBuffers[1].state          = BufferHandle::State::Free;

    // fBuffers[2].hitScoringInfo = HitScoringBuffer{fGPUHitBuffer_dev.get() + 2*fHitCapacity, 0, fHitCapacity};
    // fBuffers[2].hostBuffer     = fGPUHitBuffer_host.get() + 2*fHitCapacity;
    // fBuffers[2].state          = BufferHandle::State::Free;

    COPCORE_CUDA_CHECK(cudaGetSymbolAddress(&fHitScoringBuffer_deviceAddress, gHitScoringBuffer_dev));
    assert(fHitScoringBuffer_deviceAddress != nullptr);
    COPCORE_CUDA_CHECK(cudaMemcpy(fHitScoringBuffer_deviceAddress, &fBuffers[0].hitScoringInfo,
                                  sizeof(HitScoringBuffer), cudaMemcpyHostToDevice));
  }

  unsigned int HitCapacity() const { return fHitCapacity; }

  void SwapDeviceBuffers(cudaStream_t cudaStream)
  {
    // printf("CALLING SWAP printing states\n");
    // PrintBufferStates();
    // Ensure that host side has been processed:
    auto &currentBuffer = fBuffers[fActiveBuffer];
    if (currentBuffer.state != BufferHandle::State::OnDevice)
      throw std::logic_error(__FILE__ + std::to_string(__LINE__) + ": On-device buffer in wrong state");

    // Get new buffer info from device:
    auto &currentHitInfo = currentBuffer.hitScoringInfo;
    COPCORE_CUDA_CHECK(cudaMemcpyAsync(&currentHitInfo, fHitScoringBuffer_deviceAddress, sizeof(HitScoringBuffer),
                                       cudaMemcpyDefault, cudaStream));

    // HitScoringBuffer* deviceBuffer = static_cast<HitScoringBuffer*>(fHitScoringBuffer_deviceAddress);

    // // Copy the SlotCounterArray
    // COPCORE_CUDA_CHECK(cudaMemcpyAsync(currentHitInfo.fSlotCounter, 
    //                                    deviceBuffer->fSlotCounter,
    //                                    sizeof(unsigned int) * currentHitInfo.fNThreads,
    //                                    cudaMemcpyDeviceToDevice, cudaStream));

    // Execute the swap:
    // printf("Before Swap buffer 0: fSlotCounter = %p\n", fBuffers[0].hitScoringInfo.fSlotCounter);
    // printf("Before Swap buffer 1: fSlotCounter = %p\n", fBuffers[1].hitScoringInfo.fSlotCounter);

    fActiveBuffer          = (fActiveBuffer + 1) % fBuffers.size();
    // printf("After Swap: fSlotCounter = %p\n", fBuffers[fActiveBuffer].hitScoringInfo.fSlotCounter);
    auto &nextDeviceBuffer = fBuffers[fActiveBuffer];
    while (nextDeviceBuffer.state != BufferHandle::State::Free) {
      std::cerr << __func__ << " Warning: Another thread should have processed the hits.\n";
    }
    // assert(nextDeviceBuffer.state == BufferHandle::State::Free && nextDeviceBuffer.hitScoringInfo.fSlotCounter == 0);

    nextDeviceBuffer.state = BufferHandle::State::OnDevice;
    COPCORE_CUDA_CHECK(cudaMemcpyAsync(fHitScoringBuffer_deviceAddress, &nextDeviceBuffer.hitScoringInfo,
                                       sizeof(HitScoringBuffer), cudaMemcpyDefault, cudaStream));
    // COPCORE_CUDA_CHECK(cudaMemcpyAsync(deviceBuffer->fSlotCounter,
    //                                    nextDeviceBuffer.hitScoringInfo.fSlotCounter,
    //                                    sizeof(unsigned int) * nextDeviceBuffer.hitScoringInfo.fNThreads,
    //                                    cudaMemcpyDeviceToDevice, cudaStream));

    COPCORE_CUDA_CHECK(cudaStreamSynchronize(cudaStream));
    currentBuffer.state = BufferHandle::State::OnDeviceNeedTransferToHost;
  }

  bool ProcessHits(std::condition_variable &cvG4Workers)
  {
    std::unique_lock lock{fProcessingHitsMutex, std::defer_lock};
    bool haveNewHits = false;

    while (std::any_of(fBuffers.begin(), fBuffers.end(),
                       [](auto &buffer) { return buffer.state >= BufferHandle::State::TransferToHost; })) {
      for (auto &handle : fBuffers) {
        if (handle.state == BufferHandle::State::NeedHostProcessing) {
          if (!lock) lock.lock();
          haveNewHits = true;

          // Possible timing
          // auto start = std::chrono::high_resolution_clock::now();
          ProcessBuffer(handle, cvG4Workers, lock);
          // auto end = std::chrono::high_resolution_clock::now();
          // std::chrono::duration<double> elapsed = end - start;
          //     std::cout << "BUFFER Processing time: " << elapsed.count() << " seconds" << std::endl;

          // lock.unlock();
        }
      }
    }

      // std::cout << " Finished ProcessBuffer, states :" << std::endl;
      // PrintBufferStates();

    return haveNewHits;
  }

  bool ReadyToSwapBuffers() const
  {
    return std::any_of(fBuffers.begin(), fBuffers.end(),
                       [](const auto &handle) { return handle.state == BufferHandle::State::Free; });
  }

  std::string GetStateName(BufferHandle::State state) const {
    switch (state) {
      case BufferHandle::State::Free: return "Free";
      case BufferHandle::State::OnDevice: return "OnDevice";
      case BufferHandle::State::OnDeviceNeedTransferToHost: return "OnDeviceNeedTransferToHost";
      case BufferHandle::State::TransferToHost: return "TransferToHost";
      case BufferHandle::State::NeedHostProcessing: return "NeedHostProcessing";
      default: return "Unknown";
    }
  }

  void PrintBufferStates() const {
    std::cout << "Buffer States: ";
    
    for (const auto &handle : fBuffers) {
      std::cout << "[State: " << GetStateName(handle.state.load()) 
                << ", Refcount: " << handle.refcount.load() << "] ";
    }
    
    std::cout << std::endl;
  }

  /// Copy the current contents of the GPU hit buffer to host.
  void TransferHitsToHost(cudaStream_t cudaStreamForHitCopy)
  {
    for (auto &buffer : fBuffers) {
      if (buffer.state != BufferHandle::State::OnDeviceNeedTransferToHost) continue;

      buffer.state = BufferHandle::State::TransferToHost;
      // assert(buffer.hitScoringInfo.fSlotCounter[0] < fHitCapacity);
      // if( buffer.hitScoringInfo.fSlotCounter[0] > fHitCapacity ) {
      //   printf("Danger! buffer.hitScoringInfo.fSlotCounter[0] %u fHitCapacity %u \n", buffer.hitScoringInfo.fSlotCounter[0], fHitCapacity);
      // }

      auto bufferBegin = buffer.hitScoringInfo.hitBuffer_dev;

      // cub::DeviceMergeSort::SortKeys(fGPUSortAuxMemory.get(), fGPUSortAuxMemorySize, bufferBegin,
      //                                buffer.hitScoringInfo.fSlotCounter, CompareGPUHits{}, cudaStreamForHitCopy);

      // Copy SlotCounterArray
      COPCORE_CUDA_CHECK(cudaMemcpyAsync(buffer.hostBufferCount, buffer.hitScoringInfo.fSlotCounter,
                                         sizeof(unsigned int) * buffer.hitScoringInfo.fNThreads, cudaMemcpyDefault,
                                         cudaStreamForHitCopy));

      // unfortunately, we need to synchronize since we need to know the offsets if we want to copy only the used data.
      COPCORE_CUDA_CHECK(cudaStreamSynchronize(cudaStreamForHitCopy));

      // Copy out the hits:
      // The start address on device is always i * fNSlot (Slots per thread), and we copy always to
      // the offset of the previous copy, to get a compact buffer on host.
      unsigned int offset = 0;
      for (int i = 0; i < buffer.hitScoringInfo.fNThreads; i++) {
        COPCORE_CUDA_CHECK(cudaMemcpyAsync(buffer.hostBuffer + offset, bufferBegin + i * buffer.hitScoringInfo.fNSlot,
                                   sizeof(GPUHit) * buffer.hostBufferCount[i], cudaMemcpyDefault,
                                   cudaStreamForHitCopy));
        offset += buffer.hostBufferCount[i];
      }

        // std::cout << " offset " << offset; 
        // std::cout << " buffer.hostBufferCount[i]" << buffer.hostBufferCount[i];
        // std::cout << " Adress host begin buffer.hostBuffer + offset" << buffer.hostBuffer + offset;
        // std::cout << " address device begin bufferBegin + offset" << bufferBegin + offset;

      COPCORE_CUDA_CHECK(cudaMemsetAsync(buffer.hitScoringInfo.fSlotCounter, 0, 
                                   sizeof(unsigned int) * buffer.hitScoringInfo.fNThreads, 
                                   cudaStreamForHitCopy));

      COPCORE_CUDA_CHECK(cudaLaunchHostFunc(
          cudaStreamForHitCopy,
          [](void *arg) { static_cast<BufferHandle *>(arg)->state = BufferHandle::State::NeedHostProcessing; },
          &buffer));
    }
  }

  // comment out old function for now as we cannot keep both alive if we change the interface
  // std::shared_ptr<const std::vector<GPUHit>> GetNextHitsVector(unsigned int threadId)
  // {
  //   assert(threadId < fHitQueues.size());
  //   std::shared_lock lock{fProcessingHitsMutex};

  //   if (fHitQueues[threadId].empty())
  //     return nullptr;
  //   else {
  //     auto ret = fHitQueues[threadId].front();
  //     fHitQueues[threadId].pop_front();
  //     return ret;
  //   }
  // }

  BufferHandle* GetNextHitsHandle(unsigned int threadId)
  {
    assert(threadId < fHitQueues.size());
    std::shared_lock lock{fProcessingHitsMutex}; // read only, can use shared lock

    if (fHitQueues[threadId].empty())
      return nullptr;
    else {
      auto ret = fHitQueues[threadId].front();
      // fHitQueues[threadId].pop_front(); // don't pop the front, we still need to decrement before we can pop it
      return ret;
    }
  }

  void CloseHitsHandle(unsigned int threadId)
  {
    assert(threadId < fHitQueues.size());
    std::unique_lock lock{fProcessingHitsMutex}; // popping queue, requires unique lock

    if (fHitQueues[threadId].empty()) 
      throw std::invalid_argument{"Error, no hitQueue to close"};
    else {
      auto& ret = fHitQueues[threadId].front();
      ret->decrement(threadId);
      fHitQueues[threadId].pop_front();
    }
  }

};

// Implement Cuda-dependent functionality from PerEventScoring

void PerEventScoring::ClearGPU(cudaStream_t cudaStream)
{
  COPCORE_CUDA_CHECK(cudaMemsetAsync(fScoring_dev, 0, sizeof(GlobalCounters), cudaStream));
  COPCORE_CUDA_CHECK(cudaStreamSynchronize(cudaStream));
}

void PerEventScoring::CopyToHost(cudaStream_t cudaStream)
{
  const auto oldPointer = fScoring_dev;
  COPCORE_CUDA_CHECK(
      cudaMemcpyAsync(&fGlobalCounters, fScoring_dev, sizeof(GlobalCounters), cudaMemcpyDeviceToHost, cudaStream));
  COPCORE_CUDA_CHECK(cudaStreamSynchronize(cudaStream));
  assert(oldPointer == fScoring_dev);
  (void)oldPointer;
}

} // namespace AsyncAdePT

namespace adept_scoring {

/// @brief Record a hit
template <>
__device__ void RecordHit(AsyncAdePT::PerEventScoring * /*scoring*/, int aParentID, char aParticleType,
                          double aStepLength, double aTotalEnergyDeposit, vecgeom::NavigationState const &aPreState,
                          vecgeom::Vector3D<Precision> const &aPrePosition,
                          vecgeom::Vector3D<Precision> const &aPreMomentumDirection, double aPreEKin, double aPreCharge,
                          vecgeom::NavigationState const &aPostState, vecgeom::Vector3D<Precision> const &aPostPosition,
                          vecgeom::Vector3D<Precision> const &aPostMomentumDirection, double aPostEKin,
                          double aPostCharge, unsigned int eventID, short threadID)
{
  // Acquire a hit slot
  GPUHit &aGPUHit = AsyncAdePT::gHitScoringBuffer_dev.GetNextSlot(threadID);

  // Fill the required data
  FillHit(aGPUHit, aParentID, aParticleType, aStepLength, aTotalEnergyDeposit, aPreState, aPrePosition,
          aPreMomentumDirection, aPreEKin, aPreCharge, aPostState, aPostPosition, aPostMomentumDirection, aPostEKin,
          aPostCharge, eventID, threadID);
}

/// @brief Account for the number of produced secondaries
/// @details Atomically increase the number of produced secondaries.
template <>
__device__ void AccountProduced(AsyncAdePT::PerEventScoring *scoring, int num_ele, int num_pos, int num_gam)
{
  atomicAdd(&scoring->fGlobalCounters.numElectrons, num_ele);
  atomicAdd(&scoring->fGlobalCounters.numPositrons, num_pos);
  atomicAdd(&scoring->fGlobalCounters.numGammas, num_gam);
}

} // namespace adept_scoring

#endif
