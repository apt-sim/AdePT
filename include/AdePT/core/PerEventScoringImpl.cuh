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
  // enum class State { Free, OnDevice, OnDeviceNeedTransferToHost, TransferToHost, NeedHostProcessing };

  // enum class DeviceState { Free, Filling, NeedTransferToHost, TransferToHost };
  enum class HostState { Free, AwaitDeviceTransfer, TransferFromDevice, AwaitScoring, Scoring };

  // std::atomic<State> state;

  // std::atomic<DeviceState> deviceState;
  std::atomic<HostState> hostState;


  std::atomic<short> refCount = 0;

  void reset() {
    // std::cout << "Resetting buffer handle: " << this 
    //           << " | refCount: " << refCount.load() 
    //           << " | State: " << static_cast<int>(state.load())
    //           << " | hitScoringInfo.fSlotCounter: " << (void*)hitScoringInfo.fSlotCounter
    //           << std::endl;

    // if (!hitScoringInfo.fSlotCounter) {
    //     std::cerr << "ERROR: fSlotCounter is NULL at reset!\n";
    //     return;
    // }

    // if (refCount.load() != 0) {
    //   std::cerr << "Error: Attempting to reset a buffer with nonzero refCount!" << std::endl;
    //   std::abort();
    // }

  // if (!isValidPointer(hitScoringInfo.fSlotCounter)) {
  //   std::cerr << "ERROR: Trying to reset invalid memory in BufferHandle!" << std::endl;
  //   return;
  // }

  //   for (int i = 0; i < hitScoringInfo.fNThreads; i++) {
  //     std::cout << "Writing to: " << (void*)&hitScoringInfo.fSlotCounter[i] 
  //                 << " (before=" << hitScoringInfo.fSlotCounter[i] << ")" << std::endl;
  //     hitScoringInfo.fSlotCounter[i] = 0;
  //   }
    // FIXME reset HostState
    hostState = HostState::Free;
    // hostState.store(HostState::Free, std::memory_order_release);  // Mark buffer as free
  }

  void increment() {
    refCount += 1;
    // refCount.fetch_add(1, std::memory_order_relaxed);
    // std::cout << " incrementing refCount " << refCount.load() << std::endl; 
  }
  void decrement(unsigned int threadId) {

    // int prev = refCount.load();
    // std::cout << "Before decrement: " << prev << " | Thread: " << threadId << std::endl;



    // refCount.fetch_sub(1, std::memory_order_acq_rel);
    refCount--;
    if (refCount == 0) {
      // Last worker, reset state for reuse
      // std::cout << std::dec << "worker " << threadId << " releasing and setting to Free " << std::endl;
      reset();
    }
    // std::cout << "After decrement: " << refCount.load() << " | Thread: " << threadId << std::endl;

    // std::cout << std::dec << "worker " << threadId << " refCount after decreasing:  " << refCount.load() << std::endl;
  }

};

struct HitInfo {
  GPUHit * begin;
  GPUHit * end;
  std::shared_ptr<BufferHandle> hostBuffer;
  std::atomic_bool ScoringStarted = false;
  std::vector<GPUHit> holdoutBuffer;

  HitInfo(GPUHit* begin_, GPUHit* end_, std::shared_ptr<BufferHandle> buffer)
    : begin(begin_), end(end_), hostBuffer(std::move(buffer)) {}

  ~HitInfo() = default;

  HitInfo(const HitInfo&) = delete;
  HitInfo& operator=(const HitInfo&) = delete;

  HitInfo(HitInfo&& other) noexcept
    : begin(other.begin),
      end(other.end),
      hostBuffer(std::move(other.hostBuffer)),
      ScoringStarted(other.ScoringStarted.load()), // Read atomic value safely
      holdoutBuffer(std::move(other.holdoutBuffer)) {} // Move vector safely

  HitInfo& operator=(HitInfo&& other) noexcept {
    if (this != &other) {
      begin = other.begin;
      end = other.end;
      hostBuffer = std::move(other.hostBuffer);
      ScoringStarted.store(other.ScoringStarted.load()); // Safe atomic move
      holdoutBuffer = std::move(other.holdoutBuffer);
    }
    return *this;
  }
};

// TODO: Rename this. Maybe ScoringState? Check usage in GPUstate
class HitScoring {
  unique_ptr_cuda<GPUHit> fGPUHitBuffer_dev;
  unique_ptr_cuda<GPUHit, CudaHostDeleter<GPUHit>> fGPUHitBuffer_host;
  unique_ptr_cuda<unsigned int> fGPUHitBufferCount_dev;
  unique_ptr_cuda<unsigned int, CudaHostDeleter<unsigned int>> fGPUHitBufferCount_host;

  // FIXME have size 3 for bufferhandles
  std::array<BufferHandle, 3> fBuffers;


  enum class DeviceState { Free, Filling, NeedTransferToHost, TransferToHost };
  std::array<DeviceState, 2> fDeviceState;

  void *fHitScoringBuffer_deviceAddress = nullptr;
  unsigned int fHitCapacity;
  unsigned short fActiveBuffer = 0;
  unsigned short fActiveDeviceBuffer = 0;
  // FIXME: have fActiveDeviceBuffer and fActiveHostBuffer (let's see if this is really needed)
  unique_ptr_cuda<std::byte> fGPUSortAuxMemory;
  std::size_t fGPUSortAuxMemorySize;

  // std::vector<std::deque<std::shared_ptr<const std::vector<GPUHit>>>> fHitQueues;
  // std::vector<std::deque<BufferHandle*>> fHitQueues;
  std::vector<std::deque<HitInfo>> fHitQueues;


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
    // FIXME: use HostState
    if (handle.hostState == BufferHandle::HostState::Scoring) {

      // std::cout << "Total Memory Used in fHitQueues: " << calculateMemoryUsage(fHitQueues) / 1024.0 / 1024.0 / 1024.0
      // << " GB" << std::endl;

      // this doesn't work anymore, since the fSlotCounter is now an array!
      // auto begin = handle.hostBuffer;
      // auto end   = handle.hostBuffer + handle.hitScoringInfo.fSlotCounter;

      // OPTIONAL: print size of buffer 
      // size_t memoryUsed = (end - begin) * sizeof(GPUHit);
      // std::cout << "Memory in hit buffer to be scored: " << memoryUsed / 1024. / 1024. /1024. << " GB" << std::endl;

 #define RED "\033[31m"
#define RESET "\033[0m"
#define BOLD_RED "\033[1;31m"


      // create shared pointer with custom deleter: the shared pointer is passed to each HitInfo,
      // as soon as the last thread finishes scoring the range in HitInfo, the last shared pointer goes out of scope
      // and the custom deleter sets the HostState to free. 
      std::shared_ptr<BufferHandle> bufferHandlePtr = std::shared_ptr<BufferHandle>(&handle, [](BufferHandle* buf) {
        if (buf) {
          buf->hostState.store(BufferHandle::HostState::Free, std::memory_order_release);  // Custom deleter
        }
      });

      unsigned int offset = 0;
      for (int i = 0; i < fHitQueues.size(); i++) {

        GPUHit* begin = handle.hostBuffer + offset;
        GPUHit* end = handle.hostBuffer + offset + handle.hostBufferCount[i];


        HitInfo hitinfo{begin, end, bufferHandlePtr};

        if (begin != end) {
          lock.lock();
          // handle.increment();
          fHitQueues[i].push_back(std::move(hitinfo));
          lock.unlock();
        }
  // std::cout << BOLD_RED << "threadId " << i << " EventId " << begin->fEventId << " offset " << offset << " num hits to score " << handle.hostBufferCount[i] << RESET << std::endl;

        offset += handle.hostBufferCount[i];
      }

      // std::cout << "Notifying G4 workers..." << std::endl;
      cvG4Workers.notify_all();


      // Give G4 workers time to wake up
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(5ms);

      for (int i = 0; i < fHitQueues.size(); i++) {

        lock.lock();
        if (!fHitQueues[i].empty()) {
          auto& ret = fHitQueues[i].back(); // we just pushed to the back, so we need to check if the back is used!
          if (ret.ScoringStarted.load(std::memory_order_acquire) == true ) {
            lock.unlock();
          } else {

            assert(ret.begin && ret.end && ret.begin < ret.end);

            size_t numHits = ret.end - ret.begin;  // Compute the number of hits

            // std::cout << "G4Worker " << i << " was too slow, copying out " << numHits << " hits " << std::endl;

            ret.holdoutBuffer.resize(numHits);  // Allocate correct size
            std::copy(ret.begin, ret.end, ret.holdoutBuffer.begin());  // Copy data

            // Update pointers
            ret.begin = ret.holdoutBuffer.data();
            ret.end = ret.holdoutBuffer.data() + ret.holdoutBuffer.size();

            // Safely release the old buffer
            ret.hostBuffer.reset();

            ret.ScoringStarted.store(true, std::memory_order_release);

            lock.unlock();
          }
        
        } else {
          lock.unlock();
        }
      }

      // while (handle.refCount.load(std::memory_order_acquire)  != 0) {
      //   // std::cout << " Waiting for G4 workers... State: " 
      //   //           << GetStateName(handle.state.load()) 
      //   //           << " refCount: " << handle.refCount.load() << std::endl;
      //   // std::atomic_thread_fence(std::memory_order_seq_cst);
      //   std::this_thread::sleep_for(std::chrono::milliseconds(1));
      // }

      // lock.lock();
      // handle.reset();
      // std::cout << "G4 workers seem to have done their job! Final State: " 
      //           << GetStateName(handle.state.load()) 
      //           << " refCount: " << handle.refCount.load() << std::endl;
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
    // FIXME: have size 3 for HostBuffer
    GPUHit *gpuHits = nullptr;
    COPCORE_CUDA_CHECK(cudaMallocHost(&gpuHits, sizeof(GPUHit) * fBuffers.size() * fHitCapacity)); // 4
    fGPUHitBuffer_host.reset(gpuHits);

    auto result = cudaMalloc(&gpuHits, sizeof(GPUHit) * 2 * fHitCapacity);
    if (result != cudaSuccess) throw std::invalid_argument{"No space to allocate hit buffer."};
    fGPUHitBuffer_dev.reset(gpuHits);

    // FIXME: have size 3 for HostBuffer
    unsigned int *buffer_count = nullptr;
    COPCORE_CUDA_CHECK(cudaMallocHost(&buffer_count, sizeof(unsigned int) * fBuffers.size() * nThread)); // 4
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

    // FIXME: have HostState and DeviceState

    fDeviceState[0] = DeviceState::Filling;
    fDeviceState[1] = DeviceState::Free;


    fBuffers[0].hitScoringInfo = HitScoringBuffer{fGPUHitBuffer_dev.get(), fGPUHitBufferCount_dev.get(), fHitCapacity/nThread, nThread};
    fBuffers[0].hostBuffer     = fGPUHitBuffer_host.get();
    fBuffers[0].hostBufferCount = fGPUHitBufferCount_host.get();
    // fBuffers[0].state          = BufferHandle::State::OnDevice;
    // fBuffers[0].deviceState          = BufferHandle::DeviceState::Filling;
    fBuffers[0].hostState          = BufferHandle::HostState::Free;

    fBuffers[1].hitScoringInfo = HitScoringBuffer{fGPUHitBuffer_dev.get() + fHitCapacity, fGPUHitBufferCount_dev.get() + nThread, fHitCapacity/nThread, nThread};
    fBuffers[1].hostBuffer     = fGPUHitBuffer_host.get() + fHitCapacity;
    fBuffers[1].hostBufferCount = fGPUHitBufferCount_host.get() + nThread;
    // fBuffers[1].state          = BufferHandle::State::Free;
    // fBuffers[1].deviceState          = BufferHandle::DeviceState::Free;
    fBuffers[1].hostState          = BufferHandle::HostState::Free;

    // FIXME: have third buffer, it should point to the first GPU memory so we can get into a cycle? to be seen
    fBuffers[2].hitScoringInfo = HitScoringBuffer{fGPUHitBuffer_dev.get(), fGPUHitBufferCount_dev.get(), fHitCapacity/nThread, nThread};
    fBuffers[2].hostBuffer     = fGPUHitBuffer_host.get() + 2 * fHitCapacity;
    fBuffers[2].hostBufferCount = fGPUHitBufferCount_host.get() + 2* nThread;
    // fBuffers[2].deviceState          = BufferHandle::DeviceState::Free;
    fBuffers[2].hostState          = BufferHandle::HostState::Free;

    // fBuffers[3].hitScoringInfo = HitScoringBuffer{fGPUHitBuffer_dev.get() + fHitCapacity, fGPUHitBufferCount_dev.get() + nThread, fHitCapacity/nThread, nThread};
    // fBuffers[3].hostBuffer     = fGPUHitBuffer_host.get() + 3*fHitCapacity;
    // fBuffers[3].hostBufferCount = fGPUHitBufferCount_host.get() + 3* nThread;
    // fBuffers[3].hostState          = BufferHandle::HostState::Free;

    COPCORE_CUDA_CHECK(cudaGetSymbolAddress(&fHitScoringBuffer_deviceAddress, gHitScoringBuffer_dev));
    assert(fHitScoringBuffer_deviceAddress != nullptr);
    COPCORE_CUDA_CHECK(cudaMemcpy(fHitScoringBuffer_deviceAddress, &fBuffers[0].hitScoringInfo,
                                  sizeof(HitScoringBuffer), cudaMemcpyHostToDevice));
  }

  unsigned int HitCapacity() const { return fHitCapacity; }

  void SwapDeviceBuffers(cudaStream_t cudaStream)
  {

    // FIXME: full function

    // printf("CALLING SWAP printing states\n");
    // PrintDeviceBufferStates();
    // PrintHostBufferStates();
    // PrintBufferStates();
    // Ensure that host side has been processed:
    auto &currentBuffer = fBuffers[fActiveBuffer];
    if (fDeviceState[fActiveDeviceBuffer] != DeviceState::Filling)
      throw std::logic_error(__FILE__ + std::to_string(__LINE__) + ": On-device buffer in wrong state");


    // Get new buffer info from device:
    // auto &currentHitInfo = currentBuffer.hitScoringInfo;
    // // std::cout << " currentHitInfo.hitBuffer_dev " << currentHitInfo.hitBuffer_dev << std::endl;
    // COPCORE_CUDA_CHECK(cudaMemcpyAsync(&currentHitInfo, fHitScoringBuffer_deviceAddress, sizeof(HitScoringBuffer),
    //                                    cudaMemcpyDefault, cudaStream));


    COPCORE_CUDA_CHECK(cudaMemcpyAsync(currentBuffer.hostBufferCount, currentBuffer.hitScoringInfo.fSlotCounter,
                                        sizeof(unsigned int) * currentBuffer.hitScoringInfo.fNThreads, cudaMemcpyDefault,
                                        cudaStream));
    COPCORE_CUDA_CHECK(cudaMemsetAsync(currentBuffer.hitScoringInfo.fSlotCounter, 0, 
                                  sizeof(unsigned int) * currentBuffer.hitScoringInfo.fNThreads, 
                                  cudaStream));


    // HitScoringBuffer* deviceBuffer = static_cast<HitScoringBuffer*>(fHitScoringBuffer_deviceAddress);

    // // Copy the SlotCounterArray
    // COPCORE_CUDA_CHECK(cudaMemcpyAsync(currentHitInfo.fSlotCounter, 
    //                                    deviceBuffer->fSlotCounter,
    //                                    sizeof(unsigned int) * currentHitInfo.fNThreads,
    //                                    cudaMemcpyDeviceToDevice, cudaStream));

    // Execute the swap:
    // printf("Before Swap buffer 0: fSlotCounter = %p\n", fBuffers[0].hitScoringInfo.fSlotCounter);
    // printf("Before Swap buffer 1: fSlotCounter = %p\n", fBuffers[1].hitScoringInfo.fSlotCounter);

    // short nextActiveHostBuffer          = (fActiveBuffer + 1) % fBuffers.size();

    // Switch host buffer to next active buffer
    fActiveBuffer          = (fActiveBuffer + 1) % fBuffers.size();

    auto prevActiveDeviceBuffer = fActiveDeviceBuffer;
    fActiveDeviceBuffer          = (fActiveDeviceBuffer + 1) % fDeviceState.size();

    if (fDeviceState[fActiveDeviceBuffer] != DeviceState::Free)
      throw std::logic_error(__FILE__ + std::to_string(__LINE__) + ": Next on-device buffer in wrong state");

    // printf("After Swap: fSlotCounter = %p\n", fBuffers[fActiveBuffer].hitScoringInfo.fSlotCounter);
    auto &nextDeviceBuffer = fBuffers[fActiveBuffer];
    // while (nextDeviceBuffer.hostState != BufferHandle::HostState::Free) {
    if (currentBuffer.hostState.load(std::memory_order_acquire) != BufferHandle::HostState::Free) {
      std::cerr << __func__ << " Warning: Another thread should have processed the hits.\n";
    }

    currentBuffer.hostState.store(BufferHandle::HostState::AwaitDeviceTransfer, std::memory_order_release);
    // }
    // assert(nextDeviceBuffer.state == BufferHandle::State::Free && nextDeviceBuffer.hitScoringInfo.fSlotCounter == 0);

    nextDeviceBuffer.hitScoringInfo.hitBuffer_dev = fGPUHitBuffer_dev.get() + fActiveDeviceBuffer * fHitCapacity;
    nextDeviceBuffer.hitScoringInfo.fSlotCounter = fGPUHitBufferCount_dev.get() + fActiveDeviceBuffer * nextDeviceBuffer.hitScoringInfo.fNThreads;

    // COPCORE_CUDA_CHECK(cudaMemsetAsync(nextDeviceBuffer.hitScoringInfo.fSlotCounter, 0, nextDeviceBuffer.hitScoringInfo.fNThreads*sizeof(unsigned int), cudaStream));


    fDeviceState[fActiveDeviceBuffer] = DeviceState::Filling;
    fDeviceState[prevActiveDeviceBuffer] = DeviceState::NeedTransferToHost;
    COPCORE_CUDA_CHECK(cudaMemcpyAsync(fHitScoringBuffer_deviceAddress, &nextDeviceBuffer.hitScoringInfo,
                                       sizeof(HitScoringBuffer), cudaMemcpyDefault, cudaStream));
    // COPCORE_CUDA_CHECK(cudaMemcpyAsync(deviceBuffer->fSlotCounter,
    //                                    nextDeviceBuffer.hitScoringInfo.fSlotCounter,
    //                                    sizeof(unsigned int) * nextDeviceBuffer.hitScoringInfo.fNThreads,
    //                                    cudaMemcpyDeviceToDevice, cudaStream));

    COPCORE_CUDA_CHECK(cudaStreamSynchronize(cudaStream));
    
    // std::cout << "After copy currentHitInfo.hitBuffer_dev " << currentHitInfo.hitBuffer_dev << " fHitScoringBuffer_deviceAddress " << fHitScoringBuffer_deviceAddress << std::endl;

    // one could mark the currently active HostBuffer with a state like AwaitingDeviceTransfer and then check in TransferToHost that it is in that state.

  }

  // bool TransferAndProcessHits(cudaStream_t cudaStreamForHitCopy, std::condition_variable &cvG4Workers) {
  //   bool haveNewHits = false;

  //   while (std::any_of(fDeviceState.begin(), fDeviceState.end(),
  //                      [](auto &deviceState) { return deviceState == DeviceState::TransferFromDevice; })) {
  //     TransferHitsToHost(context->hitTransferStream);
  //     ProcessHits(cvG4Workers);
  //   }

  //   return haveNewHits;
  // }

  bool ProcessHits(std::condition_variable &cvG4Workers)
  {
    std::unique_lock lock{fProcessingHitsMutex, std::defer_lock};
    bool haveNewHits = false;

    // FIXME use HostState. Be careful, with state of scoring
    // While loop to wait for arrival of data? Need to understand why we use a while loop?
    while (std::any_of(fBuffers.begin(), fBuffers.end(),
                       [](auto &buffer) { return (buffer.hostState.load(std::memory_order_acquire) == BufferHandle::HostState::TransferFromDevice)
                                               || (buffer.hostState.load(std::memory_order_acquire) == BufferHandle::HostState::AwaitScoring); })) {
      for (auto &handle : fBuffers) {
        if (handle.hostState.load(std::memory_order_acquire) == BufferHandle::HostState::AwaitScoring) {
          // FIXME: change state to scoring
          // if (!lock) lock.lock();
          handle.hostState = BufferHandle::HostState::Scoring;

          // Possible timing
          // auto start = std::chrono::high_resolution_clock::now();
          ProcessBuffer(handle, cvG4Workers, lock);

          haveNewHits = true;

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
    return (std::any_of(fDeviceState.begin(), fDeviceState.end(),
                       [](const auto &deviceState) { return deviceState == DeviceState::Free; }) &&
            fBuffers[fActiveBuffer].hostState == BufferHandle::HostState::Free);

  }

  std::string GetDeviceStateName(DeviceState state) const {
    switch (state) {
      case DeviceState::Free: return "Free";
      case DeviceState::Filling: return "Filling";
      case DeviceState::NeedTransferToHost: return "NeedTransferToHost";
      case DeviceState::TransferToHost: return "TransferToHost";
      default: return "Unknown";
    }
  }

  std::string GetHostStateName(BufferHandle::HostState state) const {
    switch (state) {
      case BufferHandle::HostState::Free: return "Free";
      case BufferHandle::HostState::AwaitDeviceTransfer: return "AwaitDeviceTransfer";
      case BufferHandle::HostState::TransferFromDevice: return "TransferFromDevice";
      case BufferHandle::HostState::AwaitScoring: return "AwaitScoring";
      case BufferHandle::HostState::Scoring: return "Scoring";
      default: return "Unknown";
    }
  }

  void PrintDeviceBufferStates() const {
    std::cout << " DeviceBufferStates: active : " << fActiveDeviceBuffer;
    for (const auto &deviceState : fDeviceState) {
      std::cout << " [DeviceState: " << GetDeviceStateName(deviceState) << "] ";
    }
    std::cout << std::endl;
  }

  void PrintHostBufferStates() const {
    std::cout << " HostBufferStates: active : " << fActiveBuffer;
    for (const auto &handle : fBuffers) {
      std::cout << " [HostState: " << GetHostStateName(handle.hostState.load()) 
                << ", refCount: " << handle.refCount.load() << "] ";
    }
    std::cout << std::endl;
  }

  /// Copy the current contents of the GPU hit buffer to host.
  void TransferHitsToHost(cudaStream_t cudaStreamForHitCopy)
  {
    // FIXME use device state
    // for (auto &buffer : fBuffers) {
    while (std::any_of(fDeviceState.begin(), fDeviceState.end(),
                    [](auto &deviceState) { return deviceState == DeviceState::NeedTransferToHost; })) {
    auto& buffer = fBuffers[(fActiveBuffer + fBuffers.size() - 1) % fBuffers.size()]; //fActiveBuffer];
      // previous active device buffer.
      short prevActiveDeviceBuffer = (fActiveDeviceBuffer + fDeviceState.size() - 1) % fDeviceState.size();
      if (fDeviceState[prevActiveDeviceBuffer] != DeviceState::NeedTransferToHost) {
        continue;
      }

      fDeviceState[prevActiveDeviceBuffer] = DeviceState::TransferToHost;
      assert(buffer.hostState == BufferHandle::HostState::AwaitDeviceTransfer);
      buffer.hostState = BufferHandle::HostState::TransferFromDevice;

      // assert(buffer.hitScoringInfo.fSlotCounter[0] < fHitCapacity);
      // if( buffer.hitScoringInfo.fSlotCounter[0] > fHitCapacity ) {
      //   printf("Danger! buffer.hitScoringInfo.fSlotCounter[0] %u fHitCapacity %u \n", buffer.hitScoringInfo.fSlotCounter[0], fHitCapacity);
      // }

      auto bufferBegin = buffer.hitScoringInfo.hitBuffer_dev;

      // cub::DeviceMergeSort::SortKeys(fGPUSortAuxMemory.get(), fGPUSortAuxMemorySize, bufferBegin,
      //                                buffer.hitScoringInfo.fSlotCounter, CompareGPUHits{}, cudaStreamForHitCopy);

      // Copy SlotCounterArray and reset it
      // COPCORE_CUDA_CHECK(cudaMemcpyAsync(buffer.hostBufferCount, buffer.hitScoringInfo.fSlotCounter,
      //                                    sizeof(unsigned int) * buffer.hitScoringInfo.fNThreads, cudaMemcpyDefault,
      //                                    cudaStreamForHitCopy));
      // COPCORE_CUDA_CHECK(cudaMemsetAsync(buffer.hitScoringInfo.fSlotCounter, 0, 
      //                              sizeof(unsigned int) * buffer.hitScoringInfo.fNThreads, 
      //                              cudaStreamForHitCopy));
      // // unfortunately, we need to synchronize since we need to know the offsets if we want to copy only the used data.
      // COPCORE_CUDA_CHECK(cudaStreamSynchronize(cudaStreamForHitCopy));

      // Copy out the hits:
      // The start address on device is always i * fNSlot (Slots per thread), and we copy always to
      // the offset of the previous copy, to get a compact buffer on host.
      unsigned int offset = 0;
      for (int i = 0; i < buffer.hitScoringInfo.fNThreads; i++) {
        if (buffer.hostBufferCount[i] > 0) {
          COPCORE_CUDA_CHECK(cudaMemcpyAsync(buffer.hostBuffer + offset, bufferBegin + i * buffer.hitScoringInfo.fNSlot,
                                    sizeof(GPUHit) * buffer.hostBufferCount[i], cudaMemcpyDefault,
                                    cudaStreamForHitCopy));
          offset += buffer.hostBufferCount[i];
        }
      }

        // std::cout << " offset " << offset; 
        // std::cout << " buffer.hostBufferCount[i]" << buffer.hostBufferCount[i];
        // std::cout << " Adress host begin buffer.hostBuffer + offset" << buffer.hostBuffer + offset;
        // std::cout << " address device begin bufferBegin + offset" << bufferBegin + offset;

      // COPCORE_CUDA_CHECK(cudaMemsetAsync(buffer.hitScoringInfo.fSlotCounter, 0, 
      //                              sizeof(unsigned int) * buffer.hitScoringInfo.fNThreads, 
      //                              cudaStreamForHitCopy));
                                   
      // Launch the host function
      // auto* callbackData = new HostCallbackData{fDeviceState.data(), prevActiveDeviceBuffer, &buffer};
      // COPCORE_CUDA_CHECK(cudaLaunchHostFunc(cudaStreamForHitCopy, HostCallback, callbackData));

      COPCORE_CUDA_CHECK(cudaLaunchHostFunc(
          cudaStreamForHitCopy,
          [](void* arg) {
            // auto* deviceState = static_cast<std::atomic<DeviceState>*>(arg);
            static_cast<std::atomic<DeviceState>*>(arg)->store(DeviceState::Free, std::memory_order_release);
          },
          &fDeviceState[prevActiveDeviceBuffer]
      ));

      COPCORE_CUDA_CHECK(cudaLaunchHostFunc(
          cudaStreamForHitCopy,
          [](void* arg) {
            static_cast<BufferHandle*>(arg)->hostState.store(BufferHandle::HostState::AwaitScoring, std::memory_order_release);
          },
          &buffer
      ));

      // // Switch host buffer to next active buffer
      // fActiveBuffer          = (fActiveBuffer + 1) % fBuffers.size();

      // COPCORE_CUDA_CHECK(cudaLaunchHostFunc(
      //     cudaStreamForHitCopy,
      //     [this, prevActiveDeviceBuffer](void *arg) { 
      //       fDeviceState[prevActiveDeviceBuffer] = DeviceState::Free;
      //       static_cast<BufferHandle *>(arg)->hostState = BufferHandle::HostState::Scoring; },
      //     &buffer));
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

  HitInfo* GetNextHitsHandle(unsigned int threadId)
  {
    assert(threadId < fHitQueues.size());
    std::shared_lock lock{fProcessingHitsMutex}; // read only, can use shared_lock // NOT ANYMORE, need to set boolean flag

    if (fHitQueues[threadId].empty())
      return nullptr;
    else {
      auto& ret = fHitQueues[threadId].front();
      ret.ScoringStarted.store(true, std::memory_order_release);
      // fHitQueues[threadId].pop_front(); // don't pop the front, we still need to decrement before we can pop it
      return &ret;
    }
  }

  void CloseHitsHandle(unsigned int threadId)
  {
    assert(threadId < fHitQueues.size());
    std::unique_lock lock{fProcessingHitsMutex}; // popping queue, requires unique lock

    if (fHitQueues[threadId].empty()) 
      throw std::invalid_argument{"Error, no hitQueue to close"};
    else {
      // auto& ret = fHitQueues[threadId].front();
      // ret.hostBuffer->decrement(threadId);
      // std::cout << "Popping queue threadId "<<  threadId << " if I was the last, you now you may call the custom deleter! " << std::endl;
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
