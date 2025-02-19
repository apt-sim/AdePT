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
  std::array<HitScoringBuffer, 2> hitScoringInfo;
  GPUHit *hostBuffer;
  unsigned int *hostBufferCount; 
  // enum class State { Free, OnDevice, OnDeviceNeedTransferToHost, TransferToHost, NeedHostProcessing };

  // enum class DeviceState { Free, Filling, NeedTransferToHost, TransferToHost };
  enum class HostState { ReadyToBeFilled, TransferFromDevice, TransferFromDeviceFinished };

  // std::atomic<State> state;

  // std::atomic<DeviceState> deviceState;
  HostState hostState; // only touched by HitProcessingThread, doesn't need to be atomic
  std::atomic_bool hostBufferSubmitted = true; // we can only swap and copy the slot counters into the hostBufferCount, 
    // after we used the counts to submit to the hitqueues
  unsigned int offsetAtCopy = 0;

};

struct HitQueueItem {
  GPUHit * begin;
  GPUHit * end;
  std::atomic_bool ScoringStarted = false;
  std::vector<GPUHit> holdoutBuffer;

  HitQueueItem(GPUHit* begin_, GPUHit* end_)
    : begin(begin_), end(end_) {}

  ~HitQueueItem() = default;

  HitQueueItem(const HitQueueItem&) = delete;
  HitQueueItem& operator=(const HitQueueItem&) = delete;

  HitQueueItem(HitQueueItem&& other) noexcept
    : begin(other.begin),
      end(other.end),
      ScoringStarted(other.ScoringStarted.load()), // Read atomic value safely
      holdoutBuffer(std::move(other.holdoutBuffer)) {} // Move vector safely

  HitQueueItem& operator=(HitQueueItem&& other) noexcept {
    if (this != &other) {
      begin = other.begin;
      end = other.end;
      ScoringStarted.store(other.ScoringStarted.load()); // Safe atomic move
      holdoutBuffer = std::move(other.holdoutBuffer);
    }
    return *this;
  }
};

class CircularBufferManager {
 public:
  struct Segment {
    GPUHit* begin;
    GPUHit* end;

    bool operator<(const Segment& other) const {
      return begin < other.begin;
    }
  };

  CircularBufferManager(GPUHit* bufferStart, size_t capacity)
    : bufferStart(bufferStart), bufferEnd(bufferStart + capacity),
      writePtr(bufferStart), freeSpace(capacity) {}

  /// Adds a segment at the provided position. Ensures segments remain sorted.
  bool addSegment(GPUHit* begin, GPUHit* end) {
    assert(begin == writePtr && "Begin pointer must match writePtr for contiguous allocation!");
    size_t size = end - begin;
    
    if (size > freeSpace) return false; // Not enough total space

    // Insert the segment into sorted position
    segments.insert(
      std::upper_bound(segments.begin(), segments.end(), Segment{begin, end}),
      {begin, end}
    );

    writePtr = end;  // Move forward
    freeSpace -= size;
    return true;
  }

  /// Removes a segment based on its starting pointer and updates writePtr accordingly.
  void removeSegment(GPUHit* segmentPtr) {
    auto it = std::find_if(segments.begin(), segments.end(),
      [segmentPtr](const Segment& seg) { return seg.begin == segmentPtr; });

    assert(it != segments() && "Trying to remove segment that doesn't exist. They should always exist!");

    freeSpace += (it->end - it->begin);

    // If writePtr is exactly at the end of the segment being removed
    if (writePtr == it->end) {
      // Case 1: If it is the first segment, reset writePtr to bufferStart
      if (it == segments.begin()) {
        writePtr = bufferStart;
      }
      // Case 2: If there's a previous segment, move writePtr to its end
      else {
        auto prev = std::prev(it);
        writePtr = prev->end;
      }
    }

    segments.erase(it);
    return;
  }

  /// Returns the contiguous free space in front of the writePtr (or at the beginning of the buffer in case of a wraparound)
  size_t getFreeContiguousMemory(size_t transferSize) {
    if (segments.empty()) {
      return bufferEnd - bufferStart;  // Everything is free
    }

    // Find the next segment after writePtr
    auto nextSegment = std::lower_bound(segments.begin(), segments.end(), Segment{writePtr, nullptr}, 
      [](const Segment& a, const Segment& b) { return a.begin < b.begin; });

    // Free space from `writePtr` to next segment
    size_t forwardSpace = (nextSegment != segments.end()) ? nextSegment->begin - writePtr : bufferEnd - writePtr;

    if (forwardSpace >= transferSize) {
      return forwardSpace;  // Enough space in the current region
    }

    // If wrapping is needed, check space at the front
    size_t wrapAroundSpace = (segments.front().begin > bufferStart) 
                              ? (segments.front().begin - bufferStart) 
                              : 0;

    if (wrapAroundSpace >= transferSize) {
      writePtr = bufferStart; // Reset writePtr since we need to wrap around
      return wrapAroundSpace;
    }

    return 0; // Not enough contiguous space available
  }

  size_t getOffset() { return writePtr - bufferStart; }

 private:
  GPUHit* bufferStart;
  GPUHit* bufferEnd;
  GPUHit* writePtr;
  size_t freeSpace;
  std::vector<Segment> segments; // **Sorted vector instead of set**
};
// TODO: Rename this. Maybe ScoringState? Check usage in GPUstate
class HitScoring {
  unique_ptr_cuda<GPUHit> fGPUHitBuffer_dev;
  unique_ptr_cuda<GPUHit, CudaHostDeleter<GPUHit>> fGPUHitBuffer_host;
  unique_ptr_cuda<unsigned int> fGPUHitBufferCount_dev;
  unique_ptr_cuda<unsigned int, CudaHostDeleter<unsigned int>> fGPUHitBufferCount_host;

  // FIXME have size 3 for bufferhandles
  std::array<BufferHandle, 1> fBuffers;

  std::unique_ptr<CircularBufferManager> fBufferManager;

  enum class DeviceState { Free, Filling, NeedTransferToHost, TransferToHost };
  std::array<std::atomic<DeviceState>, 2> fDeviceState; // the device state must be atomic as it is touched by both the HitProcesingThread in TransferHitsToHost and by the TransportThread in SwapDeviceBuffers

  void *fHitScoringBuffer_deviceAddress = nullptr;
  unsigned int fHitCapacity;
  unsigned short fActiveBuffer = 0;
  unsigned short fActiveDeviceBuffer = 0;
  // FIXME: have fActiveDeviceBuffer and fActiveHostBuffer (let's see if this is really needed)
  unique_ptr_cuda<std::byte> fGPUSortAuxMemory;
  std::size_t fGPUSortAuxMemorySize;

  // std::vector<std::deque<std::shared_ptr<const std::vector<GPUHit>>>> fHitQueues;
  // std::vector<std::deque<BufferHandle*>> fHitQueues;
  std::vector<std::deque<HitQueueItem>> fHitQueues;


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
#define RED "\033[31m"
#define RESET "\033[0m"
#define BOLD_RED "\033[1;31m"
#define BOLD_BLUE    "\033[1;34m"
//   std::cout << BOLD_BLUE << "START PROCESS BUFFER" << RESET << std::endl;
    // std::cout << "Total Memory Used in fHitQueues: " << calculateMemoryUsage(fHitQueues) / 1024.0 / 1024.0 / 1024.0
    // << " GB" << std::endl;

    // OPTIONAL: print size of buffer 
    // size_t memoryUsed = (end - begin) * sizeof(GPUHit);
    // std::cout << "Memory in hit buffer to be scored: " << memoryUsed / 1024. / 1024. /1024. << " GB" << std::endl;


    using namespace std::chrono_literals;
    // std::this_thread::sleep_for(100ms);

    unsigned int offset = 0;
    for (int i = 0; i < fHitQueues.size(); i++) {

      GPUHit* begin = handle.hostBuffer + offset + handle.offsetAtCopy;
      GPUHit* end = handle.hostBuffer + offset + handle.offsetAtCopy + handle.hostBufferCount[i];

// std::cout << " handle.hostBuffer " << handle.hostBuffer << " offset " << offset << " size " << (end - begin) << " handle hostBuffercount " <<  handle.hostBufferCount[i] <<" Begin fStepLength " << begin->fStepLength << std::endl;
      // fBufferManager->getFreeSpace
      HitQueueItem hitItem{begin, end};

      if (begin != end) {
        lock.lock();
        // handle.increment();
        fBufferManager->addSegment(begin, end);
        fHitQueues[i].push_back(std::move(hitItem));
        lock.unlock();
      }
// std::cout << BOLD_RED << "threadId " << i << " EventId " << begin->fEventId << " offset " << offset << " num hits to score " << handle.hostBufferCount[i] << RESET << std::endl;

      offset += handle.hostBufferCount[i];
    }
    // std::cout << BOLD_RED << "Setting handle.hostBufferSubmitted to true "  << RESET << std::endl;
    handle.hostBufferSubmitted = true; // submitted hostBuffer to queue, can swap and overwrite now

    // std::cout << "Notifying G4 workers..." << std::endl;
    cvG4Workers.notify_all();


    // Give G4 workers time to wake up
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(50ms); // 50 seems to be a good value

    for (int i = 0; i < fHitQueues.size(); i++) {

      lock.lock();
      if (!fHitQueues[i].empty()) {
        auto& ret = fHitQueues[i].back(); // we just pushed to the back, so we need to check if the back is used!
        if (ret.ScoringStarted.load(std::memory_order_acquire) == true ) {
          std::cout << BOLD_BLUE << "G4worker " << i << " has taken their task and started working on " << (ret.end - ret.begin) << " hits " << RESET << std::endl;
          lock.unlock();
        } else {

          assert(ret.begin && ret.end && ret.begin < ret.end);

          size_t numHits = ret.end - ret.begin;  // Compute the number of hits

          std::cout << BOLD_RED << "G4Worker " << i << " was too slow, copying out " << numHits << " hits "  << RESET << std::endl;

          ret.holdoutBuffer.resize(numHits);  // Allocate correct size
          std::copy(ret.begin, ret.end, ret.holdoutBuffer.begin());  // Copy data

          // remove the segment first, before updating the pointers to the copied out memory
          fBufferManager->removeSegment(ret.begin);

          // Update pointers
          ret.begin = ret.holdoutBuffer.data();
          ret.end = ret.holdoutBuffer.data() + ret.holdoutBuffer.size();

          ret.ScoringStarted = true;

          lock.unlock();
        }
      
      } else {
        lock.unlock();
      }
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

    fDeviceState[0] = DeviceState::Filling;
    fDeviceState[1] = DeviceState::Free;


    fBuffers[0].hitScoringInfo[0] = HitScoringBuffer{fGPUHitBuffer_dev.get(), fGPUHitBufferCount_dev.get(), fHitCapacity/nThread, nThread};
    fBuffers[0].hitScoringInfo[1] = HitScoringBuffer{fGPUHitBuffer_dev.get() + fHitCapacity, fGPUHitBufferCount_dev.get() + nThread, fHitCapacity/nThread, nThread};
    fBuffers[0].hostBuffer     = fGPUHitBuffer_host.get();
    fBuffers[0].hostBufferCount = fGPUHitBufferCount_host.get();
    fBuffers[0].hostState          = BufferHandle::HostState::ReadyToBeFilled;

    // fBuffers[1].hitScoringInfo = HitScoringBuffer{fGPUHitBuffer_dev.get() + fHitCapacity, fGPUHitBufferCount_dev.get() + nThread, fHitCapacity/nThread, nThread};
    // fBuffers[1].hostBuffer     = fGPUHitBuffer_host.get() + fHitCapacity;
    // fBuffers[1].hostBufferCount = fGPUHitBufferCount_host.get() + nThread;
    // fBuffers[1].hostState          = BufferHandle::HostState::Free;

    // fBuffers[2].hitScoringInfo = HitScoringBuffer{fGPUHitBuffer_dev.get(), fGPUHitBufferCount_dev.get(), fHitCapacity/nThread, nThread};
    // fBuffers[2].hostBuffer     = fGPUHitBuffer_host.get() + 2 * fHitCapacity;
    // fBuffers[2].hostBufferCount = fGPUHitBufferCount_host.get() + 2* nThread;
    // fBuffers[2].hostState          = BufferHandle::HostState::Free;

    fBufferManager = std::make_unique<CircularBufferManager>(fGPUHitBuffer_host.get(), hitCapacity);

    COPCORE_CUDA_CHECK(cudaGetSymbolAddress(&fHitScoringBuffer_deviceAddress, gHitScoringBuffer_dev));
    assert(fHitScoringBuffer_deviceAddress != nullptr);
    COPCORE_CUDA_CHECK(cudaMemcpy(fHitScoringBuffer_deviceAddress, &fBuffers[0].hitScoringInfo,
                                  sizeof(HitScoringBuffer), cudaMemcpyHostToDevice));
  }

  unsigned int HitCapacity() const { return fHitCapacity; }

  void SwapDeviceBuffers(cudaStream_t cudaStream)
  {

    // FIXME: full function
//  #define RED "\033[31m"
// #define RESET "\033[0m"
// #define BOLD_RED "\033[1;31m"
//   std::cout << BOLD_RED << "START SWAP BUFFERS"  << RESET << std::endl;
    // printf("CALLING SWAP printing states\n");
    // PrintDeviceBufferStates();
    // PrintHostBufferStates();
    // PrintBufferStates();
    // Ensure that host side has been processed:
    auto &currentBuffer = fBuffers[fActiveBuffer];
    if (fDeviceState[fActiveDeviceBuffer].load(std::memory_order_acquire) != DeviceState::Filling)
      throw std::logic_error(__FILE__ + std::to_string(__LINE__) + ": On-device buffer in wrong state");


    // Get new buffer info from device:
    // auto &currentHitInfo = currentBuffer.hitScoringInfo;
    // // std::cout << " currentHitInfo.hitBuffer_dev " << currentHitInfo.hitBuffer_dev << std::endl;
    // COPCORE_CUDA_CHECK(cudaMemcpyAsync(&currentHitInfo, fHitScoringBuffer_deviceAddress, sizeof(HitScoringBuffer),
    //                                    cudaMemcpyDefault, cudaStream));


    COPCORE_CUDA_CHECK(cudaMemcpyAsync(currentBuffer.hostBufferCount, currentBuffer.hitScoringInfo[fActiveDeviceBuffer].fSlotCounter,
                                        sizeof(unsigned int) * currentBuffer.hitScoringInfo[fActiveDeviceBuffer].fNThreads, cudaMemcpyDefault,
                                        cudaStream));
    COPCORE_CUDA_CHECK(cudaMemsetAsync(currentBuffer.hitScoringInfo[fActiveDeviceBuffer].fSlotCounter, 0, 
                                  sizeof(unsigned int) * currentBuffer.hitScoringInfo[fActiveDeviceBuffer].fNThreads, 
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
    // fActiveBuffer          = (fActiveBuffer + 1) % fBuffers.size();

    auto prevActiveDeviceBuffer = fActiveDeviceBuffer;
    fActiveDeviceBuffer          = (fActiveDeviceBuffer + 1) % fDeviceState.size();

    if (fDeviceState[fActiveDeviceBuffer].load(std::memory_order_acquire) != DeviceState::Free)
      throw std::logic_error(__FILE__ + std::to_string(__LINE__) + ": Next on-device buffer in wrong state");

    // printf("After Swap: fSlotCounter = %p\n", fBuffers[fActiveBuffer].hitScoringInfo.fSlotCounter);
    // auto &nextDeviceBuffer = fBuffers[fActiveBuffer];
    // while (nextDeviceBuffer.hostState != BufferHandle::HostState::Free) {
    // if (currentBuffer.hostState.load(std::memory_order_acquire) != BufferHandle::HostState::ReadyToBeFilled) {
    //   std::cerr << __func__ << " Warning: Another thread should have processed the hits.\n";
    // }

    // currentBuffer.hostState.store(BufferHandle::HostState::AwaitDeviceTransfer, std::memory_order_release);
    // }
    // assert(nextDeviceBuffer.state == BufferHandle::State::Free && nextDeviceBuffer.hitScoringInfo.fSlotCounter == 0);

    currentBuffer.hitScoringInfo[fActiveDeviceBuffer].hitBuffer_dev = fGPUHitBuffer_dev.get() + fActiveDeviceBuffer * fHitCapacity;
    currentBuffer.hitScoringInfo[fActiveDeviceBuffer].fSlotCounter = fGPUHitBufferCount_dev.get() + fActiveDeviceBuffer * currentBuffer.hitScoringInfo[fActiveDeviceBuffer].fNThreads;

    // COPCORE_CUDA_CHECK(cudaMemsetAsync(nextDeviceBuffer.hitScoringInfo.fSlotCounter, 0, nextDeviceBuffer.hitScoringInfo.fNThreads*sizeof(unsigned int), cudaStream));


    fDeviceState[fActiveDeviceBuffer].store(DeviceState::Filling, std::memory_order_release);
    fDeviceState[prevActiveDeviceBuffer].store(DeviceState::NeedTransferToHost, std::memory_order_relaxed);
    COPCORE_CUDA_CHECK(cudaMemcpyAsync(fHitScoringBuffer_deviceAddress, &currentBuffer.hitScoringInfo[fActiveDeviceBuffer],
                                       sizeof(HitScoringBuffer), cudaMemcpyDefault, cudaStream));
    // COPCORE_CUDA_CHECK(cudaMemcpyAsync(deviceBuffer->fSlotCounter,
    //                                    nextDeviceBuffer.hitScoringInfo.fSlotCounter,
    //                                    sizeof(unsigned int) * nextDeviceBuffer.hitScoringInfo.fNThreads,
    //                                    cudaMemcpyDeviceToDevice, cudaStream));

    COPCORE_CUDA_CHECK(cudaStreamSynchronize(cudaStream));
    // std::cout << "Setting handle.hostBufferSubmitted to false " << std::endl;
    currentBuffer.hostBufferSubmitted.store(false, std::memory_order_release); // now set to false. Will be true after submission, then we can swap again
    // std::cout << "After copy currentHitInfo.hitBuffer_dev " << currentHitInfo.hitBuffer_dev << " fHitScoringBuffer_deviceAddress " << fHitScoringBuffer_deviceAddress << std::endl;

    // one could mark the currently active HostBuffer with a state like AwaitingDeviceTransfer and then check in TransferToHost that it is in that state.

  }

  bool ProcessHits(std::condition_variable &cvG4Workers)
  {
    #define RED "\033[31m"
#define RESET "\033[0m"
#define BOLD_RED "\033[1;31m"
#define BOLD_BLUE    "\033[1;34m"
  // std::cout << BOLD_BLUE << "START PROCESS HITS" << RESET << std::endl;
    std::unique_lock lock{fProcessingHitsMutex, std::defer_lock};
    bool haveNewHits = false;

    // FIXME use HostState. Be careful, with state of scoring
    // While loop to wait for arrival of data? Need to understand why we use a while loop?
    while (std::any_of(fBuffers.begin(), fBuffers.end(),
                       [](auto &buffer) { return (buffer.hostState == BufferHandle::HostState::TransferFromDevice)
                                               || (buffer.hostState == BufferHandle::HostState::TransferFromDeviceFinished); })) {
      for (auto &handle : fBuffers) {
        if (handle.hostState == BufferHandle::HostState::TransferFromDeviceFinished) {
          // FIXME: change state to scoring
          // if (!lock) lock.lock();
          // std::cout << " Setting it to scoring, why wouldn't I? " << std::endl;
          handle.hostState = BufferHandle::HostState::ReadyToBeFilled;

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
// std::cout << BOLD_BLUE << " finished process Hits " << RESET << std::endl;
    return haveNewHits;
  }

  bool ReadyToSwapBuffers() const
  {
    return (std::any_of(fDeviceState.begin(), fDeviceState.end(),
                       [](const auto &deviceState) { return deviceState.load(std::memory_order_acquire) == DeviceState::Free; })
                        && fBuffers[fActiveBuffer].hostBufferSubmitted.load(std::memory_order_acquire));

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
      case BufferHandle::HostState::ReadyToBeFilled: return "ReadytoBeFilled";
      case BufferHandle::HostState::TransferFromDevice: return "TransferFromDevice";
      case BufferHandle::HostState::TransferFromDeviceFinished: return "TransferFromDeviceFinished";
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
    std::cout << " HostBufferStates: active : " << fActiveBuffer << " Buffer already submitted: " << fBuffers[fActiveBuffer].hostBufferSubmitted;
    for (const auto &handle : fBuffers) {
      std::cout << " [HostState: " << GetHostStateName(handle.hostState) 
                << "] ";
    }
    std::cout << std::endl;
  }

  /// Copy the current contents of the GPU hit buffer to host.
  void TransferHitsToHost(cudaStream_t cudaStreamForHitCopy)
  {
#define RED "\033[31m"
#define RESET "\033[0m"
#define BOLD_RED "\033[1;31m"
#define BOLD_BLUE    "\033[1;34m"
  // std::cout << BOLD_BLUE << "START TRANSFERTOHOST" << RESET << std::endl;
    // FIXME use device state
    // for (auto &buffer : fBuffers) {
    while (std::any_of(fDeviceState.begin(), fDeviceState.end(),
                    [](auto &deviceState) { return deviceState.load() == DeviceState::NeedTransferToHost; })) {
      auto& buffer = fBuffers[fActiveBuffer]; //fActiveBuffer];
      // previous active device buffer.
      short prevActiveDeviceBuffer = (fActiveDeviceBuffer + fDeviceState.size() - 1) % fDeviceState.size();
      if (fDeviceState[prevActiveDeviceBuffer].load() != DeviceState::NeedTransferToHost) {
        std::cout << " prevActiveDeviceBuffer " << prevActiveDeviceBuffer << " not in NeedTransferToHost: ";
        PrintDeviceBufferStates(); 
        continue;
      }

      unsigned int transferSize = 0;
      for (int i = 0; i < buffer.hitScoringInfo[fActiveDeviceBuffer].fNThreads; i++) {
        transferSize += buffer.hostBufferCount[i];
      }

      {
        std::scoped_lock lock{fProcessingHitsMutex}; // need to lock the access to fBufferManager, as in the wraparound, the writePtr is moved and this needs to be locked

      if (fBufferManager->getFreeContiguousMemory(transferSize) <= transferSize ) {
        // std::cout << "Not enough free memory in buffer: " << fBufferManager->getFreeContiguousMemory(transferSize) <<  " cannot transfer yet hits of size " << transferSize << std::endl;
        continue;
      } else {
        // std::cout << "TransferHitsToHost of size " << transferSize << " Free contiguous memory in Buffer " << fBufferManager->getFreeContiguousMemory() << std::endl;
      }

      }



      fDeviceState[prevActiveDeviceBuffer].store(DeviceState::TransferToHost, std::memory_order_release);
      // assert(buffer.hostState == BufferHandle::HostState::AwaitDeviceTransfer);
      buffer.hostState = BufferHandle::HostState::TransferFromDevice;

      // std::cout << " HostBufferStates after setting to TransferFromDevice" ;
      // PrintHostBufferStates();

      // assert(buffer.hitScoringInfo.fSlotCounter[0] < fHitCapacity);
      // if( buffer.hitScoringInfo.fSlotCounter[0] > fHitCapacity ) {
      //   printf("Danger! buffer.hitScoringInfo.fSlotCounter[0] %u fHitCapacity %u \n", buffer.hitScoringInfo.fSlotCounter[0], fHitCapacity);
      // }

      auto bufferBegin = buffer.hitScoringInfo[prevActiveDeviceBuffer].hitBuffer_dev;

      buffer.offsetAtCopy = fBufferManager->getOffset();

      // Copy out the hits:
      // The start address on device is always i * fNSlot (Slots per thread), and we copy always to
      // the offset of the previous copy, to get a compact buffer on host.
      unsigned int offset = 0;
      for (int i = 0; i < buffer.hitScoringInfo[prevActiveDeviceBuffer].fNThreads; i++) {
        if (buffer.hostBufferCount[i] > 0) {
          // std::cout << " Calling cudaMemcpyAsync with buffer.offsetAtCopy " << buffer.offsetAtCopy << " offset " << offset << std::endl;
          // COPCORE_CUDA_CHECK(cudaMemcpyAsync(buffer.hostBuffer + offset, bufferBegin + i * buffer.hitScoringInfo.fNSlot,
          COPCORE_CUDA_CHECK(cudaMemcpyAsync(buffer.hostBuffer + buffer.offsetAtCopy + offset, bufferBegin + i * buffer.hitScoringInfo[prevActiveDeviceBuffer].fNSlot,
                                    sizeof(GPUHit) * buffer.hostBufferCount[i], cudaMemcpyDefault,
                                    cudaStreamForHitCopy));
          offset += buffer.hostBufferCount[i];
        }
      }

        // std::cout << " offset " << offset << " fBufferManager->getOffset() " << buffer.offsetAtCopy << std::endl;; 
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
            static_cast<BufferHandle*>(arg)->hostState = BufferHandle::HostState::TransferFromDeviceFinished;
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
    // std::cout << BOLD_BLUE << "finished transfer to host " << RESET << std::endl;
  }

  HitQueueItem* GetNextHitsHandle(unsigned int threadId, bool &dataOnBuffer)
  {
    assert(threadId < fHitQueues.size());
    std::shared_lock lock{fProcessingHitsMutex}; // read only, can use shared_lock // NOT ANYMORE, need to set boolean flag

    if (fHitQueues[threadId].empty())
      return nullptr;
    else {
      auto& ret = fHitQueues[threadId].front();
      dataOnBuffer = !ret.ScoringStarted.load();
      ret.ScoringStarted.store(true, std::memory_order_release);
      // fHitQueues[threadId].pop_front(); // don't pop the front, we still need to decrement before we can pop it
      return &ret;
    }
  }

  void CloseHitsHandle(unsigned int threadId, GPUHit* begin, const bool dataOnBuffer)
  {
    assert(threadId < fHitQueues.size());
    std::unique_lock lock{fProcessingHitsMutex}; // popping queue, requires unique lock

    if (fHitQueues[threadId].empty()) 
      throw std::invalid_argument{"Error, no hitQueue to close"};
    else {
      // std::cout << "Popping queue threadId "<<  threadId << " if I was the last, you now you may call the custom deleter! " << std::endl;
      // std::cout << "threadId " << threadId << " Popping front of hitqueue and removing segment " << begin << std::endl;
      if (dataOnBuffer) fBufferManager->removeSegment(begin); // remove used buffer memory from used-memory list
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
