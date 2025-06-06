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
#include <sstream>

// definitions for printouts and advanced debugging
// #define DEBUG
#define RESET "\033[0m"
#define BOLD_RED "\033[1;31m"
#define BOLD_BLUE "\033[1;34m"

// Comparison for sorting tracks into events on device:
struct CompareGPUHits {
  __device__ bool operator()(const GPUHit &lhs, const GPUHit &rhs) const { return lhs.threadId < rhs.threadId; }
};

namespace AsyncAdePT {

/// Struct holding GPU hits to be used both on host and device.
struct HitScoringBuffer {
  GPUHit *hitBuffer_dev      = nullptr;
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

  __device__ GPUHit &GetNextSlot(unsigned int threadId)
  {
    const auto slotIndex = atomicAdd(&fSlotCounter[threadId], 1);
    if (slotIndex >= fNSlot) {
      printf("Trying to score hit #%d with only %d slots\n", slotIndex, fNSlot);
      COPCORE_EXCEPTION("Out of slots in HitScoringBuffer::NextSlot");
    }
    return hitBuffer_dev[threadId * fNSlot + slotIndex];
  }
};

__device__ HitScoringBuffer gHitScoringBuffer_dev;

struct BufferHandle {
  std::array<HitScoringBuffer, 2> hitScoringInfo;
  GPUHit *hostBuffer;
  unsigned int *hostBufferCount;
  enum class HostState { ReadyToBeFilled, AwaitingDeviceTransfer, TransferFromDevice, TransferFromDeviceFinished };

  // the hostState is changed by the HitProcessingThread but also by the cudaHostFunction launch, so it must be atomic
  std::atomic<HostState> hostState;
  // offset in the buffer at the time of the copy (maybe redundant now?)
  unsigned int offsetAtCopy = 0;
};

struct HitQueueItem {
  // HitQueueItems are pushed into the HitQueue. They contain the begin and end pointers of the data that needs to be
  // processed. If the G4Workers are too slow to respond, the HitProcessingThread copies the data from the buffer into
  // the holdoutBuffer, so that the hostbuffer in pinned memory can be released.
  GPUHit *begin;                           // begin of GPUHit pointer
  GPUHit *end;                             // end of GPUHit pointer
  std::atomic_bool ScoringStarted = false; // whether the scoring has started. If it has not, the HitProcessingThread
                                           // will copy the data from the pinned memory host buffer to the holdoutBuffer
  std::atomic_bool IsDataOnHostBuffer =
      true; // whether the data resides in the HostBuffer in pinned memory (if false, it it is in the holdoutBuffer)
  std::vector<GPUHit>
      holdoutBuffer; // holdout buffer where the hits are copied to if the G4Worker is too slow to respond

  HitQueueItem(GPUHit *begin_, GPUHit *end_) : begin(begin_), end(end_) {}

  ~HitQueueItem() = default;

  // remove copy constructor and assigment operator
  HitQueueItem(const HitQueueItem &)            = delete;
  HitQueueItem &operator=(const HitQueueItem &) = delete;

  // write custom move constructor and assignment operator
  HitQueueItem(HitQueueItem &&other) noexcept
      : begin(other.begin), end(other.end), ScoringStarted(other.ScoringStarted.load()),
        holdoutBuffer(std::move(other.holdoutBuffer))
  {
  }

  HitQueueItem &operator=(HitQueueItem &&other) noexcept
  {
    if (this != &other) {
      begin = other.begin;
      end   = other.end;
      ScoringStarted.store(other.ScoringStarted.load());
      holdoutBuffer = std::move(other.holdoutBuffer);
    }
    return *this;
  }
};

class CircularBufferManager {
  // the CircularBufferManager manages the memory of the pinned HostBuffer fBuffer (in HostScoring).
  // It keeps track of the used space in the sorted vector of segments fSegments.
  // The HitProcessingThread adds segments when it submits items to the HitQueue (in fact, the memory is already
  // allocated as soon as the copy from TransferHitsToHost is done). The HitProcessingThread can delete segments when it
  // copies the hits to the holdoutBuffer and the G4Worker finish their work. Since both HitProcessingThread and
  // G4Workers can change the segments, a mutex is used to lock the access to the fSegments. In the TransferHitsToHost,
  // the HitProcessingThread checks whether there is enough contiguous memory in the CircularBuffer before the copy can
  // start
public:
  struct Segment {
    GPUHit *begin;
    GPUHit *end;

    bool operator<(const Segment &other) const { return begin < other.begin; }
  };

  CircularBufferManager(GPUHit *bufferStart, size_t capacity)
      : fBufferStart(bufferStart), fBufferEnd(bufferStart + capacity), fWritePtr(bufferStart),
        fFreeContiguousSpace(capacity)
  {
  }

  /// Adds a segment at the provided position. Ensures segments remain sorted.
  bool addSegment(GPUHit *begin, GPUHit *end)
  {
    std::scoped_lock lock{bufferManagerMutex};

    // Insert the segment into sorted position
    fSegments.insert(std::upper_bound(fSegments.begin(), fSegments.end(), Segment{begin, end}), {begin, end});

#ifdef DEBUG
    if (begin != fWritePtr)
      std::cout << BOLD_RED << " Begin != fWritePTr " << begin << " fWritePtr " << fWritePtr << RESET << std::endl;
    size_t size = end - begin;
    if (size > fFreeContiguousSpace)
      std::cout << BOLD_RED << " Not enough space! size " << size << " fFreeContiguousSpace " << fFreeContiguousSpace
                << RESET << std::endl;
    if (!checkForOverlaps()) std::cout << BOLD_RED << " Overlaps after AddSegment! " << RESET << std::endl;
#endif

    fWritePtr = end;

    return true;
  }

  /// Removes a segment based on its starting pointer and updates fWritePtr accordingly.
  void removeSegment(GPUHit *segmentPtr)
  {
    std::scoped_lock lock{bufferManagerMutex};

    // Find the segment
    auto it = std::find_if(fSegments.begin(), fSegments.end(),
                           [segmentPtr](const Segment &seg) { return seg.begin == segmentPtr; });

#ifdef DEBUG
    if (!segmentPtr) std::cout << BOLD_RED << " Trying to remove nullptr segment " << RESET << std::endl;
    if (it == fSegments.end())
      std::cout << BOLD_RED << " Trying to remove segment that doesn't exist !! segment : " << segmentPtr << RESET
                << std::endl;
    if (!checkForOverlaps()) std::cout << BOLD_RED << " Overlaps after removesegment! " << RESET << std::endl;
#endif

    // delete it from the list
    fSegments.erase(it);
  }

  /// Returns the contiguous free space in front of the fWritePtr (or at the beginning of the buffer in case of a
  /// wraparound)
  size_t getFreeContiguousMemory(size_t transferSize)
  {
    std::scoped_lock lock{bufferManagerMutex};

    if (fSegments.empty()) {
      // if empty, reset WritePtr to Bufferstart
      fWritePtr = fBufferStart;
      return fBufferEnd - fBufferStart; // Everything is free
    }

    // Find the next segment after fWritePtr
    auto nextSegment = std::lower_bound(fSegments.begin(), fSegments.end(), Segment{fWritePtr, nullptr},
                                        [](const Segment &a, const Segment &b) { return a.begin < b.begin; });

    // Find the previous segment before nextSegment (if it exists)
    auto prevSegment = (nextSegment != fSegments.begin()) ? std::prev(nextSegment) : fSegments.end();

    // If the fWritePtr was set on an end of a segment that was deleted, we can put it back to the last previous
    // existing segment
    if (prevSegment != fSegments.end() && fWritePtr != prevSegment->end) {
      fWritePtr = prevSegment->end;
    }

    // Free space from `fWritePtr` to next segment
    size_t forwardSpace = (nextSegment != fSegments.end()) ? nextSegment->begin - fWritePtr : fBufferEnd - fWritePtr;

    // Free space for a wraparound
    size_t wrapAroundSpace = (fSegments.front().begin > fBufferStart) ? (fSegments.front().begin - fBufferStart) : 0;

    fFreeContiguousSpace = forwardSpace + wrapAroundSpace - transferSize;

    if (forwardSpace >= transferSize) {
      return forwardSpace; // Enough space in the current region
    }

    if (wrapAroundSpace >= transferSize) {
      fWritePtr = fBufferStart; // Reset fWritePtr since we need to wrap around
      return wrapAroundSpace;
    }

#ifdef DEBUG
    std::cout << BOLD_RED
              << "Cannot transfer from Device to Host due to lack of space in CPU HostBuffer. This should never be "
                 "the case! transfersize "
              << transferSize << " forwardSpace " << forwardSpace << " wraparoundspace " << wrapAroundSpace
              << " total space : " << (fBufferEnd - fBufferStart) << " free space " << fFreeContiguousSpace << RESET
              << std::endl;
    checkForOverlaps();
#endif

    return 0; // Not enough contiguous space available
  }

  double getFillFraction() { return 1. - static_cast<double>(fFreeContiguousSpace) / (fBufferEnd - fBufferStart); }

  size_t getOffset() { return fWritePtr - fBufferStart; }

private:
  GPUHit *fBufferStart;
  GPUHit *fBufferEnd;
  GPUHit *fWritePtr;
  size_t fFreeContiguousSpace;
  std::vector<Segment> fSegments; // **Sorted vector instead of set**
  mutable std::mutex bufferManagerMutex;

  // Consistency check for debugging
  bool checkForOverlaps()
  {
    for (size_t i = 1; i < fSegments.size(); i++) {
      if (fSegments[i - 1].end > fSegments[i].begin) {
        std::cerr << "ERROR: Overlapping segments detected!\n";
        std::cerr << " Segment 1: [" << fSegments[i - 1].begin << " - " << fSegments[i - 1].end << "]\n";
        std::cerr << " Segment 2: [" << fSegments[i].begin << " - " << fSegments[i].end << "]\n";
        assert(false && "Overlapping segments in CircularBufferManager!");
        return false;
      }
    }
    return true;
  }

  void printSegments(const std::string &msg)
  {
    std::cout << msg << " | Current segments: ";
    for (const auto &seg : fSegments) {
      std::cout << "[" << seg.begin << " - " << seg.end << "] ";
    }
    std::cout << std::endl;
  }
};

// TODO: Rename this. Maybe ScoringState? Check usage in GPUstate
class HitScoring {
  unique_ptr_cuda<GPUHit> fGPUHitBuffer_dev;
  unique_ptr_cuda<GPUHit, CudaHostDeleter<GPUHit>> fGPUHitBuffer_host;
  unique_ptr_cuda<unsigned int> fGPUHitBufferCount_dev;
  unique_ptr_cuda<unsigned int, CudaHostDeleter<unsigned int>> fGPUHitBufferCount_host;

  BufferHandle fBuffer;

  std::unique_ptr<CircularBufferManager> fBufferManager;

  enum class DeviceState { Free, Filling, NeedTransferToHost, TransferToHost };
  std::array<std::atomic<DeviceState>, 2>
      fDeviceState; // the device state must be atomic as it is touched by both the HitProcesingThread in
                    // TransferHitsToHost and by the TransportThread in SwapDeviceBuffers

  void *fHitScoringBuffer_deviceAddress = nullptr;
  unsigned int fHitCapacity;
  double fCPUCapacityFactor;
  double fCPUCopyFraction;
  unsigned short fActiveBuffer = 0;
  cudaEvent_t fSwapDoneEvent; // cuda event to synchronize the swapping of the device buffers with the transport

  // HitQueue with one lock per queue
  std::vector<std::deque<HitQueueItem>> fHitQueues;
  std::vector<std::shared_mutex> fHitQueueLocks;

  void ProcessBuffer(BufferHandle &handle, std::condition_variable &cvG4Workers, int debugLevel)
  {

    // Loop over HitQueue and add HitQueueItems that contain the begin and end of the GPUhits in the HostBuffer.
    // Add the used segments in the BufferManager
    unsigned int offset = 0;
    for (int i = 0; i < fHitQueues.size(); i++) {

      GPUHit *begin = handle.hostBuffer + offset + handle.offsetAtCopy;
      GPUHit *end   = handle.hostBuffer + offset + handle.offsetAtCopy + handle.hostBufferCount[i];

      HitQueueItem hitItem{begin, end};

      if (begin != end) {
        fBufferManager->addSegment(begin, end);

        std::scoped_lock lock{fHitQueueLocks[i]};
        fHitQueues[i].push_back(std::move(hitItem));
      }
      offset += handle.hostBufferCount[i];
    }
    // release HostBuffer and notify G4Workers
    fBuffer.hostState.store(BufferHandle::HostState::ReadyToBeFilled);
    cvG4Workers.notify_all();

    for (int i = 0; i < fHitQueues.size(); i++) {

      std::unique_lock lock{fHitQueueLocks[i]};

      if (!fHitQueues[i].empty()) {
        // Check whether the last item in the HitQueue is already taken by the G4Worker
        auto &ret = fHitQueues[i].back();

        if (ret.ScoringStarted.load(std::memory_order_acquire) == true) {
          // if G4Worker has alreay started working, all good
          if (debugLevel > 5)
            std::cout << BOLD_BLUE << "G4worker " << i << " has taken their task and started working on "
                      << (ret.end - ret.begin) << " hits " << RESET << std::endl;
        } else {

          size_t numHits = ret.end - ret.begin;

          // If the circular Buffer is too full and the G4Worker didn't pick up the work, we have to copy out the hits
          // to the holdoutBuffer
          if (fBufferManager->getFillFraction() > fCPUCopyFraction) {
            if (debugLevel > 5) {
              std::cout << BOLD_RED << "FillFraction too high: " << fBufferManager->getFillFraction()
                        << ", threshold: " << fCPUCopyFraction << " copying out " << numHits << " hits for G4Worker "
                        << i << RESET << std::endl;
            }
            ret.holdoutBuffer.resize(numHits);                        // Allocate correct size
            std::copy(ret.begin, ret.end, ret.holdoutBuffer.begin()); // Copy data

            // remove the segment first, before updating the pointers to the copied out memory
            fBufferManager->removeSegment(ret.begin);

            // Update pointers
            ret.begin = ret.holdoutBuffer.data();
            ret.end   = ret.holdoutBuffer.data() + ret.holdoutBuffer.size();

            ret.ScoringStarted     = true;
            ret.IsDataOnHostBuffer = false;
          }
        }
      }
    }
  }

public:
  HitScoring(unsigned int hitCapacity, unsigned int nThread, double CPUCapacityFactor, double CPUCopyFraction)
      : fHitCapacity{hitCapacity}, fHitQueues(nThread), fHitQueueLocks(nThread), fCPUCapacityFactor(CPUCapacityFactor),
        fCPUCopyFraction(CPUCopyFraction)
  {

    if (fCPUCapacityFactor <= 2.0) {
      std::ostringstream oss;
      oss << "CPUCapacityFactor must be > 2.0 (got " << fCPUCapacityFactor << ")";
      throw std::invalid_argument(oss.str());
    }

    if (fCPUCopyFraction < 0.0 || fCPUCopyFraction > 1.0) {
      std::ostringstream oss;
      oss << "CPUCopyFraction must be between 0.0 and 1.0 (got " << fCPUCopyFraction << ")";
      throw std::invalid_argument(oss.str());
    }

    // We allocate one (circular) HostBuffer in pinned memory
    GPUHit *gpuHits = nullptr;

    // The HostBuffer is set to be fCPUCapacityFactor times the GPU buffer HitCapacity. Normally, maximally 2x of the
    // GPU hitbuffer should reside in the hostbuffer: once a full buffer that is currently processed by the G4 workers
    // and second another full buffer that is just copied from the GPU. Due to sparsity, we add another factor of .5 to
    // prevent running out of buffer. Also, the filling quota of the CPU buffer decides whether hits are processed
    // directly by the G4 workers or if they are copied out
    unsigned int hostBufferCapacity = fCPUCapacityFactor * fHitCapacity;
    COPCORE_CUDA_CHECK(cudaMallocHost(&gpuHits, sizeof(GPUHit) * hostBufferCapacity));
    fGPUHitBuffer_host.reset(gpuHits);

    // We use a single allocation for both GPU buffers:
    auto result = cudaMalloc(&gpuHits, sizeof(GPUHit) * fDeviceState.size() * fHitCapacity);
    if (result != cudaSuccess) throw std::invalid_argument{"No space to allocate hit buffer."};
    fGPUHitBuffer_dev.reset(gpuHits);

    unsigned int *buffer_count = nullptr;
    COPCORE_CUDA_CHECK(cudaMallocHost(&buffer_count, sizeof(unsigned int) * nThread));
    fGPUHitBufferCount_host.reset(buffer_count);

    result = cudaMalloc(&buffer_count, sizeof(unsigned int) * fDeviceState.size() * nThread);
    if (result != cudaSuccess) throw std::invalid_argument{"No space to allocate hit buffer."};
    fGPUHitBufferCount_dev.reset(buffer_count);

    fDeviceState[0] = DeviceState::Filling;
    fDeviceState[1] = DeviceState::Free;

    fBuffer.hitScoringInfo[0] =
        HitScoringBuffer{fGPUHitBuffer_dev.get(), fGPUHitBufferCount_dev.get(), fHitCapacity / nThread, nThread};
    fBuffer.hitScoringInfo[1] =
        HitScoringBuffer{fGPUHitBuffer_dev.get() + fHitCapacity, fGPUHitBufferCount_dev.get() + nThread,
                         fHitCapacity / nThread, nThread};
    fBuffer.hostBuffer      = fGPUHitBuffer_host.get();
    fBuffer.hostBufferCount = fGPUHitBufferCount_host.get();
    fBuffer.hostState       = BufferHandle::HostState::ReadyToBeFilled;

    fBufferManager = std::make_unique<CircularBufferManager>(fGPUHitBuffer_host.get(), hostBufferCapacity);

    COPCORE_CUDA_CHECK(cudaGetSymbolAddress(&fHitScoringBuffer_deviceAddress, gHitScoringBuffer_dev));
    assert(fHitScoringBuffer_deviceAddress != nullptr);
    COPCORE_CUDA_CHECK(cudaMemcpy(fHitScoringBuffer_deviceAddress, &fBuffer.hitScoringInfo, sizeof(HitScoringBuffer),
                                  cudaMemcpyHostToDevice));

    // create cuda event needed to tell the transport that the swap of the device buffers is executed
    cudaEventCreateWithFlags(&fSwapDoneEvent, cudaEventDisableTiming);
  }

  cudaEvent_t getSwapDoneEvent() const { return fSwapDoneEvent; }

  unsigned int HitCapacity() const { return fHitCapacity; }

  void SwapDeviceBuffers(cudaStream_t cudaStream)
  {

#ifdef DEBUG
    // Ensure that device side is free and the host side has been processed:
    if (fBuffer.hostState.load() != BufferHandle::HostState::ReadyToBeFilled)
      std::cout << BOLD_RED << " Hoststate is wrong, must be ReadyToBeFilled! " << RESET << std::endl;
    if (fDeviceState[fActiveBuffer].load(std::memory_order_acquire) != DeviceState::Filling)
      throw std::logic_error(__FILE__ + std::to_string(__LINE__) + ": On-device buffer in wrong state");
#endif

    // Get new HitBufferCounts from device:
    COPCORE_CUDA_CHECK(cudaMemcpyAsync(fBuffer.hostBufferCount, fBuffer.hitScoringInfo[fActiveBuffer].fSlotCounter,
                                       sizeof(unsigned int) * fBuffer.hitScoringInfo[fActiveBuffer].fNThreads,
                                       cudaMemcpyDefault, cudaStream));
    COPCORE_CUDA_CHECK(cudaMemsetAsync(fBuffer.hitScoringInfo[fActiveBuffer].fSlotCounter, 0,
                                       sizeof(unsigned int) * fBuffer.hitScoringInfo[fActiveBuffer].fNThreads,
                                       cudaStream));

    // Execute the swap:
    auto prevActiveDeviceBuffer = fActiveBuffer;
    fActiveBuffer               = (fActiveBuffer + 1) % fDeviceState.size();

    if (fDeviceState[fActiveBuffer].load(std::memory_order_acquire) != DeviceState::Free)
      throw std::logic_error(__FILE__ + std::to_string(__LINE__) + ": Next on-device buffer in wrong state");

    // adjust pointers to hitbuffer and slotcounter array to next active GPUbuffer
    fBuffer.hitScoringInfo[fActiveBuffer].hitBuffer_dev = fGPUHitBuffer_dev.get() + fActiveBuffer * fHitCapacity;
    fBuffer.hitScoringInfo[fActiveBuffer].fSlotCounter =
        fGPUHitBufferCount_dev.get() + fActiveBuffer * fBuffer.hitScoringInfo[fActiveBuffer].fNThreads;

    COPCORE_CUDA_CHECK(cudaMemcpyAsync(fHitScoringBuffer_deviceAddress, &fBuffer.hitScoringInfo[fActiveBuffer],
                                       sizeof(HitScoringBuffer), cudaMemcpyDefault, cudaStream));
    COPCORE_CUDA_CHECK(
        cudaEventRecord(fSwapDoneEvent, cudaStream)); // record event that the transport kernels must wait for

    // need to set the hostState to awaiting device, this prevents the next swap until the hits in the hostBuffer are
    // send to the HitQueue
    fBuffer.hostState.store(BufferHandle::HostState::AwaitingDeviceTransfer, std::memory_order_release);

    // the current active buffer can be set directly to block it from being seen as free
    fDeviceState[fActiveBuffer].store(DeviceState::Filling, std::memory_order_release);

    // However, the prevActiveDeviceBuffer state can only be advanced when the transfer is finished,
    // otherwise the TransferHitsToHost may call the transfer before the correct hostBufferCount has arrived
    COPCORE_CUDA_CHECK(cudaLaunchHostFunc(
        cudaStream,
        [](void *arg) {
          static_cast<std::atomic<DeviceState> *>(arg)->store(DeviceState::NeedTransferToHost,
                                                              std::memory_order_release);
        },
        &fDeviceState[prevActiveDeviceBuffer]));
  }

  bool ProcessHits(std::condition_variable &cvG4Workers, int debugLevel)
  {

    bool haveNewHits = false;

    // here we need to do atomic checks on the hostState, as it is modified from the cudaLaunchHostFunc
    while (fBuffer.hostState.load(std::memory_order_acquire) == BufferHandle::HostState::TransferFromDevice ||
           fBuffer.hostState.load(std::memory_order_acquire) == BufferHandle::HostState::TransferFromDeviceFinished) {

      if (fBuffer.hostState.load(std::memory_order_acquire) == BufferHandle::HostState::TransferFromDeviceFinished) {

        ProcessBuffer(fBuffer, cvG4Workers, debugLevel);
        haveNewHits = true;
      }
      // sleep shortly to reduce pressure on atomic reads
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(50us);
    }

    return haveNewHits;
  }

  bool ReadyToSwapBuffers() const
  {
    // we can swap if the next device state is free and the hostBuffer has been submitted to the HitQueue
    return (std::any_of(fDeviceState.begin(), fDeviceState.end(),
                        [](const auto &deviceState) {
                          return deviceState.load(std::memory_order_acquire) == DeviceState::Free;
                        }) &&
            fBuffer.hostState.load(std::memory_order_acquire) == BufferHandle::HostState::ReadyToBeFilled);
  }

  std::string GetDeviceStateName(DeviceState state) const
  {
    switch (state) {
    case DeviceState::Free:
      return "Free";
    case DeviceState::Filling:
      return "Filling";
    case DeviceState::NeedTransferToHost:
      return "NeedTransferToHost";
    case DeviceState::TransferToHost:
      return "TransferToHost";
    default:
      return "Unknown";
    }
  }

  std::string GetHostStateName(BufferHandle::HostState state) const
  {
    switch (state) {
    case BufferHandle::HostState::ReadyToBeFilled:
      return "ReadytoBeFilled";
    case BufferHandle::HostState::AwaitingDeviceTransfer:
      return "AwaitingDeviceTransfer";
    case BufferHandle::HostState::TransferFromDevice:
      return "TransferFromDevice";
    case BufferHandle::HostState::TransferFromDeviceFinished:
      return "TransferFromDeviceFinished";
    default:
      return "Unknown";
    }
  }

  void PrintDeviceBufferStates() const
  {
    std::cout << " DeviceBufferStates: active : " << fActiveBuffer;
    for (const auto &deviceState : fDeviceState) {
      std::cout << " [DeviceState: " << GetDeviceStateName(deviceState) << "] ";
    }
    std::cout << std::endl;
  }

  void PrintHostBufferState() const
  {
    std::cout << " HostBufferState: ";
    std::cout << " [HostState: " << GetHostStateName(fBuffer.hostState) << "] ";
    std::cout << std::endl;
  }

  /// Copy the current contents of the GPU hit buffer to host.
  void TransferHitsToHost(cudaStream_t cudaStreamForHitCopy)
  {

    while (std::any_of(fDeviceState.begin(), fDeviceState.end(),
                       [](auto &deviceState) { return deviceState.load() == DeviceState::NeedTransferToHost; })) {

#ifdef DEBUG
      if (fBuffer.hostState.load() != BufferHandle::HostState::AwaitingDeviceTransfer)
        std::cout << BOLD_RED << " Hoststate Wrong! should AwaitingDeviceTransfer but state is "
                  << GetHostStateName(fBuffer.hostState) << RESET << std::endl;
#endif

      // previous active device buffer - from this one we need to transfer the data to the host
      short prevActiveDeviceBuffer = (fActiveBuffer + fDeviceState.size() - 1) % fDeviceState.size();
      if (fDeviceState[prevActiveDeviceBuffer].load() != DeviceState::NeedTransferToHost) {
        continue;
      }

      unsigned int transferSize = 0;
      for (int i = 0; i < fBuffer.hitScoringInfo[fActiveBuffer].fNThreads; i++) {
        transferSize += fBuffer.hostBufferCount[i];
      }

      if (fBufferManager->getFreeContiguousMemory(transferSize) <= transferSize) {
        continue;
      }

      // set both states to Transfer
      fDeviceState[prevActiveDeviceBuffer].store(DeviceState::TransferToHost, std::memory_order_release);
      fBuffer.hostState.store(BufferHandle::HostState::TransferFromDevice);

      auto bufferBegin     = fBuffer.hitScoringInfo[prevActiveDeviceBuffer].hitBuffer_dev;
      fBuffer.offsetAtCopy = fBufferManager->getOffset();

      // Copy out the hits:
      // The start address on device is always i * fNSlot (Slots per thread), and we copy always to
      // the offset of the previous copy, to get a compact buffer on host.
      unsigned int offset = 0;
      for (int i = 0; i < fBuffer.hitScoringInfo[prevActiveDeviceBuffer].fNThreads; i++) {
        if (fBuffer.hostBufferCount[i] > 0) {
          COPCORE_CUDA_CHECK(cudaMemcpyAsync(fBuffer.hostBuffer + fBuffer.offsetAtCopy + offset,
                                             bufferBegin + i * fBuffer.hitScoringInfo[prevActiveDeviceBuffer].fNSlot,
                                             sizeof(GPUHit) * fBuffer.hostBufferCount[i], cudaMemcpyDefault,
                                             cudaStreamForHitCopy));
          offset += fBuffer.hostBufferCount[i];
        }
      }

      // Launch the host function to set the states correctly

      COPCORE_CUDA_CHECK(cudaLaunchHostFunc(
          cudaStreamForHitCopy,
          [](void *arg) {
            static_cast<std::atomic<DeviceState> *>(arg)->store(DeviceState::Free, std::memory_order_release);
          },
          &fDeviceState[prevActiveDeviceBuffer]));

      COPCORE_CUDA_CHECK(cudaLaunchHostFunc(
          cudaStreamForHitCopy,
          [](void *arg) {
            static_cast<BufferHandle *>(arg)->hostState.store(BufferHandle::HostState::TransferFromDeviceFinished,
                                                              std::memory_order_release);
          },
          &fBuffer));
    }
  }

  HitQueueItem *GetNextHitsHandle(unsigned int threadId, bool &dataOnBuffer)
  {
    assert(threadId < fHitQueues.size());
    std::unique_lock lock{fHitQueueLocks[threadId]}; // setting scoring started flag, need unique lock

    if (fHitQueues[threadId].empty())
      return nullptr;
    else {
      auto &ret    = fHitQueues[threadId].front();
      dataOnBuffer = ret.IsDataOnHostBuffer.load();
      ret.ScoringStarted.store(true, std::memory_order_release);
      lock.unlock(); // Unlock before returning pointer
      return &ret;
    }
  }

  void CloseHitsHandle(unsigned int threadId, GPUHit *begin, const bool dataOnBuffer)
  {
    assert(threadId < fHitQueues.size());
    std::unique_lock lock{fHitQueueLocks[threadId]}; // popping queue, requires unique lock

#ifdef DEBUG
    if (fHitQueues[threadId].empty()) {
      std::cout << BOLD_RED << "ERROR no HitQueueItem to pop, this should nevre be the case! " << RESET << std::endl;
    }
#endif

    // if data is in the hostBuffer (and not the holdoutBuffer), update the CircularBufferManager and release the memory
    if (dataOnBuffer) fBufferManager->removeSegment(begin);
    fHitQueues[threadId].pop_front();
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
__device__ void RecordHit(AsyncAdePT::PerEventScoring * /*scoring*/, uint64_t aTrackID, uint64_t aParentID,
                          short creatorProcessId, short stepLimitingProcessId, char aParticleType, double aStepLength,
                          double aTotalEnergyDeposit, float aTrackWeight,
                          vecgeom::Vector3D<Precision> const &aVertexPosition,
                          vecgeom::NavigationState const &aPreState, vecgeom::Vector3D<Precision> const &aPrePosition,
                          vecgeom::Vector3D<Precision> const &aPreMomentumDirection, double aPreEKin, double aPreCharge,
                          vecgeom::NavigationState const &aPostState, vecgeom::Vector3D<Precision> const &aPostPosition,
                          vecgeom::Vector3D<Precision> const &aPostMomentumDirection, double aPostEKin,
                          double aPostCharge, double aGlobalTime, double aLocalTime, unsigned int eventID,
                          short threadID, bool isLastStep, bool isFirstStep)
{
  // Acquire a hit slot
  GPUHit &aGPUHit = AsyncAdePT::gHitScoringBuffer_dev.GetNextSlot(threadID);

  // Fill the required data
  FillHit(aGPUHit, aTrackID, aParentID, creatorProcessId, stepLimitingProcessId, aParticleType, aStepLength,
          aTotalEnergyDeposit, aTrackWeight, aVertexPosition, aPreState, aPrePosition, aPreMomentumDirection, aPreEKin,
          aPreCharge, aPostState, aPostPosition, aPostMomentumDirection, aPostEKin, aPostCharge, aGlobalTime,
          aLocalTime, eventID, threadID, isLastStep, isLastStep);
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
