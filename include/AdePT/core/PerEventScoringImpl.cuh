// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef PER_EVENT_SCORING_CUH
#define PER_EVENT_SCORING_CUH

#include <AdePT/base/ResourceManagement.cuh>
#include <AdePT/core/GPUStep.hh>
#include <AdePT/core/HostCircularBuffer.hh>
#include <AdePT/copcore/Global.h>

#include <VecGeom/navigation/NavigationState.h>

#include <algorithm>
#include <atomic>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <array>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <sstream>
#include <vector>

// definitions for printouts and advanced debugging
// #define DEBUG
#define RESET "\033[0m"
#define BOLD_RED "\033[1;31m"
#define BOLD_BLUE "\033[1;34m"

// Comparison for sorting tracks into events on device:
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

struct BufferHandle {
  std::array<DeviceStepBufferView, 2> stepBufferViews;
  GPUStep *hostBuffer;
  unsigned int *hostStepCount;
  enum class HostState { ReadyToBeFilled, AwaitingDeviceTransfer, TransferFromDevice, TransferFromDeviceFinished };

  // the hostState is changed by the StepProcessingThread but also by the cudaHostFunction launch, so it must be atomic
  std::atomic<HostState> hostState;
  // offset in the buffer at the time of the copy (maybe redundant now?)
  unsigned int offsetAtCopy = 0;
};

struct GPUStepBatch {
  // GPUStepBatch objects are pushed into the StepQueue. They contain the begin and end pointers of the data that needs
  // to be processed. If the G4Workers are too slow to respond, the StepProcessingThread copies the data from the buffer
  // into the holdoutBuffer, so that the host buffer in pinned memory can be released.
  GPUStep *begin; // begin of GPUStep pointer
  GPUStep *end;   // end of GPUStep pointer
  std::atomic_bool ProcessingStarted =
      false; // whether the processing has started. If it has not, the StepProcessingThread
             // will copy the data from the pinned memory host buffer to the holdoutBuffer
  std::atomic_bool IsDataOnHostBuffer =
      true; // whether the data resides in the HostBuffer in pinned memory (if false, it is in the holdoutBuffer)
  std::vector<GPUStep>
      holdoutBuffer; // holdout buffer where the steps are copied to if the G4Worker is too slow to respond

  GPUStepBatch(GPUStep *begin_, GPUStep *end_) : begin(begin_), end(end_) {}

  ~GPUStepBatch() = default;

  // remove copy constructor and assignment operator
  GPUStepBatch(const GPUStepBatch &)            = delete;
  GPUStepBatch &operator=(const GPUStepBatch &) = delete;

  // write custom move constructor and assignment operator
  GPUStepBatch(GPUStepBatch &&other) noexcept
      : begin(other.begin), end(other.end), ProcessingStarted(other.ProcessingStarted.load()),
        holdoutBuffer(std::move(other.holdoutBuffer))
  {
  }

  GPUStepBatch &operator=(GPUStepBatch &&other) noexcept
  {
    if (this != &other) {
      begin = other.begin;
      end   = other.end;
      ProcessingStarted.store(other.ProcessingStarted.load());
      holdoutBuffer = std::move(other.holdoutBuffer);
    }
    return *this;
  }
};

class GPUStepTransferManager {
  unique_ptr_cuda<GPUStep> fGPUStepBuffer_dev;
  unique_ptr_cuda<GPUStep, CudaHostDeleter<GPUStep>> fGPUStepBuffer_host;
  unique_ptr_cuda<unsigned int> fGPUStepBufferCount_dev;
  unique_ptr_cuda<unsigned int, CudaHostDeleter<unsigned int>> fGPUStepBufferCount_host;

  BufferHandle fBuffer;

  std::unique_ptr<HostCircularBuffer> fBufferManager;

  enum class DeviceState { Free, Filling, NeedTransferToHost, TransferToHost };
  std::array<std::atomic<DeviceState>, 2>
      fDeviceState; // the device state must be atomic as it is touched by both the StepProcessingThread in
                    // TransferStepsToHost and by the TransportThread in SwapDeviceBuffers

  void *fDeviceStepBuffer_deviceAddress = nullptr;
  unsigned int fStepCapacity;
  double fCPUCapacityFactor;
  double fCPUCopyFraction;
  unsigned short fActiveBuffer = 0;
  ADEPT_DEVICE_API_SYMBOL(Event_t)
  fSwapDoneEvent; // cuda event to synchronize the swapping of the device buffers with the transport

  // StepQueue with one lock per queue
  std::vector<std::deque<GPUStepBatch>> fStepQueues;
  std::vector<std::shared_mutex> fStepQueueLocks;

  void ProcessBuffer(BufferHandle &handle, std::condition_variable &cvG4Workers, int debugLevel)
  {

    // Loop over StepQueue and add GPUStepBatch objects that contain the begin and end of the GPUSteps in the
    // HostBuffer. Add the used segments in the BufferManager
    unsigned int offset = 0;
    for (int i = 0; i < fStepQueues.size(); i++) {

      GPUStep *begin = handle.hostBuffer + offset + handle.offsetAtCopy;
      GPUStep *end   = handle.hostBuffer + offset + handle.offsetAtCopy + handle.hostStepCount[i];

      GPUStepBatch stepBatch{begin, end};

      if (begin != end) {
        fBufferManager->addSegment(begin, end);

        std::scoped_lock lock{fStepQueueLocks[i]};
        fStepQueues[i].push_back(std::move(stepBatch));
      }
      offset += handle.hostStepCount[i];
    }
    // release HostBuffer and notify G4Workers
    fBuffer.hostState.store(BufferHandle::HostState::ReadyToBeFilled);
    cvG4Workers.notify_all();

    for (int i = 0; i < fStepQueues.size(); i++) {

      std::unique_lock lock{fStepQueueLocks[i]};

      if (!fStepQueues[i].empty()) {
        // Check whether the last item in the StepQueue is already taken by the G4Worker
        auto &ret = fStepQueues[i].back();

        if (ret.ProcessingStarted.load(std::memory_order_acquire) == true) {
          // if G4Worker has already started working, all good
          if (debugLevel > 5)
            std::cout << BOLD_BLUE << "G4worker " << i << " has taken their task and started working on "
                      << (ret.end - ret.begin) << " steps " << RESET << std::endl;
        } else {

          size_t numSteps = ret.end - ret.begin;

          // If the circular Buffer is too full and the G4Worker didn't pick up the work, we have to copy out the steps
          // to the holdoutBuffer
          if (fBufferManager->getFillFraction() > fCPUCopyFraction) {
            if (debugLevel > 5) {
              std::cout << BOLD_RED << "FillFraction too high: " << fBufferManager->getFillFraction()
                        << ", threshold: " << fCPUCopyFraction << " copying out " << numSteps << " steps for G4Worker "
                        << i << RESET << std::endl;
            }
            ret.holdoutBuffer.resize(numSteps);                       // Allocate correct size
            std::copy(ret.begin, ret.end, ret.holdoutBuffer.begin()); // Copy data

            // remove the segment first, before updating the pointers to the copied out memory
            fBufferManager->removeSegment(ret.begin);

            // Update pointers
            ret.begin = ret.holdoutBuffer.data();
            ret.end   = ret.holdoutBuffer.data() + ret.holdoutBuffer.size();

            ret.ProcessingStarted  = true;
            ret.IsDataOnHostBuffer = false;
          }
        }
      }
    }
  }

public:
  GPUStepTransferManager(unsigned int stepCapacity, unsigned int nThread, double CPUCapacityFactor,
                         double CPUCopyFraction)
      : fStepCapacity{stepCapacity}, fStepQueues(nThread), fStepQueueLocks(nThread),
        fCPUCapacityFactor(CPUCapacityFactor), fCPUCopyFraction(CPUCopyFraction)
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
    GPUStep *gpuSteps = nullptr;

    // The HostBuffer is set to be fCPUCapacityFactor times the GPU buffer StepCapacity. Normally, maximally 2x of the
    // GPU step buffer should reside in the host buffer: once a full buffer that is currently processed by the G4
    // workers and then another full buffer that is just copied from the GPU. Due to sparsity, we add another factor of
    // .5 to prevent running out of buffer. Also, the filling quota of the CPU buffer decides whether steps are
    // processed directly by the G4 workers or if they are copied out
    unsigned int hostBufferCapacity = fCPUCapacityFactor * fStepCapacity;
    ADEPT_DEVICE_API_CALL(MallocHost(&gpuSteps, sizeof(GPUStep) * hostBufferCapacity));
    fGPUStepBuffer_host.reset(gpuSteps);

    // We use a single allocation for both GPU buffers:
    auto result = ADEPT_DEVICE_API_SYMBOL(Malloc)(&gpuSteps, sizeof(GPUStep) * fDeviceState.size() * fStepCapacity);
    if (result != ADEPT_DEVICE_API_SYMBOL(Success)) throw std::invalid_argument{"No space to allocate step buffer."};
    fGPUStepBuffer_dev.reset(gpuSteps);

    unsigned int *step_count = nullptr;
    ADEPT_DEVICE_API_CALL(MallocHost(&step_count, sizeof(unsigned int) * nThread));
    fGPUStepBufferCount_host.reset(step_count);

    result = ADEPT_DEVICE_API_SYMBOL(Malloc)(&step_count, sizeof(unsigned int) * fDeviceState.size() * nThread);
    if (result != ADEPT_DEVICE_API_SYMBOL(Success)) throw std::invalid_argument{"No space to allocate step buffer."};
    fGPUStepBufferCount_dev.reset(step_count);

    fDeviceState[0] = DeviceState::Filling;
    fDeviceState[1] = DeviceState::Free;

    fBuffer.stepBufferViews[0] =
        DeviceStepBufferView{fGPUStepBuffer_dev.get(), fGPUStepBufferCount_dev.get(), fStepCapacity / nThread, nThread};
    fBuffer.stepBufferViews[1] =
        DeviceStepBufferView{fGPUStepBuffer_dev.get() + fStepCapacity, fGPUStepBufferCount_dev.get() + nThread,
                             fStepCapacity / nThread, nThread};
    fBuffer.hostBuffer    = fGPUStepBuffer_host.get();
    fBuffer.hostStepCount = fGPUStepBufferCount_host.get();
    fBuffer.hostState     = BufferHandle::HostState::ReadyToBeFilled;

    fBufferManager = std::make_unique<HostCircularBuffer>(fGPUStepBuffer_host.get(), hostBufferCapacity);

    ADEPT_DEVICE_API_CALL(GetSymbolAddress(&fDeviceStepBuffer_deviceAddress, gDeviceStepBuffer));
    assert(fDeviceStepBuffer_deviceAddress != nullptr);
    ADEPT_DEVICE_API_CALL(Memcpy(fDeviceStepBuffer_deviceAddress, &fBuffer.stepBufferViews,
                                 sizeof(DeviceStepBufferView), ADEPT_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

    // create cuda event needed to tell the transport that the swap of the device buffers is executed
    ADEPT_DEVICE_API_CALL(EventCreateWithFlags(&fSwapDoneEvent, ADEPT_DEVICE_API_SYMBOL(EventDisableTiming)));
  }

  ADEPT_DEVICE_API_SYMBOL(Event_t) getSwapDoneEvent() const { return fSwapDoneEvent; }

  unsigned int StepCapacity() const { return fStepCapacity; }

  void SwapDeviceBuffers(ADEPT_DEVICE_API_SYMBOL(Stream_t) cudaStream)
  {

#ifdef DEBUG
    // Ensure that device side is free and the host side has been processed:
    if (fBuffer.hostState.load() != BufferHandle::HostState::ReadyToBeFilled)
      std::cout << BOLD_RED << " Hoststate is wrong, must be ReadyToBeFilled! " << RESET << std::endl;
    if (fDeviceState[fActiveBuffer].load(std::memory_order_acquire) != DeviceState::Filling)
      throw std::logic_error(__FILE__ + std::to_string(__LINE__) + ": On-device buffer in wrong state");
#endif

    // Get new StepBufferCounts from device:
    ADEPT_DEVICE_API_CALL(MemcpyAsync(fBuffer.hostStepCount, fBuffer.stepBufferViews[fActiveBuffer].fSlotCounter,
                                      sizeof(unsigned int) * fBuffer.stepBufferViews[fActiveBuffer].fNThreads,
                                      ADEPT_DEVICE_API_SYMBOL(MemcpyDefault), cudaStream));
    ADEPT_DEVICE_API_CALL(MemsetAsync(fBuffer.stepBufferViews[fActiveBuffer].fSlotCounter, 0,
                                      sizeof(unsigned int) * fBuffer.stepBufferViews[fActiveBuffer].fNThreads,
                                      cudaStream));

    // Execute the swap:
    auto prevActiveDeviceBuffer = fActiveBuffer;
    fActiveBuffer               = (fActiveBuffer + 1) % fDeviceState.size();

    if (fDeviceState[fActiveBuffer].load(std::memory_order_acquire) != DeviceState::Free)
      throw std::logic_error(__FILE__ + std::to_string(__LINE__) + ": Next on-device buffer in wrong state");

    // adjust pointers to step buffer and slotcounter array to next active GPUbuffer
    fBuffer.stepBufferViews[fActiveBuffer].stepBuffer_dev = fGPUStepBuffer_dev.get() + fActiveBuffer * fStepCapacity;
    fBuffer.stepBufferViews[fActiveBuffer].fSlotCounter =
        fGPUStepBufferCount_dev.get() + fActiveBuffer * fBuffer.stepBufferViews[fActiveBuffer].fNThreads;

    ADEPT_DEVICE_API_CALL(MemcpyAsync(fDeviceStepBuffer_deviceAddress, &fBuffer.stepBufferViews[fActiveBuffer],
                                      sizeof(DeviceStepBufferView), ADEPT_DEVICE_API_SYMBOL(MemcpyDefault),
                                      cudaStream));
    ADEPT_DEVICE_API_CALL(
        EventRecord(fSwapDoneEvent, cudaStream)); // record event that the transport kernels must wait for

    // Need to set the hostState to awaiting device. This prevents the next swap until the steps in the host buffer are
    // sent to the StepQueue.
    fBuffer.hostState.store(BufferHandle::HostState::AwaitingDeviceTransfer, std::memory_order_release);

    // the current active buffer can be set directly to block it from being seen as free
    fDeviceState[fActiveBuffer].store(DeviceState::Filling, std::memory_order_release);

    // However, the prevActiveDeviceBuffer state can only be advanced when the transfer is finished,
    // otherwise the TransferStepsToHost may call the transfer before the correct hostStepCount has arrived
    ADEPT_DEVICE_API_CALL(LaunchHostFunc(
        cudaStream,
        [](void *arg) {
          static_cast<std::atomic<DeviceState> *>(arg)->store(DeviceState::NeedTransferToHost,
                                                              std::memory_order_release);
        },
        &fDeviceState[prevActiveDeviceBuffer]));
  }

  bool ProcessSteps(std::condition_variable &cvG4Workers, int debugLevel)
  {

    bool haveNewSteps = false;

    // here we need to do atomic checks on the hostState, as it is modified from the cudaLaunchHostFunc
    while (fBuffer.hostState.load(std::memory_order_acquire) == BufferHandle::HostState::TransferFromDevice ||
           fBuffer.hostState.load(std::memory_order_acquire) == BufferHandle::HostState::TransferFromDeviceFinished) {

      if (fBuffer.hostState.load(std::memory_order_acquire) == BufferHandle::HostState::TransferFromDeviceFinished) {

        ProcessBuffer(fBuffer, cvG4Workers, debugLevel);
        haveNewSteps = true;
      }
      // sleep shortly to reduce pressure on atomic reads
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(50us);
    }

    return haveNewSteps;
  }

  bool ReadyToSwapBuffers() const
  {
    // we can swap if the next device state is free and the hostBuffer has been submitted to the StepQueue
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

  /// Copy the current contents of the GPU step buffer to host.
  void TransferStepsToHost(ADEPT_DEVICE_API_SYMBOL(Stream_t) cudaStreamForStepCopy)
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
      for (int i = 0; i < fBuffer.stepBufferViews[fActiveBuffer].fNThreads; i++) {
        transferSize += fBuffer.hostStepCount[i];
      }

      if (fBufferManager->getFreeContiguousMemory(transferSize) < transferSize) {
        continue;
      }

      // set both states to Transfer
      fDeviceState[prevActiveDeviceBuffer].store(DeviceState::TransferToHost, std::memory_order_release);
      fBuffer.hostState.store(BufferHandle::HostState::TransferFromDevice);

      auto bufferBegin     = fBuffer.stepBufferViews[prevActiveDeviceBuffer].stepBuffer_dev;
      fBuffer.offsetAtCopy = fBufferManager->getOffset();

      // Copy out the steps:
      // The start address on device is always i * fNSlot (Slots per thread), and we copy always to
      // the offset of the previous copy, to get a compact buffer on host.
      unsigned int offset = 0;
      for (int i = 0; i < fBuffer.stepBufferViews[prevActiveDeviceBuffer].fNThreads; i++) {
        if (fBuffer.hostStepCount[i] > 0) {
          ADEPT_DEVICE_API_CALL(MemcpyAsync(fBuffer.hostBuffer + fBuffer.offsetAtCopy + offset,
                                            bufferBegin + i * fBuffer.stepBufferViews[prevActiveDeviceBuffer].fNSlot,
                                            sizeof(GPUStep) * fBuffer.hostStepCount[i],
                                            ADEPT_DEVICE_API_SYMBOL(MemcpyDefault), cudaStreamForStepCopy));
          offset += fBuffer.hostStepCount[i];
        }
      }

      // Launch the host function to set the states correctly

      ADEPT_DEVICE_API_CALL(LaunchHostFunc(
          cudaStreamForStepCopy,
          [](void *arg) {
            static_cast<std::atomic<DeviceState> *>(arg)->store(DeviceState::Free, std::memory_order_release);
          },
          &fDeviceState[prevActiveDeviceBuffer]));

      ADEPT_DEVICE_API_CALL(LaunchHostFunc(
          cudaStreamForStepCopy,
          [](void *arg) {
            static_cast<BufferHandle *>(arg)->hostState.store(BufferHandle::HostState::TransferFromDeviceFinished,
                                                              std::memory_order_release);
          },
          &fBuffer));
    }
  }

  GPUStepBatch *GetNextStepBatch(unsigned int threadId, bool &dataOnBuffer)
  {
    assert(threadId < fStepQueues.size());
    std::unique_lock lock{fStepQueueLocks[threadId]}; // setting processing started flag, need unique lock

    if (fStepQueues[threadId].empty()) {
      return nullptr;
    } else {
      auto &ret    = fStepQueues[threadId].front();
      dataOnBuffer = ret.IsDataOnHostBuffer.load();
      ret.ProcessingStarted.store(true, std::memory_order_release);
      lock.unlock(); // Unlock before returning pointer
      return &ret;
    }
  }

  void CloseStepBatch(unsigned int threadId, GPUStep *begin, const bool dataOnBuffer)
  {
    assert(threadId < fStepQueues.size());
    std::unique_lock lock{fStepQueueLocks[threadId]}; // popping queue, requires unique lock

#ifdef DEBUG
    if (fStepQueues[threadId].empty()) {
      std::cout << BOLD_RED << "ERROR no GPUStepBatch to pop, this should never be the case! " << RESET << std::endl;
    }
#endif

    // if data is in the hostBuffer (and not the holdoutBuffer), update the HostCircularBuffer and release the memory
    if (dataOnBuffer) fBufferManager->removeSegment(begin);
    fStepQueues[threadId].pop_front();
  }
};

} // namespace AsyncAdePT

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
