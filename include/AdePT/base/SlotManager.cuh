// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
#ifndef SLOTMANAGER_CUH
#define SLOTMANAGER_CUH

#include "AdePT/copcore/Global.h"

// A data structure to manage slots in the track storage.
// It manages two lists:
// - A list of free slots, which can be requested using NextSlot()
// - A list of slots to be freed. Slots can be marked for freeing
//   any time, but the actual freeing has to happen at the end of an iteration when no further slots are consumed.
struct alignas(64) SlotManager {
  using value_type = unsigned int;
  value_type fSlotListSize = 0;
  value_type fFreeListSize = 0;
  value_type * fSlotList    = nullptr;
  value_type * fToFreeList  = nullptr;

  value_type fSlotCounter    = 0;
  value_type fFreeCounter    = 0;

public:
  __host__ SlotManager() {}
  __host__ __device__ SlotManager(const value_type slotListSize, const value_type freeListSize)
      : fSlotListSize{slotListSize}, fFreeListSize{freeListSize}
  {
#ifdef __CUDA_ARCH__
    Clear();
#else
    const auto memSize = sizeof(value_type) * (fSlotListSize + fFreeListSize);
    if (memSize == 0) return;

    const auto result = cudaMalloc(&fSlotList, memSize);
    if (result != cudaSuccess) {
      throw std::invalid_argument{"SlotManager: Not enough memory for " + std::to_string(fSlotListSize) + " slots"};
    }
    fToFreeList          = fSlotList + fSlotListSize;
#endif
  }
  __host__ __device__ ~SlotManager() {
#ifndef __CUDA_ARCH__
    if (fSlotList) COPCORE_CUDA_CHECK(cudaFree(fSlotList));
#endif
  }

  SlotManager(const SlotManager &) = delete;
  SlotManager & operator=(const SlotManager &) = delete;
  SlotManager(SlotManager && other) :
    SlotManager{0, 0}
  {
    *this = std::move(other);
  }
  SlotManager &operator=(SlotManager &&other)
  {
    fSlotListSize = other.fSlotListSize;
    fFreeListSize = other.fFreeListSize;
    fSlotList = other.fSlotList;
    fToFreeList = other.fToFreeList;
    fSlotCounter = other.fSlotCounter;
    fFreeCounter  = other.fFreeCounter;

    // Only one slot manager can own the device memory
    other.fSlotList = nullptr;

    return *this;
  }

  __host__ __device__ void Clear();

  __device__ unsigned int NextSlot();

  __device__ void MarkSlotForFreeing(unsigned int toBeFreed);

  __device__ value_type OccupiedSlots() const { return fSlotCounter - fFreeCounter; }
  __device__ float FillLevel() const { return float(fSlotCounter) / fSlotListSize; }

  __device__ void FreeMarkedSlotsStage1();
  __device__ void FreeMarkedSlotsStage2();
  __host__ static void SortListOfFreeSlots(int slotMgrIndex, cudaStream_t stream);
  __host__ static std::pair<value_type *, std::byte *> MemForSorting(int slotMgrIndex, value_type numItemsToSort,
                                                                     size_t sortFuncTempMemorySize);
};

__host__ __device__ void SlotManager::Clear()
{
#ifdef __CUDA_ARCH__
  for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; i < fSlotListSize; i += blockDim.x * gridDim.x) {
    fSlotList[i] = i;
  }
  if (threadIdx.x == 0) {
    fSlotCounter    = 0;
    fFreeCounter    = 0;
  }
#endif
}

__device__ unsigned int SlotManager::NextSlot()
{
  const auto slotIndex = atomicAdd(&fSlotCounter, 1);
  if (slotIndex >= fSlotListSize) {
    printf("Out of slots: slotIndex=%d slotCounter=%d slotListSize=%d freeCounter=%d\n", slotIndex, fSlotCounter,
           fSlotListSize, fFreeCounter);
    COPCORE_EXCEPTION("Out of slots in SlotManager::NextSlot");
  }

  const auto result = fSlotList[slotIndex];
  assert(result < fSlotListSize);

  return result;
}

__device__ void SlotManager::MarkSlotForFreeing(unsigned int toBeFreed)
{
  const auto idx = atomicAdd(&fFreeCounter, 1);
  if (idx >= fFreeListSize) {
    COPCORE_EXCEPTION("Out of slots in freelist in SlotManager::MarkSlotForFreeing");
  }
  fToFreeList[idx] = toBeFreed;
}

/// @brief Move the queue of freed slots into the queue of available slots.
/// Note: FreeMarkedSlotsStage2 *must* run after this.
__device__ void SlotManager::FreeMarkedSlotsStage1()
{
  const auto oldSlotCounter = fSlotCounter;

  if (oldSlotCounter < fFreeCounter && threadIdx.x == 0) {
    printf(__FILE__ ":%d (%d,%d) Error: Trying to free too many slots: free %d when allocated %d.\n", __LINE__,
           blockIdx.x, threadIdx.x, fFreeCounter, oldSlotCounter);
    for (unsigned int i = 0; i < fFreeCounter; ++i) {
      printf("%d ", fToFreeList[i]);
    }
    printf("\n");
    COPCORE_EXCEPTION("Error: Trying to free too many slots.");
  }

  const auto begin = oldSlotCounter - fFreeCounter;
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < fFreeCounter; i += blockDim.x * gridDim.x) {
    const auto slotListIndex = begin + i;
    const auto toFree        = fToFreeList[i];

    fSlotList[slotListIndex] = toFree;
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    if (fSlotCounter != oldSlotCounter) {
      printf(__FILE__ ":%d Error: New slots were allocated while trying to free slots.\n", __LINE__);
      COPCORE_EXCEPTION("Allocating and freeing slots cannot overlap");
    }
  }
}

/// @brief Finish the freeing of slots by resetting the counters.
/// Note: FreeMarkedSlotsStage1 *must* run before this.
__device__ void SlotManager::FreeMarkedSlotsStage2()
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    fSlotCounter -= fFreeCounter;
    fFreeCounter    = 0;
  }
}

#ifndef NDEBUG
__global__ void AssertConsistencyOfSlotManagers(SlotManager *mgrs, std::size_t N)
{
  for (int i = 0; i < N; ++i) {
    SlotManager &mgr       = mgrs[i];
    const auto slotCounter = mgr.fSlotCounter;
    const auto freeCounter = mgr.fFreeCounter;

    if (blockIdx.x == 0 && threadIdx.x == 0 && slotCounter < freeCounter) {
      printf("Error %s:%d: Trying to free %d slots in manager %d whereas only %d allocated\n", __FILE__, __LINE__,
             freeCounter, i, slotCounter);
      for (unsigned int i = 0; i < freeCounter; ++i) {
        printf("%d ", mgr.fToFreeList[i]);
      }
      printf("\n");
      assert(false);
    }

    bool doubleFree = false;
    for (unsigned int j = blockIdx.x; j < mgr.fFreeCounter; j += gridDim.x) {
      const auto slotToSearch = mgr.fToFreeList[j];
      for (unsigned int k = j + 1 + threadIdx.x; k < freeCounter; k += blockDim.x) {
        if (slotToSearch == mgr.fToFreeList[k]) {
          printf("Error: Manager %d: Slot %d freed both at %d and at %d\n", i, slotToSearch, k, j);
          doubleFree = true;
          break;
        }
      }
    }

    assert(slotCounter == mgr.fSlotCounter && "Race condition while checking slots");
    assert(freeCounter == mgr.fFreeCounter && "Race condition while checking slots");
    assert(!doubleFree);
  }
}
#endif

#endif //SLOTMANAGER_CUH