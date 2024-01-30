// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
#ifndef SLOTMANAGER_CUH
#define SLOTMANAGER_CUH

#include "CopCore/Global.h"

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
  value_type fSlotCounterMax = 0;

public:
  __host__ __device__ SlotManager(const value_type slotListSize, const value_type freeListSize)
      : fSlotListSize{slotListSize}, fFreeListSize{freeListSize}
  {
#ifdef __CUDA_ARCH__
    Clear();
#else
    const auto memSize = sizeof(value_type) * (fSlotListSize + fFreeListSize);
    if (memSize == 0) return;

    COPCORE_CUDA_CHECK(cudaMalloc(&fSlotList, memSize));
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
  SlotManager & operator=(SlotManager && other)
  {
    fSlotListSize = other.fSlotListSize;
    fFreeListSize = other.fFreeListSize;
    fSlotList = other.fSlotList;
    fToFreeList = other.fToFreeList;
    fSlotCounter = other.fSlotCounter;
    fFreeCounter = other.fFreeCounter;
    fSlotCounterMax = other.fSlotCounterMax;

    // Only one slot manager can own the device memory
    other.fSlotList = nullptr;

    return *this;
  }

  __device__ value_type __forceinline__ HighestOccupiedSlotIndex() const
  {
    return fSlotCounterMax > fSlotCounter ? fSlotCounterMax : fSlotCounter;
  }

  __host__ __device__ void Clear();

  __device__ unsigned int NextSlot();

  __device__ void MarkSlotForFreeing(unsigned int toBeFreed);

  __device__ value_type OccupiedSlots() const { return fSlotCounter - fFreeCounter; }

  __device__ void FreeMarkedSlots();
  __host__ static void SortListOfFreeSlots(int slotMgrIndex, cudaStream_t stream);
  __host__ static std::pair<value_type *, std::byte *> MemForSorting(int slotMgrIndex, value_type numItemsToSort,
                                                                     size_t sortFuncTempMemorySize);

  __device__ void __forceinline__ UpdateEndPtr();
};

__host__ __device__ void SlotManager::Clear()
{
#ifdef __CUDA_ARCH__
  for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; i < fSlotListSize; i += blockDim.x * gridDim.x) {
    fSlotList[i] = i;
  }
  if (threadIdx.x == 0) {
    fSlotCounterMax = 0;
    fSlotCounter    = 0;
    fFreeCounter    = 0;
  }
#endif
}

__device__ unsigned int SlotManager::NextSlot()
{
  const auto slotIndex = atomicAdd(&fSlotCounter, 1);
  if (slotIndex >= fSlotListSize) {
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

__device__ void SlotManager::FreeMarkedSlots()
{
  const auto oldSlotCounter = fSlotCounter;

  if (oldSlotCounter < fFreeCounter && threadIdx.x == 0) {
    printf(__FILE__ ":%d (%d,%d) Error: Trying to free too many slots: free %d when allocated %d.\n", __LINE__,
           blockIdx.x, threadIdx.x, fFreeCounter, oldSlotCounter);
    for (unsigned int i = 0; i < fFreeCounter; ++i) {
      printf("%d ", fToFreeList[i]);
    }
    printf("\n");
    // COPCORE_EXCEPTION("Error: Trying to free too many slots.");
  }

  const auto begin = oldSlotCounter - fFreeCounter;
  for (unsigned int i = threadIdx.x; i < fFreeCounter; i += blockDim.x) {
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
    fSlotCounterMax = fSlotCounterMax > fSlotCounter ? fSlotCounterMax : fSlotCounter;
    fSlotCounter    = begin;
    fFreeCounter    = 0;
  }
}

__device__ void __forceinline__ SlotManager::UpdateEndPtr()
{
  __shared__ value_type upperBoundShared[1024];
  auto upperBound            = fSlotCounter;
  const auto LastSlotToVisit = fSlotCounterMax;
  for (value_type i = fSlotCounter + threadIdx.x; i < LastSlotToVisit; i += blockDim.x) {
    if (fSlotList[i] != i) upperBound = i;
  }
  upperBoundShared[threadIdx.x] = upperBound;

  __syncthreads();

  if (threadIdx.x == 0) {
    for (unsigned int i = 0; i < blockDim.x; ++i) {
      auto max = [](auto a, auto b) __attribute__((always_inline))
      {
        return a > b ? a : b;
      };
      upperBound = max(upperBound, upperBoundShared[i]);
    }

    fSlotCounterMax = upperBound + 1;
    assert(fSlotCounterMax < fSlotListSize);
    assert(fSlotCounterMax >= fSlotCounter);
  }
}

#endif //SLOTMANAGER_CUH