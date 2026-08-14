// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_slot_manager.cu
 * @brief Unit tests for SlotManager ownership and device-side slot recycling.
 */

#include <AdePT/transport/containers/SlotManager.cuh>
#include <AdePT/transport/support/Portability.hh>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename T>
class ManagedBuffer {
public:
  explicit ManagedBuffer(std::size_t size) : fSize{size}
  {
    if (size == 0) return;
    const auto result = ADEPT_DEVICE_API_SYMBOL(MallocManaged)(&fData, size * sizeof(T));
    if (result != ADEPT_DEVICE_API_SYMBOL(Success)) {
      throw std::runtime_error{ADEPT_DEVICE_API_SYMBOL(GetErrorString)(result)};
    }
  }

  ~ManagedBuffer()
  {
    if (fData) ADEPT_DEVICE_API_SYMBOL(Free)(fData);
  }

  ManagedBuffer(const ManagedBuffer &)            = delete;
  ManagedBuffer &operator=(const ManagedBuffer &) = delete;

  T *data() { return fData; }
  const T *data() const { return fData; }
  std::size_t size() const { return fSize; }

  T &operator[](std::size_t index) { return fData[index]; }
  const T &operator[](std::size_t index) const { return fData[index]; }

private:
  T *fData{nullptr};
  std::size_t fSize{0};
};

class ManagedSlotManager {
public:
  ManagedSlotManager(SlotManager::value_type slotListSize, SlotManager::value_type freeListSize)
      : fStorage{sizeof(SlotManager)}
  {
    fManager = new (fStorage.data()) SlotManager{slotListSize, freeListSize};
  }

  ~ManagedSlotManager() { fManager->~SlotManager(); }

  SlotManager *get() const { return fManager; }

private:
  ManagedBuffer<std::byte> fStorage;
  SlotManager *fManager{nullptr};
};

__global__ void ClearManager(SlotManager *manager)
{
  manager->Clear();
}

__global__ void AllocateSlots(SlotManager *manager, SlotManager::value_type *slots, SlotManager::value_type numSlots)
{
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numSlots) slots[index] = manager->NextSlot();
}

__global__ void MarkSlotsForFreeing(SlotManager *manager, const SlotManager::value_type *slots,
                                    SlotManager::value_type numSlots)
{
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numSlots) manager->MarkSlotForFreeing(slots[index]);
}

__global__ void FreeMarkedSlotsStage1(SlotManager *manager)
{
  manager->FreeMarkedSlotsStage1();
}

__global__ void FreeMarkedSlotsStage2(SlotManager *manager)
{
  manager->FreeMarkedSlotsStage2();
}

__global__ void ReadMetrics(const SlotManager *manager, SlotManager::value_type *occupiedSlots, float *fillLevel)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *occupiedSlots = manager->OccupiedSlots();
    *fillLevel     = manager->FillLevel();
  }
}

void ExpectMovedFrom(const SlotManager &manager)
{
  EXPECT_EQ(0u, manager.fSlotListSize);
  EXPECT_EQ(0u, manager.fFreeListSize);
  EXPECT_EQ(nullptr, manager.fSlotList);
  EXPECT_EQ(nullptr, manager.fToFreeList);
  EXPECT_EQ(0u, manager.fSlotCounter);
  EXPECT_EQ(0u, manager.fFreeCounter);
}

std::vector<SlotManager::value_type> SortedValues(const ManagedBuffer<SlotManager::value_type> &values)
{
  std::vector<SlotManager::value_type> result(values.data(), values.data() + values.size());
  std::sort(result.begin(), result.end());
  return result;
}

TEST(SlotManagerTest, HasExclusiveNoThrowMoveOwnership)
{
  static_assert(!std::is_copy_constructible_v<SlotManager>);
  static_assert(!std::is_copy_assignable_v<SlotManager>);
  static_assert(std::is_nothrow_move_constructible_v<SlotManager>);
  static_assert(std::is_nothrow_move_assignable_v<SlotManager>);
  EXPECT_EQ(64u, alignof(SlotManager));
}

TEST(SlotManagerTest, DefaultAndZeroSizedManagersAreEmpty)
{
  SlotManager defaultManager;
  ExpectMovedFrom(defaultManager);

  SlotManager zeroSizedManager{0, 0};
  ExpectMovedFrom(zeroSizedManager);
}

TEST(SlotManagerTest, SizedConstructorAllocatesAdjacentLists)
{
  constexpr SlotManager::value_type slotListSize = 17;
  constexpr SlotManager::value_type freeListSize = 9;
  SlotManager manager{slotListSize, freeListSize};

  EXPECT_EQ(slotListSize, manager.fSlotListSize);
  EXPECT_EQ(freeListSize, manager.fFreeListSize);
  ASSERT_NE(nullptr, manager.fSlotList);
  EXPECT_EQ(manager.fSlotList + slotListSize, manager.fToFreeList);
  EXPECT_EQ(0u, manager.fSlotCounter);
  EXPECT_EQ(0u, manager.fFreeCounter);
}

TEST(SlotManagerTest, MoveConstructorTransfersAllStateAndResetsSource)
{
  SlotManager source{17, 9};
  source.fSlotCounter = 7;
  source.fFreeCounter = 3;
  auto *slotList      = source.fSlotList;
  auto *toFreeList    = source.fToFreeList;

  SlotManager destination{std::move(source)};

  EXPECT_EQ(17u, destination.fSlotListSize);
  EXPECT_EQ(9u, destination.fFreeListSize);
  EXPECT_EQ(slotList, destination.fSlotList);
  EXPECT_EQ(toFreeList, destination.fToFreeList);
  EXPECT_EQ(7u, destination.fSlotCounter);
  EXPECT_EQ(3u, destination.fFreeCounter);
  ExpectMovedFrom(source);
}

TEST(SlotManagerTest, MoveAssignmentReleasesOldAllocationAndResetsSource)
{
  SlotManager destination{31, 13};
  auto *oldDestinationAllocation = destination.fSlotList;

  SlotManager source{17, 9};
  source.fSlotCounter    = 7;
  source.fFreeCounter    = 3;
  auto *sourceSlotList   = source.fSlotList;
  auto *sourceToFreeList = source.fToFreeList;

  destination = std::move(source);

  EXPECT_EQ(sourceSlotList, destination.fSlotList);
  EXPECT_EQ(sourceToFreeList, destination.fToFreeList);
  EXPECT_EQ(17u, destination.fSlotListSize);
  EXPECT_EQ(9u, destination.fFreeListSize);
  EXPECT_EQ(7u, destination.fSlotCounter);
  EXPECT_EQ(3u, destination.fFreeCounter);
  ExpectMovedFrom(source);

  ADEPT_DEVICE_API_SYMBOL(PointerAttributes) attributes{};
  const auto result = ADEPT_DEVICE_API_SYMBOL(PointerGetAttributes)(&attributes, oldDestinationAllocation);
  if (result == ADEPT_DEVICE_API_SYMBOL(Success)) {
    EXPECT_EQ(ADEPT_DEVICE_API_SYMBOL(MemoryTypeUnregistered), attributes.type);
  } else {
    EXPECT_EQ(ADEPT_DEVICE_API_SYMBOL(ErrorInvalidValue), result);
    ADEPT_DEVICE_API_SYMBOL(GetLastError)();
  }
}

TEST(SlotManagerTest, SelfMoveAssignmentPreservesOwnershipAndState)
{
  SlotManager manager{17, 9};
  manager.fSlotCounter = 7;
  manager.fFreeCounter = 3;
  auto *slotList       = manager.fSlotList;
  auto *toFreeList     = manager.fToFreeList;
  SlotManager *alias   = &manager;

  manager = std::move(*alias);

  EXPECT_EQ(17u, manager.fSlotListSize);
  EXPECT_EQ(9u, manager.fFreeListSize);
  EXPECT_EQ(slotList, manager.fSlotList);
  EXPECT_EQ(toFreeList, manager.fToFreeList);
  EXPECT_EQ(7u, manager.fSlotCounter);
  EXPECT_EQ(3u, manager.fFreeCounter);
}

TEST(SlotManagerTest, DeviceClearInitializesEverySlotAndResetsMetrics)
{
  constexpr SlotManager::value_type slotListSize = 257;
  ManagedSlotManager manager{slotListSize, 129};

  manager.get()->fSlotCounter = 73;
  manager.get()->fFreeCounter = 19;
  ClearManager<<<4, 128>>>(manager.get());
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(DeviceSynchronize)());

  EXPECT_EQ(0u, manager.get()->fSlotCounter);
  EXPECT_EQ(0u, manager.get()->fFreeCounter);

  std::vector<SlotManager::value_type> slotList(slotListSize);
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success),
            ADEPT_DEVICE_API_SYMBOL(Memcpy)(slotList.data(), manager.get()->fSlotList,
                                            slotList.size() * sizeof(slotList.front()),
                                            ADEPT_DEVICE_API_SYMBOL(MemcpyDeviceToHost)));
  for (SlotManager::value_type index = 0; index < slotListSize; ++index)
    EXPECT_EQ(index, slotList[index]);

  ManagedBuffer<SlotManager::value_type> occupiedSlots{1};
  ManagedBuffer<float> fillLevel{1};
  ReadMetrics<<<1, 1>>>(manager.get(), occupiedSlots.data(), fillLevel.data());
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(DeviceSynchronize)());
  EXPECT_EQ(0u, occupiedSlots[0]);
  EXPECT_FLOAT_EQ(0.0f, fillLevel[0]);
}

TEST(SlotManagerTest, ZeroCapacityHasZeroFillLevel)
{
  ManagedSlotManager manager{0, 0};
  ManagedBuffer<SlotManager::value_type> occupiedSlots{1};
  ManagedBuffer<float> fillLevel{1};

  ClearManager<<<2, 32>>>(manager.get());
  ReadMetrics<<<1, 1>>>(manager.get(), occupiedSlots.data(), fillLevel.data());
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(DeviceSynchronize)());

  EXPECT_EQ(0u, occupiedSlots[0]);
  EXPECT_FLOAT_EQ(0.0f, fillLevel[0]);
}

TEST(SlotManagerTest, DeviceAllocationAndTwoStageFreeRecycleExactSlots)
{
  constexpr SlotManager::value_type capacity  = 1024;
  constexpr SlotManager::value_type allocated = 512;
  constexpr SlotManager::value_type toFree    = 128;
  constexpr SlotManager::value_type threads   = 128;

  ManagedSlotManager manager{capacity, capacity};
  ManagedBuffer<SlotManager::value_type> firstAllocation{allocated};
  ManagedBuffer<SlotManager::value_type> recycledAllocation{toFree};
  ManagedBuffer<SlotManager::value_type> occupiedSlots{1};
  ManagedBuffer<float> fillLevel{1};

  ClearManager<<<4, threads>>>(manager.get());
  AllocateSlots<<<allocated / threads, threads>>>(manager.get(), firstAllocation.data(), allocated);
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(DeviceSynchronize)());

  auto allocatedSlots = SortedValues(firstAllocation);
  for (SlotManager::value_type index = 0; index < allocated; ++index)
    EXPECT_EQ(index, allocatedSlots[index]);
  EXPECT_EQ(allocated, manager.get()->fSlotCounter);
  EXPECT_EQ(0u, manager.get()->fFreeCounter);

  MarkSlotsForFreeing<<<1, threads>>>(manager.get(), firstAllocation.data(), toFree);
  ReadMetrics<<<1, 1>>>(manager.get(), occupiedSlots.data(), fillLevel.data());
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(DeviceSynchronize)());

  EXPECT_EQ(allocated - toFree, occupiedSlots[0]);
  EXPECT_FLOAT_EQ(0.5f, fillLevel[0]);
  EXPECT_EQ(toFree, manager.get()->fFreeCounter);
  std::vector<SlotManager::value_type> expectedRecycled(firstAllocation.data(), firstAllocation.data() + toFree);
  std::sort(expectedRecycled.begin(), expectedRecycled.end());

  FreeMarkedSlotsStage1<<<4, threads>>>(manager.get());
  FreeMarkedSlotsStage2<<<1, 1>>>(manager.get());
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(DeviceSynchronize)());
  EXPECT_EQ(allocated - toFree, manager.get()->fSlotCounter);
  EXPECT_EQ(0u, manager.get()->fFreeCounter);

  AllocateSlots<<<1, threads>>>(manager.get(), recycledAllocation.data(), toFree);
  ReadMetrics<<<1, 1>>>(manager.get(), occupiedSlots.data(), fillLevel.data());
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(DeviceSynchronize)());

  EXPECT_EQ(expectedRecycled, SortedValues(recycledAllocation));
  EXPECT_EQ(allocated, occupiedSlots[0]);
  EXPECT_FLOAT_EQ(0.5f, fillLevel[0]);
  EXPECT_EQ(allocated, manager.get()->fSlotCounter);
  EXPECT_EQ(0u, manager.get()->fFreeCounter);
}

} // namespace
