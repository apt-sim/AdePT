// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_variable_size_obj_allocator.cu
 * @brief Unit tests for CPU and CUDA variable-size object allocators.
 */

#include <AdePT/transport/containers/BlockData.h>
#include <AdePT/transport/containers/VariableSizeObjAllocator.h>
#include <AdePT/transport/containers/mpmc_bounded_queue.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <initializer_list>
#include <new>
#include <stdexcept>

namespace {

using Queue = adept::mpmc_bounded_queue<int>;
using Block = adept::BlockData<int>;

struct RejectSecondConstruction {
  static inline std::size_t numAttempts{0};
  static inline std::size_t numReleased{0};

  static constexpr std::size_t SizeOfAlignAware(std::size_t) { return sizeof(RejectSecondConstruction); }
  static RejectSecondConstruction *MakeInstanceAt(std::size_t, void *address)
  {
    if (++numAttempts == 2) return nullptr;
    return new (address) RejectSecondConstruction{};
  }

  static void ReleaseInstance(RejectSecondConstruction *object)
  {
    ++numReleased;
    object->~RejectSecondConstruction();
  }
};

template <copcore::BackendType Backend>
void ExpectPartialConstructionUnwound()
{
  RejectSecondConstruction::numAttempts = 0;
  RejectSecondConstruction::numReleased = 0;
  copcore::VariableSizeObjAllocator<RejectSecondConstruction, Backend> allocator{1};

  EXPECT_THROW(allocator.allocate(3), std::runtime_error);
  EXPECT_EQ(2, RejectSecondConstruction::numAttempts);
  EXPECT_EQ(1, RejectSecondConstruction::numReleased);
}

template <copcore::BackendType Backend, typename Container>
void ExpectInvalidCapacitiesRejected()
{
  for (const std::size_t capacity : {0u, 1u, 3u, 6u}) {
    copcore::VariableSizeObjAllocator<Container, Backend> allocator{capacity};
    Container *allocated = nullptr;
    EXPECT_THROW(allocated = allocator.allocate(2), std::runtime_error) << "capacity " << capacity;
    EXPECT_EQ(nullptr, allocated);
  }
}

TEST(VariableSizeObjAllocatorTest, CPURejectsInvalidQueueCapacities)
{
  ExpectInvalidCapacitiesRejected<copcore::BackendType::CPU, Queue>();
}

TEST(VariableSizeObjAllocatorTest, CPURejectsInvalidBlockDataCapacities)
{
  ExpectInvalidCapacitiesRejected<copcore::BackendType::CPU, Block>();
}

TEST(VariableSizeObjAllocatorTest, CUDARejectsInvalidQueueCapacities)
{
  ExpectInvalidCapacitiesRejected<copcore::BackendType::CUDA, Queue>();
}

TEST(VariableSizeObjAllocatorTest, CUDARejectsInvalidBlockDataCapacities)
{
  ExpectInvalidCapacitiesRejected<copcore::BackendType::CUDA, Block>();
}

TEST(VariableSizeObjAllocatorTest, CPUUnwindsPartialConstruction)
{
  ExpectPartialConstructionUnwound<copcore::BackendType::CPU>();
}

TEST(VariableSizeObjAllocatorTest, CUDAUnwindsPartialConstruction)
{
  ExpectPartialConstructionUnwound<copcore::BackendType::CUDA>();
}

} // namespace
