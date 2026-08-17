// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_mparray.cu
 * @brief Unit tests for the host/device multi-producer array.
 */

#include <AdePT/transport/containers/MParrayT.h>
#include <AdePT/transport/support/Portability.hh>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

using Array = adept::MParrayT<int>;

struct ArrayDeleter {
  void operator()(Array *array) const { Array::ReleaseInstance(array); }
};

using ArrayPtr = std::unique_ptr<Array, ArrayDeleter>;

class ManagedArray {
public:
  explicit ManagedArray(int capacity)
  {
    const auto result = ADEPT_DEVICE_API_SYMBOL(MallocManaged)(&fStorage, Array::SizeOfInstance(capacity));
    if (result != ADEPT_DEVICE_API_SYMBOL(Success)) {
      throw std::runtime_error{ADEPT_DEVICE_API_SYMBOL(GetErrorString)(result)};
    }
    fArray = Array::MakeInstanceAt(capacity, fStorage);
  }

  ~ManagedArray()
  {
    if (fArray) Array::ReleaseInstance(fArray);
    if (fStorage) ADEPT_DEVICE_API_SYMBOL(Free)(fStorage);
  }

  Array *get() const { return fArray; }

private:
  char *fStorage{nullptr};
  Array *fArray{nullptr};
};

__global__ void PushValues(Array *array, unsigned int numValues, unsigned int *failures)
{
  const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < numValues && !array->push_back(static_cast<int>(id))) atomicAdd(failures, 1u);
}

TEST(MParrayTest, ZeroCapacityHasValidStorageAndEmptyAccessFails)
{
  EXPECT_EQ(sizeof(Array), Array::SizeOfInstance(0));
  EXPECT_EQ(0u, Array::SizeOfAlignAware(0) % alignof(Array));

  ArrayPtr array{Array::MakeInstance(0)};
  ASSERT_NE(nullptr, array);
  EXPECT_EQ(0u, array->size());
  EXPECT_TRUE(array->empty());
  EXPECT_TRUE(array->full());
  EXPECT_EQ(array->begin(), array->end());
  EXPECT_FALSE(array->push_back(1));
  EXPECT_THROW(array->front(), std::runtime_error);
  EXPECT_THROW(array->back(), std::runtime_error);
}

TEST(MParrayTest, EmptyNonzeroCapacityBackFails)
{
  ArrayPtr array{Array::MakeInstance(4)};
  ASSERT_NE(nullptr, array);
  EXPECT_THROW(array->back(), std::runtime_error);
}

TEST(MParrayTest, PushesToCapacityAndCanBeClearedAndReused)
{
  ArrayPtr array{Array::MakeInstance(4)};
  ASSERT_NE(nullptr, array);

  EXPECT_TRUE(array->push_back(10));
  EXPECT_TRUE(array->push_back(20));
  EXPECT_TRUE(array->push_back(30));
  EXPECT_TRUE(array->push_back(40));
  EXPECT_FALSE(array->push_back(50));

  EXPECT_EQ(4u, array->size());
  EXPECT_TRUE(array->full());
  EXPECT_EQ(10, array->front());
  EXPECT_EQ(40, array->back());
  EXPECT_EQ(std::vector<int>({10, 20, 30, 40}), std::vector<int>(array->begin(), array->end()));

  array->clear();
  EXPECT_TRUE(array->empty());
  EXPECT_FALSE(array->full());
  EXPECT_TRUE(array->push_back(73));
  EXPECT_EQ(73, array->front());
  EXPECT_EQ(73, array->back());
}

TEST(MParrayTest, ShrinkingCopyTruncatesWithoutOverwritingDestination)
{
  ArrayPtr source{Array::MakeInstance(4)};
  ASSERT_NE(nullptr, source);
  for (int value : {11, 22, 33, 44})
    ASSERT_TRUE(source->push_back(value));

  constexpr std::size_t canaryBytes = 64;
  constexpr unsigned char canary    = 0xa5;
  const std::size_t destinationSize = Array::SizeOfInstance(2);
  std::vector<unsigned char> storage(destinationSize + canaryBytes, canary);
  ASSERT_EQ(0u, reinterpret_cast<std::uintptr_t>(storage.data()) % alignof(Array));

  Array *copy = Array::MakeCopyAt(2, *source, storage.data());
  ASSERT_NE(nullptr, copy);
  EXPECT_EQ(2u, copy->max_size());
  EXPECT_EQ(11, (*copy)[0]);
  EXPECT_EQ(22, (*copy)[1]);
  EXPECT_TRUE(std::all_of(storage.begin() + destinationSize, storage.end(),
                          [](unsigned char value) { return value == canary; }));

  Array::ReleaseInstance(copy);
}

TEST(MParrayTest, GrowingCopyPreservesExistingValues)
{
  ArrayPtr source{Array::MakeInstance(3)};
  ASSERT_NE(nullptr, source);
  for (int value : {3, 5, 8})
    ASSERT_TRUE(source->push_back(value));

  ArrayPtr copy{Array::MakeCopy(6, *source)};
  ASSERT_NE(nullptr, copy);
  EXPECT_EQ(6u, copy->max_size());
  EXPECT_EQ(3, (*copy)[0]);
  EXPECT_EQ(5, (*copy)[1]);
  EXPECT_EQ(8, (*copy)[2]);
}

TEST(MParrayTest, ConcurrentDevicePushesStopAtCapacity)
{
  constexpr unsigned int capacity   = 1u << 12;
  constexpr unsigned int numValues  = capacity + 257;
  constexpr unsigned int numThreads = 128;
  constexpr unsigned int numBlocks  = (numValues + numThreads - 1) / numThreads;

  ManagedArray array{capacity};
  unsigned int *failures = nullptr;
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(MallocManaged)(&failures, sizeof(*failures)));
  *failures = 0;

  PushValues<<<numBlocks, numThreads>>>(array.get(), numValues, failures);
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(DeviceSynchronize)());

  EXPECT_EQ(numValues - capacity, *failures);
  ASSERT_EQ(capacity, array.get()->size());

  std::vector<int> values(array.get()->begin(), array.get()->end());
  std::sort(values.begin(), values.end());
  EXPECT_EQ(values.end(), std::adjacent_find(values.begin(), values.end()));
  EXPECT_TRUE(std::all_of(values.begin(), values.end(),
                          [](int value) { return value >= 0 && value < static_cast<int>(numValues); }));

  EXPECT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(Free)(failures));
}

} // namespace
