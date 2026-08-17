// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_queue.cu
 * @brief Unit tests for the CUDA-aware bounded multi-producer/multi-consumer queue.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#include <AdePT/transport/containers/mpmc_bounded_queue.h>
#include <AdePT/transport/support/Portability.hh>

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

namespace {

using Queue = adept::mpmc_bounded_queue<int>;

struct QueueDeleter {
  void operator()(Queue *queue) const { Queue::ReleaseInstance(queue); }
};

using QueuePtr = std::unique_ptr<Queue, QueueDeleter>;

class ManagedQueue {
public:
  explicit ManagedQueue(int capacity)
  {
    const auto result = ADEPT_DEVICE_API_SYMBOL(MallocManaged)(&fStorage, Queue::SizeOfInstance(capacity));
    if (result != ADEPT_DEVICE_API_SYMBOL(Success)) {
      throw std::runtime_error{ADEPT_DEVICE_API_SYMBOL(GetErrorString)(result)};
    }
    fQueue = Queue::MakeInstanceAt(capacity, fStorage);
  }

  ~ManagedQueue()
  {
    if (fQueue) Queue::ReleaseInstance(fQueue);
    if (fStorage) ADEPT_DEVICE_API_SYMBOL(Free)(fStorage);
  }

  Queue *get() const { return fQueue; }

private:
  char *fStorage{nullptr};
  Queue *fQueue{nullptr};
};

__global__ void PushValues(Queue *queue, unsigned int numValues, unsigned int *failures)
{
  const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < numValues && !queue->enqueue(static_cast<int>(id))) atomicAdd(failures, 1u);
}

__global__ void PopAndSum(Queue *queue, unsigned int numValues, unsigned long long *sum, unsigned int *failures)
{
  const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= numValues) return;

  int value = 0;
  if (queue->dequeue(value)) {
    atomicAdd(sum, static_cast<unsigned long long>(value));
  } else {
    atomicAdd(failures, 1u);
  }
}

TEST(BoundedQueueTest, ReportsFullAndEmptyOnHost)
{
  QueuePtr queue{Queue::MakeInstance(8)};
  ASSERT_NE(nullptr, queue);
  EXPECT_EQ(0, queue->size());

  for (int value = 0; value < 8; ++value) {
    EXPECT_TRUE(queue->enqueue(value));
  }
  EXPECT_EQ(8, queue->size());
  EXPECT_FALSE(queue->enqueue(8));

  for (int expected = 0; expected < 8; ++expected) {
    int value = -1;
    ASSERT_TRUE(queue->dequeue(value));
    EXPECT_EQ(expected, value);
  }

  int value = -1;
  EXPECT_FALSE(queue->dequeue(value));
  EXPECT_EQ(0, queue->size());
}

TEST(BoundedQueueTest, RejectsInvalidCapacitiesWithAssertionsDisabled)
{
  for (const int capacity : {0, 1, 3, 6}) {
    Queue *allocated = Queue::MakeInstance(capacity);
    EXPECT_EQ(nullptr, allocated) << "capacity " << capacity;
    if (allocated) Queue::ReleaseInstance(allocated);

    auto storage  = std::make_unique<char[]>(Queue::SizeOfInstance(capacity));
    Queue *placed = Queue::MakeInstanceAt(capacity, storage.get());
    EXPECT_EQ(nullptr, placed) << "capacity " << capacity;
    if (placed) Queue::ReleaseInstance(placed);
  }
}

TEST(BoundedQueueTest, PreservesOrderAcrossWraparoundAndClear)
{
  QueuePtr queue{Queue::MakeInstance(8)};
  ASSERT_NE(nullptr, queue);

  for (int value = 0; value < 8; ++value)
    ASSERT_TRUE(queue->enqueue(value));

  for (int expected = 0; expected < 4; ++expected) {
    int value = -1;
    ASSERT_TRUE(queue->dequeue(value));
    EXPECT_EQ(expected, value);
  }

  for (int value = 8; value < 12; ++value)
    ASSERT_TRUE(queue->enqueue(value));

  for (int expected = 4; expected < 12; ++expected) {
    int value = -1;
    ASSERT_TRUE(queue->dequeue(value));
    EXPECT_EQ(expected, value);
  }

  ASSERT_TRUE(queue->enqueue(42));
  queue->clear();
  EXPECT_EQ(0, queue->size());
  int value = -1;
  EXPECT_FALSE(queue->dequeue(value));
  EXPECT_TRUE(queue->enqueue(73));
  ASSERT_TRUE(queue->dequeue(value));
  EXPECT_EQ(73, value);
}

TEST(BoundedQueueTest, CopiesPopulatedQueueWithoutChangingCapacity)
{
  QueuePtr source{Queue::MakeInstance(8)};
  ASSERT_NE(nullptr, source);

  for (int value = 0; value < 6; ++value)
    ASSERT_TRUE(source->enqueue(value));

  for (int expected = 0; expected < 2; ++expected) {
    int value = -1;
    ASSERT_TRUE(source->dequeue(value));
    ASSERT_EQ(expected, value);
  }

  for (int value = 6; value < 10; ++value)
    ASSERT_TRUE(source->enqueue(value));

  QueuePtr copy{Queue::MakeCopy(*source)};
  ASSERT_NE(nullptr, copy);
  ASSERT_EQ(8, copy->size());

  for (int expected = 2; expected < 10; ++expected) {
    int value = -1;
    ASSERT_TRUE(copy->dequeue(value));
    EXPECT_EQ(expected, value);
  }

  int value = -1;
  EXPECT_FALSE(copy->dequeue(value));
  EXPECT_EQ(8, source->size());
}

TEST(BoundedQueueTest, RejectsCapacityChangesForPopulatedCopies)
{
  QueuePtr source{Queue::MakeInstance(8)};
  ASSERT_NE(nullptr, source);

  for (int value = 0; value < 6; ++value)
    ASSERT_TRUE(source->enqueue(value));

  EXPECT_EQ(nullptr, Queue::MakeCopy(4, *source));

  auto storage = std::make_unique<char[]>(Queue::SizeOfInstance(4));
  EXPECT_EQ(nullptr, Queue::MakeCopyAt(4, *source, storage.get()));
  EXPECT_EQ(nullptr, Queue::MakeCopy(16, *source));

  ASSERT_EQ(6, source->size());
  for (int expected = 0; expected < 6; ++expected) {
    int value = -1;
    ASSERT_TRUE(source->dequeue(value));
    EXPECT_EQ(expected, value);
  }
}

TEST(BoundedQueueTest, ConcurrentDeviceProducersAndConsumers)
{
  constexpr unsigned int numValues = 1u << 14;
  constexpr dim3 threads{128};
  constexpr dim3 blocks{numValues / threads.x};

  ManagedQueue queue{numValues};
  unsigned int *failures  = nullptr;
  unsigned long long *sum = nullptr;
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(MallocManaged)(&failures, sizeof(*failures)));
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(MallocManaged)(&sum, sizeof(*sum)));
  *failures = 0;
  *sum      = 0;

  PushValues<<<blocks, threads>>>(queue.get(), numValues, failures);
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(DeviceSynchronize)());
  EXPECT_EQ(0u, *failures);
  EXPECT_EQ(numValues, static_cast<unsigned int>(queue.get()->size()));

  PopAndSum<<<blocks, threads>>>(queue.get(), numValues, sum, failures);
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(DeviceSynchronize)());

  const auto expectedSum = static_cast<unsigned long long>(numValues) * (numValues - 1) / 2;
  EXPECT_EQ(0u, *failures);
  EXPECT_EQ(expectedSum, *sum);
  EXPECT_EQ(0, queue.get()->size());

  EXPECT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(Free)(sum));
  EXPECT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(Free)(failures));
}

} // namespace
