// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/core/HostCircularBuffer.hh>

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#define CHECK(expr)                                                                         \
  do {                                                                                      \
    if (!(expr)) {                                                                          \
      std::cerr << "CHECK failed: " #expr << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
      return 1;                                                                             \
    }                                                                                       \
  } while (0)

static bool near(double lhs, double rhs)
{
  return std::abs(lhs - rhs) < 1e-12;
}

static int test_empty_buffer_allows_exact_fit()
{
  std::vector<GPUStep> storage(8);
  AsyncAdePT::HostCircularBuffer buffer(storage.data(), storage.size());

  CHECK(buffer.getOffset() == 0);
  CHECK(buffer.getFreeContiguousMemory(8) == 8);
  CHECK(buffer.getOffset() == 0);
  CHECK(near(buffer.getFillFraction(), 0.0));

  return 0;
}

static int test_segment_add_remove_and_full_buffer()
{
  std::vector<GPUStep> storage(10);
  AsyncAdePT::HostCircularBuffer buffer(storage.data(), storage.size());

  CHECK(buffer.getFreeContiguousMemory(10) == 10);
  CHECK(buffer.addSegment(storage.data(), storage.data() + 10));
  CHECK(buffer.getOffset() == 10);

  CHECK(buffer.getFreeContiguousMemory(1) == 0);

  buffer.removeSegment(storage.data());
  CHECK(buffer.getFreeContiguousMemory(10) == 10);
  CHECK(buffer.getOffset() == 0);

  return 0;
}

static int test_fill_fraction_preserves_existing_pending_transfer_semantics()
{
  std::vector<GPUStep> storage(10);
  AsyncAdePT::HostCircularBuffer buffer(storage.data(), storage.size());

  CHECK(buffer.getFreeContiguousMemory(4) == 10);
  CHECK(buffer.addSegment(storage.data(), storage.data() + 4));

  // This intentionally documents the existing behavior: getFillFraction() is
  // based on the contiguous-space bookkeeping after a transfer-size query. It
  // is not simply occupiedSlots / capacity.
  CHECK(buffer.getFreeContiguousMemory(3) == 6);
  CHECK(near(buffer.getFillFraction(), 0.7));

  return 0;
}

static int test_wraparound_after_front_segment_is_removed()
{
  std::vector<GPUStep> storage(10);
  AsyncAdePT::HostCircularBuffer buffer(storage.data(), storage.size());

  CHECK(buffer.getFreeContiguousMemory(6) == 10);
  CHECK(buffer.addSegment(storage.data(), storage.data() + 6));

  CHECK(buffer.getFreeContiguousMemory(4) == 4);
  CHECK(buffer.getOffset() == 6);
  CHECK(buffer.addSegment(storage.data() + 6, storage.data() + 10));

  buffer.removeSegment(storage.data());
  CHECK(buffer.getFreeContiguousMemory(6) == 6);
  CHECK(buffer.getOffset() == 0);
  CHECK(near(buffer.getFillFraction(), 1.0));

  return 0;
}

static int test_tail_space_is_reused_after_tail_segment_is_removed()
{
  std::vector<GPUStep> storage(12);
  AsyncAdePT::HostCircularBuffer buffer(storage.data(), storage.size());

  CHECK(buffer.getFreeContiguousMemory(4) == 12);
  CHECK(buffer.addSegment(storage.data(), storage.data() + 4));
  CHECK(buffer.getFreeContiguousMemory(4) == 8);
  CHECK(buffer.addSegment(storage.data() + 4, storage.data() + 8));
  CHECK(buffer.getFreeContiguousMemory(4) == 4);
  CHECK(buffer.addSegment(storage.data() + 8, storage.data() + 12));

  buffer.removeSegment(storage.data() + 8);
  CHECK(buffer.getFreeContiguousMemory(4) == 4);
  CHECK(buffer.getOffset() == 8);

  return 0;
}

static int test_middle_hole_is_not_reused_from_end_write_position()
{
  std::vector<GPUStep> storage(12);
  AsyncAdePT::HostCircularBuffer buffer(storage.data(), storage.size());

  CHECK(buffer.getFreeContiguousMemory(4) == 12);
  CHECK(buffer.addSegment(storage.data(), storage.data() + 4));
  CHECK(buffer.getFreeContiguousMemory(4) == 8);
  CHECK(buffer.addSegment(storage.data() + 4, storage.data() + 8));
  CHECK(buffer.getFreeContiguousMemory(4) == 4);
  CHECK(buffer.addSegment(storage.data() + 8, storage.data() + 12));

  buffer.removeSegment(storage.data() + 4);
  CHECK(buffer.getFreeContiguousMemory(4) == 0);
  CHECK(buffer.getOffset() == 12);

  return 0;
}

static int test_fragmentation_can_block_larger_transfers()
{
  std::vector<GPUStep> storage(12);
  AsyncAdePT::HostCircularBuffer buffer(storage.data(), storage.size());

  CHECK(buffer.getFreeContiguousMemory(4) == 12);
  CHECK(buffer.addSegment(storage.data(), storage.data() + 4));
  CHECK(buffer.getFreeContiguousMemory(4) == 8);
  CHECK(buffer.addSegment(storage.data() + 4, storage.data() + 8));
  CHECK(buffer.getFreeContiguousMemory(4) == 4);
  CHECK(buffer.addSegment(storage.data() + 8, storage.data() + 12));

  buffer.removeSegment(storage.data());
  buffer.removeSegment(storage.data() + 8);
  CHECK(buffer.getFreeContiguousMemory(5) == 0);
  CHECK(buffer.getFreeContiguousMemory(4) == 4);
  CHECK(buffer.getOffset() == 8);

  return 0;
}

static int test_concurrent_allocation_and_release()
{
  std::vector<GPUStep> storage(64);
  AsyncAdePT::HostCircularBuffer buffer(storage.data(), storage.size());

  constexpr int nSegments = 10000;
  constexpr int nWorkers  = 4;

  std::atomic_bool failed{false};
  std::atomic_int removedSegments{0};
  std::atomic_int producedSegments{0};
  std::deque<GPUStep *> pointersToRelease;
  std::mutex queueMutex;
  std::condition_variable queueCondition;
  bool producerDone = false;

  auto worker = [&]() {
    while (true) {
      GPUStep *segmentBegin = nullptr;
      {
        std::unique_lock lock{queueMutex};
        queueCondition.wait(lock, [&]() { return producerDone || !pointersToRelease.empty(); });
        if (pointersToRelease.empty()) {
          if (producerDone) return;
          continue;
        }
        segmentBegin = pointersToRelease.front();
        pointersToRelease.pop_front();
      }

      buffer.removeSegment(segmentBegin);
      removedSegments.fetch_add(1, std::memory_order_relaxed);
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(nWorkers);
  for (int i = 0; i < nWorkers; ++i) {
    workers.emplace_back(worker);
  }

  for (int i = 0; i < nSegments; ++i) {
    while (buffer.getFreeContiguousMemory(1) == 0) {
      std::this_thread::yield();
    }

    const auto offset         = buffer.getOffset();
    GPUStep *segmentBegin     = storage.data() + offset;
    GPUStep *const segmentEnd = segmentBegin + 1;
    if (!buffer.addSegment(segmentBegin, segmentEnd)) {
      failed.store(true, std::memory_order_relaxed);
      break;
    }
    producedSegments.fetch_add(1, std::memory_order_relaxed);

    {
      std::scoped_lock lock{queueMutex};
      pointersToRelease.push_back(segmentBegin);
    }
    queueCondition.notify_one();
  }

  {
    std::scoped_lock lock{queueMutex};
    producerDone = true;
  }
  queueCondition.notify_all();

  for (auto &thread : workers) {
    thread.join();
  }

  CHECK(!failed.load(std::memory_order_relaxed));
  CHECK(producedSegments.load(std::memory_order_relaxed) == nSegments);
  CHECK(removedSegments.load(std::memory_order_relaxed) == nSegments);
  CHECK(buffer.getFreeContiguousMemory(storage.size()) == storage.size());
  CHECK(buffer.getOffset() == 0);

  return 0;
}

int main()
{
  if (int result = test_empty_buffer_allows_exact_fit()) return result;
  if (int result = test_segment_add_remove_and_full_buffer()) return result;
  if (int result = test_fill_fraction_preserves_existing_pending_transfer_semantics()) return result;
  if (int result = test_wraparound_after_front_segment_is_removed()) return result;
  if (int result = test_tail_space_is_reused_after_tail_segment_is_removed()) return result;
  if (int result = test_middle_hole_is_not_reused_from_end_write_position()) return result;
  if (int result = test_fragmentation_can_block_larger_transfers()) return result;
  if (int result = test_concurrent_allocation_and_release()) return result;

  std::cout << "All HostCircularBuffer tests passed.\n";
  return 0;
}
