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
  AsyncAdePT::HostCircularBuffer buffer(8);

  CHECK(buffer.GetWriteOffset() == 0);
  CHECK(buffer.GetFreeContiguousSlots(8) == 8);
  CHECK(buffer.GetWriteOffset() == 0);
  CHECK(near(buffer.GetFillFractionAfterLastRequest(), 0.0));

  return 0;
}

static int test_segment_add_remove_and_full_buffer()
{
  AsyncAdePT::HostCircularBuffer buffer(10);

  CHECK(buffer.GetFreeContiguousSlots(10) == 10);
  CHECK(buffer.AddSegment(0, 10));
  CHECK(buffer.GetWriteOffset() == 10);

  CHECK(buffer.GetFreeContiguousSlots(1) == 0);

  buffer.RemoveSegment(0);
  CHECK(buffer.GetFreeContiguousSlots(10) == 10);
  CHECK(buffer.GetWriteOffset() == 0);

  return 0;
}

static int test_fill_fraction_preserves_existing_pending_transfer_semantics()
{
  AsyncAdePT::HostCircularBuffer buffer(10);

  CHECK(buffer.GetFreeContiguousSlots(4) == 10);
  CHECK(buffer.AddSegment(0, 4));

  // This intentionally documents the existing behavior: GetFillFractionAfterLastRequest() is based on the
  // contiguous-space bookkeeping after a transfer-size query. It is not simply occupiedSlots / capacity.
  CHECK(buffer.GetFreeContiguousSlots(3) == 6);
  CHECK(near(buffer.GetFillFractionAfterLastRequest(), 0.7));

  return 0;
}

static int test_wraparound_after_front_segment_is_removed()
{
  AsyncAdePT::HostCircularBuffer buffer(10);

  CHECK(buffer.GetFreeContiguousSlots(6) == 10);
  CHECK(buffer.AddSegment(0, 6));

  CHECK(buffer.GetFreeContiguousSlots(4) == 4);
  CHECK(buffer.GetWriteOffset() == 6);
  CHECK(buffer.AddSegment(6, 10));

  buffer.RemoveSegment(0);
  CHECK(buffer.GetFreeContiguousSlots(6) == 6);
  CHECK(buffer.GetWriteOffset() == 0);
  CHECK(near(buffer.GetFillFractionAfterLastRequest(), 1.0));

  return 0;
}

static int test_tail_space_is_reused_after_tail_segment_is_removed()
{
  AsyncAdePT::HostCircularBuffer buffer(12);

  CHECK(buffer.GetFreeContiguousSlots(4) == 12);
  CHECK(buffer.AddSegment(0, 4));
  CHECK(buffer.GetFreeContiguousSlots(4) == 8);
  CHECK(buffer.AddSegment(4, 8));
  CHECK(buffer.GetFreeContiguousSlots(4) == 4);
  CHECK(buffer.AddSegment(8, 12));

  buffer.RemoveSegment(8);
  CHECK(buffer.GetFreeContiguousSlots(4) == 4);
  CHECK(buffer.GetWriteOffset() == 8);

  return 0;
}

static int test_middle_hole_is_not_reused_from_end_write_position()
{
  AsyncAdePT::HostCircularBuffer buffer(12);

  CHECK(buffer.GetFreeContiguousSlots(4) == 12);
  CHECK(buffer.AddSegment(0, 4));
  CHECK(buffer.GetFreeContiguousSlots(4) == 8);
  CHECK(buffer.AddSegment(4, 8));
  CHECK(buffer.GetFreeContiguousSlots(4) == 4);
  CHECK(buffer.AddSegment(8, 12));

  buffer.RemoveSegment(4);
  CHECK(buffer.GetFreeContiguousSlots(4) == 0);
  CHECK(buffer.GetWriteOffset() == 12);

  return 0;
}

static int test_fragmentation_can_block_larger_transfers()
{
  AsyncAdePT::HostCircularBuffer buffer(12);

  CHECK(buffer.GetFreeContiguousSlots(4) == 12);
  CHECK(buffer.AddSegment(0, 4));
  CHECK(buffer.GetFreeContiguousSlots(4) == 8);
  CHECK(buffer.AddSegment(4, 8));
  CHECK(buffer.GetFreeContiguousSlots(4) == 4);
  CHECK(buffer.AddSegment(8, 12));

  buffer.RemoveSegment(0);
  buffer.RemoveSegment(8);
  CHECK(buffer.GetFreeContiguousSlots(5) == 0);
  CHECK(buffer.GetFreeContiguousSlots(4) == 4);
  CHECK(buffer.GetWriteOffset() == 8);

  return 0;
}

static int test_concurrent_allocation_and_release()
{
  AsyncAdePT::HostCircularBuffer buffer(64);

  constexpr int nSegments = 10000;
  constexpr int nWorkers  = 4;

  std::atomic_bool failed{false};
  std::atomic_int removedSegments{0};
  std::atomic_int producedSegments{0};
  std::deque<std::size_t> offsetsToRelease;
  std::mutex queueMutex;
  std::condition_variable queueCondition;
  bool producerDone = false;

  auto worker = [&]() {
    while (true) {
      std::size_t offset = 0;
      {
        std::unique_lock lock{queueMutex};
        queueCondition.wait(lock, [&]() { return producerDone || !offsetsToRelease.empty(); });
        if (offsetsToRelease.empty()) {
          if (producerDone) return;
          continue;
        }
        offset = offsetsToRelease.front();
        offsetsToRelease.pop_front();
      }

      buffer.RemoveSegment(offset);
      removedSegments.fetch_add(1, std::memory_order_relaxed);
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(nWorkers);
  for (int i = 0; i < nWorkers; ++i) {
    workers.emplace_back(worker);
  }

  for (int i = 0; i < nSegments; ++i) {
    while (buffer.GetFreeContiguousSlots(1) == 0) {
      std::this_thread::yield();
    }

    const auto offset = buffer.GetWriteOffset();
    if (!buffer.AddSegment(offset, offset + 1)) {
      failed.store(true, std::memory_order_relaxed);
      break;
    }
    producedSegments.fetch_add(1, std::memory_order_relaxed);

    {
      std::scoped_lock lock{queueMutex};
      offsetsToRelease.push_back(offset);
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
  CHECK(buffer.GetFreeContiguousSlots(64) == 64);
  CHECK(buffer.GetWriteOffset() == 0);

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
