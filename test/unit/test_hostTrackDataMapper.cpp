// SPDX-FileCopyrightText: 2025 CERN
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <limits>
#include <vector>

#include "AdePT/integration/HostTrackDataMapper.hh"

#define CHECK(expr)                                                                         \
  do {                                                                                      \
    if (!(expr)) {                                                                          \
      std::cerr << "CHECK failed: " #expr << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
      return 1;                                                                             \
    }                                                                                       \
  } while (0)

static int test_beginEvent_clear_and_reserve()
{
  HostTrackDataMapper m;
  // Event 1: create something
  m.beginEvent(/*eventID*/ 1, /*expectedTracks*/ 8);
  auto &d1 = m.create(/*gpuId*/ 100u, /*useNewId=*/false); // g4id = 100
  CHECK(d1.gpuId == 100u);
  CHECK(d1.g4id == 100);
  CHECK(&d1 == &m.get(100u)); // alias check
  CHECK(m.contains(100u));
  uint64_t gotGpu{};
  CHECK(m.getGPUId(/*g4id*/ 100, gotGpu) == true);
  CHECK(gotGpu == 100u);

  // Calling beginEvent with same ID must NOT clear
  m.beginEvent(1);
  CHECK(m.contains(100u));
  CHECK(m.getGPUId(100, gotGpu) == true && gotGpu == 100u);

  // New event ID must clear everything and reset counters
  m.beginEvent(2);
  CHECK(!m.contains(100u));
  CHECK(m.getGPUId(100, gotGpu) == false); // returns gpuId=100 by convention
  CHECK(gotGpu == 100u);

  // Verify currentGpuReturnG4ID reset behavior by creating two with useNewId=true
  auto &a = m.create(/*gpuId*/ 1u, /*useNewId=*/true);
  auto &b = m.create(/*gpuId*/ 2u, /*useNewId=*/true);
  CHECK(a.g4id == std::numeric_limits<int>::max());
  CHECK(b.g4id == std::numeric_limits<int>::max() - 1);

  return 0;
}

static int test_create_and_lookup()
{
  HostTrackDataMapper m;
  m.beginEvent(1, 8);

  auto &d = m.create(/*gpuId*/ 42u, /*useNewId=*/false); // g4id = 42
  CHECK(d.gpuId == 42u);
  CHECK(d.g4id == 42);
  CHECK(m.contains(42u));

  uint64_t gotGpu{};
  CHECK(m.getGPUId(42, gotGpu) == true);
  CHECK(gotGpu == 42u);

  // get() must return same slot
  auto &d2 = m.get(42u);
  CHECK(&d == &d2);

  return 0;
}

static int test_removeTrack_swap_erase()
{
  HostTrackDataMapper m;
  m.beginEvent(1, 8);

  auto &d1 = m.create(1u, false); // g4id=1
  auto &d2 = m.create(2u, false); // g4id=2
  CHECK(m.contains(1u) && m.contains(2u));
  CHECK(d1.g4id == 1 && d2.g4id == 2);

  // Remove front (1u), should swap-erase with last (2u)
  m.removeTrack(1u);
  CHECK(!m.contains(1u));
  CHECK(m.contains(2u));

  // Reverse map should have erased g4id=1; getting it should return false
  uint64_t got{};
  CHECK(m.getGPUId(1, got) == false);
  CHECK(got == 1u);

  // Still consistent for the survivor
  CHECK(m.getGPUId(2, got) == true && got == 2u);
  auto &d2_again = m.get(2u);
  CHECK(d2_again.g4id == 2 && d2_again.gpuId == 2u);

  // Removing non-existent should be safe (no crash)
  m.removeTrack(123456u);

  return 0;
}

static int test_retire_reactivate_preserve_reverse()
{
  HostTrackDataMapper m;
  m.beginEvent(1, 8);

  // Case A: CPU-born track (useNewId=false => g4id == gpuId)
  auto &d = m.create(/*gpuId*/ 10u, /*useNewId=*/false);
  CHECK(d.g4id == 10);
  CHECK(m.contains(10u));

  // Retire: removes slot, keeps reverse map
  m.retireToCPU(10u);
  CHECK(!m.contains(10u));

  uint64_t gotGpu{};
  CHECK(m.getGPUId(/*g4id*/ 10, gotGpu) == true);
  CHECK(gotGpu == 10u);

  // Reactivate with haveReverse=true; must not change mapping; must re-create slot
  auto &revived = m.activateForGPU(/*gpuId*/ 10u, /*g4id*/ 10, /*haveReverse*/ true);
  CHECK(m.contains(10u));
  CHECK(revived.gpuId == 10u && revived.g4id == 10);

  // Double-activate should be a no-op (same address, no growth)
  auto &same = m.activateForGPU(10u, 10, true);
  CHECK(&same == &revived);

  // Case B: First activation via activateForGPU with haveReverse=false
  m.beginEvent(2, 8);
  auto &x = m.activateForGPU(/*gpuId*/ 77u, /*g4id*/ 77, /*haveReverse*/ false);
  CHECK(x.gpuId == 77u && x.g4id == 77);
  CHECK(m.contains(77u));
  CHECK(m.getGPUId(77, gotGpu) == true && gotGpu == 77u);

  return 0;
}

static int test_gpu_born_reactivate_preserve_g4id()
{
  HostTrackDataMapper m;
  m.beginEvent(1, 8);

  // GPU-born: useNewId=true assigns a special decreasing g4id (not equal to gpuId)
  auto &born           = m.create(/*gpuId*/ 555u, /*useNewId=*/true);
  const int originalG4 = born.g4id;
  CHECK(originalG4 == std::numeric_limits<int>::max());

  // Retire slot (keep reverse map)
  m.retireToCPU(555u);
  CHECK(!m.contains(555u));

  // Reactivate with the SAME g4id (this is how CPU would call it)
  auto &rev = m.activateForGPU(/*gpuId*/ 555u, /*g4id*/ originalG4, /*haveReverse*/ true);
  CHECK(rev.g4id == originalG4);
  CHECK(rev.gpuId == 555u);
  CHECK(m.contains(555u));

  // Reverse map still points from originalG4 -> 555
  uint64_t gotGpu{};
  CHECK(m.getGPUId(originalG4, gotGpu) == true && gotGpu == 555u);

  return 0;
}

static int test_contains_and_getGPUId_contract()
{
  HostTrackDataMapper m;
  m.beginEvent(1, 8);

  // Non-existent query
  uint64_t gp{};
  CHECK(m.getGPUId(42, gp) == false);
  CHECK(gp == 42u); // contract: default gpuId returned equals g4id

  // After create
  m.create(42u, false);
  CHECK(m.contains(42u));
  CHECK(m.getGPUId(42, gp) == true && gp == 42u);

  return 0;
}

int main()
{
  if (int r = test_beginEvent_clear_and_reserve()) return r;
  if (int r = test_create_and_lookup()) return r;
  if (int r = test_removeTrack_swap_erase()) return r;
  if (int r = test_retire_reactivate_preserve_reverse()) return r;
  if (int r = test_gpu_born_reactivate_preserve_g4id()) return r;
  if (int r = test_contains_and_getGPUId_contract()) return r;

  std::cout << "All HostTrackDataMapper tests passed.\n";
  return 0;
}
