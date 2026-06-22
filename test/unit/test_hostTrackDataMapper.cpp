// SPDX-FileCopyrightText: 2025 CERN
// SPDX-License-Identifier: Apache-2.0

#include "AdePT/g4integration/returned_steps/HostTrackDataMapper.hh"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>

TEST(HostTrackDataMapper, BeginEventClearsOnlyWhenEventChanges)
{
  HostTrackDataMapper mapper;

  // A new event clears old state and prepares storage. Entries created in the
  // event must remain addressable through both GPU-id and G4-id lookups.
  mapper.beginEvent(/*eventID*/ 1, /*expectedTracks*/ 8);
  auto &first = mapper.create(/*gpuId*/ 100u, /*useNewId=*/false);
  EXPECT_EQ(first.gpuId, 100u);
  EXPECT_EQ(first.g4id, 100);
  EXPECT_EQ(&first, &mapper.get(100u));
  EXPECT_TRUE(mapper.contains(100u));

  uint64_t gotGpu{};
  EXPECT_TRUE(mapper.getGPUId(/*g4id*/ 100, gotGpu));
  EXPECT_EQ(gotGpu, 100u);

  // Re-entering beginEvent() with the same event ID is intentionally a no-op:
  // ProcessTrack can call it repeatedly inside one event without losing mapper state.
  mapper.beginEvent(1);
  EXPECT_TRUE(mapper.contains(100u));
  EXPECT_TRUE(mapper.getGPUId(100, gotGpu));
  EXPECT_EQ(gotGpu, 100u);

  // A different event ID starts a fresh event. Old live entries and reverse-map
  // entries must disappear, and missing G4 IDs fall back to gpuId == g4id.
  mapper.beginEvent(2);
  EXPECT_FALSE(mapper.contains(100u));
  EXPECT_FALSE(mapper.getGPUId(100, gotGpu));
  EXPECT_EQ(gotGpu, 100u);

  // GPU-born tracks use descending synthetic G4 IDs. The counter must reset at
  // event boundaries so runs remain reproducible event by event.
  auto &a = mapper.create(/*gpuId*/ 1u, /*useNewId=*/true);
  auto &b = mapper.create(/*gpuId*/ 2u, /*useNewId=*/true);
  EXPECT_EQ(a.g4id, std::numeric_limits<int>::max());
  EXPECT_EQ(b.g4id, std::numeric_limits<int>::max() - 1);
}

TEST(HostTrackDataMapper, DefaultBeginEventUsesSmallRetainedCapacity)
{
  HostTrackDataMapper mapper;

  // The default reserve should cover ordinary events without keeping the old
  // million-track buffer alive in every worker thread.
  mapper.beginEvent(1);

  EXPECT_GE(mapper.hostDataCapacity(), HostTrackDataMapper::kDefaultExpectedTracks);
  EXPECT_LE(mapper.hostDataCapacity(), HostTrackDataMapper::kDefaultMaxRetainedTrackCapacity);
  EXPECT_GE(mapper.gpuToIndexRetainedCapacity(), HostTrackDataMapper::kDefaultExpectedTracks);
  EXPECT_LE(mapper.gpuToIndexRetainedCapacity(), HostTrackDataMapper::kDefaultMaxRetainedTrackCapacity);
  EXPECT_GE(mapper.g4idToGpuIdRetainedCapacity(), HostTrackDataMapper::kDefaultExpectedTracks);
  EXPECT_LE(mapper.g4idToGpuIdRetainedCapacity(), HostTrackDataMapper::kDefaultMaxRetainedTrackCapacity);
}

TEST(HostTrackDataMapper, LargeEventCapacityIsReleasedBeforeNextDefaultEvent)
{
  HostTrackDataMapper mapper;
  constexpr size_t largeEventTracks = HostTrackDataMapper::kDefaultMaxRetainedTrackCapacity + 1;

  // Rare large events are allowed to grow beyond the default retained capacity.
  // This keeps correctness for pathological showers; the reserve is not a hard cap.
  mapper.beginEvent(1, largeEventTracks);
  EXPECT_GE(mapper.hostDataCapacity(), largeEventTracks);
  EXPECT_GE(mapper.gpuToIndexRetainedCapacity(), largeEventTracks);
  EXPECT_GE(mapper.g4idToGpuIdRetainedCapacity(), largeEventTracks);

  // The next default-sized event should release that oversized storage before
  // reserving the normal capacity again, reducing long-lived worker RSS.
  mapper.beginEvent(2);
  EXPECT_LE(mapper.hostDataCapacity(), HostTrackDataMapper::kDefaultMaxRetainedTrackCapacity);
  EXPECT_LE(mapper.gpuToIndexRetainedCapacity(), HostTrackDataMapper::kDefaultMaxRetainedTrackCapacity);
  EXPECT_LE(mapper.g4idToGpuIdRetainedCapacity(), HostTrackDataMapper::kDefaultMaxRetainedTrackCapacity);

  // After shrinking, the mapper must still be fully usable for new entries.
  auto &data = mapper.create(/*gpuId*/ 7u, /*useNewId=*/false);
  EXPECT_EQ(data.gpuId, 7u);
  EXPECT_EQ(data.g4id, 7);
  EXPECT_TRUE(mapper.contains(7u));
}

TEST(HostTrackDataMapper, CreateAndLookupReturnStableSlot)
{
  HostTrackDataMapper mapper;
  mapper.beginEvent(1, 8);

  // create(..., false) represents a CPU-born track entering the GPU: its GPU ID
  // is the original G4 track ID, and both maps should point back to one slot.
  auto &data = mapper.create(/*gpuId*/ 42u, /*useNewId=*/false);
  EXPECT_EQ(data.gpuId, 42u);
  EXPECT_EQ(data.g4id, 42);
  EXPECT_TRUE(mapper.contains(42u));

  uint64_t gotGpu{};
  EXPECT_TRUE(mapper.getGPUId(42, gotGpu));
  EXPECT_EQ(gotGpu, 42u);

  // No extra insertion happened, so get() must return the exact same vector slot.
  auto &same = mapper.get(42u);
  EXPECT_EQ(&data, &same);
}

TEST(HostTrackDataMapper, RemoveTrackSwapErasesLiveSlotAndReverseMap)
{
  HostTrackDataMapper mapper;
  mapper.beginEvent(1, 8);

  auto &first  = mapper.create(1u, false);
  auto &second = mapper.create(2u, false);
  EXPECT_TRUE(mapper.contains(1u));
  EXPECT_TRUE(mapper.contains(2u));
  EXPECT_EQ(first.g4id, 1);
  EXPECT_EQ(second.g4id, 2);

  // Removing a non-last vector element uses swap-erase. The moved survivor must
  // keep a correct gpuId -> vector-slot entry after its index changes.
  mapper.removeTrack(1u);
  EXPECT_FALSE(mapper.contains(1u));
  EXPECT_TRUE(mapper.contains(2u));

  // removeTrack() means the track is done, so both the live slot and reverse map
  // are erased. A later query falls back to gpuId == g4id.
  uint64_t got{};
  EXPECT_FALSE(mapper.getGPUId(1, got));
  EXPECT_EQ(got, 1u);

  EXPECT_TRUE(mapper.getGPUId(2, got));
  EXPECT_EQ(got, 2u);
  auto &survivor = mapper.get(2u);
  EXPECT_EQ(survivor.g4id, 2);
  EXPECT_EQ(survivor.gpuId, 2u);

  // Removing an already-absent track is part of the cleanup contract and must be harmless.
  EXPECT_NO_THROW(mapper.removeTrack(123456u));
}

TEST(HostTrackDataMapper, RetireToCPUCanReactivateWithPreservedReverseMap)
{
  HostTrackDataMapper mapper;
  mapper.beginEvent(1, 8);

  // A CPU-born track that leaves the GPU temporarily loses its live HostTrackData
  // slot, but it keeps the reverse map so a future GPU handoff reuses the same ID.
  auto &data = mapper.create(/*gpuId*/ 10u, /*useNewId=*/false);
  EXPECT_EQ(data.g4id, 10);
  EXPECT_TRUE(mapper.contains(10u));

  mapper.retireToCPU(10u);
  EXPECT_FALSE(mapper.contains(10u));

  uint64_t gotGpu{};
  EXPECT_TRUE(mapper.getGPUId(/*g4id*/ 10, gotGpu));
  EXPECT_EQ(gotGpu, 10u);

  // Reactivation with an existing reverse-map entry must recreate the live slot
  // without changing the preserved ID relationship.
  auto &revived = mapper.activateForGPU(/*gpuId*/ 10u, /*g4id*/ 10, /*haveReverse*/ true);
  EXPECT_TRUE(mapper.contains(10u));
  EXPECT_EQ(revived.gpuId, 10u);
  EXPECT_EQ(revived.g4id, 10);

  // A second activation for an already-live track is a no-op and returns the same slot.
  auto &same = mapper.activateForGPU(10u, 10, true);
  EXPECT_EQ(&same, &revived);

  // A first activation with no reverse-map entry should create both the live slot
  // and the reverse mapping in one pass.
  mapper.beginEvent(2, 8);
  auto &created = mapper.activateForGPU(/*gpuId*/ 77u, /*g4id*/ 77, /*haveReverse*/ false);
  EXPECT_EQ(created.gpuId, 77u);
  EXPECT_EQ(created.g4id, 77);
  EXPECT_TRUE(mapper.contains(77u));
  EXPECT_TRUE(mapper.getGPUId(77, gotGpu));
  EXPECT_EQ(gotGpu, 77u);
}

TEST(HostTrackDataMapper, GPUBornTrackReactivationPreservesSyntheticG4Id)
{
  HostTrackDataMapper mapper;
  mapper.beginEvent(1, 8);

  // GPU-born secondaries get synthetic decreasing G4 IDs. If they later return to
  // CPU and are handed back to the GPU, reproducibility requires preserving that ID.
  auto &born           = mapper.create(/*gpuId*/ 555u, /*useNewId=*/true);
  const int originalG4 = born.g4id;
  EXPECT_EQ(originalG4, std::numeric_limits<int>::max());

  mapper.retireToCPU(555u);
  EXPECT_FALSE(mapper.contains(555u));

  auto &revived = mapper.activateForGPU(/*gpuId*/ 555u, /*g4id*/ originalG4, /*haveReverse*/ true);
  EXPECT_EQ(revived.g4id, originalG4);
  EXPECT_EQ(revived.gpuId, 555u);
  EXPECT_TRUE(mapper.contains(555u));

  uint64_t gotGpu{};
  EXPECT_TRUE(mapper.getGPUId(originalG4, gotGpu));
  EXPECT_EQ(gotGpu, 555u);
}

TEST(HostTrackDataMapper, GetGPUIdFallsBackToG4IdForUnknownTracks)
{
  HostTrackDataMapper mapper;
  mapper.beginEvent(1, 8);

  // Unknown G4 IDs are CPU-born by convention, so getGPUId() returns false while
  // still giving the caller gpuId == g4id as the handoff ID to use.
  uint64_t gpuId{};
  EXPECT_FALSE(mapper.getGPUId(42, gpuId));
  EXPECT_EQ(gpuId, 42u);

  // Once the track is created, the same lookup becomes an existing reverse-map hit.
  mapper.create(42u, false);
  EXPECT_TRUE(mapper.contains(42u));
  EXPECT_TRUE(mapper.getGPUId(42, gpuId));
  EXPECT_EQ(gpuId, 42u);
}
