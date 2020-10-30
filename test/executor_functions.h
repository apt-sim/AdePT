// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <cassert>
#include <AdePT/BlockData.h>
inline namespace COPCORE_IMPL {

struct MyTrack {
  int index{0};
  float energy{0};
};

struct Hit {
  float edep{0};
};

/** @brief Generate a number of primaries */
VECCORE_ATT_DEVICE
void generateAndStorePrimary(int id, adept::BlockData<MyTrack> *tracks)
{
  auto track = tracks->NextElement();
  if (!track) COPCORE_EXCEPTION("generateAndStorePrimary: Not enough space for tracks");

  track->index  = id;
  track->energy = 100.;
}

COPCORE_CALLABLE_FUNC(generateAndStorePrimary)

namespace devfunc {
VECCORE_ATT_DEVICE
void selectTrack(int id, adept::BlockData<MyTrack> *tracks, adept::mpmc_bounded_queue<int> *queue)
{
  auto track    = (*tracks)[id];
  bool selected = (track.index % 2 == 0);
  if (selected) queue->enqueue(id);
}
COPCORE_CALLABLE_FUNC(selectTrack)
} // end namespace devfunc

VECCORE_ATT_DEVICE
void processTrack(MyTrack const &track, Hit &hit)
{
  hit.edep += 0.1 * track.energy;
}

VECCORE_ATT_DEVICE
void updateTrack(MyTrack &track)
{
  track.energy -= 0.1 * track.energy;
}

///______________________________________________________________________________________
template <copcore::BackendType backend>
int simplePipeline()
{
  std::cout << "Executing pipeline on " << copcore::BackendName<backend>::name << "\n";

  using TrackBlock     = adept::BlockData<MyTrack>;
  using TrackAllocator = copcore::VariableSizeObjAllocator<TrackBlock, backend>;
  using HitBlock       = adept::BlockData<Hit>;
  using HitAllocator   = copcore::VariableSizeObjAllocator<HitBlock, backend>;
  using Queue_t        = adept::mpmc_bounded_queue<int>;
  using QueueAllocator = copcore::VariableSizeObjAllocator<Queue_t, backend>;
  using StreamStruct   = copcore::StreamType<backend>;
  using Stream_t       = typename StreamStruct::value_type;

  // Boilerplate to get the pointers to the device functions to be used
  COPCORE_CALLABLE_DECLARE(generateFunc, generateAndStorePrimary);
  COPCORE_CALLABLE_IN_NAMESPACE_DECLARE(selectTrackFunc, devfunc, selectTrack);

  //  const char *result[2] = {"FAILED", "OK"};
  // Track capacity of the block
  constexpr int capacity = 1 << 20;

  //  bool testOK  = true;
  bool success = true;

  // Boilerplate to create the data structures that we need
  TrackAllocator trackAlloc(capacity);
  auto blockT = trackAlloc.allocate(1);

  HitAllocator hitAlloc(1024);
  auto blockH = hitAlloc.allocate(1);

  QueueAllocator queueAlloc(capacity);
  auto queue = queueAlloc.allocate(1);

  // Create a stream to work with
  Stream_t stream;
  StreamStruct::CreateStream(stream);

  // Allocate some tracks in parallel
  copcore::Executor<backend> generate(stream);
  generate.Launch(generateFunc, capacity, {0, 0}, blockT);

  // Allow memory to reach the device
  generate.Wait();

  std::cout << "Generated " << blockT->GetNused() << " tracks\n";

  copcore::Executor<backend> selector(stream);
  selector.Launch(selectTrackFunc, blockT->GetNused(), {0, 0}, blockT, queue);

  selector.Wait();
  std::cout << "Selected " << queue->size() << " tracks\n";

  // Allow all warps to finish
  // COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  trackAlloc.deallocate(blockT, 1);
  hitAlloc.deallocate(blockH, 1);
  queueAlloc.deallocate(queue, 1);

  if (!success) return 1;
  return 0;
}

} // End namespace COPCORE_IMPL
