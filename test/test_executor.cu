// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_track_executor.cu
 * @brief Unit test for the CUDA executor.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#include <iostream>
#include <cassert>
#include <AdePT/BlockData.h>
 
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
  if (!track)
    COPCORE_EXCEPTION("generateAndStorePrimary: Not enough space for tracks");
  
  track->index = id;
  track->energy = 100.;
}

COPCORE_CALLABLE_FUNC(generateAndStorePrimary)

namespace devfunc {
VECCORE_ATT_DEVICE
void selectTrack(int id, adept::BlockData<MyTrack> *tracks, adept::mpmc_bounded_queue<int> *queue)
{
  auto track = (*tracks)[id];
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
int test()
{
  //const auto backend   = copcore::BackendType::CPU;

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
  generate.Launch(generateFunc, capacity, {0,0},
                  blockT);

  // Allow memory to reach the device
  COPCORE_CUDA_CHECK(cudaStreamSynchronize(stream));

  std::cout << "Generated " << blockT->GetNused() <<" tracks\n";

  copcore::Executor<backend> selector(stream);
  selector.Launch(selectTrackFunc, blockT->GetNused(), {0,0},
                  blockT, queue);

  COPCORE_CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cout << "Selected " << queue->size() << " tracks\n";

  // Allow all warps to finish
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  trackAlloc.deallocate(blockT, 1);
  hitAlloc.deallocate(blockH, 1);
  queueAlloc.deallocate(queue, 1);

  if (!success) return 1;
  return 0;
}

///______________________________________________________________________________________
int main(void)
{
  int result;
  result = test<copcore::BackendType::CUDA>();
  //result += test<copcore::BackendType::CPU>();
  return result;
}