// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sim_kernels.h"

///______________________________________________________________________________________
template <copcore::BackendType backend>
int runSimulation()
{
  std::cout << "Executing simulation on " << copcore::BackendName<backend>::name << "\n";

  using TrackBlock     = adept::BlockData<MyTrack>;
  using TrackAllocator = copcore::VariableSizeObjAllocator<TrackBlock, backend>;
  using HitBlock       = adept::BlockData<MyHit>;
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

  // Boilerplate to allocate the data structures that we need
  TrackAllocator trackAlloc(capacity);
  auto blockT = trackAlloc.allocate(1);

  HitAllocator hitAlloc(1024);
  auto blockH = hitAlloc.allocate(1);

  QueueAllocator queueAlloc(capacity);
  auto queue = queueAlloc.allocate(1);

  // Create a stream to work with. On the CPU backend, this will be equivalent with: int stream = 0;
  Stream_t stream;
  StreamStruct::CreateStream(stream);

  // Allocate some tracks in parallel
  copcore::Launcher<backend> generate(stream);
  generate.Run(generateFunc, capacity, {0, 0}, blockT);

  // Synchronize stream if we need memory to reach the device
  generate.Wait();

  std::cout << "Generated " << blockT->GetNused() << " tracks\n";

  copcore::Launcher<backend> selector(stream);
  selector.Run(selectTrackFunc, blockT->GetNused(), {0, 0}, blockT, queue);

  selector.Wait();

  std::cout << "Selected " << queue->size() << " tracks\n";

  trackAlloc.deallocate(blockT, 1);
  hitAlloc.deallocate(blockH, 1);
  queueAlloc.deallocate(queue, 1);

  if (!success) return 1;
  return 0;
}
