// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sim_kernels.h"

///______________________________________________________________________________________
template <copcore::BackendType backend>
int runSimulation()
{
  // Track capacity of the block
  constexpr int capacity = 1 << 24;

  using TrackBlock     = adept::BlockData<MyTrack>;
  using TrackAllocator = copcore::VariableSizeObjAllocator<TrackBlock, backend>;
  using HitBlock       = adept::BlockData<MyHit>;
  using HitAllocator   = copcore::VariableSizeObjAllocator<HitBlock, backend>;
  using Array_t        = adept::MParray;
  using ArrayAllocator = copcore::VariableSizeObjAllocator<Array_t, backend>;
  using StreamStruct   = copcore::StreamType<backend>;
  using Stream_t       = typename StreamStruct::value_type;
  using Launcher_t     = copcore::Launcher<backend>;

  // Boilerplate to get the pointers to the device functions to be used
  COPCORE_CALLABLE_DECLARE(generateFunc, generateAndStorePrimary);
  COPCORE_CALLABLE_DECLARE(elossFunc, elossTrack);
  COPCORE_CALLABLE_IN_NAMESPACE_DECLARE(selectTrackFunc, devfunc, selectTrack);

  std::cout << "Executing simulation on " << copcore::BackendName(backend) << "\n";
  //  const char *result[2] = {"FAILED", "OK"};

  //  bool testOK  = true;
  bool success = true;

  copcore::Allocator<MyTrack, backend> trackAlloc;
  MyTrack *tr = trackAlloc.allocate(10, 100.); // allocate array of 10 tracks with energy = 100 GeV
  assert(tr[9].energy == 100.);

  copcore::Allocator<int, backend> intAlloc;
  int *int_array = intAlloc.allocate(32, 0); // array of 32 integers, initialized to 0

  using Atomic_int = adept::Atomic_t<int>;
  copcore::Allocator<Atomic_int, backend> atomicAllocator;
  Atomic_int *at_index = atomicAllocator.allocate(1); // an atomic integer, initialized by its default ctor with 0

  // Boilerplate to allocate the data structures that we need
  TrackAllocator trackBlockAlloc(capacity);
  TrackBlock *blockT = trackBlockAlloc.allocate(1);

  HitAllocator hitAlloc(1024);
  HitBlock *blockH = hitAlloc.allocate(1);

  ArrayAllocator arrayAlloc(capacity);
  Array_t *selection1 = arrayAlloc.allocate(1);

  // Create a stream to work with. On the CPU backend, this will be equivalent with: int stream = 0;
  Stream_t stream;
  StreamStruct::CreateStream(stream);

  // A launcher that runs a lambda function that fills an array with the current thread number
  Launcher_t fillArray(stream);
  fillArray.Run([] __device__ (int thread_id, Atomic_int *index,
                                      int *array) { array[thread_id] = (*index)++; }, // lambda being run
                32,                                                                   // number of elements
                {2, 16},              // run with 2 block of 16 threads (if backend=CUDA)
                at_index, int_array); // parameters passed to the lambda (thread_id is automatic)
  fillArray.WaitStream();
  std::cout << "Filled array: {" << int_array[0];
  for (auto i = 1; i < 32; ++i)
    std::cout << ", " << int_array[i];
  std::cout << "}\n";

  // Allocate some tracks in parallel
  Launcher_t generate(stream);
  generate.Run(generateFunc, capacity, {0, 0}, blockT);

  // Synchronize stream if we need memory to reach the device
  generate.WaitStream();

  std::cout << "Generated " << blockT->GetNused() << " tracks\n";

  Launcher_t selector(stream);
  // This will select each 2'nd track from a container (see selectTrack impl)
  selector.Run(selectTrackFunc, blockT->GetNused(), {0, 0}, blockT, 2, selection1);

  selector.WaitStream();

  std::cout << "Selected " << selection1->size() << " tracks\n";

  Launcher_t process_tracks(stream);
  process_tracks.Run(elossFunc, selection1->size(), {1000, 32}, selection1, blockT, blockH);

  process_tracks.WaitStream();

  // Sum up total energy loss (on host)
  float sum_eloss = 0.;
  for (auto i = 0; i < 1024; ++i) {
    auto &hit = (*blockH)[i];
    sum_eloss += hit.edep.load();
  }

  std::cout << "Total eloss computed on host: " << sum_eloss << "\n";

  Launcher_t::WaitDevice();

  trackAlloc.deallocate(tr, 10);  // Will call the destructor for all 10 elements.
  intAlloc.deallocate(int_array); // no destructor called
  atomicAllocator.deallocate(at_index);
  trackBlockAlloc.deallocate(blockT, 1);
  hitAlloc.deallocate(blockH, 1);
  arrayAlloc.deallocate(selection1, 1);

  if (!success) return 1;
  return 0;
}
