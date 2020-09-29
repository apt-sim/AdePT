// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_track_block.cu
 * @brief Unit test for the BlockData concurrent container.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#include <iostream>
#include <cassert>
#include <AdePT/BlockData.h>

struct MyTrack {
  int index{0};
  double pos[3]{0};
  double dir[3]{0};
  bool flag1;
  bool flag2;
};

// Kernel function to process the next free track in a block
__global__ void testTrackBlock(adept::BlockData<MyTrack> *block)
{
  auto track = block->NextElement();
  if (!track) return;
  int id       = blockIdx.x * blockDim.x + threadIdx.x;
  track->index = id;
}

// Kernel function to process the next free track in a block
__global__ void releaseTrack(adept::BlockData<MyTrack> *block)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  block->ReleaseElement(id);
}

///______________________________________________________________________________________
int main(void)
{
  using Block_t         = adept::BlockData<MyTrack>;
  const char *result[2] = {"FAILED", "OK"};
  // Track capacity of the block
  constexpr int capacity = 1 << 20;

  // Define the kernels granularity: 10K blocks of 32 treads each
  constexpr dim3 nblocks(10000), nthreads(32);

  // Allocate a block of tracks with capacity larger than the total number of spawned threads
  // Note that if we want to allocate several consecutive block in a buffer, we have to use
  // Block_t::SizeOfAlignAware rather than SizeOfInstance to get the space needed per block
  size_t blocksize = Block_t::SizeOfInstance(capacity);
  char *buffer     = nullptr;
  cudaMallocManaged(&buffer, blocksize);
  auto block = adept::BlockData<MyTrack>::MakeInstanceAt(capacity, buffer);

  bool testOK = true;
  std::cout << "   test_track_block ... ";
  // Allow memory to reach the device
  cudaDeviceSynchronize();
  // Launch a kernel processing tracks
  testTrackBlock<<<nblocks, nthreads>>>(block);
  // Allow all warps to finish
  cudaDeviceSynchronize();
  // The number of used tracks should be equal to the number of spawned threads
  testOK &= block->GetNused() == nblocks.x * nthreads.x;

  // Compute the sum of assigned track indices, which has to match the sum from 0 to nthreads-1
  // (the execution order is arbitrary, but all thread indices must be distributed)
  unsigned long long counter1 = 0, counter2 = 0;

  for (auto i = 0; i < nblocks.x * nthreads.x; ++i) {
    counter1 += i;
    counter2 += (*block)[i].index;
  }

  testOK &= counter1 == counter2;

  // Now release 32K tracks
  releaseTrack<<<1000, 32>>>(block);
  cudaDeviceSynchronize();
  testOK &= block->GetNused() == nblocks.x * nthreads.x - 32000;
  testOK &= block->GetNholes() == 32000;
  // Now allocate in the holes
  testTrackBlock<<<10, 32>>>(block);
  cudaDeviceSynchronize();
  testOK &= block->GetNholes() == (32000 - 320);
  std::cout << result[testOK] << "\n";
  cudaFree(buffer);
  if (!testOK) return 1;
  return 0;
}
