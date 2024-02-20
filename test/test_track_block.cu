// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_track_block.cu
 * @brief Unit test for the BlockData concurrent container.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#include <iostream>
#include <cassert>
#include <AdePT/base/BlockData.h>

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

  bool testOK  = true;
  bool success = true;

  // Test simple allocation/de-allocation on host
  std::cout << "   host allocation MakeInstance ... ";
  auto h_block = Block_t::MakeInstance(1024);
  testOK &= h_block != nullptr;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Test using the slots on the block (more than existing)
  std::cout << "   host NextElement             ... ";
  testOK           = true;
  size_t checksum1 = 0;
  for (auto i = 0; i < 2048; ++i) {
    auto track = h_block->NextElement();
    if (i >= 1024) testOK &= track == nullptr;
    // Assign current index to the current track
    if (track) {
      track->index = i;
      checksum1 += i;
    }
  }
  testOK = h_block->GetNused() == 1024;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Create another block into adopted memory on host
  testOK          = true;
  char *buff_host = new char[Block_t::SizeOfInstance(2048)];
  std::cout << "   host MakeCopyAt              ... ";
  // Test copying a block into another
  auto h_block2    = Block_t::MakeCopyAt(*h_block, buff_host);
  size_t checksum2 = 0;
  for (auto i = 0; i < 1024; ++i) {
    auto track = h_block2->NextElement();
    assert(track);
    checksum2 += track->index;
  }
  testOK = checksum1 == checksum2;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Release some elements end validate
  testOK = true;
  std::cout << "   host ReleaseElement          ... ";
  for (auto i = 0; i < 10; ++i)
    h_block2->ReleaseElement(i);
  testOK &= h_block2->GetNused() == (1024 - 10);
  testOK &= h_block2->GetNholes() == 10;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Release allocated blocks
  Block_t::ReleaseInstance(h_block);  // mandatory, frees memory for blocks allocated with MakeInstance
  Block_t::ReleaseInstance(h_block2); // will not do anything since block adopted memory
  delete[] buff_host;                 // Only this will actually free the memory

  // Create a large block on the device
  testOK = true;
  std::cout << "   host MakeInstanceAt          ... ";
  size_t blocksize = Block_t::SizeOfInstance(capacity);
  char *buffer     = nullptr;
  cudaMallocManaged(&buffer, blocksize);
  auto block = Block_t::MakeInstanceAt(capacity, buffer);
  testOK &= block != nullptr;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  std::cout << "   device NextElement           ... ";
  testOK = true;
  // Allow memory to reach the device
  cudaDeviceSynchronize();
  // Launch a kernel processing tracks
  testTrackBlock<<<nblocks, nthreads>>>(block); ///< note that we are passing a host block type allocated on device
                                                ///< memory - works because the layout is the same
  // Allow all warps to finish
  cudaDeviceSynchronize();
  // The number of used tracks should be equal to the number of spawned threads
  testOK &= block->GetNused() == nblocks.x * nthreads.x;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Compute the sum of assigned track indices, which has to match the sum from 0 to nthreads-1
  // (the execution order is arbitrary, but all thread indices must be distributed)
  unsigned long long counter1 = 0, counter2 = 0;
  testOK = true;
  std::cout << "   device concurrency checksum  ... ";
  for (auto i = 0; i < nblocks.x * nthreads.x; ++i) {
    counter1 += i;
    counter2 += (*block)[i].index;
  }
  testOK &= counter1 == counter2;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Now release 32K tracks
  testOK = true;
  std::cout << "   device ReleaseElement        ... ";
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
  if (!success) return 1;
  return 0;
}
