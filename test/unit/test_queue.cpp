// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_queue.cu
 * @brief Unit test for queue operations.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#include <iostream>
#include <cassert>
#include <AdePT/base/mpmc_bounded_queue.h>
#include <AdePT/core/Portability.hh>

// Kernel function to perform atomic addition
__global__ void pushData(adept::mpmc_bounded_queue<int> *queue)
{
  // Push the thread index in the queue
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  queue->enqueue(id);
}

// Kernel function to dequeue a value and add it atomically
__global__ void popAndAdd(adept::mpmc_bounded_queue<int> *queue, adept::Atomic_t<unsigned long long> *sum)
{
  // Push the thread index in the queue
  int id = 0;
  if (!queue->dequeue(id)) id = 0;
  sum->fetch_add(id);
}

///______________________________________________________________________________________
int main(void)
{
  using Queue_t      = adept::mpmc_bounded_queue<int>;
  using AtomicLong_t = adept::Atomic_t<unsigned long long>;

  const char *result[2] = {"FAILED", "OK"};
  bool success          = true;
  // Define the kernels granularity: 10K blocks of 32 treads each
  dim3 nblocks(1000), nthreads(32);

  int capacity      = 1 << 15; // 32768 - accomodates values pushed by all threads
  size_t buffersize = Queue_t::SizeOfInstance(capacity);
  char *buffer      = nullptr;
  ADEPT_DEVICE_API_CALL(MallocManaged(&buffer, buffersize));
  auto queue = Queue_t::MakeInstanceAt(capacity, buffer);

  char *buffer_atomic = nullptr;
  ADEPT_DEVICE_API_CALL(MallocManaged(&buffer_atomic, sizeof(AtomicLong_t)));
  auto sum = new (buffer_atomic) AtomicLong_t;

  bool testOK = true;
  std::cout << "   test_queue ... ";
  // Allow memory to reach the device
  ADEPT_DEVICE_API_CALL(DeviceSynchronize());
  // Launch a kernel queueing thread id's
  pushData<<<nblocks, nthreads>>>(queue);
  // Allow all warps in the stream to finish
  ADEPT_DEVICE_API_CALL(DeviceSynchronize());
  // Make sure all threads managed to queue their id
  testOK &= queue->size() == nblocks.x * nthreads.x;
  // Launch a kernel top collect queued data
  popAndAdd<<<nblocks, nthreads>>>(queue, sum);
  // Wait work to finish and memory to reach the host
  ADEPT_DEVICE_API_CALL(DeviceSynchronize());
  // Check if all data was dequeued
  testOK &= queue->size() == 0;
  // Check if the sum of all dequeued id's matches the sum of thread indices
  unsigned long long sumref = 0;
  for (auto i = 0; i < nblocks.x * nthreads.x; ++i)
    sumref += i;
  testOK &= sum->load() == sumref;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  ADEPT_DEVICE_API_CALL(Free(buffer));
  ADEPT_DEVICE_API_CALL(Free(buffer_atomic));
  if (!success) return 1;
  return 0;
}

