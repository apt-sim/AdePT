// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_atomic.cu
 * @brief Unit test for atomic operations.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#include <iostream>
#include <cassert>
#include <AdePT/Atomic.h>

// Example data structure containing several atomics
struct SomeStruct {
  adept::Atomic_t<int> var_int;
  adept::Atomic_t<float> var_float;

  __host__ __device__
  SomeStruct() {}

  __host__ __device__
  static SomeStruct *MakeInstanceAt(void *addr)
  {
    SomeStruct *obj = new (addr) SomeStruct();
    return obj;
  }
};

// Kernel function to perform atomic addition
__global__ void testAdd(SomeStruct *s)
{
  // Test fetch_add, fetch_sub
  s->var_int.fetch_add(1);
  s->var_float.fetch_add(1);
}

// Kernel function to perform atomic subtraction
__global__ void testSub(SomeStruct *s)
{
  // Test fetch_add, fetch_sub
  s->var_int.fetch_sub(1);
  s->var_float.fetch_sub(1);
}

// Kernel function to test compare_exchange
__global__ void testCompareExchange(SomeStruct *s)
{
  // Read the content of the atomic
  auto expected = s->var_int.load();
  bool success  = false;
  while (!success) {
    // Try to decrement the content, if zero try to replace it with 100
    while (expected > 0) {
      success = s->var_int.compare_exchange_strong(expected, expected - 1);
      if (success) return;
    }
    while (expected == 0) {
      success = s->var_int.compare_exchange_strong(expected, 100);
    }
  }
}

///______________________________________________________________________________________
int main(void)
{
  const char *result[2] = {"FAILED", "OK"};
  bool success          = true;
  // Define the kernels granularity: 10K blocks of 32 treads each
  dim3 nblocks(10000), nthreads(32);

  // Allocate the content of SomeStruct in a buffer
  char *buffer = nullptr;
  cudaMallocManaged(&buffer, sizeof(SomeStruct));
  SomeStruct *a = SomeStruct::MakeInstanceAt(buffer);

  // Launch a kernel doing additions
  bool testOK = true;
  std::cout << "   testAdd ... ";
  // Wait memory to reach device
  cudaDeviceSynchronize();
  testAdd<<<nblocks, nthreads>>>(a);
  // Wait all warps to finish and sync memory
  cudaDeviceSynchronize();

  testOK &= a->var_int.load() == nblocks.x * nthreads.x;
  testOK &= a->var_float.load() == float(nblocks.x * nthreads.x);
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Launch a kernel doing subtraction
  testOK = true;
  std::cout << "   testSub ... ";
  a->var_int.store(nblocks.x * nthreads.x);
  a->var_float.store(nblocks.x * nthreads.x);
  cudaDeviceSynchronize();
  testSub<<<nblocks, nthreads>>>(a);
  cudaDeviceSynchronize();

  testOK &= a->var_int.load() == 0;
  testOK &= a->var_float.load() == 0;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Launch a kernel testing compare and swap operations
  std::cout << "   testCAS ... ";
  a->var_int.store(99);
  cudaDeviceSynchronize();
  testCompareExchange<<<nblocks, nthreads>>>(a);
  cudaDeviceSynchronize();
  testOK = a->var_int.load() == 99;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  cudaFree(buffer);
  if (!success) return 1;
  return 0;
}
