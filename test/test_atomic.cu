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
#include <VecGeom/backend/cuda/Interface.h>

// Example data structure containing several atomics
struct SomeStruct {
  adept::Atomic_t<int> var_int{nullptr};
  adept::Atomic_t<float> var_float{nullptr};

  void EmplaceDataAt(void *addr)
  {
    char *add_next = (char *)addr;
    // Recipe of calculating next free address in the buffer
    add_next = var_int.EmplaceDataAt(add_next) + adept::Atomic_t<int>::SizeOfData();
    add_next = var_float.EmplaceDataAt(add_next) + adept::Atomic_t<float>::SizeOfData();
  }

  static size_t SizeOfData() { return adept::Atomic_t<int>::SizeOfData() + adept::Atomic_t<float>::SizeOfData(); }
};

// Kernel function to perform atomic addition
__global__ void testAdd(SomeStruct s)
{
  // Test fetch_add, fetch_sub
  s.var_int.fetch_add(1);
  s.var_float.fetch_add(1);
}

// Kernel function to perform atomic subtraction
__global__ void testSub(SomeStruct s)
{
  // Test fetch_add, fetch_sub
  s.var_int.fetch_sub(1);
  s.var_float.fetch_sub(1);
}

// Kernel function to test compare_exchange
__global__ void testCompareExchange(SomeStruct s)
{
  // Read the content of the atomic
  auto expected = s.var_int.load();
  bool success  = false;
  while (!success) {
    // Try to decrement the content, if zero try to replace it with 100
    while (expected > 0) {
      success = s.var_int.compare_exchange_strong(expected, expected - 1);
      if (success) return;
    }
    while (expected == 0) {
      success = s.var_int.compare_exchange_strong(expected, 100);
    }
  }
}

///______________________________________________________________________________________
int main(void)
{
  const char *result[2] = {"FAILED", "OK"};
  SomeStruct a;
  bool success = true;

  // Allocate the content of SomeStruct in a buffer
  auto buff_size = SomeStruct::SizeOfData();
  char *buffer   = nullptr;
  cudaMallocManaged(&buffer, buff_size);

  a.EmplaceDataAt(buffer);

  // Wait for GPU to finish before accessing on host

  // Launch a kernel doing additions (10K blocks of 32 treads each)
  bool testOK = true;
  dim3 nblocks(10000), nthreads(32);
  std::cout << "   testAdd ... ";
  // Wait memory to reach device/host
  cudaDeviceSynchronize();
  testAdd<<<nblocks, nthreads>>>(a);
  cudaDeviceSynchronize();

  testOK &= a.var_int.load() == nblocks.x * nthreads.x;
  testOK &= a.var_float.load() == float(nblocks.x * nthreads.x);
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Launch a kernel doing subtraction (10K blocks of 32 treads each)
  testOK = true;
  std::cout << "   testSub ... ";
  a.var_int.store(nblocks.x * nthreads.x);
  a.var_float.store(nblocks.x * nthreads.x);
  cudaDeviceSynchronize();
  testSub<<<nblocks, nthreads>>>(a);
  cudaDeviceSynchronize();

  testOK &= a.var_int.load() == 0;
  testOK &= a.var_float.load() == 0;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  // Launch a kernel testing compare and swap operations
  std::cout << "   testCAS ... ";
  a.var_int.store(99);
  cudaDeviceSynchronize();
  testCompareExchange<<<nblocks, nthreads>>>(a);
  cudaDeviceSynchronize();
  testOK = a.var_int.load() == 99;
  std::cout << result[testOK] << "\n";
  success &= testOK;

  cudaFree(buffer);
  if (!success) return 1;
  return 0;
}
