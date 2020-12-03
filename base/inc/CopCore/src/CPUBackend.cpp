/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/

#ifdef TARGET_DEVICE_CPU

#include "CopCore/backend/CPUBackend.h"
#include <cstring>
#include <cstdlib>

thread_local GridDimensions gridDim;
thread_local BlockIndices blockIdx;

dim3::dim3(const unsigned int &x) : x(x) {}
dim3::dim3(const unsigned int &x, const unsigned int &y) : x(x), y(y) {}
dim3::dim3(const unsigned int &x, const unsigned int &y, const unsigned int &z) : x(x), y(y), z(z) {}

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
  posix_memalign(devPtr, 64, size);
  return 0;
}

cudaError_t cudaMallocHost(void **ptr, size_t size)
{
  posix_memalign(ptr, 64, size);
  return 0;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind)
{
  std::memcpy(dst, src, count);
  return 0;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind, cudaStream_t)
{
  std::memcpy(dst, src, count);
  return 0;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count)
{
  std::memset(devPtr, value, count);
  return 0;
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t)
{
  std::memset(devPtr, value, count);
  return 0;
}

cudaError_t cudaPeekAtLastError()
{
  return 0;
}

cudaError_t cudaEventCreate(cudaEvent_t *)
{
  return 0;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *, int)
{
  return 0;
}

cudaError_t cudaEventSynchronize(cudaEvent_t)
{
  return 0;
}

cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t)
{
  return 0;
}

cudaError_t cudaFreeHost(void *ptr)
{
  free(ptr);
  return 0;
}

cudaError_t cudaFree(void *ptr)
{
  free(ptr);
  return 0;
}

cudaError_t cudaDeviceReset()
{
  return 0;
}

cudaError_t cudaStreamCreate(cudaStream_t *)
{
  return 0;
}

cudaError_t cudaMemcpyToSymbol(void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind)
{
  std::memcpy(symbol, reinterpret_cast<const char *>(src) + offset, count);
  return 0;
}

unsigned int atomicInc(unsigned int *address, unsigned int val)
{
  unsigned int old = *address;
  *address         = ((old >= val) ? 0 : (old + 1));
  return old;
}

namespace Configuration {
unsigned verbosity_level;
}

cudaError_t cudaHostUnregister(void *)
{
  return 0;
}

cudaError_t cudaHostRegister(void *, size_t, unsigned int)
{
  return 0;
}

#endif
