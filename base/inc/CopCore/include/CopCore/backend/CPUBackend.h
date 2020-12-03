/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/

#pragma once

#ifdef TARGET_DEVICE_CPU

#include <stdexcept>
#include <cassert>
#include <cmath>
#include <cstring>
#include <algorithm>

using std::copysignf;
using std::signbit;

#define __host__
#define __device__
#define __shared__
#define __global__
#define __constant__
#define __syncthreads()
#define __syncwarp()
#define __launch_bounds__(_i)
#define cudaError_t int
#define cudaEvent_t int
#define cudaStream_t int
#define cudaSuccess 0
#define cudaErrorMemoryAllocation 2
#define __popc __builtin_popcount
#define __popcll __builtin_popcountll
#define __ffs __builtin_ffs
#define __clz __builtin_clz
#define cudaEventBlockingSync 0x01
#define __forceinline__ inline
#define CUDART_PI_F M_PI

enum cudaMemcpyKind {
  cudaMemcpyHostToHost,
  cudaMemcpyHostToDevice,
  cudaMemcpyDeviceToHost,
  cudaMemcpyDeviceToDevice,
  cudaMemcpyDefault
};

struct float3 {
  float x;
  float y;
  float z;
};

struct float2 {
  float x;
  float y;
};

struct dim3 {
  unsigned int x = 1;
  unsigned int y = 1;
  unsigned int z = 1;

  dim3()             = default;
  dim3(const dim3 &) = default;

  dim3(const unsigned int &x);
  dim3(const unsigned int &x, const unsigned int &y);
  dim3(const unsigned int &x, const unsigned int &y, const unsigned int &z);
};

struct GridDimensions {
  unsigned int x;
  unsigned int y;
  unsigned int z;
};

struct BlockIndices {
  unsigned int x;
  unsigned int y;
  unsigned int z;
};

struct BlockDimensions {
  unsigned int x = 1;
  unsigned int y = 1;
  unsigned int z = 1;
};

struct ThreadIndices {
  unsigned int x = 0;
  unsigned int y = 0;
  unsigned int z = 0;
};

extern thread_local GridDimensions gridDim;
extern thread_local BlockIndices blockIdx;
constexpr BlockDimensions blockDim{1, 1, 1};
constexpr ThreadIndices threadIdx{0, 0, 0};

cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream);
cudaError_t cudaPeekAtLastError();
cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, int flags);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaFreeHost(void *ptr);
cudaError_t cudaFree(void *ptr);
cudaError_t cudaDeviceReset();
cudaError_t cudaStreamCreate(cudaStream_t *pStream);
cudaError_t cudaMemcpyToSymbol(void *symbol, const void *src, size_t count, size_t offset = 0,
                               enum cudaMemcpyKind kind = cudaMemcpyDefault);
cudaError_t cudaHostUnregister(void *ptr);
cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags);

// CUDA accepts more bindings to cudaMemcpyTo/FromSymbol
template <class T>
cudaError_t cudaMemcpyToSymbol(T &symbol, const void *src, size_t count, size_t offset = 0,
                               enum cudaMemcpyKind = cudaMemcpyHostToDevice)
{
  std::memcpy(reinterpret_cast<void *>(((char *)&symbol) + offset), src, count);
  return 0;
}

template <class T>
cudaError_t cudaMemcpyFromSymbol(void *dst, const T &symbol, size_t count, size_t offset = 0,
                                 enum cudaMemcpyKind = cudaMemcpyHostToDevice)
{
  std::memcpy(dst, reinterpret_cast<void *>(((char *)&symbol) + offset), count);
  return 0;
}

template <class T, class S>
T atomicAdd(T *address, S val)
{
  const T old = *address;
  *address += val;
  return old;
}

template <class T, class S>
T atomicOr(T *address, S val)
{
  const T old = *address;
  *address |= val;
  return old;
}

template <class T>
T max(const T &a, const T &b)
{
  return std::max(a, b);
}

template <class T>
T min(const T &a, const T &b)
{
  return std::min(a, b);
}

unsigned int atomicInc(unsigned int *address, unsigned int val);

uint16_t __float2half(const float f);

float __half2float(const uint16_t h);

#ifdef CPU_USE_REAL_HALF

/**
 * @brief half_t with int16_t backend (real half).
 */
struct half_t {
private:
  uint16_t m_value;

public:
  half_t()               = default;
  half_t(const half_t &) = default;

  half_t(const float f) { m_value = __float2half(f); }

  inline operator float() const { return __half2float(m_value); }

  inline bool operator<(const half_t &a) const
  {
    const auto sign   = (m_value >> 15) & 0x01;
    const auto sign_a = (a.get() >> 15) & 0x01;
    return (sign & sign_a & operator!=(a)) ^ (m_value < a.get());
  }

  inline bool operator>(const half_t &a) const
  {
    const auto sign   = (m_value >> 15) & 0x01;
    const auto sign_a = (a.get() >> 15) & 0x01;
    return (sign & sign_a & operator!=(a)) ^ (m_value > a.get());
  }

  inline bool operator<=(const half_t &a) const { return !operator>(a); }

  inline bool operator>=(const half_t &a) const { return !operator<(a); }

  inline bool operator==(const half_t &a) const { return m_value == a.get(); }

  inline bool operator!=(const half_t &a) const { return !operator==(a); }
};

#else

/**
 * @brief half_t with float backend.
 */
using half_t = float;

#endif

#define cudaCheck(stmt)                                \
  {                                                    \
    cudaError_t err = stmt;                            \
    if (err != cudaSuccess) {                          \
      std::cerr << "Failed to run " << #stmt << "\n";  \
      throw std::invalid_argument("cudaCheck failed"); \
    }                                                  \
  }

#define cudaCheckKernelCall(stmt)                                \
  {                                                              \
    cudaError_t err = stmt;                                      \
    if (err != cudaSuccess) {                                    \
      std::cerr << "Failed to invoke kernel.\n";                 \
      throw std::invalid_argument("cudaCheckKernelCall failed"); \
    }                                                            \
  }

#endif
