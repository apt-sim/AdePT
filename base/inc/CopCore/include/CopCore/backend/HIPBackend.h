/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/

#pragma once

#ifdef TARGET_DEVICE_HIP

#if !defined(__HCC__) && !defined(__HIP__)
#define __HIP_PLATFORM_HCC__
#include <hip/hip_runtime_api.h>
#else
#include <hip/hip_runtime.h>
#include <hip/math_functions.h>
#endif

#include <hip/hip_fp16.h>
#define half_t half

#include <cmath>
#define CUDART_PI_F M_PI

// CUDA to HIP conversion
#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventRecord hipEventRecord
#define cudaFreeHost hipHostFree
#define cudaDeviceReset hipDeviceReset
#define cudaStreamCreate hipStreamCreate
#define cudaMemGetInfo hipMemGetInfo
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp hipDeviceProp_t
#define cudaError_t hipError_t
#define cudaEvent_t hipEvent_t
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define cudaErrorMemoryAllocation hipErrorMemoryAllocation
#define cudaEventBlockingSync hipEventBlockingSync
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaFuncCachePreferL1 hipFuncCachePreferL1
#define cudaDeviceGetByPCIBusId hipDeviceGetByPCIBusId
#define cudaDeviceSetCacheConfig hipDeviceSetCacheConfig
#define cudaHostUnregister hipHostUnregister

#define cudaCheck(stmt)                                                                                      \
  {                                                                                                          \
    hipError_t err = stmt;                                                                                   \
    if (err != hipSuccess) {                                                                                 \
      fprintf(stderr, "Failed to run %s\n%s (%d) at %s: %d\n", #stmt, hipGetErrorString(err), err, __FILE__, \
              __LINE__);                                                                                     \
      throw std::invalid_argument("cudaCheck failed");                                                       \
    }                                                                                                        \
  }

#define cudaCheckKernelCall(stmt)                                                                            \
  {                                                                                                          \
    cudaError_t err = stmt;                                                                                  \
    if (err != cudaSuccess) {                                                                                \
      fprintf(stderr, "Failed to invoke kernel\n%s (%d) at %s: %d\n", hipGetErrorString(err), err, __FILE__, \
              __LINE__);                                                                                     \
      throw std::invalid_argument("cudaCheckKernelCall failed");                                             \
    }                                                                                                        \
  }

#endif