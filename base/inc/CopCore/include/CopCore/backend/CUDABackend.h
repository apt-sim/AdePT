/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/

#pragma once

#if defined(TARGET_DEVICE_CUDA) || defined(TARGET_DEVICE_CUDACLANG)

#if !defined(DEVICE_COMPILER)
#include <cuda_runtime_api.h>

#if defined(TARGET_DEVICE_CUDACLANG)
inline const char *cudaGetErrorString(cudaError_t error)
{
  return "";
}
#endif

#endif

#include <cuda_fp16.h>
#define half_t half

#include "math_constants.h"

/**
 * @brief Macro to check cuda calls.
 */
#define cudaCheck(stmt)                                \
  {                                                    \
    cudaError_t err = stmt;                            \
    if (err != cudaSuccess) {                          \
      std::cerr << "Failed to run " << #stmt << "\n";  \
      std::cerr << cudaGetErrorString(err) << "\n";    \
      throw std::invalid_argument("cudaCheck failed"); \
    }                                                  \
  }

#define cudaCheckKernelCall(stmt)                                                                             \
  {                                                                                                           \
    cudaError_t err = stmt;                                                                                   \
    if (err != cudaSuccess) {                                                                                 \
      fprintf(stderr, "Failed to invoke kernel\n%s (%d) at %s: %d\n", cudaGetErrorString(err), err, __FILE__, \
              __LINE__);                                                                                      \
      throw std::invalid_argument("cudaCheckKernelCall failed");                                              \
    }                                                                                                         \
  }

#endif