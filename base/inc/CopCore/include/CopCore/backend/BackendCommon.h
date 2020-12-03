/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/

#pragma once

#include <tuple>
#include <string>
#include <cassert>

// Host / device compiler identification
#if defined(TARGET_DEVICE_CPU) || (defined(TARGET_DEVICE_CUDA) && defined(__CUDACC__)) || \
    (defined(TARGET_DEVICE_CUDACLANG) && defined(__clang__) && defined(__CUDA__)) ||      \
    (defined(TARGET_DEVICE_HIP) && (defined(__HCC__) || defined(__HIP__)))
#define DEVICE_COMPILER
#endif

// Host / device compilation trajectory identification
// Only for CUDA at present...
#if defined(TARGET_DEVICE_CUDA) && defined(__CUDA_ARCH__)
#define DEVICE_COMPILATION_TRAJECTORY
#endif

// Dispatch to the right backend
#if defined(TARGET_DEVICE_CPU)
#include "CPUBackend.h"
#elif defined(TARGET_DEVICE_HIP)
#include "HIPBackend.h"
#elif defined(TARGET_DEVICE_CUDA) || defined(TARGET_DEVICE_CUDACLANG)
#include "CUDABackend.h"
#endif

