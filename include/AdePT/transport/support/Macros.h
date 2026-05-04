// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file AdePT/transport/support/Macros.h
 * @brief CopCore global macros
 *
 * @details Compiling mixtures of C/C++/CUDA, we run into the cases of compiling for
 * the host and the device. This involves both marking functions as accessible
 * from the host/device/both, as well as (potential) different compile paths
 * when compiling for the host vs for the device. The macros in this file
 * assist this markup process. Only C/C++ and NVidia CUDA are currently supported,
 * but other "backends" such as HIP can be added as time goes on.
 *
 * AdePT follows LHCb's Allen project in using macros for host/device/inline functions that match the CUDA keywords.
 */
/**
 * @brief Macro documentation notes
 *
 * `COPCORE_CUDA_COMPILER`
 * Defined when using a CUDA compiler (effectively a proxy for `__CUDACC__`).
 *
 * `COPCORE_DEVICE_COMPILATION`
 * Defined when compiling CUDA code for device (effectively a proxy for
 * `__CUDA_ARCH__`).
 *
 * `__host__`
 * Marks a function as callable from host code. For non-CUDA compilation this
 * macro expands to nothing.
 *
 * `__device__`
 * Marks a function as callable from device code. For non-CUDA compilation this
 * macro expands to nothing.
 *
 * `__forceinline__`
 * Marks a function to always inline. If not provided by CUDA headers, it
 * expands to `inline __attribute__((always_inline))` on host compilers.
 */

#ifndef COPCORE_MACROS_H_
#define COPCORE_MACROS_H_

// Macros for separating CUDA compiler from others
#ifdef __CUDACC__
#define COPCORE_CUDA_COMPILER
#endif

// Define function keywords if not already present.
#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif

// Definition of __CUDA_ARCH__ means we are compiling CUDA *and*
// on the device compile pass
// COPCORE_DEVICE_COMPILATION is defined in this case
#ifdef __CUDA_ARCH__
#define COPCORE_DEVICE_COMPILATION
#endif

// Pragmas to locally suppress -Wpedantic for __int128 or other non-standard extensions
#if defined(__GNUC__) && !defined(__clang__) && !defined(COPCORE_DEVICE_COMPILATION)
#define DISABLE_PEDANTIC_WARNINGS _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wpedantic\"")
#define ENABLE_PEDANTIC_WARNINGS _Pragma("GCC diagnostic pop")
#elif defined(__clang__) && !defined(COPCORE_DEVICE_COMPILATION)
#define DISABLE_PEDANTIC_WARNINGS _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wpedantic\"")
#define ENABLE_PEDANTIC_WARNINGS _Pragma("clang diagnostic pop")
#else
#define DISABLE_PEDANTIC_WARNINGS
#define ENABLE_PEDANTIC_WARNINGS
#endif

#endif // COPCORE_MACROS_H_
