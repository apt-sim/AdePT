// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file CopCore/Macros.h
 * @brief CopCore global macros
 *
 * @details Compiling mixtures of C/C++/CUDA, we run into the cases of compiling for
 * the host and the device. This involves both marking functions as accessible
 * from the host/device/both, as well as (potential) different compile paths
 * when compiling for the host vs for the device. The macros in this file
 * assist this markup process. Only C/C++ and NVidia CUDA are currently supported,
 * but other "backends" such as HIP can be added as time goes on.
 *
 * CopCore/AdePT follow LHCb's Allen project in using macros for host/device/inline
 * functions that match the CUDA keywords.
 */
/**
 * @def COPCORE_CUDA_COMPILER
 * @brief Defined when using a CUDA compiler
 *
 * Effectively a proxy for `__CUDACC__`
 */
/**
 * @def COPCORE_DEVICE_COMPILATION
 * @brief Defined when compiling CUDA code for the device
 *
 * Effectively a proxy for `__CUDA_ARCH__`
 */
/**
 * @def __host__
 * @brief mark a function as accesible from the host
 *
 * Expands to nothing when not compiling CUDA
 */
/**
 * @def __device__
 * @brief mark a function as accesible from the host
 *
 * Expands to nothing when not compiling CUDA
 */
/**
 * @def __forceinline__
 * @brief mark a function to always be inline
 *
 * Expands to the relevant CUDA keyword or host compiler attribute
 */


#ifndef COPCORE_MACROS_H_
#define COPCORE_MACROS_H_

// Macros for separating CUDA compiler from others
#ifdef __CUDACC__
#  define COPCORE_CUDA_COMPILER
#endif

// Define function keywords if not already present.
#ifndef __host__
#  define __host__
#endif

#ifndef __device__
#  define __device__
#endif

#ifndef __forceinline__
#  define __forceinline__ inline __attribute__((always_inline))
#endif

// Definition of __CUDA_ARCH__ means we are compiling CUDA *and*
// on the device compile pass
// COPCORE_DEVICE_COMPILATION is defined in this case
#ifdef __CUDA_ARCH__
#  define COPCORE_DEVICE_COMPILATION
#endif


#endif // COPCORE_MACROS_H_
