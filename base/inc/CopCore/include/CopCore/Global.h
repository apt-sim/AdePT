// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file CopCore/Global.h
 * @brief CopCore global macros and types
 */

#ifndef COPCORE_GLOBAL_H_
#define COPCORE_GLOBAL_H_

#include <cstdio>
#include <type_traits>
#include <CopCore/backend/BackendCommon.h>

#ifdef TARGET_DEVICE_CUDA
// Compiling with nvcc
#define COPCORE_IMPL cuda
#else
#define COPCORE_IMPL cxx
#endif

namespace copcore {

/** @brief Backend types enumeration */
enum BackendType { CPU = 0, CUDA, HIP };

/** @brief CUDA error checking */
#ifndef TARGET_DEVICE_CUDA
inline void error_check(int, const char *, int) {}
#else
inline void error_check(cudaError_t err, const char *file, int line)
{
  if (err != cudaSuccess) {
    ::fprintf(stderr, "CUDA ERROR at %s[%d] : %s\n", file, line, cudaGetErrorString(err));
    abort();
  }
}
#endif
#define COPCORE_CUDA_CHECK(err)                    \
  do {                                             \
    copcore::error_check(err, __FILE__, __LINE__); \
  } while (0)

/** @brief Trigger a runtime error depending on the backend */
#ifndef DEVICE_COMPILATION_TRAJECTORY
#define COPCORE_EXCEPTION(message) throw std::runtime_error(message)
#else
#define COPCORE_EXCEPTION(message) asm("trap;") // can do better here
#endif

template <BackendType T>
struct StreamType {
  using value_type = int;
  static void CreateStream(value_type &stream) { stream = 0; }
};

#ifdef TARGET_DEVICE_CUDA
template <>
struct StreamType<BackendType::CUDA> {
  using value_type = cudaStream_t;
  static void CreateStream(value_type &stream) { COPCORE_CUDA_CHECK(cudaStreamCreate(&stream)); }
};
#endif

/** @brief Getting the backend name in templated constructs */
template <typename Backend>
const char *BackendName(Backend const &backend)
{
  switch (backend) {
  case BackendType::CPU:
    return "BackendType::CPU";
  case BackendType::CUDA:
    return "BackendType::CUDA";
  case BackendType::HIP:
    return "BackendType::HIP";
  default:
    return "Unknown backend";
  };
};

} // End namespace copcore

/** @brief Macro to template-specialize on a specific compile-time requirement */
#define COPCORE_REQUIRES(...) typename std::enable_if<(__VA_ARGS__)>::type * = nullptr

/** @brief macro to declare device callable functions usable in executors */
#define COPCORE_CALLABLE_FUNC(FUNC) __device__ auto _ptr_##FUNC = FUNC;

/** @brief macro to pass callable function to executors */
#ifdef TARGET_DEVICE_CUDA
#define COPCORE_CALLABLE_DECLARE(HVAR, FUNC)                    \
  auto HVAR = FUNC;                                             \
  /*printf("cudaMemcpyFromSymbol for function: %s\n", #FUNC);*/ \
  cudaMemcpyFromSymbol(&HVAR, _ptr_##FUNC, sizeof(decltype(_ptr_##FUNC)));
#else
#define COPCORE_CALLABLE_DECLARE(HVAR, FUNC) auto HVAR = FUNC;
#endif

#ifdef TARGET_DEVICE_CUDA
#define COPCORE_CALLABLE_IN_NAMESPACE_DECLARE(HVAR, NAMESPACE, FUNC)            \
  auto HVAR = NAMESPACE::FUNC;                                                  \
  /*printf("cudaMemcpyFromSymbol for function: %s::%s\n", #NAMESPACE, #FUNC);*/ \
  cudaMemcpyFromSymbol(&HVAR, NAMESPACE::_ptr_##FUNC, sizeof(decltype(NAMESPACE::_ptr_##FUNC)));
#else
#define COPCORE_CALLABLE_IN_NAMESPACE_DECLARE(HVAR, NAMESPACE, FUNC) auto HVAR = NAMESPACE::FUNC;
#endif

#endif // COPCORE_GLOBAL_H_
