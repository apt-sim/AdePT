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
#include <CopCore/Macros.h>

#ifdef COPCORE_CUDA_COMPILER
// Compiling with nvcc
#define COPCORE_IMPL cuda
#else
#define COPCORE_IMPL cxx
#endif

namespace copcore {

/** @brief Backend types enumeration */
enum BackendType { CPU = 0, CUDA, HIP };

/** @brief CUDA error checking */
#ifndef COPCORE_CUDA_COMPILER
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
#ifndef COPCORE_DEVICE_COMPILATION
#define COPCORE_EXCEPTION(message) throw std::runtime_error(message)
#else
#define COPCORE_EXCEPTION(message)      \
  do {                                  \
    printf("Exception: %s\n", message); \
    asm("trap;");                       \
  } while (0)
#endif

/** @brief Check if pointer id device-resident */
#ifndef COPCORE_CUDA_COMPILER
inline bool is_device_pointer(void *ptr) { return false; }
#else
inline bool is_device_pointer(void *ptr)
{
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
  COPCORE_CUDA_CHECK(err);
  return (attr.type == cudaMemoryTypeDevice);
}
#endif

/** @brief Get number of SMs on the current device */
#ifndef COPCORE_CUDA_COMPILER
inline int get_num_SMs() { return 0; }
#else
inline int get_num_SMs()
{
  int deviceId, numSMs;
  COPCORE_CUDA_CHECK(cudaGetDevice(&deviceId));
  COPCORE_CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
  return numSMs;
}
#endif

template <BackendType T>
struct StreamType {
  using value_type = int;
  static void CreateStream(value_type &stream) { stream = 0; }
};

#ifdef COPCORE_CUDA_COMPILER
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
#ifdef COPCORE_CUDA_COMPILER
#define COPCORE_CALLABLE_DECLARE(HVAR, FUNC)                    \
  auto HVAR = FUNC;                                             \
  /*printf("cudaMemcpyFromSymbol for function: %s\n", #FUNC);*/ \
  cudaMemcpyFromSymbol(&HVAR, _ptr_##FUNC, sizeof(decltype(_ptr_##FUNC)));
#else
#define COPCORE_CALLABLE_DECLARE(HVAR, FUNC) auto HVAR = FUNC;
#endif

#ifdef COPCORE_CUDA_COMPILER
#define COPCORE_CALLABLE_IN_NAMESPACE_DECLARE(HVAR, NAMESPACE, FUNC)            \
  auto HVAR = NAMESPACE::FUNC;                                                  \
  /*printf("cudaMemcpyFromSymbol for function: %s::%s\n", #NAMESPACE, #FUNC);*/ \
  cudaMemcpyFromSymbol(&HVAR, NAMESPACE::_ptr_##FUNC, sizeof(decltype(NAMESPACE::_ptr_##FUNC)));
#else
#define COPCORE_CALLABLE_IN_NAMESPACE_DECLARE(HVAR, NAMESPACE, FUNC) auto HVAR = NAMESPACE::FUNC;
#endif

#endif // COPCORE_GLOBAL_H_
