// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file CopCore/Global.h
 * @brief CopCore global macros and types
 */

#ifndef COPCORE_GLOBAL_H_
#define COPCORE_GLOBAL_H_

#include <VecCore/VecCore>

#if (defined(__CUDACC__) || defined(__NVCC__))
// Compiling with nvcc
#define COPCORE_IMPL cuda
#else
#define COPCORE_IMPL cxx
#endif

namespace copcore {

/** @brief Backend types enumeration */
enum BackendType { CPU = 0, CUDA, HIP };

/** @brief CUDA error checking */
#ifndef VECCORE_CUDA
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
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
#define COPCORE_EXCEPTION(message) throw std::runtime_error(message)
#else
#define COPCORE_EXCEPTION(message) asm("trap;") // can do better here
#endif

template <BackendType T>
struct StreamType {
  using value_type = int;
  static void CreateStream(value_type &stream) { stream = 0; }
};

#ifdef VECCORE_CUDA
template <>
struct StreamType<BackendType::CUDA> {
  using value_type = cudaStream_t;
  static void CreateStream(value_type &stream) { COPCORE_CUDA_CHECK(cudaStreamCreate(&stream)); }
};
#endif

inline namespace COPCORE_IMPL {

/** @brief Getting the backend name in templated constructs */
template <BackendType T>
struct BackendName;

#define REGISTER_BACKEND_NAME(X) \
  template <>                    \
  struct BackendName<X> {        \
    static const char *name;     \
  };                             \
  const char *BackendName<X>::name = #X

REGISTER_BACKEND_NAME(BackendType::CPU);
REGISTER_BACKEND_NAME(BackendType::CUDA);
REGISTER_BACKEND_NAME(BackendType::HIP);

} // End namespace COPCORE_IMPL
} // End namespace copcore

/** @brief macro to declare device callable functions usable in executors */
#define COPCORE_CALLABLE_FUNC(FUNC) VECCORE_ATT_DEVICE auto _ptr_##FUNC = FUNC;

/** @brief macro to pass callable function to executors */
#ifdef VECCORE_CUDA
#define COPCORE_CALLABLE_DECLARE(HVAR, FUNC) \
  auto HVAR = FUNC;                          \
  cudaMemcpyFromSymbol(&HVAR, _ptr_##FUNC, sizeof(decltype(_ptr_##FUNC)));
#else
#define COPCORE_CALLABLE_DECLARE(HVAR, FUNC) auto HVAR = FUNC;
#endif

#ifdef VECCORE_CUDA
#define COPCORE_CALLABLE_IN_NAMESPACE_DECLARE(HVAR, NAMESPACE, FUNC) \
  auto HVAR = NAMESPACE::FUNC;                                       \
  cudaMemcpyFromSymbol(&HVAR, NAMESPACE::_ptr_##FUNC, sizeof(decltype(NAMESPACE::_ptr_##FUNC)));
#else
#define COPCORE_CALLABLE_IN_NAMESPACE_DECLARE(HVAR, NAMESPACE, FUNC) auto HVAR = NAMESPACE::FUNC;
#endif

#endif // COPCORE_GLOBAL_H_
