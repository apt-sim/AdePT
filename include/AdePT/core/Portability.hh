// SPDX-FileCopyrightText: Celeritas contributors
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see https://github.com/celeritas-project/celeritas?tab=License-1-ov-file
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/*!
 * \file Assert.h
 * \brief Macros, exceptions, and helpers for assertions and error handling.
 *
 * This defines host- and device-compatible assertion macros that are
 * currenntly toggled on the \c NDEBUG configure macros.
 *
 * Derived from celeritas corecel: DeviceRuntimeApi.hh, Macros.hh, Assert.hh
 */
//---------------------------------------------------------------------------/

// Portability macros used to switch between cuda/hip

#ifndef PORTABILITY_HH
#define PORTABILITY_HH

#ifndef __CUDA_ARCH__
#include <stdexcept>
#include <sstream>
#else
#include <cassert>
#endif

#define ADEPT_ATT_HOST __host__
#define ADEPT_ATT_DEVICE __device__
#define ADEPT_ATT_HOST_DEVICE __host__ __device__

// NOTE: if we want more debug granularity, these could be defined via CMake config in the future
#ifdef NDEBUG
#define ADEPT_DEBUG 0
#define ADEPT_DEVICE_DEBUG 0
#else
#define ADEPT_DEBUG 1
#define ADEPT_DEVICE_DEBUG 1
#endif

/*!
 * \def ADEPT_DEVICE_PLATFORM
 *
 * API prefix token for the device offloading type.
 */
/*!
 * \def ADEPT_DEVICE_API_SYMBOL
 *
 * Add a prefix "hip" or "cuda" to a code token.
 */
#if defined(__CUDACC__)
#define ADEPT_DEVICE_PLATFORM cuda
#define ADEPT_DEVICE_PLATFORM_UPPER_STR "CUDA"
#define ADEPT_DEVICE_API_SYMBOL(TOK) cuda##TOK
#elif defined(__HIP__)
#define ADEPT_DEVICE_PLATFORM hip
#define ADEPT_DEVICE_PLATFORM_UPPER_STR "HIP"
#define ADEPT_DEVICE_API_SYMBOL(TOK) hip##TOK
#else
#define ADEPT_DEVICE_PLATFORM none
#define ADEPT_DEVICE_PLATFORM_UPPER_STR ""
#define ADEPT_DEVICE_API_SYMBOL(TOK) void
#endif

/*!
 * \def ADEPT_DEVICE_SOURCE
 *
 * Defined and true if building a HIP or CUDA source file. This is a generic
 * replacement for \c __CUDACC__ .
 */
/*!
 * \def ADEPT_DEVICE_COMPILE
 *
 * Defined and true if building device code in HIP or CUDA. This is a generic
 * replacement for \c __CUDA_ARCH__ .
 */
#if defined(__CUDACC__) || defined(__HIP__)
#define ADEPT_DEVICE_SOURCE 1
#elif defined(__DOXYGEN__)
#define ADEPT_DEVICE_SOURCE 0
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define ADEPT_DEVICE_COMPILE 1
#elif defined(__DOXYGEN__)
#define ADEPT_DEVICE_COMPILE 0
#endif

/*!
 * \def ADEPT_UNLIKELY(condition)
 *
 * Mark the result of this condition to be "unlikely".
 *
 * This asks the compiler to move the section of code to a "cold" part of the
 * instructions, improving instruction locality. It should be used primarily
 * for error checking conditions.
 */
#if defined(__clang__) || defined(__GNUC__)
// GCC and Clang support the same builtin
#define ADEPT_UNLIKELY(COND) __builtin_expect(!!(COND), 0)
#else
// No other compilers seem to have a similar builtin
#define ADEPT_UNLIKELY(COND) (COND)
#endif

/*!
 * \def ADEPT_ASSERT
 *
 * Internal debug assertion macro. This replaces standard \c assert usage.
 */
/*!
 * \def ADEPT_NOT_CONFIGURED
 *
 * Assert if the code point is reached because an optional feature is disabled.
 * This generally should be used for the constructors of dummy class
 * definitions in, e.g., \c Foo.nocuda.cc:
 * \code
    Foo::Foo()
    {
        ADEPT_NOT_CONFIGURED("CUDA");
    }
 * \endcode
 */
/*!
 * \def ADEPT_DEBUG_FAIL
 *
 * Throw a debug assertion regardless of the \c ADEPT_DEBUG setting. This
 * is used internally but is also useful for catching subtle programming errors
 * in downstream code.
 */
/*!
 * \def ADEPT_ASSERT_UNREACHABLE
 *
 * Throw an assertion if the code point is reached. When debug assertions are
 * turned off, this changes to a compiler hint that improves optimization (and
 * may force the code to exit uncermoniously if the point is encountered,
 * rather than continuing on with undefined behavior).
 */
/*!
 * \def ADEPT_VALIDATE
 *
 * Always-on runtime assertion macro. This can check user input and input data
 * consistency, and will raise std::runtime_error on failure with a descriptive error
 * message that is streamed as the second argument. If used
 * in \c __device__ -annotated code, the second argument *must* be a single C string.
 *
 * An always-on debug-type assertion without a detailed message can be
 * constructed by omitting the stream (but leaving the comma):
 * \code
    ADEPT_VALIDATE(file_stream,);
 * \endcode
 */

#if !defined(__HIP__) && !defined(__CUDA_ARCH__)
// Throw in host code
#define ADEPT_DEBUG_THROW_(MSG, WHICH) throw ::portability::make_debug_error(#WHICH, MSG, __FILE__, __LINE__)
#elif defined(__CUDA_ARCH__) && !defined(NDEBUG)
// Use the assert macro for CUDA when supported
#define ADEPT_DEBUG_THROW_(MSG, WHICH) assert(false && sizeof(#WHICH ": " MSG))
#else
// Use a special device function to emulate assertion failure if HIP
// (assertion from multiple threads simultaneously can cause unexpected device
// failures on AMD hardware) or if NDEBUG is in use with CUDA
#define ADEPT_DEBUG_THROW_(MSG, WHICH) ::portability::device_debug_error(#WHICH, MSG, __FILE__, __LINE__)
#endif

#define ADEPT_DEBUG_FAIL(MSG, WHICH) \
  do {                               \
    ADEPT_DEBUG_THROW_(MSG, WHICH);  \
    ::portability::unreachable();    \
  } while (0)
#define ADEPT_DEBUG_ASSERT_(COND, WHICH) \
  do {                                   \
    if (ADEPT_UNLIKELY(!(COND))) {       \
      ADEPT_DEBUG_THROW_(#COND, WHICH);  \
    }                                    \
  } while (0)
#define ADEPT_NOASSERT_(COND) \
  do {                        \
    if (false && (COND)) {    \
    }                         \
  } while (0)

#if (ADEPT_DEBUG && !ADEPT_DEVICE_COMPILE) || (ADEPT_DEVICE_DEBUG && ADEPT_DEVICE_COMPILE)
#define ADEPT_ASSERT(COND) ADEPT_DEBUG_ASSERT_(COND, internal)
#define ADEPT_ASSERT_UNREACHABLE() ADEPT_DEBUG_FAIL("unreachable code point encountered", unreachable)
#else
#define ADEPT_ASSERT(COND) ADEPT_NOASSERT_(COND)
#define ADEPT_ASSERT_UNREACHABLE() ::portability::unreachable()
#endif

#if !ADEPT_DEVICE_COMPILE
#define ADEPT_RUNTIME_THROW(WHICH, WHAT, COND) \
  throw ::portability::make_runtime_error(WHICH, WHAT, COND, __FILE__, __LINE__)
#else
#define ADEPT_RUNTIME_THROW(WHICH, WHAT, COND) \
  ADEPT_DEBUG_FAIL("Runtime errors cannot be thrown from device code", unreachable);
#endif

#if !ADEPT_DEVICE_COMPILE
#define ADEPT_VALIDATE(COND, MSG)                                           \
  do {                                                                      \
    if (ADEPT_UNLIKELY(!(COND))) {                                          \
      std::ostringstream vg_runtime_msg_;                                   \
      vg_runtime_msg_ MSG;                                                  \
      ADEPT_RUNTIME_THROW("runtime", vg_runtime_msg_.str().c_str(), #COND); \
    }                                                                       \
  } while (0)
#else
#define ADEPT_VALIDATE(COND, MSG)                                                                \
  do {                                                                                           \
    if (ADEPT_UNLIKELY(!(COND))) {                                                               \
      ADEPT_RUNTIME_THROW("runtime", (::portability::detail::StreamlikeIdentity {} MSG), #COND); \
    }                                                                                            \
  } while (0)
#endif

// #define ADEPT_NOT_CONFIGURED(WHAT) ADEPT_RUNTIME_THROW("not configured", WHAT, nullptr)

/*!
 * \def ADEPT_DEVICE_API_CALL
 *
 * Safely and portably dispatch a CUDA/HIP API call.
 *
 * When CUDA or HIP support is enabled, execute the wrapped statement
 * prepend the argument with "cuda" or "hip" and throw a
 * std::runtime_error if it fails. If no device platform is enabled, throw an
 * unconfigured assertion.
 *
 * Example:
 *
 * \code
   ADEPT_DEVICE_API_CALL(Malloc(&ptr_gpu, 100 * sizeof(float)));
   ADEPT_DEVICE_API_CALL(DeviceSynchronize());
 * \endcode
 */
// #if defined(__HIP__) || defined(__CUDA_ARCH__)
#define ADEPT_DEVICE_API_CALL(STMT)                                                                                  \
  do {                                                                                                               \
    using ErrT_   = ADEPT_DEVICE_API_SYMBOL(Error_t);                                                                \
    ErrT_ result_ = ADEPT_DEVICE_API_SYMBOL(STMT);                                                                   \
    if (ADEPT_UNLIKELY(result_ != ADEPT_DEVICE_API_SYMBOL(Success))) {                                               \
      result_ = ADEPT_DEVICE_API_SYMBOL(GetLastError)();                                                             \
      ADEPT_RUNTIME_THROW(ADEPT_DEVICE_PLATFORM_UPPER_STR, ADEPT_DEVICE_API_SYMBOL(GetErrorString)(result_), #STMT); \
    }                                                                                                                \
  } while (0)
/*
#else
#define ADEPT_DEVICE_API_CALL(STMT)      \
 do {                                   \
    ADEPT_NOT_CONFIGURED("CUDA or HIP"); \
  } while (0)
#endif
*/
namespace portability {
//---------------------------------------------------------------------------//
// FUNCTION DECLARATIONS
//---------------------------------------------------------------------------//

// #ifndef __CUDA_ARCH__ // Not defined for code that runs on device
// [[nodiscard]] std::logic_error make_debug_error(char const *which, char const *condition, char const *file, int
// line);

// [[nodiscard]] std::runtime_error make_runtime_error(char const *which, char const *what, char const *condition,
//                                                     char const *file, int line);
// #endif

//---------------------------------------------------------------------------//
// FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//

// #ifndef ADEPT_DEVICE_COMPILE // Not defined for code that runs on device
// TODO: It doesn't make sense in principle that this file is included in a translation unit compiled
// by gcc, because gcc wouldn't be able to compile API calls either. Also, the macro changes between
// cuda/hip by detecting the compiler, so it won't even work with gcc. Then, where does the error if
// this is not done come from?
/*
Due to VecGeom's function definitions varying depending on the compiler used (Host or device compiler),
AdePT's code is compiled using two translation units. One includes all AdePT device headers and is
compiled by nvcc/hipcc, the other includes all other headers and is handled by the host compiler.
The macros defined in Portability.hh are needed in both translation units, in order to avoid multiple-definitions
of these functions, this code is only compiled by nvcc/hip, and the host compiler will only see the
declarations.
*/
// #ifdef ADEPT_DEVICE_SOURCE
#ifndef __CUDA_ARCH__
[[nodiscard]] inline std::logic_error make_debug_error(char const *which, char const *condition, char const *file,
                                                       int line)
{
  std::string msg;
  msg += which;
  msg += ": ";
  msg += condition;
  msg += " failed at ";
  msg += file;
  msg += ":";
  msg += std::to_string(line);
  return std::logic_error(std::move(msg));
}

[[nodiscard]] inline std::runtime_error make_runtime_error(char const *which, char const *what, char const *condition,
                                                           char const *file, int line)
{
  std::string msg;
  if (which) {
    msg += which;
  } else {
    msg += "unknown";
  }
  msg += " error: ";
  msg += what;
  msg += ": ";
  if (condition) {
    msg += condition;
    msg += " failed";
  }
  msg += " at ";
  msg += file;
  msg += ":";
  msg += std::to_string(line);
  return std::runtime_error(std::move(msg));
}
#endif
// #endif // ADEPT_DEVICE_SOURCE
// #endif // ADEPT_DEVICE_COMPILE

//---------------------------------------------------------------------------//
// INLINE FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//

//! Invoke undefined behavior
[[noreturn]] inline ADEPT_ATT_HOST_DEVICE void unreachable()
{
#if (!defined(__CUDA_ARCH__) && (defined(__clang__) || defined(__GNUC__))) || defined(__NVCOMPILER) || \
    (defined(__CUDA_ARCH__) && CUDART_VERSION >= 11030) || defined(__HIP_DEVICE_COMPILE__)
  __builtin_unreachable();
#elif defined(_MSC_VER)
  __assume(false);
#else
  ADEPT_UNREACHABLE;
#endif
}

#if defined(__CUDA_ARCH__) && defined(NDEBUG)
//! Host+device definition for CUDA when \c assert is unavailable
inline __attribute__((noinline)) __host__ __device__ void device_debug_error(char const *, char const *condition,
                                                                             char const *file, int line)
{
  printf("%s:%u:\nvecgeom: internal assertion failed: %s\n", file, line, condition);
  __trap();
}
#elif defined(__HIP__)
//! Host-only HIP call (whether or not NDEBUG is in use)
inline __host__ void device_debug_error(char const *which, char const *condition, char const *file, int line)
{
  throw make_debug_error(which, condition, file, line);
}

//! Device-only call for HIP (must always be declared; only used if
//! NDEBUG)
inline __attribute__((noinline)) __device__ void device_debug_error(char const *, char const *condition,
                                                                    char const *file, int line)
{
  printf("%s:%u:\nvecgeom: internal assertion failed: %s\n", file, line, condition);
  abort();
}
#endif

namespace detail {
//! Allow passing a single string into a streamlike operator for device-compatible ADEPT_VALIDATE messages
struct StreamlikeIdentity {
  ADEPT_ATT_HOST_DEVICE operator char const *() const { return ""; }
};
inline ADEPT_ATT_HOST_DEVICE char const *operator<<(StreamlikeIdentity const &, char const *s)
{
  return s;
}
} // namespace detail

} // namespace portability

#endif
