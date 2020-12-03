// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file Utils.h
 * @brief Portable atomic structure.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#ifndef ADEPT_ATOMIC_H_
#define ADEPT_ATOMIC_H_

#include <CopCore/CopCore.h>
#include <cassert>

#ifndef DEVICE_COMPILATION_TRAJECTORY
#include <atomic>
#endif

namespace adept {
/**
 * @brief A portable atomic type. Not all base types are supported by CUDA.
 */
template <typename Type>
struct AtomicBase_t {

#ifndef DEVICE_COMPILATION_TRAJECTORY
  /// Standard library atomic behaviour.
  using AtomicType_t = std::atomic<Type>;
#else
  /// CUDA atomic operations.
  using AtomicType_t = Type;
#endif

  AtomicType_t fData{0}; ///< Atomic data

  /** @brief Constructor taking an address */
  __host__ __device__
  AtomicBase_t() : fData{0} {}

  /** @brief Copy constructor */
  __host__ __device__
  AtomicBase_t(AtomicBase_t const &other) { store(other.load()); }

  /** @brief Assignment */
  __host__ __device__
  AtomicBase_t &operator=(AtomicBase_t const &other) { store(other.load()); }

  /** @brief Emplace the data at a given address */
  __host__ __device__
  static AtomicBase_t *MakeInstanceAt(void *addr)
  {
    assert(addr != nullptr && "cannot allocate at nullptr address");
    assert((((unsigned long long)addr) % alignof(AtomicType_t)) == 0 && "addr does not satisfy alignment");
    AtomicBase_t *obj = new (addr) AtomicBase_t();
    return obj;
  }

  /// Implementation-dependent
#ifndef DEVICE_COMPILATION_TRAJECTORY

  /** @brief Atomically replaces the current value with desired. */
  void store(Type desired) { fData.store(desired); }

  /** @brief Atomically loads and returns the current value of the atomic variable. */
  Type load() const { return fData.load(); }

  /** @brief Atomically replaces the underlying value with desired. */
  Type exchange(Type desired) { return fData.exchange(desired); }

  /** @brief Atomically compares the stored value with the expected one, and if equal stores desired.
   * If not loads the old value into expected. Returns true if swap was successful. */
  bool compare_exchange_strong(Type &expected, Type desired)
  {
    return fData.compare_exchange_strong(expected, desired);
  }

#else

  /** @brief Atomically replaces the current value with desired. */
  __device__
  void store(Type desired) { atomicExch(&fData, desired); }

  /** @brief Atomically loads and returns the current value of the atomic variable. */
  __device__
  Type load() const
  {
    // There is no atomic load on CUDA. Issue a memory fence to get the required
    // semantics and make sure that the value is really loaded from memory.
    __threadfence();
    return fData;
  }

  /** @brief Atomically replaces the underlying value with desired. */
  __device__
  Type exchange(Type desired) { return atomicExch(&fData, desired); }

  /** @brief Atomically compares the stored value with the expected one, and if equal stores desired.
   * If not loads the old value into expected. Returns true if swap was successful. */
  __device__
  bool compare_exchange_strong(Type &expected, Type desired)
  {
    Type old    = atomicCAS(&fData, expected, desired);
    bool worked = (old == expected);
    if (!worked) expected = fData;
    return worked;
  }

  /** @brief Atomically replaces the current value with the result of arithmetic addition of the value and arg. */
  __device__
  Type fetch_add(Type arg) { return atomicAdd(&fData, arg); }

  /** @brief Atomically replaces the current value with the result of arithmetic subtraction of the value and arg. */
  __device__
  Type fetch_sub(Type arg) { return atomicAdd(&fData, -arg); }

  /** @brief Atomically replaces the current value with the result of bitwise AND of the value and arg. */
  __device__
  Type fetch_and(Type arg) { return atomicAnd(&fData, arg); }

  /** @brief Atomically replaces the current value with the result of bitwise OR of the value and arg. */
  __device__
  Type fetch_or(Type arg) { return atomicOr(&fData, arg); }

  /** @brief Atomically replaces the current value with the result of bitwise XOR of the value and arg. */
  __device__
  Type fetch_xor(Type arg) { return atomicXor(&fData, arg); }

  /** @brief Performs atomic add. */
  __device__
  Type operator+=(Type arg) { return (fetch_add(arg) + arg); }

  /** @brief Performs atomic subtract. */
  __device__
  Type operator-=(Type arg) { return (fetch_sub(arg) - arg); }

  /** @brief Performs atomic pre-increment. */
  __device__
  Type operator++() { return (fetch_add(1) + 1); }

  /** @brief Performs atomic post-increment. */
  __device__
  Type operator++(int) { return fetch_add(1); }

  /** @brief Performs atomic pre-decrement. */
  __device__
  Type operator--() { return (fetch_sub(1) - 1); }

  /** @brief Performs atomic post-decrement. */
  __device__
  Type operator--(int) { return fetch_sub(1); }

#endif
};

/** @brief Atomic_t generic implementation specialized using SFINAE mechanism */
template <typename Type, typename Enable = void>
struct Atomic_t;

/** @brief Specialization for integral types. */
template <typename Type>
struct Atomic_t<Type, typename std::enable_if<std::is_integral<Type>::value>::type> : public AtomicBase_t<Type> {
  using AtomicBase_t<Type>::fData;

#ifndef DEVICE_COMPILATION_TRAJECTORY
  /// Standard library atomic operations for integral types

  /** @brief Atomically assigns the desired value to the atomic variable. */
  Type operator=(Type desired) { return fData.operator=(desired); }

  /** @brief Atomically replaces the current value with the result of arithmetic addition of the value and arg. */
  Type fetch_add(Type arg) { return fData.fetch_add(arg); }

  /** @brief Atomically replaces the current value with the result of arithmetic subtraction of the value and arg. */
  Type fetch_sub(Type arg) { return fData.fetch_sub(arg); }

  /** @brief Atomically replaces the current value with the result of bitwise AND of the value and arg. */
  Type fetch_and(Type arg) { return fData.fetch_and(arg); }

  /** @brief Atomically replaces the current value with the result of bitwise OR of the value and arg. */
  Type fetch_or(Type arg) { return fData.fetch_or(arg); }

  /** @brief Atomically replaces the current value with the result of bitwise XOR of the value and arg. */
  Type fetch_xor(Type arg) { return fData.fetch_xor(arg); }

  /** @brief Performs atomic pre-increment. */
  Type operator+=(Type arg) { return (fetch_add(arg) + arg); }

  /** @brief Performs atomic subtract. */
  Type operator-=(Type arg) { return (fetch_sub(arg) - arg); }

  /** @brief Performs atomic pre-increment. */
  Type operator++() { return (fetch_add(1) + 1); }

  /** @brief Performs atomic post-increment. */
  Type operator++(int) { return fetch_add(1); }

  /** @brief Performs atomic pre-decrement. */
  Type operator--() { return (fetch_sub(1) - 1); }

  /** @brief Performs atomic post-decrement. */
  Type operator--(int) { return fetch_sub(1); }
#else
  /** @brief Atomically assigns the desired value to the atomic variable. */
  __device__
  Type operator=(Type desired)
  {
    atomicExch(&fData, desired);
    return desired;
  }

#endif

}; // End specialization for integral types of Atomic_t

/** @brief Specialization for non-integral types. */
template <typename Type>
struct Atomic_t<Type, typename std::enable_if<!std::is_integral<Type>::value>::type> : public AtomicBase_t<Type> {
  using Base_t = AtomicBase_t<Type>;
  using Base_t::fData;

#ifndef DEVICE_COMPILATION_TRAJECTORY

  /** @brief Atomically assigns the desired value to the atomic variable. */
  Type operator=(Type desired) { return fData.operator=(desired); }

  /** @brief Atomically replaces the current value with the result of arithmetic addition of the value and arg. */
  Type fetch_add(Type arg)
  {
    Type current = fData.load();
    while (!fData.compare_exchange_weak(current, current + arg))
      ;
    return current;
  }

  /** @brief Atomically replaces the current value with the result of arithmetic subtraction of the value and arg. */
  Type fetch_sub(Type arg) { return fetch_add(-arg); }

  /** @brief Performs atomic pre-increment. */
  Type operator+=(Type arg) { return (fetch_add(arg) + arg); }

  /** @brief Performs atomic subtract. */
  Type operator-=(Type arg) { return (fetch_sub(arg) - arg); }

  /** @brief Performs atomic pre-increment. */
  Type operator++() { return (fetch_add(1) + 1); }

  /** @brief Performs atomic post-increment. */
  Type operator++(int) { return fetch_add(1); }

  /** @brief Performs atomic pre-decrement. */
  Type operator--() { return (fetch_sub(1) - 1); }

  /** @brief Performs atomic post-decrement. */
  Type operator--(int) { return fetch_sub(1); }

#else

  /** @brief Atomically assigns the desired value to the atomic variable. */
  __device__
  Type operator=(Type desired)
  {
    atomicExch(&fData, desired);
    return desired;
  }

#endif
};

} // End namespace adept
#endif // ADEPT_ATOMIC_H_
