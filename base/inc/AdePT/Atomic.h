// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file Utils.h
 * @brief Portable atomic structure.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#ifndef ADEPT_ATOMIC_H_
#define ADEPT_ATOMIC_H_

#include <VecCore/CUDA.h>
#include <AdePT/Utils.h>
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
#include <atomic>
#endif

namespace adept {
/**
 * @brief A portable atomic type. Not all base types are supported by CUDA.
 */
template <typename Type>
struct Atomic_t {

  /** @brief Constructor taking an address */
  Atomic_t(void *addr) { EmplaceDataAt(addr); }

  /** @brief Emplace the data at a given address */
  char *EmplaceDataAt(void *addr)
  {
    if (addr) {
      fData = (AtomicType_t *)utils::round_up_align((char *)addr, alignof(AtomicType_t));
      store(0);
    }
    return (char *)fData;
  }

  /** @brief Compute size of the data, including the alignment offset */
  static size_t SizeOfData() { return (sizeof(AtomicType_t) + alignof(AtomicType_t)); }

#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  /// Standard library atomic behaviour.
  using AtomicType_t = std::atomic<Type>;

  /** @brief Atomically assigns the desired value to the atomic variable. */
  Type operator=(Type desired) { return fData->operator=(desired); }

  /** @brief Atomically replaces the current value with desired. */
  void store(Type desired) { fData->store(desired); }

  /** @brief Atomically loads and returns the current value of the atomic variable. */
  Type load() { return fData->load(); }

  /** @brief Atomically replaces the underlying value with desired. */
  Type exchange(Type desired) { return fData->exchange(desired); }

  /** @brief Atomically replaces the current value with the result of arithmetic addition of the value and arg. */
  Type fetch_add(Type arg) { return fData->fetch_add(arg); }

  /** @brief Atomically replaces the current value with the result of arithmetic subtraction of the value and arg. */
  Type fetch_sub(Type arg) { return fData->fetch_sub(arg); }

  /** @brief Atomically replaces the current value with the result of bitwise AND of the value and arg. */
  Type fetch_and(Type arg) { return fData->fetch_and(arg); }

  /** @brief Atomically replaces the current value with the result of bitwise OR of the value and arg. */
  Type fetch_or(Type arg) { return fData->fetch_or(arg); }

  /** @brief Atomically replaces the current value with the result of bitwise XOR of the value and arg. */
  Type fetch_xor(Type arg) { return fData->fetch_xor(arg); }

  /** @brief Atomically compares the stored value with the expected one, and if equal stores desired.
   * If not loads the old value into expected. Returns true if swap was successful. */
  bool compare_exchange_strong(Type &expected, Type desired)
  {
    return fData->compare_exchange_strong(expected, desired);
  }

  /** @brief Performs atomic pre-increment. */
  Type operator++() { return (fetch_add(1) + 1); }

  /** @brief Performs atomic post-increment. */
  Type operator++(int) { return fetch_add(1); }

  /** @brief Performs atomic pre-decrement. */
  Type operator--() { return (fetch_sub(1) - 1); }

  /** @brief Performs atomic post-decrement. */
  Type operator--(int) { return fetch_sub(1); }

#else
  /// CUDA atomic operations.
  using AtomicType_t = Type;

  /** @brief Atomically assigns the desired value to the atomic variable. */
  VECCORE_ATT_DEVICE
  Type operator=(Type desired)
  {
    atomicExch(fData, desired);
    return desired;
  }

  /** @brief Atomically replaces the current value with desired. */
  VECCORE_ATT_DEVICE
  void store(Type desired) { atomicExch(fData, desired); }

  /** @brief Atomically loads and returns the current value of the atomic variable. */
  VECCORE_ATT_DEVICE
  Type load() { return *fData; }

  /** @brief Atomically replaces the underlying value with desired. */
  VECCORE_ATT_DEVICE
  Type exchange(Type desired) { return atomicExch(fData, desired); }

  /** @brief Atomically replaces the current value with the result of arithmetic addition of the value and arg. */
  VECCORE_ATT_DEVICE
  Type fetch_add(Type arg) { return atomicAdd(fData, arg); }

  /** @brief Atomically replaces the current value with the result of arithmetic subtraction of the value and arg. */
  VECCORE_ATT_DEVICE
  Type fetch_sub(Type arg) { return atomicAdd(fData, -arg); }

  /** @brief Atomically replaces the current value with the result of bitwise AND of the value and arg. */
  VECCORE_ATT_DEVICE
  Type fetch_and(Type arg) { return atomicAnd(fData, arg); }

  /** @brief Atomically replaces the current value with the result of bitwise OR of the value and arg. */
  VECCORE_ATT_DEVICE
  Type fetch_or(Type arg) { return atomicOr(fData, arg); }

  /** @brief Atomically replaces the current value with the result of bitwise XOR of the value and arg. */
  VECCORE_ATT_DEVICE
  Type fetch_xor(Type arg) { return atomicXor(fData, arg); }

  /** @brief Atomically compares the stored value with the expected one, and if equal stores desired.
   * If not loads the old value into expected. Returns true if swap was successful. */
  VECCORE_ATT_DEVICE
  bool compare_exchange_strong(Type &expected, Type desired)
  {
    Type old    = atomicCAS(fData, expected, desired);
    bool worked = (old == expected);
    if (!worked) expected = *fData;
    return worked;
  }

  /** @brief Performs atomic pre-increment. */
  VECCORE_ATT_DEVICE
  Type operator++() { return (fetch_add(1) + 1); }

  /** @brief Performs atomic post-increment. */
  VECCORE_ATT_DEVICE
  Type operator++(int) { return fetch_add(1); }

  /** @brief Performs atomic pre-decrement. */
  VECCORE_ATT_DEVICE
  Type operator--() { return (fetch_sub(1) - 1); }

  /** @brief Performs atomic post-decrement. */
  VECCORE_ATT_DEVICE
  Type operator--(int) { return fetch_sub(1); }
#endif

  AtomicType_t *fData{nullptr}; ///< Atomic data

}; // End struct Atomic_t

} // End namespace adept
#endif // ADEPT_ATOMIC_H_
