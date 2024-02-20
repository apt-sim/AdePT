// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file MParrayT.h
 * @brief Multi-producer array of arbitrary type that can be filled concurrently
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#ifndef ADEPT_MPARRAYT_H_
#define ADEPT_MPARRAYT_H_

#include <AdePT/copcore/CopCore.h>
#include <AdePT/base/Atomic.h>

namespace adept {

/** @brief A variable-size array having elements added in an atomic way */
template <typename T>
class MParrayT : protected copcore::VariableSizeObjectInterface<MParrayT<T>, T> {
public:
  using value_type      = T;
  using pointer         = value_type *;
  using const_pointer   = const value_type *;
  using reference       = value_type &;
  using const_reference = const value_type &;
  using iterator        = value_type *;
  using const_iterator  = const value_type *;
  using size_t          = std::size_t;
  using AtomicInt_t     = adept::Atomic_t<int>;
  using Base_t          = copcore::VariableSizeObjectInterface<MParrayT<T>, T>;
  using ArrayData_t     = copcore::VariableSizeObj<T>;

private:
  size_t fCapacity{0};  ///< Maximum number of elements
  AtomicInt_t fNbooked; ///< Number of booked elements
  AtomicInt_t fNused;   ///< Number of used elements
  ArrayData_t fData;    ///< Data follows, has to be last

private:
  friend Base_t;

  /** @brief Functions required by VariableSizeObjectInterface */
  __host__ __device__ __forceinline__ ArrayData_t &GetVariableData() { return fData; }

  __host__ __device__ __forceinline__ const ArrayData_t &GetVariableData() const { return fData; }

  // constructors and assignment operators are private
  // states have to be constructed using MakeInstance() function
  __host__ __device__ __forceinline__ MParrayT(size_t nvalues) : fCapacity(nvalues), fData(nvalues) {}

  __host__ __device__ __forceinline__ MParrayT(MParrayT const &other) : MParrayT(other.fCapacity, other) {}

  __host__ __device__ __forceinline__ MParrayT(size_t new_size, MParrayT const &other)
      : Base_t(other), fCapacity(new_size), fData(new_size, other.fData)
  {
  }

  __forceinline__ __host__ __device__ ~MParrayT() {}

public:
  ///< Enumerate the part of the private interface, we want to expose.
  using Base_t::MakeCopy;
  using Base_t::MakeCopyAt;
  using Base_t::MakeInstance;
  using Base_t::MakeInstanceAt;
  using Base_t::ReleaseInstance;
  using Base_t::SizeOf;
  using Base_t::SizeOfAlignAware;

  /** @brief Maximum number of elements */
  __host__ __device__ __forceinline__ size_t size() const { return fNused.load(); }

  /** @brief Maximum number of elements */
  __host__ __device__ __forceinline__ constexpr size_t max_size() const { return fCapacity; }

  /** @brief Clear the content */
  __host__ __device__ __forceinline__ void clear()
  {
    fNused.store(0);
    fNbooked.store(0);
  }

  /** @brief Read-only index operator */
  __host__ __device__ __forceinline__ const_reference operator[](size_t index) const { return fData[index]; }

  /** @brief Dispatch next free element, nullptr if none left */
  __host__ __device__ __forceinline__ bool push_back(const_reference val)
  {
    // Operation may fail if the max size is exceeded. Has to be checked by the user.
    int index = fNbooked.fetch_add(1);
    if (index >= fCapacity) return false;
    fData[index] = val;
    fNused++;
    return true;
  }

  /** @brief Check if container is fully distributed */
  __host__ __device__ __forceinline__ bool full() const { return (size() == fCapacity); }

  __host__ __device__ __forceinline__ const_iterator begin() const { return const_iterator(&fData[0]); }

  __host__ __device__ __forceinline__ const_iterator end() const { return const_iterator(&fData[fNused.load()]); }

  __host__ __device__ __forceinline__ const_reference front() const { return *begin(); }

  __host__ __device__ __forceinline__ const_reference back() const { return fCapacity ? *(end() - 1) : *end(); }

  __host__ __device__ __forceinline__ const_pointer data() const { return &fData[0]; }

  /** @brief Returns the size in bytes of a BlockData object with given capacity */
  __host__ __device__ __forceinline__ static size_t SizeOfInstance(int capacity) { return Base_t::SizeOf(capacity); }

}; // End class MParrayT
} // End namespace adept

#endif // ADEPT_MPARRAYT_H_
