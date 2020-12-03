// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file MParray.h
 * @brief Multi-producer array that can be filled concurrently
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#ifndef ADEPT_MPARRAY_H_
#define ADEPT_MPARRAY_H_

#include <CopCore/CopCore.h>
#include <AdePT/Atomic.h>

namespace adept {

/** @brief A variable-size array having elements added in an atomic way */
class MParray : protected copcore::VariableSizeObjectInterface<MParray, int> {
public:
  using value_type      = int;
  using pointer         = value_type *;
  using const_pointer   = const value_type *;
  using reference       = value_type &;
  using const_reference = const value_type &;
  using iterator        = value_type *;
  using const_iterator  = const value_type *;
  using size_t          = std::size_t;
  using AtomicInt_t     = adept::Atomic_t<int>;
  using Base_t          = copcore::VariableSizeObjectInterface<MParray, int>;
  using ArrayData_t     = copcore::VariableSizeObj<int>;

private:
  size_t fCapacity{0};  ///< Maximum number of elements
  AtomicInt_t fNbooked; ///< Number of booked elements
  AtomicInt_t fNused;   ///< Number of used elements
  ArrayData_t fData;    ///< Data follows, has to be last

private:
  friend Base_t;

  /** @brief Functions required by VariableSizeObjectInterface */
  __host__ __device__
  COPCORE_FORCE_INLINE
  ArrayData_t &GetVariableData() { return fData; }

  __host__ __device__
  COPCORE_FORCE_INLINE
  const ArrayData_t &GetVariableData() const { return fData; }

  // constructors and assignment operators are private
  // states have to be constructed using MakeInstance() function
  __host__ __device__
  COPCORE_FORCE_INLINE
  MParray(size_t nvalues) : fCapacity(nvalues), fData(nvalues) {}

  __host__ __device__
  COPCORE_FORCE_INLINE
  MParray(MParray const &other) : MParray(other.fCapacity, other) {}

  __host__ __device__
  COPCORE_FORCE_INLINE
  MParray(size_t new_size, MParray const &other) : Base_t(other), fCapacity(new_size), fData(new_size, other.fData) {}

  COPCORE_FORCE_INLINE
  __host__ __device__
  ~MParray() {}

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
  __host__ __device__
  COPCORE_FORCE_INLINE
  size_t size() const { return fNused.load(); }

  /** @brief Maximum number of elements */
  __host__ __device__
  COPCORE_FORCE_INLINE
  constexpr size_t max_size() const { return fCapacity; }

  /** @brief Clear the content */
  __host__ __device__
  COPCORE_FORCE_INLINE
  void clear()
  {
    fNused.store(0);
    fNbooked.store(0);
  }

  /** @brief Read-only index operator */
  __host__ __device__
  COPCORE_FORCE_INLINE
  const_reference operator[](size_t index) const { return fData[index]; }

  /** @brief Dispatch next free element, nullptr if none left */
  __host__ __device__
  COPCORE_FORCE_INLINE
  bool push_back(value_type val)
  {
    // Operation may fail if the max size is exceeded. Has to be checked by the user.
    int index = fNbooked.fetch_add(1);
    if (index >= fCapacity) return false;
    fData[index] = val;
    fNused++;
    return true;
  }

  /** @brief Check if container is fully distributed */
  __host__ __device__
  COPCORE_FORCE_INLINE
  bool full() const { return (size() == fCapacity); }

  __host__ __device__
  COPCORE_FORCE_INLINE
  const_iterator begin() const { return const_iterator(&fData[0]); }

  __host__ __device__
  COPCORE_FORCE_INLINE
  const_iterator end() const { return const_iterator(&fData[fNused.load()]); }

  __host__ __device__
  COPCORE_FORCE_INLINE
  const_reference front() const { return *begin(); }

  __host__ __device__
  COPCORE_FORCE_INLINE
  const_reference back() const { return fCapacity ? *(end() - 1) : *end(); }

  __host__ __device__
  COPCORE_FORCE_INLINE
  const_pointer data() const { return &fData[0]; }

  /** @brief Returns the size in bytes of a BlockData object with given capacity */
  __host__ __device__
  COPCORE_FORCE_INLINE
  static size_t SizeOfInstance(int capacity) { return Base_t::SizeOf(capacity); }

}; // End class MParray
} // End namespace adept

#endif // ADEPT_MPARRAY_H_
