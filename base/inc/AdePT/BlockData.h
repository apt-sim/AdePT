// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file BlockData.h
 * @brief Templated data structure storing a contiguous block of data.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#ifndef ADEPT_BLOCKDATA_H_
#define ADEPT_BLOCKDATA_H_

#include <AdePT/Atomic.h>
#include <VecGeom/base/VariableSizeObj.h>

namespace adept {

/** @brief A container of data that adopts memory. It is caller responsibility to allocate
  at least SizeOfInstance bytes for a single object, and SizeOfAlignAware for multiple objects.
  Write access to data elements in the block is given atomically, up to the capacity.
 */
template <typename Type>
class BlockData : protected vecgeom::VariableSizeObjectInterface<BlockData<Type>, Type> {

public:
  using AtomicInt_t = adept::Atomic_t<int>;
  using Value_t     = Type;
  using Base_t      = vecgeom::VariableSizeObjectInterface<BlockData<Value_t>, Value_t>;
  using ArrayData_t = vecgeom::VariableSizeObj<Value_t>;

private:
  int fCapacity{0};     ///< Maximum number of elements
  AtomicInt_t fNbooked; ///< Number of booked elements
  AtomicInt_t fNused; ///< Number of used elements
  ArrayData_t fData;    ///< Data follows, has to be last

private:
  friend Base_t;

  /** @brief Functions required by VariableSizeObjectInterface */
  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  ArrayData_t &GetVariableData() { return fData; }

  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  const ArrayData_t &GetVariableData() const { return fData; }

  // constructors and assignment operators are private
  // states have to be constructed using MakeInstance() function
  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  BlockData(size_t nvalues) : fCapacity(nvalues), fData(nvalues) {}

  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  BlockData(size_t /*new_size*/, BlockData & /*other*/) {}

public:
  ///< Enumerate the part of the private interface, we want to expose.
  using Base_t::MakeCopy;
  using Base_t::MakeCopyAt;
  using Base_t::MakeInstance;
  using Base_t::MakeInstanceAt;
  using Base_t::ReleaseInstance;
  using Base_t::SizeOf;
  using Base_t::SizeOfAlignAware;

  /** @brief Returns the size in bytes of a BlockData object with given capacity */
  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  static size_t SizeOfInstance(int capacity) { return Base_t::SizeOf(capacity); }

  /** @brief Size of container in bytes */
  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  int SizeOf() const { return BlockData<Value_t>::SizeOfInstance(fCapacity); }

  /** @brief Maximum number of elements */
  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  int Capacity() const { return fCapacity; }

  /** @brief Clear the content */
  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  void Clear()
  {
    fNused.store(0);
    fNbooked.store(0);
  }

  /** @brief Read-only index operator */
  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  Type const &operator[](const int index) const { return fData[index]; }

  /** @brief Read/write index operator */
  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  Type &operator[](const int index) { return fData[index]; }

  /** @brief Dispatch next free element, nullptr if none left */
  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  Type *NextElement()
  {
    int index = fNbooked.fetch_add(1);
    if (index >= fCapacity) return nullptr;
    fNused++;
    return &fData[index];
  }

  /** @brief Number of elements currently distributed */
  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  int GetNused() { return fNused.load(); }

  /** @brief Check if container is fully distributed */
  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  bool IsFull() const { return (GetNused() == fCapacity); }

}; // End BlockData
} // End namespace adept

#endif // ADEPT_BLOCKDATA_H_
