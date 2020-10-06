// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file mpmc_bounded_queue.h
 * @brief Implementation of CUDA-aware multi-producer, multi-consumer bounded queue
 * @author Andrei Gheata
 * based on http://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue
 */

#ifndef ADEPT_MPMC_BOUNDED_QUEUE
#define ADEPT_MPMC_BOUNDED_QUEUE

#include <stdint.h>
#include <cassert>
#include <AdePT/Atomic.h>
#include <VecGeom/base/VariableSizeObj.h>

namespace adept {
namespace internal {
/** @brief Internal data structure to handle the data sequence */
template <typename Type>
struct Cell_t {
  adept::Atomic_t<int> fSequence; ///< Atomic sequence counter
  Type fData;                     ///< Data stored in the cell

  /** @brief Cell constructor */
  Cell_t() {}
};
} // namespace internal

/** @brief Class MPMC bounded queue */
template <typename Type>
class mpmc_bounded_queue
    : protected vecgeom::VariableSizeObjectInterface<mpmc_bounded_queue<Type>, internal::Cell_t<Type>> {
public:
  using AtomicInt_t = adept::Atomic_t<int>;
  using Value_t     = internal::Cell_t<Type>;
  using Base_t      = vecgeom::VariableSizeObjectInterface<mpmc_bounded_queue<Type>, Value_t>;
  using ArrayData_t = vecgeom::VariableSizeObj<Value_t>;

private:
  int const fCapacity;  ///< Capacity of the queue
  int const fMask;      ///< Mask used to navigate fast in the circular buffer
  AtomicInt_t fEnqueue; ///< Enqueueing index
  AtomicInt_t fDequeue; ///< Dequeueing index
  AtomicInt_t fNstored; ///< Number of stored elements
  ArrayData_t fBuffer;  ///< Buffer of cells with atomic access to data elements

private:
  friend Base_t;

  /** @brief Functions required by VariableSizeObjectInterface */
  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  ArrayData_t &GetVariableData() { return fBuffer; }

  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  const ArrayData_t &GetVariableData() const { return fBuffer; }

  // constructors and assignment operators are private
  // states have to be constructed using MakeInstance() function

  /**
   * @brief MPMC bounded queue constructor
   * @param fBuffersize Maximum number of elements in the queue
   * @param addr Address where the queue is allocated.
   * The user should make sure to allocate at least SizeofInstance(fBuffersize) bytes
   */
  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  mpmc_bounded_queue(int nvalues) : fCapacity(nvalues), fMask(nvalues - 1), fBuffer(nvalues)
  {
    // The queue size must be a power of 2 (for fast access)
    assert((nvalues >= 2) && ((nvalues & (nvalues - 1)) == 0) && "buffer size has to be a power of 2");
    for (int i = 0; i < nvalues; ++i)
      fBuffer[i].fSequence.store(i);
  }

  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  mpmc_bounded_queue(int /*nvalues*/, mpmc_bounded_queue const & /*other*/) {}
  /** @brief MPMC bounded queue copy constructor */
  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  mpmc_bounded_queue(mpmc_bounded_queue const &other) : mpmc_bounded_queue(other.fCapacity, other) {}

  /** @brief MPMC bounded queue copy constructor with given size */
  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  mpmc_bounded_queue(size_t new_size, mpmc_bounded_queue const &other)
      : fCapacity(new_size), fMask(new_size - 1), fEnqueue(other.fEnqueue), fDequeue(other.fDequeue),
        fNstored(other.fNstored), fBuffer(new_size, other.fBuffer)
  {
    assert((new_size >= 2) && ((new_size & (new_size - 1)) == 0) && "buffer size has to be a power of 2");
  }

  /** @brief Operator = */
  void operator=(mpmc_bounded_queue const &) = delete;

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

  /** @brief Maximum number of elements */
  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  int Capacity() const { return fCapacity; }

  /** @brief Size function */
  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  int size() const { return fNstored.load(); }

  /** @brief MPMC enqueue function */
  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  bool enqueue(Type const &data)
  {
    Value_t *cell;
    int pos = fEnqueue.load();
    for (;;) {
      cell         = &fBuffer[pos & fMask];
      int seq      = cell->fSequence.load();
      intptr_t dif = (intptr_t)seq - (intptr_t)pos;
      if (dif == 0) {
        if (fEnqueue.compare_exchange_strong(pos, pos + 1)) break;
      } else if (dif < 0)
        return false;
      else
        pos = fEnqueue.load();
    }
    cell->fData = data;
    fNstored++;
    cell->fSequence.store(pos + 1);
    return true;
  }

  /** @brief MPMC dequeue function */
  VECCORE_ATT_HOST_DEVICE
  VECCORE_FORCE_INLINE
  bool dequeue(Type &data)
  {
    Value_t *cell;
    int pos = fDequeue.load();
    for (;;) {
      cell         = &fBuffer[pos & fMask];
      int seq      = cell->fSequence.load();
      intptr_t dif = (intptr_t)seq - (intptr_t)(pos + 1);
      if (dif == 0) {
        if (fDequeue.compare_exchange_strong(pos, pos + 1)) break;
      } else if (dif < 0)
        return false;
      else
        pos = fDequeue.load();
    }
    data = cell->fData;
    fNstored--;
    cell->fSequence.store(pos + fMask + 1);
    return true;
  }
}; // End mpmc_bounded_queue
} // End namespace adept
#endif // ADEPT_MPMC_BOUNDED_QUEUE
