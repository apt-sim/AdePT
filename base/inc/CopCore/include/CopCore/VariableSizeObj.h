// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file VariableSizeObj.h
 * @brief Array/vector of items with all information stored contiguously in memory.
 * @author Philippe Canal (pcanal@fnal.gov). Copied/adapted for VecGeom library (andrei.gheata@cern.ch).
 *
 * @details Resizing operations require explicit new creation and copy.
 * This is decomposed in the content part (VariableSizeObj) and the interface
 * part (VariableSizeObjectInterface) so that the variable size part can be placed
 * at the end of the object. Derived types must decorate the implementation methods with
 * appropriate __host__ __device__ qualifiers (or use the macros pre-defined in VecCore) for
 * usage on GPU.
 *
 * Use in derived classes
 * ======================
 * The derivation should be done on `VariableSizeObjectInterface` like:
 *
 *    class MyVariableSizeType : protected copcore::VariableSizeObjectInterface<MyVariableSizeType, V>
 *
 * where `V` is the variable data type. The derived class must have the LAST data member (say fData)
 * of type `copcore::VariableSizeObj<V>`. It has to declare the base class as friend to its private
 * methods, and has to mandatory implement the following private methods:
 *
 *    using ArrayData_t = copcore::VariableSizeObj<Value_t>;
 *    ArrayData_t &GetVariableData() { return fData; }
 *    const ArrayData_t &GetVariableData() const { return fData; }
 *
 * but also a private constructor initializing the capacity of the variable size array:
 *
 *    MyVariableSizeType(size_t nvalues, ...param) : ... , fData(nvalues)
 *
 * The derived class should have no public constructors, but should enumerate the public base class
 * static construction methods:
 *
 *    using Base_t = copcore::VariableSizeObjectInterface<MyVariableSizeType, V>;
 *    using Base_t::MakeInstance;
 *    using Base_t::MakeInstanceAt;
 *
 * These methods internally invoke the private constructor, their invokation being dependent on the
 * extra parameters taken by this constructor and explained in the `Usage` section.
 *
 * In case the derived type should be copiable, it has to expose the Base_t copy interfaces:
 *
 *      using Base_t::MakeCopy;
 *      using Base_t::MakeCopyAt;
 *
 * and implement private copy constructors:
 *
 *    MyVariableSizeType(MyVariableSizeType const &other) : MyVariableSizeType(other.fData.fN, other) {...}
 *    // Copy and resize
 *    MyVariableSizeType(size_t new_size, MyVariableSizeType const &other) : fData(new_size, other.fData) {...}
 *
 * Derived types must implement a public static method returning the size of objects for a given number
 * of elements, which must be called mandatory by the user before allocating memory for the object. This may
 * be just an alias to the SizeOf helper method from the base class as below, or has to be extended to handle
 * additional custom requirements:
 *
 *     static size_t SizeOfInstance(int capacity) { return Base_t::SizeOf(capacity); }
 *
 * Important note: the derived type should not contain any pointer data members that have to be dynamically
 * allocated, or if it does then it has to be supported in the constructors and in the SizeOfInstance method.
 * The base helper class is taylored for data member types for which the size can be computed at compilation
 * time (i.e. sizeof will return the actual object size).
 *
 * Composition using data members of types derived from VariableSizeObject is allowed. In such case it is
 * mandatory to implement the private method:
 *
 *    static size_t SizeOfExtra(int capacity) { ...; return extra_size; }
 *
 * The extra size has to be computed by summing the sizes of all VariableSizeObjectInterface-derived data
 * members as returned by their method SizeOfAlignedAware(capacity).
 *
 * Implementation examples for derived VariableSizeObjectInterface types can be seen
 * [here](https://github.com/apt-sim/AdePT/tree/master/base/inc/AdePT)
 *
 *
 *   Usage of VariableSize objects
 *   =============================
 *
 * The most important use case the helper was designed for is allocation of variable size objects in a
 * pre-allocated memory buffer. To do this, one needs to reteive the container size for the desired array
 * capacity via:
 *
 *    size_t buffer_size = MyVariableSizeType::SizeOfInstance(size_t capacity);
 *
 * or:
 *
 *    size_t buffer_size = MyVariableSizeType::SizeOfAlignAware(size_t capacity);
 *
 * The latter version has to be used for allocating multiple objects of type MyVariableSizeType in a
 * contiguous buffer.
 *
 *    char *buffer = nullptr;
 *    host_or_device_malloc_method(buffer, [N *] buffer_size);
 *
 * To allocate a single object, use:
 *
 *    auto obj = MyVariableSizeType::MakeInstanceAt(capacity, buffer, params...);
 *
 * where params... are the private constructor parameters. In case of multiple contiguous objects, the
 * padding in the buffer must be equal to buffer_size.
 *
 * In case standard allocation on the heap is needed, one has to use the method:
 *
 *    auto obj = MyVariableSizeType::MakeInstance(capacity, param...);
 *
 * The cleanup of VariableSize objects should not be done via `delete`, but rather using:
 *
 *    MyVariableSizeType::ReleaseInstance(obj);
 *
 * This will call the actual `delete` in case the object is allocated on the heap, or do nothing for the
 * buffer adoption case, when the buffer has to be de-allocated separately.
 *
 * Copying VariableSize objects is possible in case the copy constructors are defined as described above.
 * The copying can be done together with resizing the container, using:
 *
 *    auto copy = MyVariableSizeType::MakeCopy([new_size], MyVariableSizeType const &source); // using malloc
 *    auto copy = MyVariableSizeType::MakeCopy([new_size], MyVariableSizeType const &source, void *addr); // adopting
 * memory
 *
 * If omitting new_size above, the copy will have the same size as the source.
 */

#ifndef COPCORE_VARIABLESIZEOBJ_H
#define COPCORE_VARIABLESIZEOBJ_H

#include <CopCore/backend/BackendCommon.h>

// For memset and memcpy
#include <string.h>

class TRootIOCtor;

namespace copcore {

template <typename V>
class VariableSizeObj {
public:
  using Index_t = unsigned int;

  bool fSelfAlloc : 1;   //! Record whether the memory is externally managed.
  const Index_t fN : 31; // Number of elements.
  V fRealArray[1];       //! Beginning address of the array values -- real size: [fN]

  VariableSizeObj(TRootIOCtor *) : fSelfAlloc(false), fN(0) {}

  COPCORE_FORCE_INLINE
  __host__ __device__
  VariableSizeObj(unsigned int nvalues) : fSelfAlloc(false), fN(nvalues) {}

  COPCORE_FORCE_INLINE
  __host__ __device__
  VariableSizeObj(const VariableSizeObj &other) : fSelfAlloc(false), fN(other.fN)
  {
    if (other.fN) memcpy(GetValues(), other.GetValues(), (other.fN) * sizeof(V));
  }

  COPCORE_FORCE_INLINE
  __host__ __device__
  VariableSizeObj(size_t new_size, const VariableSizeObj &other) : fSelfAlloc(false), fN(new_size)
  {
    if (other.fN) memcpy(GetValues(), other.GetValues(), (other.fN) * sizeof(V));
  }

  COPCORE_FORCE_INLINE __host__ __device__ V *GetValues() { return &fRealArray[0]; }
  COPCORE_FORCE_INLINE __host__ __device__ const V *GetValues() const { return &fRealArray[0]; }

  COPCORE_FORCE_INLINE __host__ __device__ V &operator[](Index_t index) { return GetValues()[index]; };
  COPCORE_FORCE_INLINE __host__ __device__ const V &operator[](Index_t index) const { return GetValues()[index]; };

  COPCORE_FORCE_INLINE __host__ __device__ VariableSizeObj &operator=(const VariableSizeObj &rhs)
  {
    // Copy data content using memcpy, limited by the respective size
    // of the the object.  If this is smaller there is data truncation,
    // if this is larger the extra datum are zero to zero.

    if (rhs.fN == fN) {
      memcpy(GetValues(), rhs.GetValues(), rhs.fN * sizeof(V));
    } else if (rhs.fN < fN) {
      memcpy(GetValues(), rhs.GetValues(), rhs.fN * sizeof(V));
      memset(GetValues() + rhs.fN, 0, (fN - rhs.fN) * sizeof(V));
    } else {
      // Truncation!
      memcpy(GetValues(), rhs.GetValues(), fN * sizeof(V));
    }
    return *this;
  }
};

template <typename Cont, typename V>
class VariableSizeObjectInterface {
protected:
  VariableSizeObjectInterface()  = default;
  ~VariableSizeObjectInterface() = default;

public:
  // The static maker to be used to create an instance of the variable size object.

  template <typename... T>
  __host__ __device__ static Cont *MakeInstance(size_t nvalues, const T &... params)
  {
    // Make an instance of the class which allocates the node array. To be
    // released using ReleaseInstance.
    size_t needed = SizeOf(nvalues);
    char *ptr     = new char[needed];
    if (!ptr) return 0;
    assert((((unsigned long long)ptr) % alignof(Cont)) == 0 && "alignment error");
    Cont *obj                         = new (ptr) Cont(nvalues, params...);
    obj->GetVariableData().fSelfAlloc = true;
    return obj;
  }

  template <typename... T>
  __host__ __device__ static Cont *MakeInstanceAt(size_t nvalues, void *addr, const T &... params)
  {
    // Make an instance of the class which allocates the node array. To be
    // released using ReleaseInstance. If addr is non-zero, the user promised that
    // addr contains at least that many bytes:  size_t needed = SizeOf(nvalues);
    if (!addr) {
      return MakeInstance(nvalues, params...);
    } else {
      assert((((unsigned long long)addr) % alignof(Cont)) == 0 && "addr does not satisfy alignment");
      Cont *obj                         = new (addr) Cont(nvalues, params...);
      obj->GetVariableData().fSelfAlloc = false;
      return obj;
    }
  }

  // The equivalent of the copy constructor
  __host__ __device__
  static Cont *MakeCopy(const Cont &other)
  {
    // Make a copy of the variable size array and its container.

    size_t needed = SizeOf(other.GetVariableData().fN);
    char *ptr     = new char[needed];
    if (!ptr) return 0;
    Cont *copy                         = new (ptr) Cont(other);
    copy->GetVariableData().fSelfAlloc = true;
    return copy;
  }

  __host__ __device__
  static Cont *MakeCopy(size_t new_size, const Cont &other)
  {
    // Make a copy of a the variable size array and its container with
    // a new_size of the content.

    size_t needed = SizeOf(new_size);
    char *ptr     = new char[needed];
    if (!ptr) return 0;
    Cont *copy                         = new (ptr) Cont(new_size, other);
    copy->GetVariableData().fSelfAlloc = true;
    return copy;
  }

  // The equivalent of the copy constructor
  __host__ __device__
  static Cont *MakeCopyAt(const Cont &other, void *addr)
  {
    // Make a copy of a the variable size array and its container at the location (if indicated)
    if (addr) {
      Cont *copy                         = new (addr) Cont(other);
      copy->GetVariableData().fSelfAlloc = false;
      return copy;
    } else {
      return MakeCopy(other);
    }
  }

  // The equivalent of the copy constructor
  __host__ __device__
  static Cont *MakeCopyAt(size_t new_size, const Cont &other, void *addr)
  {
    // Make a copy of a the variable size array and its container at the location (if indicated)
    if (addr) {
      Cont *copy                         = new (addr) Cont(new_size, other);
      copy->GetVariableData().fSelfAlloc = false;
      return copy;
    } else {
      return MakeCopy(new_size, other);
    }
  }

  // The equivalent of the destructor
  __host__ __device__
  static void ReleaseInstance(Cont *obj)
  {
    // Releases the space allocated for the object
    obj->~Cont();
    if (obj->GetVariableData().fSelfAlloc) delete[](char *) obj;
  }

  // Equivalent of sizeof function (not taking into account padding for alignment)
  __host__ __device__
  static constexpr size_t SizeOf(size_t nvalues)
  {
    return (sizeof(Cont) + Cont::SizeOfExtra(nvalues) + sizeof(V) * (nvalues - 1));
  }

  // Size of the allocated derived type data members that are also variable size
  __host__ __device__
  static constexpr size_t SizeOfExtra(size_t nvalues) { return 0; }

  // equivalent of sizeof function taking into account padding for alignment
  // this function should be used when making arrays of VariableSizeObjects
  __host__ __device__
  static constexpr size_t SizeOfAlignAware(size_t nvalues) { return SizeOf(nvalues) + RealFillUp(nvalues); }

private:
  __host__ __device__
  static constexpr size_t FillUp(size_t nvalues) { return alignof(Cont) - SizeOf(nvalues) % alignof(Cont); }

  __host__ __device__
  static constexpr size_t RealFillUp(size_t nvalues)
  {
    return (FillUp(nvalues) == alignof(Cont)) ? 0 : FillUp(nvalues);
  }
};
} // namespace copcore

#endif //  COPCORE_VARIABLESIZEOBJ_H
