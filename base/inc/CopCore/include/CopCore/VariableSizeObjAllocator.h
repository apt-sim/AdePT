// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file VariableSizeObjAllocator.h
 * @brief Backend-aware allocator for variable size objects
 * @author Andrei Gheata (andrei.gheata@cern.ch).
 *
 * @details A standard allocator providing allocate/deallocate interface
 * for VariableSizeObj objects. Specializations are provided for CPU, CUDA and *TODO* HIP.
 */

#include <cstddef>
#include <stdexcept>
#include <iostream>

#include <CopCore/backend/BackendCommon.h>

namespace copcore {

template <class T, BackendType backend>
class VariableSizeObjAllocator {
};

#ifdef TARGET_DEVICE_CUDA

/** @brief Partial variable-size allocator specialization for the CUDA backend */
template <class T>
class VariableSizeObjAllocator<T, BackendType::CUDA> {
public:
  using value_type = T;

  VariableSizeObjAllocator(std::size_t capacity, int device = 0) : fCapacity(capacity), fDeviceId(device) {}

  VariableSizeObjAllocator() : VariableSizeObjAllocator(0, 0) {}

  VariableSizeObjAllocator(const VariableSizeObjAllocator &) = default;

  template <class U>
  VariableSizeObjAllocator(const VariableSizeObjAllocator<U, BackendType::CUDA> &other) : fDeviceId(other.device())
  {
  }

  bool operator==(const VariableSizeObjAllocator &other) const { return fDeviceId == other.fDeviceId; }

  bool operator!=(const VariableSizeObjAllocator &other) const { return !(*this == other); }

  template <typename... P>
  value_type *allocate(std::size_t n, const P &... params) const
  {
    int old_device = SetDevice(fDeviceId);

    value_type *result   = nullptr;
    std::size_t obj_size = T::SizeOfAlignAware(fCapacity);
    COPCORE_CUDA_CHECK(cudaMallocManaged(&result, n * obj_size));
    char *buff = (char *)result;

    // allocate all objects at their aligned positions in the buffer
    for (auto i = 0; i < n; ++i) {
      T::MakeInstanceAt(fCapacity, buff, params...);
      buff += obj_size;
    }

    SetDevice(old_device);

    return result;
  }

  void deallocate(value_type *ptr, std::size_t n = 0) const
  {
    int old_device       = SetDevice(fDeviceId);
    std::size_t obj_size = T::SizeOfAlignAware(fCapacity);
    char *buff           = (char *)ptr;

    // Call destructor for all allocated objects
    for (auto i = 0; i < n; ++i) {
      T::ReleaseInstance((T *)buff);
      buff += obj_size;
    }

    // Release the memory
    COPCORE_CUDA_CHECK(cudaFree(ptr));
    SetDevice(old_device);
  }

  int GetDevice() const { return fDeviceId; }

  void SetCapacity(std::size_t capacity) { fCapacity = capacity; }

private:
  static int SetDevice(int new_device)
  {
    int old_device = -1;
    COPCORE_CUDA_CHECK(cudaGetDevice(&old_device));
    COPCORE_CUDA_CHECK(cudaSetDevice(new_device));
    return old_device;
  }

  std::size_t fCapacity{0}; ///< Capacity of each VariableSizeObj container
  int fDeviceId{0};         ///< Device id
};
#endif

/** @brief Partial variable-size allocator specialization for the CPU backend */
template <class T>
class VariableSizeObjAllocator<T, BackendType::CPU> {
public:
  using value_type = T;

  VariableSizeObjAllocator(std::size_t capacity, int device = 0) : fCapacity(capacity), fDeviceId(device) {}

  VariableSizeObjAllocator() : VariableSizeObjAllocator(0, 0) {}

  VariableSizeObjAllocator(const VariableSizeObjAllocator &) = default;

  template <class U>
  VariableSizeObjAllocator(const VariableSizeObjAllocator<U, BackendType::CPU> &other) : fDeviceId(other.device())
  {
  }

  bool operator==(const VariableSizeObjAllocator &other) const { return fDeviceId == other.fDeviceId; }

  bool operator!=(const VariableSizeObjAllocator &other) const { return !(*this == other); }

  template <typename... P>
  value_type *allocate(std::size_t n, const P &... params) const
  {
    value_type *result   = nullptr;
    std::size_t obj_size = T::SizeOfAlignAware(fCapacity);
    result               = (value_type *)malloc(n * obj_size);
    char *buff           = (char *)result;

    // allocate all objects at their aligned positions in the buffer
    for (auto i = 0; i < n; ++i) {
      T::MakeInstanceAt(fCapacity, buff, params...);
      buff += obj_size;
    }

    return result;
  }

  void deallocate(value_type *ptr, std::size_t n = 0) const
  {
    std::size_t obj_size = T::SizeOfAlignAware(fCapacity);
    char *buff           = (char *)ptr;

    // Call destructor for all allocated objects
    for (auto i = 0; i < n; ++i) {
      T::ReleaseInstance((T *)buff);
      buff += obj_size;
    }

    // Release the memory
    free(ptr);
  }

  int GetDevice() const { return fDeviceId; }

  void SetCapacity(std::size_t capacity) { fCapacity = capacity; }

private:
  std::size_t fCapacity{0}; ///< Capacity of each VariableSizeObj container
  int fDeviceId{0};         ///< Device id
};

} // End namespace copcore
