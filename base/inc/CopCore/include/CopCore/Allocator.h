// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file Allocator.h
 * @brief Backend-aware allocator for objects and arrays of types not doing internal allocations.
 * @author Andrei Gheata (andrei.gheata@cern.ch).
 *
 * @details A standard allocator providing allocate/deallocate interface.
 * Specializations are provided for CPU, CUDA and *TODO* HIP.
 */

#include <cstddef>
#include <stdexcept>
#include <iostream>

#include <CopCore/backend/BackendCommon.h>

namespace copcore {

template <class T, BackendType backend>
class Allocator {
};

#ifdef TARGET_DEVICE_CUDA

/** @brief Partial allocator specialization for the CUDA backend */
template <class T>
class Allocator<T, BackendType::CUDA> {
public:
  using value_type = T;

  Allocator(int device = 0) : fDeviceId(device) {}

  Allocator(const Allocator &) = default;

  template <class U>
  Allocator(const Allocator<U, BackendType::CUDA> &other) : fDeviceId(other.device())
  {
  }

  bool operator==(const Allocator &other) const { return fDeviceId == other.fDeviceId; }

  bool operator!=(const Allocator &other) const { return !(*this == other); }

  template <typename... P>
  value_type *allocate(std::size_t n, const P &... params) const
  {
    int old_device = SetDevice(fDeviceId);

    value_type *result = nullptr;
    auto obj_size      = sizeof(T);
    COPCORE_CUDA_CHECK(cudaMallocManaged(&result, n * obj_size));
    value_type *current = result;

    // allocate all objects at their aligned positions in the buffer
    for (auto i = 0; i < n; ++i)
      new (current++) T(params...);

    SetDevice(old_device);

    return result;
  }

  void deallocate(value_type *ptr, std::size_t n = 0) const
  {
    int old_device = SetDevice(fDeviceId);

    // Call destructor for all allocated objects
    value_type *current = ptr;
    for (auto i = 0; i < n; ++i) {
      current->~T();
      current++;
    }

    // Release the memory
    COPCORE_CUDA_CHECK(cudaFree(ptr));
    SetDevice(old_device);
  }

  int GetDevice() const { return fDeviceId; }

private:
  static int SetDevice(int new_device)
  {
    int old_device = -1;
    COPCORE_CUDA_CHECK(cudaGetDevice(&old_device));
    COPCORE_CUDA_CHECK(cudaSetDevice(new_device));
    return old_device;
  }

  int fDeviceId{0}; ///< Device id
};
#endif

/** @brief Partial allocator specialization for the CPU backend */
template <class T>
class Allocator<T, BackendType::CPU> {
public:
  using value_type = T;

  Allocator(int device = 0) : fDeviceId(device) {}

  Allocator(const Allocator &) = default;

  template <class U>
  Allocator(const Allocator<U, BackendType::CPU> &other) : fDeviceId(other.device())
  {
  }

  bool operator==(const Allocator &other) const { return fDeviceId == other.fDeviceId; }

  bool operator!=(const Allocator &other) const { return !(*this == other); }

  template <typename... P>
  value_type *allocate(std::size_t n, const P &... params) const
  {
    auto obj_size       = sizeof(T);
    value_type *result  = (value_type *)malloc(n * obj_size);
    value_type *current = result;

    // allocate all objects at their aligned positions in the buffer
    for (auto i = 0; i < n; ++i)
      new (current++) T(params...);

    return result;
  }

  void deallocate(value_type *ptr, std::size_t n = 0) const
  {
    // Call destructor for all allocated objects
    value_type *current = ptr;
    for (auto i = 0; i < n; ++i) {
      current->~T();
      current++;
    }

    // Release the memory
    free(ptr);
  }

  int GetDevice() const { return fDeviceId; }

private:
  int fDeviceId{0}; ///< Device id
};

} // End namespace copcore
